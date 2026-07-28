# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.gpu.block_table import BlockTables, _load_ptr
from vllm.utils.math_utils import cdiv


# GPUModelRunner.initialize_kv_cache() only passes scalar block-table
# dimensions to BlockTables. DeepSeek-V4 needs the original group specs as
# well: compressed cache groups have a smaller logical block-table capacity.
# Keep this scoped to initialization so the upstream runner control flow does
# not need to be copied into vllm-ascend.
_KV_CACHE_GROUPS_FOR_BLOCK_TABLE: ContextVar[
    tuple[KVCacheGroupSpec, ...] | None
] = ContextVar("kv_cache_groups_for_ascend_block_table", default=None)


@contextmanager
def block_table_kv_cache_groups_context(kv_cache_groups: Sequence[KVCacheGroupSpec]):
    token = _KV_CACHE_GROUPS_FOR_BLOCK_TABLE.set(tuple(kv_cache_groups))
    try:
        yield
    finally:
        _KV_CACHE_GROUPS_FOR_BLOCK_TABLE.reset(token)


def _get_compress_ratio(kv_cache_group: KVCacheGroupSpec) -> int:
    """Get the one compression ratio represented by a KV-cache group."""
    spec = kv_cache_group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        ratios = {
            max(int(getattr(item, "compress_ratio", 1) or 1), 1)
            for item in spec.kv_cache_specs.values()
        }
        if len(ratios) != 1:
            raise ValueError(
                "A KV cache group must have one compress_ratio for block-table "
                f"addressing, got {sorted(ratios)}."
            )
        return ratios.pop()
    return max(int(getattr(spec, "compress_ratio", 1) or 1), 1)


class AscendBlockTables(BlockTables):
    """Block table for Ascend NPUs."""

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int] | None = None,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ):
        if kernel_block_sizes is None:
            kernel_block_sizes = block_sizes

        kv_cache_groups = _KV_CACHE_GROUPS_FOR_BLOCK_TABLE.get()
        if kv_cache_groups is None:
            # Non-DSA callers keep the upstream behavior. The V2 patch is
            # process-wide, so this fallback is needed for other models.
            compress_ratios = [1] * len(block_sizes)
        else:
            if len(kv_cache_groups) != len(block_sizes):
                raise ValueError(
                    "KV cache group count must match block-table group count: "
                    f"{len(kv_cache_groups)} != {len(block_sizes)}"
                )
            compress_ratios = [_get_compress_ratio(group) for group in kv_cache_groups]

        # Mirror MRv1 BlockTable: the scheduler's DCP/alignment-adjusted
        # capacity is converted to the compressed cache's logical capacity
        # before BlockTables expands physical blocks into kernel blocks.
        max_num_blocks_per_group = [
            max(cdiv(num_blocks, ratio), 1)
            for num_blocks, ratio in zip(max_num_blocks_per_group, compress_ratios)
        ]

        super().__init__(
            block_sizes,
            max_num_reqs,
            max_num_batched_tokens,
            max_num_blocks_per_group,
            device,
            kernel_block_sizes,
            cp_size,
            cp_rank,
            cp_interleave,
        )
        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.slot_mappings
        # vllm-ascend' reshape_and_cache function requires slot_mappings to be int32.
        # so we need to redefine slot_mappings to be int32.
        self.slot_mappings: torch.Tensor = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.compress_ratios = compress_ratios
        # The DSA compressor custom op owns c4/c128 compressed slot mapping.
        # Generic V2 slot mapping must not index those shortened block tables
        # with raw token positions.
        self.generic_slot_mapping_mask = torch.tensor(
            [ratio == 1 for ratio in compress_ratios],
            dtype=torch.bool,
            device=self.device,
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            self.max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.generic_slot_mapping_mask,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return self.slot_mappings[:, :num_tokens_padded]


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    generic_slot_mapping_mask,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)
    use_generic_slot_mapping = tl.load(generic_slot_mapping_mask + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        # Type conversion of 'position' to int32 to be compatible with npu
        # otherwise, it will degrade to scalar computation
        positions = positions.to(tl.int32)
        block_indices = positions // (block_size * CP_SIZE)

        # block_offset = positions % (block_size * CP_SIZE)
        # The % operation on int32 type will degrade to scalar computation
        # replace the % operation with sub and mul instead
        block_offsets = positions - (block_size * CP_SIZE) * block_indices

        # c4/c128 cache groups use a compressed table. Their exact compressed
        # slot mapping is produced by the DSA compressor custom op, so the
        # generic V2 mapping must be PAD and, critically, must not read the
        # shortened table using raw positions. For ordinary/SWA groups retain
        # upstream addressing and mask the table boundary.
        valid_slot = (
            (offset < end_idx)
            & use_generic_slot_mapping
            & (block_indices < block_table_stride)
        )
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=valid_slot,
            other=0,
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        slot_ids = tl.where(valid_slot, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
