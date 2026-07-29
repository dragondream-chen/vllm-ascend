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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.gpu.block_table import (
    BlockTables,
    _compute_slot_mappings_kernel as upstream_compute_slot_mappings_kernel,
)
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
        self.generic_slot_mapping_groups = [
            group_id for group_id, ratio in enumerate(compress_ratios) if ratio == 1
        ]
    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        # c4/c128 compressed groups do not consume the generic slot mapping:
        # the DSA compressor custom op produces their exact mapping. Fill them
        # with PAD and only launch the generic kernel for ratio==1 groups.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        for group_id in self.generic_slot_mapping_groups:
            # Reuse the upstream kernel unchanged. Passing one-group views
            # makes its program_id(0) remain zero while still targeting this
            # group's persistent table and int32 Ascend slot-mapping row.
            upstream_compute_slot_mappings_kernel[(1, num_reqs + 1)](
                self.max_num_batched_tokens,
                idx_mapping,
                query_start_loc,
                positions,
                self.block_table_ptrs[group_id : group_id + 1],
                self.block_table_strides[group_id : group_id + 1],
                self.block_sizes_tensor[group_id : group_id + 1],
                self.slot_mappings[group_id : group_id + 1],
                self.slot_mappings[group_id : group_id + 1].stride(0),
                self.cp_rank,
                CP_SIZE=self.cp_size,
                CP_INTERLEAVE=self.cp_interleave,
                PAD_ID=PAD_SLOT_ID,
                TRITON_BLOCK_SIZE=1024,  # type: ignore
            )
        return self.slot_mappings[:, :num_tokens_padded]
