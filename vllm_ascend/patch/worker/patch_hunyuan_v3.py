#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Patch Hunyuan V3 to use NPU fused RoPE+Norm+KV FP8 path."""

from __future__ import annotations

import torch

from vllm import envs as vllm_envs
from vllm.config import get_current_vllm_config
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.models.hunyuan_v3 import HYV3Attention
from vllm.v1.worker.tpsp_utils import fused_all_gather_qkv_proj

from vllm_ascend.ops.rope_norm import NpuRopeNorm

logger = init_logger(__name__)

_original_init = HYV3Attention.__init__
_original_forward = HYV3Attention.forward


def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)

    vllm_config = get_current_vllm_config()
    cache_config = vllm_config.cache_config
    kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"

    self.npu_rope_norm: NpuRopeNorm | None = None
    if (
        vllm_envs.VLLM_PRECISION_MODE == "default"
        and kv_cache_dtype in ("fp8", "fp8_e4m3")
        and NpuRopeNorm.support(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            kv_cache_dtype,
        )
        and hasattr(self, "rotary_emb")
    ):
        self.npu_rope_norm = NpuRopeNorm(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            use_qk_norm=self.use_qk_norm,
            fallback_qnorm=self.q_norm if self.use_qk_norm else None,
            fallback_knorm=self.k_norm if self.use_qk_norm else None,
            kv_cache_dtype=kv_cache_dtype,
            qk_norm_policy=2,
            enable_hadamard=getattr(vllm_envs, "VLLM_ENABLE_HPC_QK_HADAMARD", False),
        )
        self.npu_rope_norm.register_layer_name(self.attn.layer_name)
        if self.npu_rope_norm.use_fp8:
            self.attn.query_quant = None
            self.attn.impl.use_npu_rope_norm = True
        logger.info_once(
            "[NpuRopeNorm] enabled for Hunyuan V3 on NPU, kv_cache_dtype=%s, use_fp8=%s",
            kv_cache_dtype,
            self.npu_rope_norm.use_fp8,
        )


def _patched_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if getattr(self, "npu_rope_norm", None) is None:
        return _original_forward(self, positions, hidden_states)

    if vllm_envs.VLLM_ENABLE_TPSP:
        if self.use_fused_gemm_comm:
            qkv = fused_all_gather_qkv_proj(
                hidden_states, self.qkv_proj, self.tp_group_name
            )
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            qkv, _ = self.qkv_proj(hidden_states)
    else:
        qkv, _ = self.qkv_proj(hidden_states)

    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = self.npu_rope_norm(qkv, self.attn.layer_name)
    q = q.view(-1, self.num_heads * self.head_dim)

    attn_output = self.attn(q, k, v, None)
    if vllm_envs.VLLM_PRECISION_MODE == "default":
        attn_output = attn_output.view(q.shape[0], -1)

    if vllm_envs.VLLM_ENABLE_TPSP:
        if self.use_fused_gemm_comm:
            from vllm.v1.worker.tpsp_utils import fused_o_proj_reduce_scatter

            output = fused_o_proj_reduce_scatter(
                attn_output,
                self.o_proj,
                self.tp_group_name,
            )
        else:
            from vllm.distributed import tensor_model_parallel_reduce_scatter

            output, _ = self.o_proj(attn_output)
            output = tensor_model_parallel_reduce_scatter(output, 0)
    else:
        output, _ = self.o_proj(attn_output)
    return output


HYV3Attention.__init__ = _patched_init
HYV3Attention.forward = _patched_forward
