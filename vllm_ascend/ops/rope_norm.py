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
"""NPU fused RoPE + QK-Norm + KV-Cache-Write + FP8 Q quant."""

from __future__ import annotations

import torch
import torch_npu

from vllm.config.cache import KVCacheQuantConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from vllm_ascend.device.device_op import DeviceOperator

logger = init_logger(__name__)


class NpuRopeNorm(torch.nn.Module):
    """Hunyuan V3 NPU equivalent of GPU HpcRopeNorm.

    The fused operator owns QK-Norm, RoPE, KV cache write, and FP8 Q/K/V
    quantization. The normal attention backend then consumes the processed
    query and the populated KV cache.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
        use_qk_norm: bool,
        fallback_qnorm: torch.nn.Module | None,
        fallback_knorm: torch.nn.Module | None,
        kv_cache_dtype: str,
        qk_norm_policy: int = 1,
        enable_hadamard: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.cos_sin_cache = cos_sin_cache.float()
        self.fallback_qnorm = fallback_qnorm
        self.fallback_knorm = fallback_knorm
        self.head_per_group = num_heads // num_kv_heads
        self.qnorm_weight: torch.Tensor | None = None
        self.knorm_weight: torch.Tensor | None = None
        self.qk_norm_policy = qk_norm_policy
        self.enable_hadamard = enable_hadamard
        self.use_fp8 = "fp8" in kv_cache_dtype
        self.layer_name: str | None = None
        self._kv_cache_quant_config: KVCacheQuantConfig | None = None
        self._quant_type: int | None = None

        if self.use_fp8:
            self._resolve_quant_config()

    @classmethod
    def support(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: str,
    ) -> bool:
        if kv_cache_dtype not in ("fp8_e4m3", "fp8", "bfloat16", "auto"):
            return False
        if head_dim != 128:
            return False
        head_per_group = num_heads // num_kv_heads
        return head_per_group in (4, 8)

    def _resolve_quant_config(self) -> None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.cache_config is not None:
            self._kv_cache_quant_config = (
                vllm_config.cache_config.kv_cache_quant_config
            )

        if self._kv_cache_quant_config is None:
            from vllm.config.cache import KVCacheQuantConfig, KVQuantSpec

            self._kv_cache_quant_config = KVCacheQuantConfig(
                k_quant=KVQuantSpec(
                    dtype="fp8_e4m3",
                    granularity="per_token_per_head",
                ),
                v_quant=KVQuantSpec(dtype="fp8_e4m3", granularity="per_head"),
            )

        # Policy 1 maps to Q per-token-per-head, K per-token-per-head, V per-head
        # in the current NPU fused operator prototype.
        self._quant_type = 1
        if self.enable_hadamard:
            logger.warning(
                "NpuRopeNorm received enable_hadamard=True, but the current "
                "quant policy does not expose a separate Hadamard mode."
            )

    def process_weights_after_loading(self, act_dtype: torch.dtype | None = None) -> None:
        if not self.use_qk_norm:
            return
        if self.fallback_qnorm is not None:
            self.qnorm_weight = self.fallback_qnorm.weight.detach().float()
        if self.fallback_knorm is not None:
            self.knorm_weight = self.fallback_knorm.weight.detach().float()

    def register_layer_name(self, layer_name: str) -> None:
        self.layer_name = layer_name

    def forward(self, qkv: torch.Tensor, layer_name: str) -> torch.Tensor:
        self.layer_name = layer_name
        num_tokens = qkv.shape[0]
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float8_e4m3fn if self.use_fp8 else qkv.dtype,
            device=qkv.device,
        )
        self._forward_impl(qkv, output)
        return output

    def _get_kv_cache(self):
        forward_context = get_forward_context()
        assert self.layer_name is not None
        attn_layer = forward_context.no_compile_layers[self.layer_name]
        kv_cache = attn_layer.kv_cache[0]
        if isinstance(kv_cache, (tuple, list)):
            return attn_layer, kv_cache[0], kv_cache[1]
        return attn_layer, kv_cache[0], kv_cache[1]

    def _get_kv_scales(
        self,
        layer,
        k_scale_from_cache: torch.Tensor | None,
        v_scale_from_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_qcfg = self._kv_cache_quant_config
        if kv_qcfg is None:
            return layer._k_scale.reshape(1), layer._v_scale.reshape(1)

        if kv_qcfg.k_quant is not None and kv_qcfg.k_quant.granularity == "per_token_per_head":
            k_scale = (
                k_scale_from_cache.view(torch.float32)
                if k_scale_from_cache is not None
                else layer._k_scale
            )
        elif kv_qcfg.k_quant is not None and kv_qcfg.k_quant.granularity == "per_head":
            k_scale = layer._k_scale
        else:
            k_scale = layer._k_scale.reshape(1)

        if kv_qcfg.v_quant is not None and kv_qcfg.v_quant.granularity == "per_token_per_head":
            v_scale = (
                v_scale_from_cache.view(torch.float32)
                if v_scale_from_cache is not None
                else layer._v_scale
            )
        elif kv_qcfg.v_quant is not None and kv_qcfg.v_quant.granularity == "per_head":
            v_scale = layer._v_scale
        else:
            v_scale = layer._v_scale.reshape(1)
        return k_scale, v_scale

    def _forward_impl(self, qkv: torch.Tensor, output: torch.Tensor) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            assert self.layer_name is not None
            attn_metadata = attn_metadata[self.layer_name]

        if attn_metadata is None:
            output.zero_()
            return

        attn_layer, key_cache, value_cache = self._get_kv_cache()
        if key_cache.numel() == 0:
            output.zero_()
            return

        qkv = qkv[: attn_metadata.num_actual_tokens]
        if self.use_fp8:
            key_cache_fp8 = (
                key_cache.view(torch.float8_e4m3fn)
                if key_cache.dtype != torch.float8_e4m3fn
                else key_cache
            )
            value_cache_fp8 = (
                value_cache.view(torch.float8_e4m3fn)
                if value_cache.dtype != torch.float8_e4m3fn
                else value_cache
            )
            k_data, k_scale_from_cache = DeviceOperator.split_fp8_kv_cache_and_scale(
                key_cache_fp8, self.head_dim, self._kv_cache_quant_config
            )
            v_data, v_scale_from_cache = DeviceOperator.split_fp8_kv_cache_and_scale(
                value_cache_fp8, self.head_dim, self._kv_cache_quant_config
            )
            k_scale, v_scale = self._get_kv_scales(
                attn_layer, k_scale_from_cache, v_scale_from_cache
            )
            result = torch_npu.mytest_rope_norm_store_kv_fp8(
                key_cache=k_data,
                value_cache=v_data,
                qkv=qkv,
                cos_sin=self.cos_sin_cache,
                num_seqlen_per_req=attn_metadata.seq_lens,
                q_index=attn_metadata.query_start_loc,
                kvcache_indices=attn_metadata.block_tables,
                is_prefill=attn_metadata.num_decodes == 0,
                k_scale=k_scale,
                v_scale=v_scale,
                quant_policy=self._quant_type,
                max_seqlens=attn_metadata.max_query_len or 1,
                q_norm_weight=self.qnorm_weight if self.qk_norm_policy > 0 else None,
                k_norm_weight=self.knorm_weight if self.qk_norm_policy > 0 else None,
                qk_norm_policy=self.qk_norm_policy,
                out_q=output[: qkv.shape[0]],
            )
            if isinstance(result, tuple):
                if len(result) > 1:
                    attn_metadata.npu_prefill_q_scale = result[1]
                    attn_metadata.npu_decode_q_scale = result[1]
                if len(result) > 2:
                    attn_metadata.npu_split_k_flag = result[2]
        else:
            k_data = key_cache
            v_data = value_cache
            torch_npu.mytest_rope_norm_store_kv(
                k_data,
                v_data,
                qkv,
                self.cos_sin_cache,
                attn_metadata.seq_lens,
                attn_metadata.query_start_loc,
                attn_metadata.block_tables,
                attn_metadata.num_decodes == 0,
                q_norm_weight=self.qnorm_weight if self.qk_norm_policy > 0 else None,
                k_norm_weight=self.knorm_weight if self.qk_norm_policy > 0 else None,
                qk_norm_policy=self.qk_norm_policy,
                out_q=output[: qkv.shape[0]],
            )
