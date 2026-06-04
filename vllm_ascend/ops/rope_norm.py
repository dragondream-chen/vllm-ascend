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
"""NPU fused RoPE + QK-Norm + KV-Cache-Write + FP8 Q quant.

Equivalent to GPU's HpcRopeNorm, using mytest_rope_norm_store_kv_fp8.
"""

from __future__ import annotations

import torch
import torch_npu

from vllm.config.cache import KVCacheQuantConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_npu_rope_norm_instances: dict[str, "NpuRopeNorm"] = {}


class NpuRopeNorm:
    """NPU fused RoPE + QK-Norm + KV-Cache-Write + FP8 Q quant.

    Registered as a sub-module in model layers (e.g. HunYuanAttention).
    Norm weights are extracted from fallback norm modules via
    process_weights_after_loading() after all weights are loaded.
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
        self.use_fp8 = "fp8" in kv_cache_dtype
        self.layer_name: str | None = None
        self._kv_cache_quant_config: KVCacheQuantConfig | None = None
        self._quant_type = None

        if self.use_fp8:
            self._resolve_quant_config()

    def _resolve_quant_config(self):
        from vllm.config import get_current_vllm_config_or_none

        _vllm_cfg = get_current_vllm_config_or_none()
        if _vllm_cfg is not None and _vllm_cfg.cache_config is not None:
            self._kv_cache_quant_config = _vllm_cfg.cache_config.kv_cache_quant_config

        if self._kv_cache_quant_config is None:
            from vllm.config.cache import KVCacheQuantConfig, KVQuantSpec

            self._kv_cache_quant_config = KVCacheQuantConfig(
                k_quant=KVQuantSpec(dtype="fp8_e4m3", granularity="per_token_per_head"),
                v_quant=KVQuantSpec(dtype="fp8_e4m3", granularity="per_head"),
            )

        self._quant_type = 1

    @classmethod
    def support(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: str,
    ) -> bool:
        if kv_cache_dtype not in ("fp8_e4m3", "bfloat16", "auto"):
            return False
        if head_dim not in (128,):
            return False
        head_per_group = num_heads // num_kv_heads
        if head_per_group not in (4, 8):
            return False
        return True

    def process_weights_after_loading(self, model: torch.nn.Module = None) -> None:
        if self.use_qk_norm:
            if self.fallback_qnorm is not None:
                self.qnorm_weight = self.fallback_qnorm.weight.data.float()
            if self.fallback_knorm is not None:
                self.knorm_weight = self.fallback_knorm.weight.data.float()

    def register_layer_name(self, layer_name: str) -> None:
        self.layer_name = layer_name
        _npu_rope_norm_instances[layer_name] = self

    def forward(
        self,
        qkv: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        num_tokens = qkv.shape[0]
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float8_e4m3fn if self.use_fp8 else qkv.dtype,
            device=qkv.device,
        )
        self._forward_impl(qkv, output)
        return output

    def _forward_impl(
        self,
        qkv: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        from vllm.forward_context import get_forward_context

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]

        if attn_metadata is None:
            output.zero_()
            return

        attn_layer = forward_context.no_compile_layers[self.layer_name]
        kv_cache_tuple = attn_layer.kv_cache[0]

        if isinstance(kv_cache_tuple, (tuple, list)):
            key_cache, value_cache = kv_cache_tuple[0], kv_cache_tuple[1]
        else:
            key_cache, value_cache = kv_cache_tuple[0], kv_cache_tuple[1]

        if key_cache.numel() == 0:
            output.zero_()
            return

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        qkv = qkv[:num_actual_tokens]
        num_prefill_tokens = num_actual_tokens - num_decode_tokens

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

            from vllm_ascend.device.device_op import BaseDeviceAdaptor

            k_data, k_scale = BaseDeviceAdaptor.split_fp8_kv_cache_and_scale(
                key_cache_fp8, self.head_dim, self._kv_cache_quant_config
            )
            v_data, v_scale = BaseDeviceAdaptor.split_fp8_kv_cache_and_scale(
                value_cache_fp8, self.head_dim, self._kv_cache_quant_config
            )

            k_cache_data = k_data
            v_cache_data = v_data
        else:
            k_cache_data = key_cache
            v_cache_data = value_cache

        if attn_metadata.num_prefills > 0:
            qkv_prefill = qkv[num_decode_tokens:]

            if self.use_fp8:
                k_scale, v_scale = self._get_kv_scales(attn_layer, k_scale, v_scale)
                torch_npu.mytest_rope_norm_store_kv_fp8(
                    key_cache=k_cache_data,
                    value_cache=v_cache_data,
                    qkv=qkv_prefill,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=attn_metadata.seq_lens_prefill,
                    q_index=attn_metadata.qo_indptr,
                    kvcache_indices=attn_metadata.block_table_prefill,
                    is_prefill=True,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=self._quant_type,
                    max_seqlens=attn_metadata.max_query_len,
                    q_norm_weight=self.qnorm_weight if self.qk_norm_policy > 0 else None,
                    k_norm_weight=self.knorm_weight if self.qk_norm_policy > 0 else None,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=output[num_decode_tokens : num_decode_tokens + num_prefill_tokens],
                )
            else:
                torch_npu.mytest_rope_norm_store_kv(
                    k_cache_data,
                    v_cache_data,
                    qkv_prefill,
                    self.cos_sin_cache,
                    attn_metadata.seq_lens_prefill,
                    attn_metadata.qo_indptr,
                    attn_metadata.block_table_prefill,
                    True,
                    q_norm_weight=self.qnorm_weight if self.qk_norm_policy > 0 else None,
                    k_norm_weight=self.knorm_weight if self.qk_norm_policy > 0 else None,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=output[num_decode_tokens : num_decode_tokens + num_prefill_tokens],
                )

        if attn_metadata.num_decodes > 0:
            qkv_decode = qkv[:num_decode_tokens]

            if self.use_fp8:
                k_scale, v_scale = self._get_kv_scales(attn_layer, k_scale, v_scale)
                torch_npu.mytest_rope_norm_store_kv_fp8(
                    key_cache=k_cache_data,
                    value_cache=v_cache_data,
                    qkv=qkv_decode,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=attn_metadata.seq_lens_decode,
                    q_index=attn_metadata.qo_indptr_decode,
                    kvcache_indices=attn_metadata.block_table_decode,
                    is_prefill=False,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=self._quant_type,
                    max_seqlens=1,
                    q_norm_weight=self.qnorm_weight if self.qk_norm_policy > 0 else None,
                    k_norm_weight=self.knorm_weight if self.qk_norm_policy > 0 else None,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=output[:num_decode_tokens],
                )
            else:
                torch_npu.mytest_rope_norm_store_kv(
                    k_cache_data,
                    v_cache_data,
                    qkv_decode,
                    self.cos_sin_cache,
                    attn_metadata.seq_lens_decode,
                    attn_metadata.qo_indptr_decode,
                    attn_metadata.block_table_decode,
                    False,
                    q_norm_weight=self.qnorm_weight if self.qk_norm_policy > 0 else None,
                    k_norm_weight=self.knorm_weight if self.qk_norm_policy > 0 else None,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=output[:num_decode_tokens],
                )

    def _get_kv_scales(self, layer, k_scale_from_cache, v_scale_from_cache):
        kv_qcfg = self._kv_cache_quant_config

        if kv_qcfg is not None:
            if kv_qcfg.k_quant is not None and kv_qcfg.k_quant.granularity == "per_token_per_head":
                k_scale = k_scale_from_cache.view(torch.float32) if k_scale_from_cache is not None else layer._k_scale
            elif kv_qcfg.k_quant is not None and kv_qcfg.k_quant.granularity == "per_head":
                k_scale = layer._k_scale
            else:
                k_scale = layer._k_scale.reshape(1)

            if kv_qcfg.v_quant is not None and kv_qcfg.v_quant.granularity == "per_token_per_head":
                v_scale = v_scale_from_cache.view(torch.float32) if v_scale_from_cache is not None else layer._v_scale
            elif kv_qcfg.v_quant is not None and kv_qcfg.v_quant.granularity == "per_head":
                v_scale = layer._v_scale
            else:
                v_scale = layer._v_scale.reshape(1)
        else:
            k_scale = layer._k_scale.reshape(1)
            v_scale = layer._v_scale.reshape(1)

        return k_scale, v_scale
