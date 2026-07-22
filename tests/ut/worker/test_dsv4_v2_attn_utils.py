# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for DeepSeek-V4 v2 cache views."""

import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.attn_utils import bind_kv_cache


class _DSABackend:
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        return num_blocks, block_size, num_kv_heads, head_size


_DSAAttentionOwner = type(
    "DSAAttention",
    (),
    {"__module__": "vllm_ascend.models.layer.attention.layer"},
)
_DSAStateCacheOwner = type(
    "AscendCompressorStateCache",
    (),
    {"__module__": "vllm_ascend.models.deepseek_v4"},
)


def test_dsv4_v2_bind_kv_cache_keeps_v1_cache_list_abi() -> None:
    main_cache = object()
    state_cache = object()
    mtp_cache = object()
    forward_context = {
        "model.layers.1.self_attn.attn": _DSAAttentionOwner(),
        "model.layers.1.self_attn.compressor.state_cache": _DSAStateCacheOwner(),
        "model.mtp.layers.0.self_attn.attn": _DSAAttentionOwner(),
    }
    kv_caches = {
        "model.mtp.layers.0.self_attn.attn": mtp_cache,
        "model.layers.1.self_attn.compressor.state_cache": state_cache,
        "model.layers.1.self_attn.attn": main_cache,
    }
    runner_kv_caches = []

    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    assert runner_kv_caches == [main_cache, state_cache, mtp_cache]
    assert forward_context["model.layers.1.self_attn.attn"].kv_cache == [main_cache]
    assert forward_context["model.layers.1.self_attn.compressor.state_cache"].kv_cache == [state_cache]
    assert forward_context["model.mtp.layers.0.self_attn.attn"].kv_cache == [mtp_cache]


def test_dsv4_v2_main_cache_is_one_tensor() -> None:
    spec = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        model_version="deepseek_v4",
        compress_ratio=4,
    )
    raw = torch.zeros(spec.page_size_bytes * 2, dtype=torch.int8)

    cache = attn_utils._view_dsv4_cache(raw, spec, _DSABackend)

    assert isinstance(cache, torch.Tensor)
    assert cache.shape == (2, 4, 1, 8)


def test_dsv4_v2_cache_view_keeps_physical_block_geometry() -> None:
    dtype_size = torch.empty((), dtype=torch.float32).element_size()
    spec = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        model_version="deepseek_v4",
        compress_ratio=4,
        # A merged hybrid group can carry a larger planning value than this
        # owner actually allocates per page.  DSA still has to use its own
        # physical page layout.
        page_size_padded=2 * 4 * 8 * dtype_size,
    )
    raw = torch.zeros(spec.page_size_bytes * 2, dtype=torch.int8)

    cache = attn_utils._view_dsv4_cache(raw, spec, _DSABackend)

    assert cache.shape == (2, spec.block_size, 1, spec.head_size)
    assert cache.stride(0) == spec.page_size_bytes // dtype_size


def test_dsv4_v2_indexer_keeps_k_and_scale_views(monkeypatch) -> None:
    spec = AscendMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.int8,
        model_version="deepseek_v4",
        compress_ratio=4,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    raw = torch.zeros(spec.page_size_bytes * 2, dtype=torch.int8)

    monkeypatch.setattr(attn_utils, "get_ascend_device_type", lambda: None)
    caches = attn_utils._view_dsv4_cache(raw, spec, _DSABackend)

    assert isinstance(caches, tuple)
    k_cache, scale_cache = caches
    assert k_cache.shape == (2, 4, 1, 8)
    assert scale_cache.shape == (2, 4, 1, 1)
    assert k_cache.stride(0) == spec.page_size_bytes
    assert scale_cache.stride(0) == spec.page_size_bytes // torch.empty((), dtype=torch.float16).element_size()
