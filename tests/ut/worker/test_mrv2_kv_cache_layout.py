from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2 import attn_utils


def _group(backend, layer_name, spec, group_id):
    return SimpleNamespace(
        backend=backend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=group_id,
    )


def _patch_attention_layers(monkeypatch, layers):
    config = SimpleNamespace(kv_transfer_config=None, quant_config=None)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(mla_v1, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(mla_v1, "enable_fa_quant", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda _config, _layer_type, layer_names=None: (
            layers if layer_names is None else {name: layers[name] for name in layer_names}
        ),
    )


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_ascend_mla_storage_block_size_uses_physical_block_size(
    compress_ratio: int,
):
    spec = AscendMLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        compress_ratio=compress_ratio,
    )
    assert spec.storage_block_size == 32


def test_dsa_mrv2_cache_lifecycle_supports_component_layouts(monkeypatch):
    swa_name = "model.layers.7.self_attn.swa_cache"
    state_name = "model.layers.7.self_attn.compressor_state"
    indexer_name = "model.layers.7.self_attn.indexer.k_cache"
    single_spec = AscendMLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    indexer_spec = AscendMLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.int8,
        compress_ratio=4,
        scale_dim=1,
        scale_dtype=torch.float16,
    )
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: AscendDSABackend)
        for name in (swa_name, state_name, indexer_name)
    }
    _patch_attention_layers(monkeypatch, layers)

    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * single_spec.page_size_bytes,
                shared_by=[swa_name, state_name],
            ),
            KVCacheTensor(
                size=2 * indexer_spec.page_size_bytes,
                shared_by=[indexer_name],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[swa_name], kv_cache_spec=single_spec),
            KVCacheGroupSpec(layer_names=[state_name], kv_cache_spec=single_spec),
            KVCacheGroupSpec(layer_names=[indexer_name], kv_cache_spec=indexer_spec),
        ],
    )
    groups = [
        _group(AscendDSABackend, swa_name, single_spec, 0),
        _group(AscendDSABackend, state_name, single_spec, 1),
        _group(AscendDSABackend, indexer_name, indexer_spec, 2),
    ]

    raw_caches = attn_utils._allocate_kv_cache(config, {}, torch.device("cpu"))
    assert raw_caches[swa_name].data_ptr() == raw_caches[state_name].data_ptr()
    assert raw_caches[swa_name].numel() == 2 * single_spec.page_size_bytes
    indexer_raw = raw_caches[indexer_name]
    assert isinstance(indexer_raw, tuple)
    assert sum(component.numel() for component in indexer_raw) == 2 * indexer_spec.page_size_bytes

    caches = attn_utils._reshape_kv_cache_v2(
        groups,
        raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[32, 32, 32],
        shared_kv_cache_layers={},
        kv_cache_config=config,
    )
    assert single_spec.storage_block_size == 32
    assert caches[swa_name].shape == (2, 32, 1, 128)
    assert caches[swa_name].dtype == torch.bfloat16
    assert caches[swa_name].data_ptr() == caches[state_name].data_ptr()
    indexer_cache = caches[indexer_name]
    assert isinstance(indexer_cache, tuple)
    assert indexer_cache[0].shape == (2, 32, 1, 128)
    assert indexer_cache[0].dtype == torch.int8
    assert indexer_cache[1].shape == (2, 32, 1, 1)
    assert indexer_cache[1].dtype == torch.float16


@pytest.mark.parametrize("layout", ["gqa", "mla"])
def test_mrv2_cache_lifecycle_preserves_existing_layouts(monkeypatch, layout):
    layer_name = f"model.layers.1.self_attn.{layout}"
    if layout == "gqa":
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=64,
            head_size_v=32,
            dtype=torch.float16,
        )
        layer = SimpleNamespace(get_attn_backend=lambda: AscendAttentionBackend)
        backend = AscendAttentionBackend
        expected_shapes = ((2, 16, 2, 64), (2, 16, 2, 32))
    else:
        spec = AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        )
        layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(layer)
        layer.layer_name = layer_name
        layer.kv_lora_rank = 512
        layer.qk_rope_head_dim = 64
        layer.get_attn_backend = lambda: AscendMLABackend
        backend = AscendMLABackend
        expected_shapes = ((2, 128, 1, 512), (2, 128, 1, 64))

    _patch_attention_layers(monkeypatch, {layer_name: layer})
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * spec.page_size_bytes,
                shared_by=[layer_name],
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )

    raw_caches = attn_utils._allocate_kv_cache(config, {}, torch.device("cpu"))
    caches = attn_utils._reshape_kv_cache_v2(
        [_group(backend, layer_name, spec, 0)],
        raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[spec.block_size],
        shared_kv_cache_layers={},
        kv_cache_config=config,
    )

    assert isinstance(caches[layer_name], tuple)
    assert tuple(component.shape for component in caches[layer_name]) == expected_shapes
