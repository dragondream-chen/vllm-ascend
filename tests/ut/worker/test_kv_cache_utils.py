from types import SimpleNamespace

from vllm_ascend.worker.kv_cache_utils import (
    bind_deepseek_v4_kv_cache_v1,
    bind_kv_cache,
)


def test_bind_kv_cache_allows_multiple_modules_per_layer():
    kv_caches = {
        "model.layers.1.self_attn.state_cache": "layer-1-state",
        "model.layers.0.self_attn.swa_cache": "layer-0-swa",
        "model.layers.1.self_attn.indexer_cache": "layer-1-indexer",
    }
    forward_context = {layer_name: SimpleNamespace(kv_cache=None) for layer_name in kv_caches}
    runner_kv_caches = []

    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

    assert runner_kv_caches == [
        "layer-0-swa",
        "layer-1-state",
        "layer-1-indexer",
    ]
    assert all(forward_context[layer_name].kv_cache == kv_cache for layer_name, kv_cache in kv_caches.items())


def test_bind_deepseek_v4_kv_cache_v1_orders_mtp_and_wraps_caches():
    kv_caches = {
        "model.mtp.0.self_attn.swa_cache": "mtp-0-swa",
        "model.layers.1.self_attn.swa_cache": "layer-1-swa",
        "model.layers.0.self_attn.swa_cache": "layer-0-swa",
        "model.layers.0.self_attn.indexer_cache": "layer-0-indexer",
    }
    forward_context = {layer_name: SimpleNamespace(kv_cache=None) for layer_name in kv_caches}
    runner_kv_caches = []
    hf_text_config = SimpleNamespace(num_hidden_layers=2)

    bind_deepseek_v4_kv_cache_v1(
        kv_caches,
        forward_context,
        runner_kv_caches,
        hf_text_config,
    )

    assert runner_kv_caches == [
        "layer-0-indexer",
        "layer-0-swa",
        "layer-1-swa",
        "mtp-0-swa",
    ]
    assert all(forward_context[layer_name].kv_cache == [kv_cache] for layer_name, kv_cache in kv_caches.items())
