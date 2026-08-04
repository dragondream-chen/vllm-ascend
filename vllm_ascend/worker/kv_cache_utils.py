# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from typing import Any

from vllm.model_executor.models.utils import extract_layer_index

from vllm_ascend.utils import extract_dsv4_layer_index


def bind_kv_cache(
    kv_caches: dict[str, Any],
    forward_context: dict[str, Any],
    runner_kv_caches: list[Any],
    num_attn_module: int = 1,
) -> None:
    """Bind KV caches while allowing multiple cache modules per layer."""
    assert len(runner_kv_caches) == 0

    index2name: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        layer_index = extract_layer_index(layer_name, num_attn_module)
        index2name[layer_index].append(layer_name)

    for layer_index in sorted(index2name):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = kv_cache


def bind_deepseek_v4_kv_cache_v1(
    kv_caches: dict[str, Any],
    forward_context: dict[str, Any],
    runner_kv_caches: list[Any],
    hf_text_config: Any,
) -> None:
    """Bind DeepSeek-V4 V1 caches in model/MTP order.

    V1 keeps each module cache in a one-element container for its legacy
    virtual-engine-aware execution path. MRV2 binds its final tensor or tuple
    directly and therefore uses :func:`bind_kv_cache` instead.
    """
    assert len(runner_kv_caches) == 0

    layer_names = sorted(
        kv_caches,
        key=lambda name: (
            extract_dsv4_layer_index(hf_text_config, name),
            name,
        ),
    )
    runner_kv_caches.extend(kv_caches[layer_name] for layer_name in layer_names)

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].kv_cache = [kv_cache]
