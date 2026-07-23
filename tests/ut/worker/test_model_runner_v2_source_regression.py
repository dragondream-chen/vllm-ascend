# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the Ascend v2 model runner."""

from pathlib import Path


def _prepare_inputs_source() -> str:
    source_path = (
        Path(__file__).resolve().parents[3]
        / "vllm_ascend"
        / "worker"
        / "v2"
        / "model_runner.py"
    )
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def prepare_inputs(")
    end = source.index("    def postprocess(", start)
    return source[start:end]


def _source_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3].joinpath("vllm_ascend", *parts)


def test_full_graph_uses_padded_device_attention_metadata() -> None:
    """CPU and device attention metadata must describe the same graph batch."""
    source = _prepare_inputs_source()

    assert "query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]" in source
    assert "query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]" in source
    assert "seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]" in source


def test_graph_capture_populates_ascend_moe_context_with_input_ids() -> None:
    """DSV4 hash routing needs graph-buffer token IDs while capturing."""
    source = _source_path("worker", "v2", "aclgraph_utils.py").read_text(encoding="utf-8")

    assert "ModelWithContext(model, vllm_config=self.vllm_config)" in source
    assert "with set_ascend_forward_context(" in source
    assert "input_ids=input_ids," in source


def test_model_pre_hook_populates_mrv2_hash_routing_context() -> None:
    """The V2 model call itself must cover eager, warmup, and capture."""
    source = _source_path("worker", "v2", "model_runner.py").read_text(encoding="utf-8")

    assert "self.model.register_forward_pre_hook(" in source
    assert "self._populate_ascend_moe_forward_context" in source
    assert "_EXTRA_CTX.input_ids = input_ids" in source
    assert "_EXTRA_CTX.moe_comm_method = get_moe_comm_method(moe_comm_type)" in source


def test_v2_metadata_preserves_cpu_sequence_length_upper_bound() -> None:
    """Builders must receive the same upper bound as upstream MRV2."""
    model_state = _source_path("worker", "v2", "model_states", "default.py").read_text(encoding="utf-8")
    attn_utils = _source_path("worker", "v2", "attn_utils.py").read_text(encoding="utf-8")

    assert "seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound" in model_state
    assert "seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound," in model_state
    assert "seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound[:num_reqs]," in attn_utils


def test_full_graph_metadata_uses_real_tokens_and_padded_input_buffer() -> None:
    """Mirror MRV1's DSA graph contract for persistent slot mappings."""
    source = _source_path("worker", "v2", "model_states", "default.py").read_text(encoding="utf-8")

    assert "num_tokens = input_batch.num_tokens" in source
    assert "num_input_tokens=input_batch.num_tokens_after_padding," in source
