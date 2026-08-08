from pathlib import Path


def test_model_pre_hook_populates_mrv2_hash_routing_context() -> None:
    """MRV2 must initialize the hash-router context before model forward."""
    source_path = Path(__file__).parents[3] / "vllm_ascend" / "worker" / "v2" / "model_runner.py"
    source = source_path.read_text(encoding="utf-8")

    assert "def load_model(self, *args, **kwargs) -> None:" in source
    assert source.index("super().load_model(*args, **kwargs)") < source.index("self.model.register_forward_pre_hook(")
    assert "self._populate_ascend_moe_forward_context" in source
    assert "forward_context = get_forward_context()" in source
    assert "forward_context.input_ids = input_ids" in source
    assert "forward_context.moe_comm_method = moe_comm_method" in source
    assert "_EXTRA_CTX.input_ids = input_ids" in source
    assert "_EXTRA_CTX.moe_comm_method = moe_comm_method" in source
