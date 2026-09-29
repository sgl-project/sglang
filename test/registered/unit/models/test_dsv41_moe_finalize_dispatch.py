from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "rows,expected", [(0, False), (384, True), (385, True), (1024, True), (1025, False)]
)
@pytest.mark.parametrize("capacity", [4, 12])
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_moe_dispatch_and_capacity(monkeypatch, rows, expected, capacity, tp_size):
    import sglang.srt.models.deepseek_v2 as model
    from sglang.srt import batch_invariant_ops
    from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as moe

    method = object.__new__(moe.Mxfp4FlashinferTrtllmMoEMethod)
    method.flashinfer_mxfp4_moe_precision = "default"
    experts = SimpleNamespace(
        quant_method=method, should_fuse_routed_scaling_factor_in_topk=True
    )
    layer = SimpleNamespace(
        _fuse_finalize_all_reduce=True,
        _shared_expert_tp1=False,
        tp_size=tp_size,
        experts=experts,
    )
    comm = SimpleNamespace(
        config=SimpleNamespace(num_push_blocks=1024), max_push_size=capacity * 1024**2
    )
    parallel = SimpleNamespace(tp_group=SimpleNamespace(world_size=tp_size))
    monkeypatch.setattr(moe, "_fused_finalize_all_reduce_comm", comm)
    monkeypatch.setattr(
        moe, "_fused_finalize_all_reduce_comm_world_size", lambda: tp_size
    )
    monkeypatch.setattr(moe, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        batch_invariant_ops, "is_batch_invariant_mode_enabled", lambda: False
    )
    monkeypatch.setattr(
        model, "should_skip_post_experts_all_reduce", lambda **kw: False
    )
    hidden_states = torch.empty(rows, 5120, device="meta")
    can_fuse = model.DeepseekV2MoE._can_fuse_finalize_all_reduce
    assert can_fuse(layer, hidden_states, True) == (
        expected and rows * 5120 * 2 <= comm.max_push_size
    )
    assert not can_fuse(layer, hidden_states, False)
    comm.config.num_push_blocks = 383
    assert not can_fuse(layer, hidden_states, True)
    comm.config.num_push_blocks = 1024
    monkeypatch.setattr(
        batch_invariant_ops, "is_batch_invariant_mode_enabled", lambda: True
    )
    assert not can_fuse(layer, hidden_states, True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
