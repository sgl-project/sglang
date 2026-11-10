import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The FP16 PLE fusion requires V100",
)


def reference(gate, value):
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    return torch.sigmoid(gate) * value.unsqueeze(-2)


def test_all_finite_fp16_gate_values(monkeypatch):
    from sglang.srt.layers.qwen4_ple import fused_qwen4_gate_value

    monkeypatch.setenv("SGLANG_SM70_QWEN_FUSIONS", "1")
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    gates = bits.view(torch.float16)
    gates = gates[torch.isfinite(gates)].reshape(-1, 4, 1)
    values = torch.ones((gates.shape[0], 2560), device="cuda", dtype=torch.float16)
    result = fused_qwen4_gate_value(gates, values)
    torch.testing.assert_close(result, reference(gates, values), rtol=0, atol=0)


def test_ple_gate_graph_replay_and_dispatch(monkeypatch):
    from sglang.srt.layers.qwen4_ple import (
        can_fuse_qwen4_gate_value,
        fused_qwen4_gate_value,
    )

    monkeypatch.setenv("SGLANG_SM70_QWEN_FUSIONS", "1")
    gate = torch.randn((1, 4, 1), device="cuda", dtype=torch.float16)
    value = torch.randn((1, 2560), device="cuda", dtype=torch.float16)
    assert can_fuse_qwen4_gate_value(gate, value)
    fused_qwen4_gate_value(gate, value)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fused_qwen4_gate_value(gate, value)
    for scale in [1.0, -2.0, 0.0]:
        gate.mul_(scale)
        value.mul_(-0.5)
        graph.replay()
        torch.testing.assert_close(result, reference(gate, value), rtol=0, atol=0)
    monkeypatch.setenv("SGLANG_SM70_QWEN_FUSIONS", "0")
    assert not can_fuse_qwen4_gate_value(gate, value)
