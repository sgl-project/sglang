"""Opt-in Qwen4 gate dispatch and replay regression on an idle NPU."""

import os
from types import SimpleNamespace

import pytest

if os.getenv("SGLANG_TEST_QWEN4_GATE_NPU") != "1":
    pytest.skip("Requires an idle NPU and the NPU kernel package", allow_module_level=True)

import sgl_kernel_npu  # Register vendors before device initialization.
import torch
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.srt.hardware_backend.npu.utils import init_npu_backend

init_npu_backend()

from sglang.srt.models import qwen4_exp


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "rows,layout",
    [(1, "contiguous"), (4, "strided3d"), (32, "strided2d"),
     (128, "contiguous"), (4096, "contiguous"), (4096, "strided3d")],
)
def test_gate_dispatch_and_changed_input_replay(dtype, rows, layout, monkeypatch):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    x = torch.randn(rows, 768, device="npu", dtype=dtype)
    if layout == "strided3d":
        gate = torch.randn(rows, 3, 512, device="npu", dtype=dtype)[:, :, 256:]
    elif layout == "strided2d":
        gate = torch.randn(rows, 1536, device="npu", dtype=dtype)[:, 768:]
    else:
        gate = torch.randn_like(x)
    # Saturation and zero exercise sigmoid behavior beyond random small inputs.
    gate[0].fill_(0)
    if rows > 1:
        gate[-1].fill_(-20)
    assert x.is_cuda and x.device.type == "npu"

    # Exercise the real model method without loading weights. Other stages are
    # identity stubs; cloning attention output keeps repeated forwards independent.
    layer = SimpleNamespace(
        is_qsa=False,
        alt_stream=None,
        _prepare_qkv_gate=lambda **kwargs: (None, None, None, gate),
        attn=lambda *args, **kwargs: x.clone(),
        o_proj=lambda output: (output, None),
    )

    def wrong_cuda_branch(*args, **kwargs):
        raise AssertionError("NPU gate selected the CUDA row-wise implementation")

    monkeypatch.setattr(qwen4_exp, "fused_sigmoid_mul", wrong_cuda_branch)

    def run():
        return qwen4_exp.Qwen4ExpAttentionDecoderLayer.self_attention(layer, None, x, None)

    out = run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=False):
        captured = run()
    for iteration in range(3):
        if iteration:
            x.mul_(0.7)
            gate.add_(0.25)
        graph.replay()
        torch.npu.synchronize()
        expected = fused_sigmoid_mul(x, gate)
        promoted = (x.float() * torch.sigmoid(gate.reshape(rows, 768).float())).to(dtype)
        tolerance = 1e-5 if dtype == torch.float32 else 0.016 if dtype == torch.bfloat16 else 0.001
        # Compare both the previous execution path and an independent expression.
        torch.testing.assert_close(captured, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(captured, promoted, atol=tolerance, rtol=tolerance)
        if iteration == 0:
            torch.testing.assert_close(out, expected, atol=tolerance, rtol=tolerance)
