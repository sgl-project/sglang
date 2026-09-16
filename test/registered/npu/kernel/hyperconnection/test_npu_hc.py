"""Compare HC fusions against the eager Torch intermediate-dtype contract."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.kernels.qwen3_8_flash_next import hc
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


def norm_reference(x, weight, group_size, eps):
    dim = group_size or x.shape[-1]
    grouped = x.float().reshape(*x.shape[:-1], x.shape[-1] // dim, dim)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, True) + eps)
    return (normalized.flatten(-2) * (1 + weight.float())).to(x.dtype)


def mix_reference(x, down, up, count, dim):
    gates = F.linear(F.silu(F.linear(x, down) / count), up).sigmoid()
    return (gates.unflatten(-1, (count, dim)) * x.unflatten(-1, (count, dim))).mean(-2)


def combine_reference(block, residual, normed, weight, count, dim):
    gate = 2 * torch.sigmoid(F.linear(normed, weight) / count)
    return (
        residual.unflatten(-1, (count, dim)) + block.unsqueeze(-2) * gate.unsqueeze(-1)
    ).flatten(-2)


def close(actual, expected):
    # Match the existing HC test tolerances; FP32 checks are stricter.
    tol = {
        torch.float32: (2e-5, 2e-5),
        torch.float16: (1e-3, 2e-3),
        torch.bfloat16: (5e-3, 1e-2),
    }[expected.dtype]
    torch.testing.assert_close(actual, expected, atol=tol[0], rtol=tol[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,count,dim",
    [
        (0, 4, 128),
        (1, 3, 127),
        (2, 1, 128),
        (2, 2, 128),
        (2, 8, 8192),
        (8, 4, 2560),
        (32, 5, 2560),
        (129, 4, 512),
    ],
)
def test_hc_reference(rows, count, dim, dtype):
    torch.manual_seed(81)
    x = torch.randn(rows, count * dim, device="npu", dtype=dtype)
    down = torch.randn(32, count * dim, device="npu", dtype=dtype) * 0.02
    up = torch.randn(count * dim, 32, device="npu", dtype=dtype) * 0.02
    weight = torch.randn(count, count * dim, device="npu", dtype=dtype) * 0.02
    block = torch.randn(rows, dim, device="npu", dtype=dtype)
    assert hc.can_run_mix(x, down, up, count, dim) == (rows <= 32)
    assert hc.can_run_combine(block, x, x, weight, count, dim) == (rows <= 32)
    close(hc.mix(x, down, up, count, dim), mix_reference(x, down, up, count, dim))
    close(
        hc.combine(block, x, x, weight, count, dim),
        combine_reference(block, x, x, weight, count, dim),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,group",
    [
        ((0, 512), 128),
        ((3, 4, 2560), None),
        ((8, 10240), 2560),
        ((2, 381), 127),
        ((1, 8192), None),
    ],
)
def test_norm_reference(shape, group, dtype):
    torch.manual_seed(91)
    x = torch.randn(shape, device="npu", dtype=dtype)
    weight = torch.randn(shape[-1], device="npu", dtype=torch.float32)
    assert hc.can_run_norm(x, weight, group)
    close(
        hc.grouped_norm(x, weight, group, 1e-6), norm_reference(x, weight, group, 1e-6)
    )


def test_graph_replay():
    x = torch.randn(8, 4 * 2560, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(4 * 2560, device="npu")
    down = torch.randn(320, 4 * 2560, device="npu", dtype=x.dtype) * 0.02
    up = down.t().contiguous()
    inject = torch.randn(4, 4 * 2560, device="npu", dtype=x.dtype) * 0.02

    def run():
        normed = hc.grouped_norm(x, weight, 2560, 1e-6)
        mixed = hc.mix(normed, down, up, 4, 2560)
        return hc.combine(mixed, x, normed, inject, 4, 2560)

    for _ in range(2):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    for scale in (0, 0.1, 2):
        x.normal_().mul_(scale)
        weight.normal_()
        graph.replay()
        torch.npu.synchronize()
        normed = norm_reference(x, weight, 2560, 1e-6)
        mixed = mix_reference(normed, down, up, 4, 2560)
        close(output, combine_reference(mixed, x, normed, inject, 4, 2560))


def test_unsupported_layout():
    x = torch.empty(3, 1024, device="npu")[:, ::2]
    weight = torch.empty(512, device="npu")
    assert not hc.can_run_norm(x, weight, 128)
    assert not hc.can_run_norm(x.cpu(), weight.cpu(), 128)
    assert not hc.can_run_norm(x.contiguous(), weight, 127)


def test_combine_intermediate_rounding_under_cancellation():
    torch.manual_seed(81)
    block = torch.randn(1, 2560, device="npu", dtype=torch.bfloat16) * 3
    normed = torch.zeros(1, 10240, device="npu", dtype=block.dtype)
    normed[:, 0] = 1
    weight = torch.zeros(4, 10240, device="npu", dtype=block.dtype)
    weight[:, 0] = torch.tensor(
        [-0.62109375, 0.9453125, -0.51171875, -2.65625],
        device="npu",
        dtype=block.dtype,
    )
    gate = 2 * torch.sigmoid(F.linear(normed, weight) / 4)
    residual = -(block.unsqueeze(1) * gate.unsqueeze(-1)).flatten(1)
    actual = hc.combine(block, residual, normed, weight, 4, 2560)
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)


@pytest.mark.parametrize("per_branch", [False, True])
@pytest.mark.parametrize("rows", [4, 128])
def test_public_dispatch_and_fallback(monkeypatch, per_branch, rows):
    from sglang.srt.layers import hyperconnection

    monkeypatch.setattr(hyperconnection, "_is_npu", True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("npu"))
    config = hyperconnection.HyperConnectionConfig(
        hidden_size=128, hc_count=4, hc_lowrank=32, hc_per_branch_norm=per_branch
    )
    module = hyperconnection.GatedResidual(config).to(device="npu")
    x = torch.randn(rows, 512, device="npu", dtype=torch.bfloat16)
    calls = []
    for name in ("grouped_norm", "mix", "combine"):
        original = getattr(hc, name)

        def counted(*args, _name=name, _fn=original):
            calls.append(_name)
            return _fn(*args)

        monkeypatch.setattr(hc, name, counted)

    with torch.no_grad():
        mixed, residuals = module.mix(x)
        actual = module.combine(mixed, residuals)
        expected_calls = ["grouped_norm"]
        if rows <= 32:
            expected_calls += ["mix", "combine"]
        assert calls == expected_calls
        # A strided 3-D input uses the original eager paths throughout.
        calls.clear()
        strided = torch.stack((x, x), -1)[..., 0].view(2, rows // 2, 512)
        fallback_mix, fallback_residuals = module.mix(strided)
        fallback = module.combine(fallback_mix, fallback_residuals)
        assert "mix" not in calls and "combine" not in calls
        close(actual, fallback.reshape_as(actual))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
