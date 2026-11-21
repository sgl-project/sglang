"""The fused loop message controllers must match the unfused op chain they replace
(RMSNorm, down linear, gain / forget GEMMs, tanh mix) up to rounding."""

import sys

import pytest
import torch

from sglang.kernels.ops.elementwise.inkling2_loop import message_controllers
from sglang.kernels.ops.gemm.inkling2_linear import linear
from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

WIDTH, RANK = 2304, 64


def _reference(dest, message, down, gain, forget):
    normalized = rms_norm(
        torch.stack((dest, message), dim=1), None, 1e-6, output_dtype=torch.bfloat16
    ).flatten(1)
    hidden = (normalized.double() @ down.double().T).bfloat16()
    g = torch.tanh(linear(hidden, gain, out_dtype=torch.float32))
    f = torch.tanh(linear(hidden, forget, out_dtype=torch.float32))
    return (1.0 - f) * dest + g * message


@pytest.mark.parametrize("tokens", [1, 7, 64, 300, 1500, 5000])
@pytest.mark.parametrize("layout", ["repeated", "views", "separate"])
def test_message_controllers_match_unfused_chain(tokens, layout):
    _check(tokens, layout, RANK)


@pytest.mark.parametrize("tokens", [1, 300, 1500])
@pytest.mark.parametrize("rank", [8, 48])
def test_message_controllers_any_rank(tokens, rank):
    _check(tokens, "views", rank)


def _check(tokens, layout, rank):
    torch.manual_seed(tokens)
    num = 3
    down = torch.randn(num, rank, 2 * WIDTH, device="cuda").mul(0.02).bfloat16()
    gain = torch.randn(num, WIDTH, rank, device="cuda").mul(0.2).bfloat16()
    forget = torch.randn(num, WIDTH, rank, device="cuda").mul(0.2).bfloat16()
    stacked = torch.randn(num, tokens, WIDTH, device="cuda")
    if layout == "repeated":
        dests = (stacked[0],) * num
    elif layout == "views":
        dests = tuple(stacked.unbind(0))
    else:
        dests = tuple(x.clone() for x in stacked.unbind(0))
    messages = tuple(torch.randn(num, tokens, WIDTH, device="cuda").unbind(0))
    outs = message_controllers(dests, messages, down, gain, forget)
    assert outs.shape == (num, tokens, WIDTH)
    for i in range(num):
        ref = _reference(dests[i], messages[i], down[i], gain[i], forget[i])
        diff = (outs[i] - ref).abs()
        assert diff.max() < 0.05 * ref.abs().max(), (i, diff.max().item())
        assert diff.mean() < 1e-3 * ref.abs().mean(), (i, diff.mean().item())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
