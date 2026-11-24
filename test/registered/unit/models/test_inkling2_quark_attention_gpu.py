"""FP8 attention projections vs an FP32 reference (gfx950)."""

import collections
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.inkling2_quark_kernel_test_utils import rel_l2
from sglang.test.inkling2_quark_test_utils import (
    mini_layout,
    mini_text_config,
    real_attention_export,
)

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

GFX95 = is_gfx95_supported()
pytestmark = pytest.mark.skipif(not GFX95, reason="needs an AMD gfx95x GPU")

if GFX95:  # a failing import must fail the file on the machines that run it
    from sglang.srt.models.inkling_common.inkling2_attn_hip import apply_wo_ud

TOKENS = (1, 8, 64, 513)
VARIANTS = ("UINT4-AWQ-AttnFP8", "MXFP4-AttnFP8")
# a BF16 GEMM on the dequantized weights (no activation quantizer) is 1.9e-2 off
FP8_RELATIVE_L2 = 1e-2
BF16_RELATIVE_L2 = 1e-3
FP8_PROJECTIONS = {"q": "wq_du", "k": "wk_dv", "v": "wv_dv", "gate": "gate_attn"}
BF16_PROJECTIONS = {"r": "wr_du", "sinks": "attn_sink_proj"}


@pytest.fixture(
    scope="module",
    params=[(variant, layer) for variant in VARIANTS for layer in (0, 5)],
    ids=lambda case: f"{case[0]}-layer{case[1]}",
)
def attention(request):
    """(the loaded and post-processed attention, its checkpoint tensors)."""
    from sglang.srt.runtime_context import reset_context
    from sglang.srt.server_args import ServerArgs
    from sglang.test.parallel_groups import publish

    variant, layer_id = request.param
    publish(ServerArgs(model_path="dummy"), role="test")
    attn, target, tensors = real_attention_export(
        mini_text_config(), mini_layout(variant), layer_id, device="cuda"
    )
    target.load_weights(list(tensors.items()))
    DefaultModelLoader.postprocess_weights(target, torch.device("cuda"))
    yield attn, tensors
    reset_context()


@pytest.fixture
def fp8_calls(monkeypatch):
    """Call counts of the FP8 quantizer and GEMM functions."""
    calls = collections.Counter()
    names = ("per_token_group_quant_fp8", "gemm_a8w8_bpreshuffle", "smallm_fp8_gemm")
    for owner, name in [
        *((fp8_utils, name) for name in (*names, "_apply_fallback_scaled_mm")),
        (torch, "_scaled_mm"),
    ]:
        function = getattr(owner, name, None)
        if function is not None:
            monkeypatch.setattr(owner, name, _counted(calls, name, function))
    return calls


def _counted(calls, name, function):
    def counted(*args, **kwargs):
        calls[name] += 1
        return function(*args, **kwargs)

    return counted


def _one_fp8_linear(calls):
    """One FP8 linear call: one activation quantizer, one FP8 GEMM, nothing else."""
    gemms = calls["gemm_a8w8_bpreshuffle"] + calls["_scaled_mm"]
    assert calls["per_token_group_quant_fp8"] == 1 and gemms == 1, dict(calls)
    assert sum(calls.values()) == 2, dict(calls)


def _inputs(tokens, columns, seed):
    """Hidden-state-like activations: per-token amplitudes and a few outlier channels."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, columns, generator=generator)
    x *= 0.3 + 1.7 * torch.rand(tokens, 1, generator=generator)
    x[:, torch.randperm(columns, generator=generator)[:6]] *= 25.0
    return x.to(torch.bfloat16).cuda()


def _fp8_reference(x, weight, scale):
    """x [T, K] against FP8 weight [N, K] with one scale per output channel."""
    x = x.float()
    token_scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    quantized = (x * (1.0 / token_scale)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (quantized.float() @ weight.float().T) * token_scale * scale.float()


def _checkpoint(attn, tensors, name):
    base = f"model.llm.layers.{attn.layer_id}.attn.{name}"
    scale = tensors.get(f"{base}.weight_scale")
    return tensors[f"{base}.weight"].cuda(), None if scale is None else scale.cuda()


def _check_projections(attn, tensors, got, inputs):
    """inputs[piece] is the activation the piece was projected from."""
    for piece, name in FP8_PROJECTIONS.items():
        weight, scale = _checkpoint(attn, tensors, name)
        error = rel_l2(got[piece], _fp8_reference(inputs[piece], weight, scale))
        assert error < FP8_RELATIVE_L2, (piece, error)
    for piece, name in BF16_PROJECTIONS.items():
        weight, _ = _checkpoint(attn, tensors, name)
        error = rel_l2(got[piece], F.linear(inputs[piece], weight))
        assert error < BF16_RELATIVE_L2, (piece, error)


@pytest.mark.parametrize("tokens", TOKENS)
def test_qkvr_projects_like_the_fp8_reference(attention, fp8_calls, tokens):
    attn, tensors = attention
    x = _inputs(tokens, attn.hidden_size, seed=tokens)
    pieces = attn._project_qkvr(x)
    torch.cuda.synchronize()
    _one_fp8_linear(fp8_calls)
    assert len({piece.stride() for piece in pieces}) == 1  # the prologue's row stride
    got = dict(zip(("q", "k", "v", "r", "gate", "sinks"), pieces))
    _check_projections(attn, tensors, got, dict.fromkeys(got, x))


@pytest.mark.parametrize("tokens", TOKENS)
def test_wo_ud_projects_like_the_fp8_reference(attention, fp8_calls, tokens):
    attn, tensors = attention
    weight, scale = _checkpoint(attn, tensors, "wo_ud")
    x = _inputs(tokens, weight.shape[1], seed=100 + tokens)
    out = apply_wo_ud(attn.wo_ud, x)
    torch.cuda.synchronize()
    _one_fp8_linear(fp8_calls)
    assert rel_l2(out, _fp8_reference(x, weight, scale)) < FP8_RELATIVE_L2


def test_cuda_graph_replay_equals_eager(attention):
    attn, _ = attention
    x = _inputs(8, attn.hidden_size, seed=31)
    xo = _inputs(8, attn.wo_ud.input_size_per_partition, seed=32)

    def run():
        return [*attn._project_qkvr(x), apply_wo_ud(attn.wo_ud, xo)]

    eager = [t.clone() for t in run()]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up outside the capture
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(eager, captured))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
