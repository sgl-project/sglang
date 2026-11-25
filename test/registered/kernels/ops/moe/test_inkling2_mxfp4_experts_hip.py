"""MXFP4 Inkling2QuarkExperts vs a torch reference (gfx950)."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.inkling2_quark_kernel_test_utils import (
    TOP_K,
    assert_no_weight_sized_allocation,
    build_experts,
    ep_group,
    make_inputs,
    rel_l2,
    synth_mxfp4,
    triton_gap,
)
from sglang.test.inkling2_quark_test_utils import quantization_config
from sglang.test.kernels.gluon_moe_reference import quantize_dequantize

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

GFX95 = is_gfx95_supported()
TRITON_GAP = triton_gap("dot_scaled") if GFX95 else ""
pytestmark = [
    pytest.mark.skipif(not GFX95, reason="needs an AMD gfx95x GPU"),
    pytest.mark.skipif(bool(TRITON_GAP), reason=TRITON_GAP),
]

EXPERTS, HIDDEN, INTERMEDIATE = 128, 2304, 1152
GATE_CAP, UP_CAP = 4.0, 25.0
TOKENS = (1, 8, 64, 513, 4096)
FP32_TOL = 1e-6  # relative L2 error of the FP32 sums
CONFIG = SimpleNamespace(
    hidden_size=HIDDEN,
    intermediate_size=INTERMEDIATE,
    mlp_gate_softcap=GATE_CAP,
    mlp_up_softcap=UP_CAP,
    num_experts_per_tok=TOP_K,
    route_scale=8.0,
)

if GFX95:  # a failing import must fail the file on the machines that run it
    from sglang.kernels.ops.moe import inkling2_quark_gemm_hip as gemm
    from sglang.srt.layers.quantization.inkling2_quark import Inkling2QuarkLayout
    from sglang.srt.models.inkling_common import inkling2_moe
    from sglang.srt.models.inkling_common import inkling2_moe_quark_mxfp4 as mxfp4

    LAYOUT = Inkling2QuarkLayout.from_quantization_dict(
        quantization_config("MXFP4-RTN"), 1
    )


def quark_fake_quantize(x):
    """Quark's MXFP4 activation quantizer (E8M0 scale per 32 elements by the "even"
    rule, E2M1 elements with ties to even), as in the GLM MXFP4 tests."""
    return quantize_dequantize(x, shared=True).to(x.dtype)


def oracle_sums(x, ids, gammas, w13, w2, *, local=None):
    """FP32 sums [tokens, H] of the Quark recipe over the experts `local` = (start, count),
    all experts by default."""
    tokens = x.shape[0]
    local_ids = ids.long()
    if local is not None:
        start, count = local
        local_ids = torch.where(
            (local_ids >= start) & (local_ids < start + count), local_ids, -1
        )
    xq = quark_fake_quantize(x)
    down = torch.zeros(tokens, TOP_K, HIDDEN, dtype=torch.float32, device=x.device)
    for expert in torch.unique(local_ids).tolist():
        if expert < 0:
            continue
        token, k = (local_ids == expert).nonzero(as_tuple=True)
        gate_up = (xq[token].double() @ w13[expert].double().T).float().bfloat16()
        ones = torch.ones_like(gammas[token, k])  # the route weight comes after w2
        hidden = inkling2_moe._capped_swiglu(gate_up, ones, GATE_CAP, UP_CAP)
        out = quark_fake_quantize(hidden).double() @ w2[expert].double().T
        down[token, k] = out.float().bfloat16().float()
    sums = torch.zeros(tokens, HIDDEN, dtype=torch.float32, device=x.device)
    for k in range(TOP_K):
        sums += down[:, k] * gammas[:, k : k + 1]
    return sums


@pytest.fixture(scope="module")
def checkpoint():
    """w13 and w2 of the real shapes (scale bytes in the range of the real exports)."""
    w13 = synth_mxfp4(
        EXPERTS,
        2 * INTERMEDIATE,
        HIDDEN,
        seed=1,
        scale_lo=118,
        scale_hi=125,
        device="cuda",
    )
    w2 = synth_mxfp4(
        EXPERTS, HIDDEN, INTERMEDIATE, seed=2, scale_lo=118, scale_hi=125, device="cuda"
    )
    return {"w13": w13, "w2": w2}


@pytest.fixture
def experts(checkpoint):
    with ep_group():
        yield build_experts(
            CONFIG, LAYOUT, EXPERTS, checkpoint["w13"], checkpoint["w2"]
        )


def inputs(tokens, seed=0, outliers=False):
    """Hidden states, route weights (summing to the route scale 8) and expert ids [T, 8]."""
    return make_inputs(
        tokens, HIDDEN, EXPERTS, seed=1000 * seed + tokens, outliers=outliers
    )


@pytest.mark.parametrize("tokens", TOKENS)
def test_forward_equals_the_torch_pipeline(experts, checkpoint, tokens):
    if tokens == 1:  # these token counts must cover the split-K FP32 slabs ...
        assert (
            gemm.default_config_quark(gemm.MXFP4, tokens * TOP_K, EXPERTS).split_k > 1
        )
    if tokens == 4096:  # ... and the plain BF16 epilogue
        assert (
            gemm.default_config_quark(gemm.MXFP4, tokens * TOP_K, EXPERTS).split_k == 1
        )
    for seed, outliers in ((0, False), (1, True)):
        x, gammas, ids = inputs(tokens, seed, outliers)
        x_before = x.clone()
        expected = oracle_sums(
            x, ids, gammas, checkpoint["w13"]["expected"], checkpoint["w2"]["expected"]
        )
        sums = experts(x, gammas, ids, combine=False).rows
        assert torch.equal(x, x_before)  # quantized into a new tensor, never in place
        assert sums.dtype == torch.float32 and sums.shape == (tokens, HIDDEN)
        assert torch.isfinite(sums).all()
        assert rel_l2(sums, expected) < FP32_TOL
        assert torch.equal(
            sums, experts(x, gammas, ids, combine=False).rows
        )  # deterministic
        combined = experts(
            x, gammas, ids
        )  # combine=True: the FP32 sums rounded once to BF16
        assert combined.dtype == torch.bfloat16 and torch.equal(
            combined, sums.bfloat16()
        )
        # the BF16 outputs differ only where an FP32 sum sits at a rounding boundary
        flips = int((combined != expected.bfloat16()).sum())
        assert flips <= 1e-3 * combined.numel()


@pytest.mark.parametrize("tokens", [1, 64, 513])
def test_expert_parallel_ranks_add_up_to_the_full_result(
    monkeypatch, checkpoint, tokens
):
    x, gammas, ids = inputs(tokens)
    x_before = x.clone()
    num_local = EXPERTS // 2
    full = oracle_sums(
        x, ids, gammas, checkpoint["w13"]["expected"], checkpoint["w2"]["expected"]
    )
    flat_ids = ids.flatten()
    partial = []
    for rank in range(2):
        start = rank * num_local
        foreign = (flat_ids < start) | (flat_ids >= start + num_local)
        real_gemm = gemm.expert_gemm_quark

        def poisoned(*args, _real=real_gemm, _foreign=foreign, **kwargs):
            out = _real(
                *args, **kwargs
            )  # [slabs, rows, N]: rows of foreign experts are unset memory
            out[:, _foreign] = float("nan")
            return out

        with ep_group(2, rank):
            experts = build_experts(
                CONFIG, LAYOUT, EXPERTS, checkpoint["w13"], checkpoint["w2"]
            )
            monkeypatch.setattr(mxfp4.gemm, "expert_gemm_quark", poisoned)
            sums = experts(x, gammas, ids, combine=False).rows
            monkeypatch.setattr(mxfp4.gemm, "expert_gemm_quark", real_gemm)
        assert torch.isfinite(sums).all()
        assert torch.equal(x, x_before)
        expected = oracle_sums(
            x,
            ids,
            gammas,
            checkpoint["w13"]["expected"],
            checkpoint["w2"]["expected"],
            local=(start, num_local),
        )
        assert rel_l2(sums, expected) < FP32_TOL
        partial.append(sums)
    assert rel_l2(partial[0] + partial[1], full) < FP32_TOL


@pytest.mark.parametrize("tokens", [1, 64, 513])
def test_cuda_graph_replay_equals_eager_and_nothing_syncs(experts, tokens):
    x, gammas, ids = inputs(tokens)

    def run():
        return experts(x, gammas, ids, combine=False).rows

    eager = run()  # warm-up outside a capture builds the decode tables
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(
        "error"
    )  # a prototype: it does not see every kind of sync
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)


@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_a_decode_size_forward_allocates_no_weight_sized_scratch(experts, tokens):
    """No dequantized copy of the weights: one layer's w13 alone would be 1.36 GB."""
    x, gammas, ids = inputs(tokens)

    def run():
        return experts(x, gammas, ids, combine=False).rows

    assert_no_weight_sized_allocation(run, experts.packed_bytes())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
