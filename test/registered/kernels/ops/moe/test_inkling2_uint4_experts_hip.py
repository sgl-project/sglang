"""UINT4-AWQ Inkling2QuarkExperts vs a torch reference (gfx950)."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization.inkling2_quark import Inkling2QuarkLayout
from sglang.srt.models.inkling_common import inkling2_moe as moe
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.inkling2_quark_kernel_test_utils import (
    TOP_K,
    assert_no_weight_sized_allocation,
    build_experts,
    ep_group,
    experts_slice,
    make_inputs,
    reference_grouped,
    rel_l2,
    synth_uint4,
    triton_gap,
)
from sglang.test.inkling2_quark_test_utils import quantization_config

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

GFX95 = is_gfx95_supported()
TRITON_GAP = triton_gap("join") if GFX95 else ""
pytestmark = [
    pytest.mark.skipif(not GFX95, reason="needs an AMD gfx95x GPU"),
    pytest.mark.skipif(bool(TRITON_GAP), reason=TRITON_GAP),
]

H, I, E = 2304, 1152, 16  # hidden, intermediate, experts (top-8 routing needs 8)
GATE_CAP, UP_CAP, ROUTE_SCALE = 4.0, 25.0, 8.0  # the values of the model
X_SCALE = 0.25  # pre-activations of about 3: the caps act on a share of the entries
TOL = 1e-3  # relative L2 error of the BF16 output
TOL_SUMS = 5e-4  # and of the FP32 sums
# split-K slabs (1, 8), BF16 output (48+), the F32-scale config (130), 4104 routed rows
TOKENS = (1, 8, 48, 130, 513)
KINDS = ("bf16", "bf16_as_f32", "f32")  # dtype of the scale tensors the kernels read
CONFIG = SimpleNamespace(
    hidden_size=H,
    intermediate_size=I,
    mlp_gate_softcap=GATE_CAP,
    mlp_up_softcap=UP_CAP,
    num_experts_per_tok=TOP_K,
    route_scale=ROUTE_SCALE,
)
LAYOUT = Inkling2QuarkLayout.from_quantization_dict(quantization_config("UINT4-AWQ"), 1)


@pytest.fixture(scope="module")
def fixtures():
    """(w13, w2) with BF16 and with full-mantissa F32 scales."""
    return {
        name: (
            synth_uint4(E, 2 * I, H, seed=31, scale_dtype=dtype, device="cuda"),
            synth_uint4(E, H, I, seed=37, scale_dtype=dtype, device="cuda"),
        )
        for name, dtype in (("bf16", torch.bfloat16), ("f32", torch.float32))
    }


def tensors_of(fixtures, kind):
    return fixtures["f32" if kind == "f32" else "bf16"]


def repeated(data, times):
    return {k: v.repeat(times, 1, 1) for k, v in data.items() if k != "expected"}


def uint4_experts(fixtures, kind, num_experts=E):
    """Inkling2QuarkExperts of the current ep_group(). kind: "f32" or "bf16_as_f32" (the
    loader's upcast of BF16 scales), or "bf16" scale tensors as on disk."""
    w13, w2 = tensors_of(fixtures, kind)
    if num_experts > E:
        w13, w2 = (repeated(data, num_experts // E) for data in (w13, w2))
    experts = build_experts(CONFIG, LAYOUT, num_experts, w13, w2)
    if kind == "bf16":
        part = experts_slice(experts)
        experts.w13_scale.data = w13["weight_scale"][part]
        experts.w2_scale.data = w2["weight_scale"][part]
    return experts


def inputs(tokens, seed, num_experts=E):
    return make_inputs(
        tokens, H, num_experts, seed=seed, x_scale=X_SCALE, route_scale=ROUTE_SCALE
    )


def oracle_sums(fixtures, kind, x, weights, ids, part=slice(0, E)):
    """FP32 per-token sums of the BF16 expert pipeline over the experts ``part``."""
    w13, w2 = tensors_of(fixtures, kind)
    tokens, start = ids.shape[0], part.start
    gate_up, _ = reference_grouped(x, w13["expected"][part], ids, TOP_K, start)
    gate_up = gate_up.bfloat16().float()  # BF16 values held in FP32
    hidden = moe._capped_swiglu(gate_up, weights.flatten(), GATE_CAP, UP_CAP)
    hidden = hidden.bfloat16().float()
    down, _ = reference_grouped(hidden, w2["expected"][part], ids, 1, start)
    down = down.bfloat16().float().view(tokens, TOP_K, H)
    total = down[:, 0]
    for k in range(1, TOP_K):
        total = total + down[:, k]
    return total


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tokens", TOKENS)
def test_forward_equals_the_torch_pipeline(fixtures, kind, tokens):
    x, weights, ids = inputs(tokens, seed=tokens)
    ref_sums = oracle_sums(fixtures, kind, x, weights, ids)
    with ep_group():
        experts = uint4_experts(fixtures, kind)
        out = experts(x, weights, ids)
        routed = experts(x, weights, ids, combine=False)
    assert out.dtype == torch.bfloat16 and out.shape == (tokens, H)
    assert rel_l2(out, ref_sums.to(torch.bfloat16)) <= TOL
    # combine=False: the FP32 sums in token order, identity src2dst, as the native path
    assert routed.rows.dtype == torch.float32 and routed.top_k == 1
    assert torch.equal(routed.src2dst, torch.arange(tokens, device="cuda").int())
    assert torch.equal(routed.rows.to(torch.bfloat16), out)
    assert rel_l2(routed.rows, ref_sums) <= TOL_SUMS


@pytest.mark.parametrize("tokens", (3, 48))
def test_expert_parallel_ranks_add_up_to_the_full_forward(fixtures, tokens):
    x, weights, ids = inputs(tokens, seed=100 + tokens)
    with ep_group():
        full = uint4_experts(fixtures, "f32")(x, weights, ids, combine=False).rows
    reduced, partial = [], []
    for rank in (0, 1):
        reduced.clear()

        def all_reduce(sums):
            reduced.append((sums.dtype, tuple(sums.shape)))
            return sums

        with ep_group(2, rank, all_reduce):
            experts = uint4_experts(fixtures, "f32")
            part = experts_slice(experts)
            partial.append(experts(x, weights, ids, combine=False).rows)
        assert reduced == [(torch.float32, (tokens, H))]  # one FP32 all-reduce
        want = oracle_sums(fixtures, "f32", x, weights, ids, part)
        assert rel_l2(partial[rank], want) <= TOL_SUMS
    assert rel_l2(partial[0] + partial[1], full) <= 1e-6
    # the reduced sums the model returns: what the other rank's partial adds to ours
    with ep_group(2, 0, lambda sums: sums + partial[1]):
        out = uint4_experts(fixtures, "f32")(x, weights, ids)
    assert torch.equal(out, (partial[0] + partial[1]).to(torch.bfloat16))


@pytest.mark.parametrize("tokens", (1, 64))
def test_cuda_graph_replay_follows_new_inputs(fixtures, tokens):
    static = [t.clone() for t in inputs(tokens, seed=1)]
    with ep_group():
        experts = uint4_experts(fixtures, "f32")
        experts(*static)  # compile outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = experts(*static)
        for seed in (1, 2, 3):  # same buffers, new tokens and routing
            for buf, new in zip(static, inputs(tokens, seed)):
                buf.copy_(new)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, experts(*static))


REAL_E = 128  # experts of a real layer: 545 MB of packed weights
# the scratch of a decode forward is at most 5.6 % of that, a BF16 copy of the packed
# words of w2 alone 15.6 %
SCRATCH_SHARE = 0.10


@pytest.mark.parametrize("tokens", (1, 8, 64))
def test_decode_forward_allocates_no_weight_sized_scratch(fixtures, tokens):
    """The first forward on these experts (no warm-up) builds and keeps no weight-sized
    temporary."""
    with ep_group():
        experts = uint4_experts(fixtures, "f32", REAL_E)
        x, weights, ids = inputs(tokens, seed=tokens, num_experts=REAL_E)
        assert_no_weight_sized_allocation(
            lambda: (
                experts(x, weights, ids),
                experts(x, weights, ids, combine=False),
            ),
            experts.packed_bytes(),
            share=SCRATCH_SHARE,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
