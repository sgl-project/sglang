"""MiniMax-M3.1 low-batch FFN kernels against the stock path they replace, on one
TP rank's layer (768 columns: TP4, 384: TP8) at 1-32 rows, eager and CUDA graph.

The routed side is compared with the stock router (``moe_fused_gate``) and
TRT-LLM-gen routed experts stopped after GEMM2, on weights SGLang's own NVFP4
loader processed; the shared side with an MXFP8 torch reference. Both chains
round through NVFP4 / MXFP8 and accumulate in different orders, so a row may
move by about a percent when a rounding flips; a wrong expert, scale, clamp,
weight or row moves it by far more. The oracle gives every expert its own
global scale and sizes the weights so the FC1 clamp decides part of the
epilogue; ``test_oracle_reaches_clamp_and_expert_scales`` checks both, and
``test_wrong_expert_scale_is_caught`` that the comparison fails a kernel
reading another expert's ``alpha``.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or torch.cuda.get_device_capability()[1] > 3,
    reason="the low-batch FFN kernels run on SM100-family GPUs",
)

ROUTED_SCALING_FACTOR = 2.0
# Per-row relative L2 error allowed between two NVFP4 / MXFP8 chains that
# accumulate in different orders: a rounding flip moves a row by under a
# percent; a wrong expert, scale, clamp or row by far more (the mutation tests).
TOLERANCE = 0.02
# Absolute error allowed on a routing weight (a row's four weights sum to
# ROUTED_SCALING_FACTOR). The front recovers each score as ``biased - bias``
# in FP32, as the routing policy defines it, so a score carries up to one FP32
# ulp of ``biased`` (2.4e-7 below 4); stock keeps the sigmoid itself. A small
# weight then differs by about 1e-6 absolute (up to 1e-3 relative).
WEIGHT_ATOL = 1e-5


@pytest.fixture(scope="module", params=[768, 384], ids=["tp4", "tp8"])
def layer(request):
    from sglang.kernels.ops.moe.minimax_m3_low_batch import ffn
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    columns = request.param
    generator = torch.Generator(device="cuda").manual_seed(columns)
    routed = ref.build_routed_layer(columns, generator)
    shared = ref.build_shared(columns, generator)
    records = ffn.shared_records(torch.device("cuda"))
    ffn.precompile_low_batch_ffn(torch.cuda.current_device(), columns)
    yield routed, shared, records
    del routed, shared
    torch.cuda.empty_cache()


def _inputs(kind: str, rows: int, generator: torch.Generator):
    """BF16 rows, their FP32 router logits and the FP32 score bias."""
    from sglang.kernels.ops.moe.minimax_m3_low_batch.shapes import EXPERTS, HIDDEN

    x = torch.randn(rows, HIDDEN, generator=generator, device="cuda") * 2
    gate = torch.randn(EXPERTS, HIDDEN, generator=generator, device="cuda") * 0.02
    bias = torch.rand(EXPERTS, generator=generator, device="cuda") * 2
    if kind == "random" and rows > 1:
        x[1] = 0  # a zero row
    x = x.bfloat16()
    # The stock BF16 router GEMM with FP32 output.
    logits = torch.mm(x, gate.bfloat16().t(), out_dtype=torch.float32)
    if kind == "one_expert_set":
        # Every row on experts 5-8: full tiles at 8 rows, several per expert above.
        bias[5:9] += 20
    elif kind == "distinct":
        # Row r on experts 4r .. 4r + 3: all 128 experts at 32 rows.
        bias.zero_()
        logits.zero_()
        for row in range(rows):
            logits[row, 4 * row : 4 * row + 4] = 8
    return x, logits.contiguous(), bias


def _run(x, logits, bias, routed, shared, records):
    from sglang.kernels.ops.moe.minimax_m3_low_batch import ffn

    columns = routed.intermediate_size_per_partition
    out, weights, permuted = ffn.low_batch_routed_ffn(
        x,
        logits,
        bias=bias,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        w13=routed.w13_weight,
        w13_scale=routed.w13_weight_scale,
        w2=routed.w2_weight,
        w2_scale=routed.w2_weight_scale,
        alpha1=routed.g1_alphas,
        alpha2=routed.g2_alphas,
        columns=columns,
    )
    shared_out = ffn.low_batch_shared_ffn(
        x,
        gate_up=shared.gate_up,
        gate_up_scale=shared.gate_up_swizzled,
        down=shared.down,
        down_scale=shared.down_swizzled,
        fc1_records=records[0],
        fc2_records=records[1],
        columns=columns,
    )
    return out, weights, permuted, shared_out


def _stock_rows(x, logits, bias, routed, quant_info=None):
    """The stock routed rows (FP32 weighted sum over the top-k), the stock
    routing weights, and the rows whose top-k set is not a near tie."""
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    rows = x.shape[0]
    stock_weights, stock_ids = ref.stock_topk(logits, bias, ROUTED_SCALING_FACTOR)
    stock = ref.stock_routed_deferred(x, stock_weights, stock_ids, routed, quant_info)
    want = ref.weighted_rows(
        stock.gemm2_out,
        stock.expert_weights,
        stock.expanded_idx_to_permuted_idx,
        rows,
    )
    return want, stock_weights, ~ref.near_tie_rows(logits, bias)


def _check(result, x, logits, bias, routed, shared):
    from sglang.kernels.ops.moe.moe_finalize_fuse_shared import (
        moe_finalize_fuse_shared,
    )
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    out, weights, permuted, shared_out = result
    rows = x.shape[0]
    assert out.dtype == torch.bfloat16 and out.shape[1] == x.shape[1]
    assert weights.dtype == torch.float32 and weights.shape == (rows, 4)
    assert permuted.dtype == torch.int32 and permuted.shape == (rows, 4)
    # Every slot points at a written row, the four of a row distinct.
    assert bool(((permuted >= 0) & (permuted < out.shape[0])).all())
    assert bool((permuted.sort(-1).values.diff(dim=-1) > 0).all())

    want, stock_weights, exact = _stock_rows(x, logits, bias, routed)
    torch.testing.assert_close(
        weights.sort(-1).values[exact],
        stock_weights.float().reshape(-1, 4)[:rows].sort(-1).values[exact],
        rtol=1e-5,
        atol=WEIGHT_ATOL,
    )

    mine = ref.weighted_rows(out, weights, permuted, rows)
    routed_error = ref.relative_errors(mine, want)
    shared_error = ref.relative_errors(shared_out, ref.shared_reference(x, shared))
    for row in range(rows):
        if not bool(x[row].any()):
            assert not bool(mine[row].any()) and not bool(shared_out[row].any()), (
                f"zero row {row} has a nonzero output"
            )
            continue
        assert shared_error[row] < TOLERANCE, (row, float(shared_error[row]))
        if exact[row]:
            assert routed_error[row] < TOLERANCE, (row, float(routed_error[row]))

    # The consumer's unfused fallback reads the handoff as the stock one.
    finalized = moe_finalize_fuse_shared(
        out, permuted.reshape(-1), weights, shared_out, 4
    )
    expected = mine + shared_out.float()
    assert ref.relative_errors(finalized, expected).max() < 1e-2


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("kind", ["random", "one_expert_set", "distinct"])
@pytest.mark.parametrize("rows", [1, 3, 8, 12, 16, 24, 32])
def test_low_batch_ffn_matches_stock(layer, rows, kind, graph):
    """At each front width (8, 16, 32 rows) and both FC2 schedules (rows <= 8
    after FC1, above it overlapped). A graph is replayed twice, on other rows
    the second time, so a record the front does not reset fails it."""
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    routed, shared, records = layer
    generator = torch.Generator(device="cuda").manual_seed(rows)
    x, logits, bias = _inputs(kind, rows, generator)
    ref.poison_allocator()
    if not graph:
        result = _run(x, logits, bias, routed, shared, records)
        torch.cuda.synchronize()
        _check(result, x, logits, bias, routed, shared)
        return
    static_x, static_logits = torch.zeros_like(x), torch.zeros_like(logits)
    _run(static_x, static_logits, bias, routed, shared, records)
    torch.cuda.synchronize()
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        result = _run(static_x, static_logits, bias, routed, shared, records)
    for step_x, step_logits in ((x, logits), (x.roll(1, 0), logits.roll(1, 0))):
        static_x.copy_(step_x)
        static_logits.copy_(step_logits)
        cuda_graph.replay()
        torch.cuda.synchronize()
        _check(result, step_x, step_logits, bias, routed, shared)


def test_oracle_reaches_clamp_and_expert_scales(layer):
    """The comparisons above can see the per-expert scales and the FC1 clamp:
    the experts' alphas differ, part of the FC1 pre-activations pass the
    SwiGLU-OAI limit, and lifting the stock clamp moves the stock rows far
    past ``TOLERANCE``."""
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    routed, _, _ = layer
    for name in ("g1_alphas", "g2_alphas"):
        alphas = getattr(routed, name).float()
        assert float(alphas.max() / alphas.min()) > 3, (name, alphas)

    generator = torch.Generator(device="cuda").manual_seed(11)
    x, logits, bias = _inputs("random", 32, generator)
    fraction = ref.fc1_clamp_fraction(x, routed)
    assert 0.02 < fraction < 0.5, fraction

    want, _, exact = _stock_rows(x, logits, bias, routed)
    unclamped, _, _ = _stock_rows(
        x, logits, bias, routed, quant_info=ref.without_fc1_clamp(routed)
    )
    keep = exact & x.ne(0).any(-1)
    moved = ref.relative_errors(unclamped, want)[keep]
    assert float((moved > 5 * TOLERANCE).float().mean()) > 0.9, moved


@pytest.mark.parametrize("wrong", ["alpha1", "alpha2"])
def test_wrong_expert_scale_is_caught(layer, wrong):
    """Mutation check of the comparison: the low-batch routed chain fed the
    previous expert's FC1 or FC2 alpha misses the stock rows by more than
    ``TOLERANCE`` on (nearly) every row."""
    from sglang.kernels.ops.moe.minimax_m3_low_batch import ffn
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    routed, _, _ = layer
    generator = torch.Generator(device="cuda").manual_seed(13)
    x, logits, bias = _inputs("random", 16, generator)
    alphas = dict(alpha1=routed.g1_alphas, alpha2=routed.g2_alphas)
    alphas[wrong] = alphas[wrong].roll(1).contiguous()
    out, weights, permuted = ffn.low_batch_routed_ffn(
        x,
        logits,
        bias=bias,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        w13=routed.w13_weight,
        w13_scale=routed.w13_weight_scale,
        w2=routed.w2_weight,
        w2_scale=routed.w2_weight_scale,
        columns=routed.intermediate_size_per_partition,
        **alphas,
    )
    torch.cuda.synchronize()
    want, _, exact = _stock_rows(x, logits, bias, routed)
    mine = ref.weighted_rows(out, weights, permuted, x.shape[0])
    keep = exact & x.ne(0).any(-1)
    errors = ref.relative_errors(mine, want)[keep]
    assert float((errors > TOLERANCE).float().mean()) > 0.9, errors


@pytest.mark.parametrize("wrong", ["gate_up_order", "down_scale"])
def test_wrong_shared_operand_is_caught(layer, wrong):
    """Mutation check of the shared comparison: the shared expert run with its
    gate and up halves swapped, or with every down block scale taken from the
    neighboring block, misses the reference by more than ``TOLERANCE`` on
    (nearly) every row."""
    from flashinfer import block_scale_interleave

    from sglang.kernels.ops.moe.minimax_m3_low_batch import ffn
    from sglang.test.kernels import minimax_m3_low_batch_ffn as ref

    _, shared, records = layer
    generator = torch.Generator(device="cuda").manual_seed(17)
    x, _, _ = _inputs("random", 16, generator)
    gate_up, gate_up_scale = shared.gate_up, shared.gate_up_scale
    down_scale = shared.down_scale
    if wrong == "gate_up_order":
        # torch has no FP8 roll: move the E4M3 bytes as uint8.
        gate_up = gate_up.view(torch.uint8).roll(shared.columns, 0).view(gate_up.dtype)
        gate_up_scale = gate_up_scale.roll(shared.columns, 0).contiguous()
    else:
        down_scale = down_scale.roll(1, 1).contiguous()
    shared_out = ffn.low_batch_shared_ffn(
        x,
        gate_up=gate_up,
        gate_up_scale=block_scale_interleave(gate_up_scale).contiguous(),
        down=shared.down,
        down_scale=block_scale_interleave(down_scale).contiguous(),
        fc1_records=records[0],
        fc2_records=records[1],
        columns=shared.columns,
    )
    torch.cuda.synchronize()
    keep = x.ne(0).any(-1)
    errors = ref.relative_errors(shared_out, ref.shared_reference(x, shared))[keep]
    assert float((errors > TOLERANCE).float().mean()) > 0.9, errors


def test_bf16_weights_variant(layer):
    """The BF16 routing weights (``weights_dtype=torch.bfloat16``) round the
    FP32 ones."""
    from sglang.kernels.ops.moe.minimax_m3_low_batch import ffn

    routed, _, _ = layer
    generator = torch.Generator(device="cuda").manual_seed(7)
    x, logits, bias = _inputs("random", 16, generator)
    ffn.precompile_low_batch_ffn(
        torch.cuda.current_device(), routed.intermediate_size_per_partition
    )
    common = dict(
        bias=bias,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        w13=routed.w13_weight,
        w13_scale=routed.w13_weight_scale,
        w2=routed.w2_weight,
        w2_scale=routed.w2_weight_scale,
        alpha1=routed.g1_alphas,
        alpha2=routed.g2_alphas,
        columns=routed.intermediate_size_per_partition,
    )
    out32, w32, p32 = ffn.low_batch_routed_ffn(x, logits, **common)
    out16, w16, p16 = ffn.low_batch_routed_ffn(
        x, logits, weights_dtype=torch.bfloat16, **common
    )
    torch.cuda.synchronize()
    assert w16.dtype == torch.bfloat16
    assert torch.equal(w16, w32.bfloat16())
    assert torch.equal(p16, p32)
    assert torch.equal(out16[p16.reshape(-1).long()], out32[p32.reshape(-1).long()])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
