"""Independent test data and FP64 interval oracle for compact FP4 scores.

No production decode/score helper is used by the oracle. The two visible BF16
rounds and final BF16 round are explicit. FP32 dot uncertainty is bounded by
gamma(128)*sum(abs(q*k)); head reduction by gamma(31)*sum(abs(weighted)).
These standard rounding bounds cover either parity accumulation order and any
FP32 reduction tree, for these finite, nonoverflowing normal-range fixtures.
Monotone BF16 rounding propagates interval endpoints through each round; signed
weights swap endpoints. Exact analytic cases additionally require zero error.
"""

import math
import os
from dataclasses import dataclass

import torch

SEED = 40860
SUPPORTED_TOPK = (512, 1024)
GATES = {
    "cuda_triton": "FP64 interval and full-row selection contract; bitwise delta reported; optional require_bitwise parity gate; exact -inf mask",
    "oracle": "FP64 full dot with explicit BF16 rounds; propagated gamma128/gamma31 interval",
    "analytic": "exact scores and masks",
    "topk": "512/1024 on every row/all candidates; exact higher-than-cutoff membership; equal-cutoff alternatives accepted",
    "no_post_result_relaxation": True,
    "reference_semantics_erratum": "Before CUDA testing: Triton SM90 exp2 FTZ maps exponent byte0 to scale0; mathematical UE8M0 remains2^-127, recorded separately",
    "fixture_domain": "H32 D128, finite BF16 Q/W; random K exponents119..135 plus analytic0/27/127/227; no overflow",
}


@dataclass
class Inputs:
    q: torch.Tensor
    weights: torch.Tensor
    req: torch.Tensor
    req_table: torch.Tensor
    lens: torch.Tensor
    table: torch.Tensor
    page_size: int
    max_len: int
    ratio: int
    blocks: torch.Tensor

    def args(self):
        return (
            self.q,
            self.weights,
            self.req,
            self.req_table,
            self.lens,
            self.table,
            self.page_size,
            self.max_len,
            self.ratio,
            self.blocks,
        )

    def bytes(self):
        return sum(
            x.numel() * x.element_size()
            for x in self.args()
            if isinstance(x, torch.Tensor)
        )


def implementations(baseline_only=False):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        fp4_index_logits_candidates_triton,
    )

    if baseline_only or os.environ.get("FP4_CANDIDATE_TEST_IMPL") == "triton":
        return {"triton": fp4_index_logits_candidates_triton}
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_cuda import (
        fp4_index_logits_candidates_cuda,
    )

    return {
        "triton": fp4_index_logits_candidates_triton,
        "cuda": fp4_index_logits_candidates_cuda,
    }


def make_inputs(
    requests=2,
    queries=2,
    candidates=72,
    *,
    max_len=137,
    ratio=1,
    mapping="random",
    q_kind="arbitrary",
    dtype=torch.int32,
    padded=False,
    seed=SEED,
    device="cuda",
    analytic=False,
    strided=False,
):
    assert candidates % 8 == 0
    g = torch.Generator().manual_seed(seed)
    rows, page_size = requests * queries, 64
    slots_per_req = math.ceil(max(max_len, 1) / page_size) * page_size
    slots = requests * slots_per_req
    pages = slots // page_size
    # Guard bytes survive H2D and expose incorrect cache stride assumptions.
    backing = torch.full(
        (pages, page_size * 68 + (37 if padded else 0)), 0xA5, dtype=torch.uint8
    )
    table = backing[:, : page_size * 68]
    payload = torch.randint(
        0, 256, (pages, page_size * 64), generator=g, dtype=torch.uint8
    )
    exps = torch.randint(
        119, 136, (pages, page_size * 4), generator=g, dtype=torch.uint8
    )
    table[:, : page_size * 64] = payload
    table[:, page_size * 64 :] = exps
    q = torch.randn(rows, 32, 128, generator=g).to(torch.bfloat16)
    weights = (torch.randn(rows, 32, generator=g) / 32).to(torch.bfloat16)
    # Mixed zero, cancellation and independent scales are intentional.
    if rows:
        q[0, 0] = 0
        weights[0, 1:3] = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
        q[0, 2] = q[0, 1]
    req = torch.arange(requests, dtype=dtype).repeat_interleave(queries)
    req = req.flip(0)
    tb = torch.full(
        (requests, max_len * ratio + (19 if padded else 0)), -31337, dtype=dtype
    )
    req_table = tb[:, : max_len * ratio]
    for r in range(requests):
        physical = torch.arange(r * slots_per_req, (r + 1) * slots_per_req)
        if mapping == "random":
            physical = physical[torch.randperm(slots_per_req, generator=g)]
        req_table[r, ::ratio] = physical[:max_len].to(dtype) * ratio
    lens = torch.full((rows,), max_len, dtype=dtype)
    nblocks = candidates // 8
    blocks = torch.empty((rows, nblocks), dtype=dtype)
    for row in range(rows):
        block_ids = torch.arange(max(1, math.ceil(max_len / 8)))
        if mapping == "random":
            block_ids = block_ids[torch.randperm(block_ids.numel(), generator=g)]
        # Different verify rows can carry different candidate lists for one req.
        blocks[row] = block_ids.roll(row % max(block_ids.numel(), 1)).repeat(
            math.ceil(nblocks / max(block_ids.numel(), 1))
        )[:nblocks]
    if analytic:
        table[:, : page_size * 64] = 0
        table[:, page_size * 64 :] = 127
        # First K coordinate cycles through exactly representable E2M1 values.
        table[:, : page_size * 64].reshape(pages, page_size, 64)[:, :, 0] = (
            torch.arange(slots).reshape(pages, page_size).remainder(8).byte()
        )
        q.zero_()
        weights.zero_()
        if rows:
            q[:, 0, 0] = 1
            weights[:, 0] = 1

    def transfer(x):
        if padded and x is table:
            return backing.to(device)[:, : page_size * 68]
        if padded and x is req_table:
            return tb.to(device)[:, : max_len * ratio]
        return x.to(device)

    result = Inputs(
        *(transfer(x) for x in (q, weights, req, req_table, lens, table)),
        page_size,
        max_len,
        ratio,
        transfer(blocks),
    )
    if q_kind == "fakequant":
        # Exercise the actual production query preparation; oracle is independent.
        from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4

        result.q = fake_quant_fp4(result.q)
    if strided:
        for name in ("q", "weights", "req", "lens", "blocks"):
            x = getattr(result, name)
            bigger = torch.empty(
                (*x.shape[:-1], x.shape[-1] * 2), dtype=x.dtype, device=device
            )
            bigger[..., ::2].copy_(x)
            setattr(result, name, bigger[..., ::2])
    return result


def coordinates(inputs):
    blocks = inputs.blocks.cpu().long()
    logical = (blocks[:, :, None] * 8 + torch.arange(8)).flatten(1)
    valid = (
        (blocks.repeat_interleave(8, 1) >= 0)
        & (logical < inputs.lens.cpu().long()[:, None])
        & (logical < inputs.max_len)
    )
    slots = torch.zeros_like(logical)
    if valid.any():
        rows = inputs.req.cpu().long()[:, None].expand_as(logical)
        slots[valid] = (
            inputs.req_table.cpu().long()[rows[valid], logical[valid] * inputs.ratio]
            // inputs.ratio
        )
    return logical, slots, valid


def decode_slots(table, slots, page_size, *, reference_ftz=True):
    """Explicit layout and 16-entry E2M1 LUT, independent of Triton helpers."""
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float64,
    )
    table = table.cpu()
    page, off = slots // page_size, slots % page_size
    packed = table[page[:, None], off[:, None] * 64 + torch.arange(64)].long()
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(1)
    exp = table[
        page[:, None], page_size * 64 + off[:, None] * 4 + torch.arange(4)
    ].long()
    scale = torch.pow(2.0, exp.double() - 127)
    # Explicit baseline hardware semantic, independently established by a
    # Triton-only magnitude2/Q2^100 probe before any CUDA acceptance run.
    # Mathematical UE8M0 byte0 is2^-127; SM90 ex2.approx.ftz flushes that scale.
    if reference_ftz:
        scale = torch.where(exp == 0, 0.0, scale)
    scale = scale.repeat_interleave(32, 1)
    return (lut[codes] * scale).to(torch.bfloat16).double()


def bf16(x):
    return x.to(torch.bfloat16).double()


def oracle(inputs, row_ids=None, col_ids=None):
    logical, slots, valid = coordinates(inputs)
    rows = list(range(inputs.q.shape[0])) if row_ids is None else list(row_ids)
    cols = (
        torch.arange(inputs.blocks.shape[1] * 8)
        if col_ids is None
        else torch.tensor(col_ids)
    )
    shape = (len(rows), len(cols))
    exact = torch.full(shape, -torch.inf, dtype=torch.float64)
    lower, upper = exact.clone(), exact.clone()
    u = 2.0**-24
    gamma_dot, gamma_sum = 128 * u / (1 - 128 * u), 31 * u / (1 - 31 * u)
    q, w = inputs.q.cpu().double(), inputs.weights.to(torch.bfloat16).cpu().double()
    for i, row in enumerate(rows):
        mask = valid[row, cols]
        if not mask.any():
            continue
        k = decode_slots(inputs.table, slots[row, cols[mask]], inputs.page_size)
        dot = q[row] @ k.T
        error = gamma_dot * (q[row].abs() @ k.abs().T)
        center = bf16(bf16(dot).clamp_min(0) * w[row, :, None])
        lo = bf16(dot - error).clamp_min(0) * w[row, :, None]
        hi = bf16(dot + error).clamp_min(0) * w[row, :, None]
        lo, hi = bf16(torch.minimum(lo, hi)), bf16(torch.maximum(lo, hi))
        sum_error = gamma_sum * torch.maximum(lo.abs(), hi.abs()).sum(0)
        exact[i, mask] = bf16(center.sum(0))
        lower[i, mask] = bf16(lo.sum(0) - sum_error)
        upper[i, mask] = bf16(hi.sum(0) + sum_error)
    return exact, lower, upper, valid[rows][:, cols]


def ulp_distance(a, b):
    def ordered(x):
        bits = x.to(torch.bfloat16).contiguous().view(torch.int16).long()
        return torch.where(bits < 0, -32768 - bits, bits)

    return (ordered(a) - ordered(b)).abs()


def assert_topk_ties(actual, expected, k=16):
    for a, e in zip(actual.cpu(), expected.cpu()):
        finite = torch.isfinite(e)
        count = min(k, int(finite.sum()))
        if not count:
            continue
        cutoff = e[finite].topk(count).values[-1]
        chosen = a.topk(count).indices
        assert (e[chosen] >= cutoff).all(), "top-k selected a below-cutoff score"
        required = (e > cutoff).nonzero().flatten()
        assert torch.isin(required, chosen).all(), "top-k omitted an above-cutoff score"


def assert_topk_intervals(actual, lower, upper, k):
    """Reject a chosen candidate that is provably worse under the frozen bounds."""
    for got, lo, hi in zip(actual.cpu().double(), lower, upper):
        count = min(k, int(torch.isfinite(lo).sum()))
        if not count:
            continue
        chosen = got.topk(count).indices
        selected = torch.zeros(got.numel(), dtype=torch.bool)
        selected[chosen] = True
        unchosen = ~selected & torch.isfinite(lo)
        if unchosen.any():
            assert not bool(lo[unchosen].max() > hi[chosen].min()), (
                "top-k violates independent FP64 selection intervals"
            )


def check(inputs, funcs=None, *, analytic=False, sample=False, require_bitwise=False):
    """Apply frozen arithmetic/mapping/selection gates; report exactness separately.

    Row sampling never samples candidate columns. Historical parity experiments
    can additionally request the original CUDA/Triton bitwise requirement.
    """
    funcs = implementations() if funcs is None else funcs
    outputs = {name: fn(*inputs.args()) for name, fn in funcs.items()}
    torch.cuda.synchronize()
    logical, slots, valid = coordinates(inputs)
    for name, out in outputs.items():
        assert out.shape == valid.shape and out.dtype == torch.float32, name
        assert torch.equal(torch.isfinite(out).cpu(), valid), name
        assert torch.equal(torch.isneginf(out).cpu(), ~valid), name

    exact, different, different_finite, different_zero = False, 0, 0, 0
    cuda_triton_ulp = 0
    if "cuda" in outputs and "triton" in outputs:
        cuda, triton = outputs["cuda"], outputs["triton"]
        changed = cuda.view(torch.int32) != triton.view(torch.int32)
        exact = not bool(changed.any())
        different = int(changed.sum())
        different_finite = int((changed & valid.to(changed.device)).sum())
        different_zero = int((changed & (cuda == 0) & (triton == 0)).sum())
        if valid.any():
            cuda_triton_ulp = int(
                ulp_distance(
                    cuda[valid.to(cuda.device)], triton[valid.to(triton.device)]
                ).max()
            )
        if require_bitwise:
            assert exact, "CUDA/Triton bitwise mismatch; finite BF16 ULP max=" + str(
                cuda_triton_ulp
            )
        for k in SUPPORTED_TOPK:
            assert_topk_ties(cuda, triton, k=k)

    rows = (
        sorted(set([0, max(0, inputs.q.shape[0] - 1)]))
        if sample and inputs.q.shape[0]
        else None
    )
    # Every candidate in each selected row contributes to the numerical and
    # selection checks, including narrow score-boundary regressions.
    expected, lo, hi, mask = oracle(inputs, row_ids=rows)
    max_ulp, max_abs = 0, 0.0
    for name, out in outputs.items():
        got = out.cpu().double()
        if rows is not None:
            got = got[rows]
        assert ((got[mask] >= lo[mask]) & (got[mask] <= hi[mask])).all(), (
            f"{name}: independent FP64 interval failure"
        )
        if analytic:
            assert torch.equal(got, expected), f"{name}: exact analytic score failure"
        if mask.any():
            max_ulp = max(max_ulp, int(ulp_distance(got[mask], expected[mask]).max()))
            max_abs = max(max_abs, float((got[mask] - expected[mask]).abs().max()))
        for k in SUPPORTED_TOPK:
            assert_topk_intervals(got, lo, hi, k)
            if torch.equal(lo, hi):
                assert_topk_ties(got, expected, k=k)
    return {
        "max_oracle_bf16_ulps": max_ulp,
        "max_oracle_abs": max_abs,
        "oracle_rows": rows if rows is not None else "all",
        "oracle_columns": "all",
        "topk": list(SUPPORTED_TOPK),
        "cuda_triton_exact": exact,
        "cuda_triton_different_elements": different,
        "cuda_triton_different_finite_elements": different_finite,
        "cuda_triton_signed_zero_differences": different_zero,
        "max_cuda_triton_bf16_ulps": cuda_triton_ulp,
    }
