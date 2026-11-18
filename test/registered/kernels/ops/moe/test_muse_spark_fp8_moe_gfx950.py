"""Muse Spark Quark PTPC-FP8 routed experts on gfx950 (``rocm_muse_spark.muse_moe_fp8``).

The expert path composes the public Triton FP8 MoE GEMM (per-token activation
quantization, per-channel weight scales, one output row per routed slot), the
SGLang SiLU-and-mul and Muse's finalize (per-slot post-expert norm, then the
weighted top-k sum). The cases run the EP8 rank shapes of the release (32 local
experts at a non-zero offset, K = I = H = 4096, top-8) with heterogeneous
channel scales and token magnitudes:

* each stage is teacher-forced: fed production's own quantized codes / scales
  and the previous stage's production output, and held to the floating-point
  bounds in ``sglang.test.kernels.muse_spark_fp8_moe``;
* an eps-dominant variant (expert outputs far below sqrt(eps)) and a no-norm
  variant make scale, routed-weight and eps errors visible through the
  finalize, which an RMSNorm of O(1) rows would normalize away;
* graph replay with new inputs and routes is bit-equal to eager;
* an independent end-to-end oracle characterizes gross errors.
"""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.muse_spark_fp8_moe import (
    U_BF16,
    exceeds,
    finalize_bound,
    finalize_reference,
    gemm_row_bounds,
    muse_moe_fp8_reference,
    quantize_rows,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

NUM_EXPERTS, LOCAL, LOCAL_START, TOP_K = 256, 32, 96, 8
HIDDEN = INTER = 4096
EPS, GAIN = 1e-5, 1.0
ROWS = (1, 8, 9, 16, 32, 64, 128, 512, 513)
# Production SiLU-and-mul rounds silu(gate) to BF16 and then the product (two roundings, see
# kernels/aot/csrc/elementwise/activation.cu); FP32 expf / division add well under 1e-6 relative.
SILU_REL = 2 * U_BF16 + U_BF16**2 + 1e-6
SILU_EXP_OVERFLOW = 88.72
BF16_HALF_SUBNORMAL = 2.0**-134
# Gross-error characterization of the independent oracle, aggregated over all cases. Pending a derived bar.
E2E_REL_L2_PENDING = 2e-2
E2E_MIN_COS_PENDING = 0.995


def _routes(m: int, gen: torch.Generator, local_share: float = 0.5):
    """Distinct top-8 ids per token, about ``local_share`` of them on this rank; renormalized scores."""
    ids = torch.empty(m, TOP_K, dtype=torch.int64)
    local_ids = torch.arange(LOCAL_START, LOCAL_START + LOCAL)
    remote_ids = torch.cat(
        [torch.arange(0, LOCAL_START), torch.arange(LOCAL_START + LOCAL, NUM_EXPERTS)]
    )
    for t in range(m):
        n_local = int(
            torch.binomial(
                torch.tensor(float(TOP_K)), torch.tensor(local_share), generator=gen
            )
        )
        pick_local = local_ids[torch.randperm(LOCAL, generator=gen)[:n_local]]
        pick_remote = remote_ids[
            torch.randperm(remote_ids.numel(), generator=gen)[: TOP_K - n_local]
        ]
        ids[t] = torch.cat([pick_local, pick_remote])[
            torch.randperm(TOP_K, generator=gen)
        ]
    scores = torch.rand(m, TOP_K, generator=gen) + 0.05
    return ids.cuda(), (scores / scores.sum(-1, keepdim=True)).cuda()


def _tokens(m: int, gen: torch.Generator) -> torch.Tensor:
    """Token rows with magnitudes spread over 0.01..100."""
    magnitude = 10.0 ** (torch.rand(m, 1, generator=gen) * 4 - 2)
    return (torch.randn(m, HIDDEN, generator=gen) * magnitude).cuda().bfloat16()


def _bank(rows: int, cols: int, gen: torch.Generator):
    """FP8 codes and scales of a weight whose output channels span 0.1..10 in magnitude."""
    spread = 10.0 ** (torch.rand(rows, 1, generator=gen) * 2 - 1)
    return quantize_rows(
        (torch.randn(rows, cols, generator=gen) * 0.02 * spread).cuda()
    )


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 only")
class TestMuseSparkFp8MoE(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.rocm_muse_spark.muse_moe_fp8 import muse_moe_fp8

        cls.moe = staticmethod(muse_moe_fp8)
        gen = torch.Generator().manual_seed(0)
        w13, s13, w2, s2 = [], [], [], []
        for _ in range(LOCAL):
            codes, scale = _bank(2 * INTER, HIDDEN, gen)
            w13.append(codes), s13.append(scale)
            codes, scale = _bank(HIDDEN, INTER, gen)
            w2.append(codes), s2.append(scale)
        cls.w13, cls.s13 = torch.stack(w13), torch.stack(s13)
        cls.w2, cls.s2 = torch.stack(w2), torch.stack(s2)
        cls.norm = (torch.randn(HIDDEN, generator=gen) * 0.1).cuda().bfloat16()

    def run_moe(self, x, scores, ids, *, w2_scale=None, has_norm=True):
        return self.moe(
            hidden_states=x,
            topk_scores=scores,
            topk_ids=ids,
            w13=self.w13,
            w13_scale=self.s13,
            w2=self.w2,
            w2_scale=self.s2 if w2_scale is None else w2_scale,
            local_expert_start=LOCAL_START,
            post_norm_weight=self.norm,
            eps=EPS,
            post_norm_gain=GAIN,
            has_norm=has_norm,
        )

    def local_slots(self, ids):
        flat = ids.reshape(-1) - LOCAL_START
        slots = ((flat >= 0) & (flat < LOCAL)).nonzero().squeeze(1)
        return slots, flat[slots]

    def stages(self, x, scores, ids, w2_scale):
        """Production stage outputs with the alignment and configs the glue uses."""
        return self.stages_at(x, scores, ids, LOCAL_START, w2_scale)

    def stages_at(self, x, scores, ids, local_start, w2_scale=None):
        from sglang.kernels.ops.activation import silu_and_mul
        from sglang.srt.rocm_muse_spark.muse_moe_fp8 import moe_fp8_gemm, pick_configs
        from sglang.srt.rocm_muse_spark.muse_moe_triton import moe_align_block_size

        w2_scale = self.s2 if w2_scale is None else w2_scale
        m = x.shape[0]
        ids32, scores32 = ids.to(torch.int32), scores.float()
        up, down = pick_configs(m * TOP_K)
        align = moe_align_block_size(
            topk_ids=ids32,
            num_local_experts=LOCAL,
            local_start=local_start,
            block_m=up["BLOCK_SIZE_M"],
        )
        gate_up = torch.empty(m * TOP_K, 2 * INTER, device="cuda", dtype=torch.bfloat16)
        moe_fp8_gemm(
            a=x,
            w=self.w13,
            w_scale=self.s13,
            out=gate_up,
            scores=scores32,
            ids=ids32,
            align=align,
            top_k=TOP_K,
            config=up,
        )
        inter = silu_and_mul(gate_up)
        y = torch.empty(m * TOP_K, HIDDEN, device="cuda", dtype=torch.bfloat16)
        moe_fp8_gemm(
            a=inter,
            w=self.w2,
            w_scale=w2_scale,
            out=y,
            scores=scores32,
            ids=ids32,
            align=align,
            top_k=1,
            config=down,
        )
        return gate_up, inter, y

    def assert_gemm(self, out, a_codes, a_scale, w, w_scale, rows, experts, what):
        """Rows ``rows`` of ``out`` against exact float64 products of the production codes and scales."""
        a = a_codes.double() * a_scale.double()
        failures = []
        for e in experts.unique().tolist():
            sel = experts == e
            b = w[e].double() * w_scale[e].double().unsqueeze(-1)
            self.assertTrue(
                bool(torch.isfinite(out[sel]).all()), f"{what} expert {e} not finite"
            )
            bounds = gemm_row_bounds(out[sel], a[rows[sel]], b)
            for key in ("deterministic", "statistical"):
                bad = exceeds(bounds["err"], bounds[key])
                if bad.any():
                    i = int(bad.nonzero()[0])
                    failures.append(
                        f"{what} expert {e} {key}: err {float(bounds['err'][i]):.3e} > "
                        f"{float(bounds[key][i]):.3e} (|ref| {float(bounds['ref_norm'][i]):.3e}, "
                        f"R {float(bounds['cancellation'][i]):.1f}, K {bounds['k']})"
                    )
        self.assertFalse(failures, failures[:4])

    def assert_finalize(self, out, y, scores, slots, has_norm=True):
        m = out.shape[0]
        ref, cancellation = finalize_reference(
            y[slots], scores, slots, m, self.norm, GAIN, EPS, has_norm=has_norm
        )
        has_local = torch.zeros(m, dtype=torch.bool, device="cuda")
        has_local[slots // TOP_K] = True
        self.assertTrue(bool(torch.isfinite(out).all()), "finalize output not finite")
        err = (out.double() - ref).norm(dim=-1)
        bound = finalize_bound(ref, cancellation, HIDDEN)
        bad = has_local & exceeds(err, bound)
        self.assertFalse(
            bool(bad.any()),
            f"finalize: rows {bad.nonzero().squeeze(1).tolist()[:4]} err {err[bad][:4].tolist()} "
            f"bound {bound[bad][:4].tolist()}",
        )
        self.assertTrue(bool((out[~has_local] == 0).all()))

    def check_stages(self, m, gen, *, w2_scale=None, has_norm=True, local_share=0.5):
        from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant

        w2_scale = self.s2 if w2_scale is None else w2_scale
        x = _tokens(m, gen)
        ids, scores = _routes(m, gen, local_share)
        slots, experts = self.local_slots(ids)
        gate_up, inter, y = self.stages(x, scores, ids, w2_scale)
        # The wrapper quantizes A per token with scaled_fp8_quant; the same call yields the same codes.
        xq, xs = scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
        self.assert_gemm(
            gate_up[slots], xq, xs, self.w13, self.s13, slots // TOP_K, experts, "gemm1"
        )
        gate, up = gate_up[slots].double().chunk(2, dim=-1)
        exact = torch.nn.functional.silu(gate) * up
        got = inter[slots].double()
        # silu computes x / (1 + expf(-x)) in FP32: below -88.72 expf overflows and silu is exactly -0.
        overflow = gate < -SILU_EXP_OVERFLOW
        self.assertTrue(bool((got[overflow] == 0).all()), "silu_and_mul exp overflow")
        # Gradual underflow: each BF16 rounding also adds up to half the subnormal spacing, 2^-134.
        subnormal = (up.abs() + 1) * BF16_HALF_SUBNORMAL
        silu_ok = (got - exact).abs() <= SILU_REL * exact.abs() + subnormal
        self.assertTrue(bool(silu_ok[~overflow].all()), "silu_and_mul")
        iq, isc = scaled_fp8_quant(inter, None, use_per_token_if_dynamic=True)
        self.assert_gemm(
            y[slots],
            iq[slots],
            isc[slots],
            self.w2,
            w2_scale,
            torch.arange(slots.numel(), device="cuda"),
            experts,
            "gemm2",
        )
        out = self.run_moe(x, scores, ids, w2_scale=w2_scale, has_norm=has_norm)
        self.assert_finalize(out, y, scores, slots, has_norm=has_norm)
        # The finalize must not read rows of slots routed to other ranks: fill them with NaN first.
        from sglang.srt.rocm_muse_spark.muse_moe_fp8 import finalize_slots

        remote = torch.ones(y.shape[0], dtype=torch.bool, device="cuda")
        remote[slots] = False
        poisoned = y.masked_fill(remote.unsqueeze(-1), float("nan"))
        ids32, scores32 = ids.to(torch.int32), scores.float()
        out_poisoned = finalize_slots(
            y=poisoned,
            scores32=scores32,
            ids32=ids32,
            local_expert_start=LOCAL_START,
            num_local=LOCAL,
            post_norm_weight=self.norm,
            eps=EPS,
            post_norm_gain=GAIN,
            has_norm=has_norm,
            dtype=torch.bfloat16,
        )
        self.assert_finalize(out_poisoned, y, scores, slots, has_norm=has_norm)
        return x, scores, ids, out

    def test_stages_on_production_inputs(self):
        gen = torch.Generator().manual_seed(1)
        for m in ROWS:
            with self.subTest(m=m):
                self.check_stages(m, gen)

    def test_eps_dominant_and_unnormalized_finalize(self):
        """Expert outputs with RMS ~1e-3 against eps = 1e-5 (the norm no longer cancels scale or routed-weight
        errors), and the has_norm=False combine."""
        gen = torch.Generator().manual_seed(2)
        x = _tokens(64, gen)
        ids, scores = _routes(64, gen)
        slots, _ = self.local_slots(ids)
        _, _, y = self.stages(x, scores, ids, self.s2)
        tiny = self.s2 * (1e-3 / float(y[slots].float().pow(2).mean().sqrt()))
        for m in (1, 64, 513):
            for has_norm in (True, False):
                with self.subTest(m=m, has_norm=has_norm):
                    self.check_stages(m, gen, w2_scale=tiny, has_norm=has_norm)

    def test_ep_ranks_sum_to_the_global_routes(self):
        """256 experts over eight EP ranks of 32 (offsets 0, 32, ..., 224; each rank's bank differs): every route
        is computed by exactly the rank owning its expert, so the rank outputs sum to the per-rank exact
        finalizes of the teacher-forced slots within the sum of their bounds (triangle inequality)."""
        gen = torch.Generator().manual_seed(7)
        banks = [
            [_bank(2 * INTER, HIDDEN, gen) for _ in range(LOCAL)],
            [_bank(HIDDEN, INTER, gen) for _ in range(LOCAL)],
        ]
        # Ranks permute and rescale one set of 32 experts, so no two global experts share weights.
        m = 64
        x = _tokens(m, gen)
        ids = torch.stack(
            [torch.randperm(NUM_EXPERTS, generator=gen)[:TOP_K] for _ in range(m)]
        ).cuda()
        scores = torch.rand(m, TOP_K, generator=gen).cuda() + 0.05
        scores = scores / scores.sum(-1, keepdim=True)
        total = torch.zeros(m, HIDDEN, dtype=torch.float64, device="cuda")
        ref_total = torch.zeros_like(total)
        bound_total = torch.zeros(m, dtype=torch.float64, device="cuda")
        for rank in range(NUM_EXPERTS // LOCAL):
            perm = torch.roll(torch.arange(LOCAL), rank)
            self.w13 = torch.stack([banks[0][i][0] for i in perm])
            self.s13 = torch.stack([banks[0][i][1] for i in perm]) * (1 + rank / 8)
            self.w2 = torch.stack([banks[1][i][0] for i in perm])
            self.s2 = torch.stack([banks[1][i][1] for i in perm])
            start = rank * LOCAL
            out = self.moe(
                hidden_states=x,
                topk_scores=scores,
                topk_ids=ids,
                w13=self.w13,
                w13_scale=self.s13,
                w2=self.w2,
                w2_scale=self.s2,
                local_expert_start=start,
                post_norm_weight=self.norm,
                eps=EPS,
                post_norm_gain=GAIN,
            )
            self.assertTrue(bool(torch.isfinite(out).all()))
            flat = ids.reshape(-1) - start
            slots = ((flat >= 0) & (flat < LOCAL)).nonzero().squeeze(1)
            _, _, y = self.stages_at(x, scores, ids, start)
            ref, cancellation = finalize_reference(
                y[slots], scores, slots, m, self.norm, GAIN, EPS
            )
            total += out.double()
            ref_total += ref
            bound_total += finalize_bound(ref, cancellation, HIDDEN)
        err = (total - ref_total).norm(dim=-1)
        self.assertFalse(
            bool(exceeds(err, bound_total).any()), (err / bound_total).max()
        )

    def test_empty_and_all_non_local(self):
        gen = torch.Generator().manual_seed(3)
        out = self.run_moe(
            torch.empty(0, HIDDEN, device="cuda", dtype=torch.bfloat16),
            torch.empty(0, TOP_K, device="cuda"),
            torch.empty(0, TOP_K, device="cuda", dtype=torch.int64),
        )
        self.assertEqual(out.shape, (0, HIDDEN))
        for m in (1, 16, 513):
            with self.subTest(m=m):
                ids, scores = _routes(m, gen, local_share=0.0)
                out = self.run_moe(_tokens(m, gen), scores, ids)
                self.assertTrue(bool((out == 0).all()))

    def test_uninitialised_buffers_do_not_leak(self):
        """Intermediate buffers come from the caching allocator; rows of slots the GEMMs skip are never read."""
        gen = torch.Generator().manual_seed(4)
        for m in (8, 64, 513):
            x = _tokens(m, gen)
            ids, scores = _routes(m, gen, local_share=0.3)
            clean = self.run_moe(x, scores, ids)
            for fill in (float("nan"), 0.0):
                with self.subTest(fill=fill, m=m):
                    junk = torch.full(
                        (m * TOP_K * 3, 2 * INTER),
                        fill,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    del junk
                    self.assertTrue(torch.equal(self.run_moe(x, scores, ids), clean))

    def test_graph_replay_with_new_routes_equals_eager(self):
        """Replay reads new inputs and routes from the captured buffers: bit-equal to an eager call on them."""
        gen = torch.Generator().manual_seed(5)
        for m in (1, 16, 64):
            with self.subTest(m=m):
                ids, scores = _routes(m, gen)
                static = [_tokens(m, gen), scores.clone(), ids.clone()]
                for _ in range(2):  # compile and warm up outside capture
                    self.run_moe(*static)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = self.run_moe(*static)
                for share in (0.5, 1.0, 0.0, 0.2):
                    new_x = _tokens(m, gen)
                    new_ids, new_scores = _routes(m, gen, local_share=share)
                    for buf, value in zip(static, (new_x, new_scores, new_ids)):
                        buf.copy_(value)
                    graph.replay()
                    eager = self.run_moe(new_x, new_scores, new_ids)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(out, eager), f"share={share}")
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    remote = ~(
                        (new_ids >= LOCAL_START) & (new_ids < LOCAL_START + LOCAL)
                    ).any(-1)
                    self.assertTrue(bool((out[remote] == 0).all()))
                del graph

    def test_independent_oracle_gross_errors(self):
        """Aggregate agreement with an oracle that quantizes and rounds every stage itself (pending threshold)."""
        gen = torch.Generator().manual_seed(6)
        got, ref = [], []
        for m in ROWS:
            x = _tokens(m, gen)
            ids, scores = _routes(m, gen)
            has_local = ((ids >= LOCAL_START) & (ids < LOCAL_START + LOCAL)).any(-1)
            got.append(self.run_moe(x, scores, ids).float()[has_local])
            ref.append(
                muse_moe_fp8_reference(
                    x,
                    scores,
                    ids,
                    self.w13,
                    self.s13,
                    self.w2,
                    self.s2,
                    LOCAL_START,
                    self.norm,
                    EPS,
                    post_norm_gain=GAIN,
                ).float()[has_local]
            )
        got, ref = torch.cat(got), torch.cat(ref)
        self.assertLess(float((got - ref).norm() / ref.norm()), E2E_REL_L2_PENDING)
        cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
        self.assertGreater(float(cos.min()), E2E_MIN_COS_PENDING)


if __name__ == "__main__":
    unittest.main()
