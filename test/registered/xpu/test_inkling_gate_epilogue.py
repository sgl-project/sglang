"""Inkling MoE gate epilogue on Intel XPU.

Covers the three things that were broken or untested on XPU:

  A. Dispatch. `sigmoid_gate_topk_renorm` gated its CUDA JIT fast path on
     `torch.version.hip is None` ("not ROCm"), which is also true on XPU, so it
     selected a CUDA kernel and raised `AssertionError` on every call. The gate
     is now `.is_cuda and torch.version.hip is None` -- ROCm reports `.is_cuda`,
     so neither conjunct alone is right. Guarded by
     `test_dispatch_reaches_a_triton_kernel`.

  B. Underflow. The fallback `tl.topk` kernel normalized as
     `sigmoid(x) / sum(sigmoid(x))`. fp32 sigmoid flushes to zero below
     x ~ -104, so all-negative active logits made that a 0/0 and every weight
     came back NaN. Guarded by `test_underflow_logits_are_nan_free`.

  C. The `LOGSIGMOID_SINK` epilogue in the unified router, which replaces the
     `tl.topk` sort with iterative masked-max, plus the requirement that the
     router's default `SHARED_SINK == 0` behaviour is untouched. Guarded by the
     oracle / invariant / packed / non-regression cases below.

This file lives under `test/registered/xpu/` and registers XPU CI only, so no
per-class device skip is needed -- see the `register_xpu_ci` BKM. Do not add
`register_cuda_ci` here: CI collects by file, so that would make CUDA runners
pick up these XPU-only cases. A CUDA-side counterpart for the shared
`_router_triton_kernel` epilogue belongs in
`test/registered/kernels/ops/moe/test_moe_fused_gate.py`.

The oracle is an fp64 recompute of the gate contract:

    sel_j = sigmoid(logits[:, j]) + bias[j]      for j < n_routed  # selection only
    idx   = topk(sel, k)                                           # lowest id wins ties
    act   = logits[idx] ++ logits[n_routed:]                       # RAW logits, k + s
    w     = exp(logsigmoid(act) - logsumexp(logsigmoid(act))) * route_scale * global_scale
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import sigmoid_gate_topk_renorm
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"
N_ROUTED = 256
N_SHARED = 2
G_COLS = N_ROUTED + N_SHARED  # 258
G_PAD = 264  # the production gate GEMM writes a [T, 264] fp32 row pitch
TOPK = 6
ROUTE_SCALE = 8.0
GLOBAL_SCALE = 1.3

# `SGLANG_OPT_USE_ROUTER_GATE_EPILOGUE` selects the dispatch branch inside
# `sigmoid_gate_topk_renorm`: True (default) is the unified router's
# LOGSIGMOID_SINK epilogue, False is the older tl.topk kernel, which stays
# reachable as a fallback. Both must satisfy the contract, so both are exercised.


def _make_logits(tokens: int, seed: int):
    """Production-shaped inputs: a [T, 264] pad sliced to [:, :258], stride (264, 1).

    The real gate logits are a non-contiguous but column-contiguous slice of a
    padded fp32 GEMM output, so the kernels must not require contiguity.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    pad = torch.randn((tokens, G_PAD), generator=g, device=DEVICE, dtype=torch.float32)
    logits = pad[:, :G_COLS]
    bias = (
        torch.randn((N_ROUTED,), generator=g, device=DEVICE, dtype=torch.float32) * 0.1
    )
    global_scale = torch.tensor([GLOBAL_SCALE], device=DEVICE, dtype=torch.float32)
    return logits, bias, global_scale


def _oracle(logits, bias, global_scale, k=TOPK, s=N_SHARED):
    """fp64 recompute of the contract. Returns (routed_w, shared_w, indices)."""
    l64 = logits.double()
    sel = torch.sigmoid(l64[:, :-s]) + bias.double()
    idx = sel.topk(k, dim=-1).indices
    act = torch.cat([l64[:, :-s].gather(-1, idx), l64[:, -s:]], dim=-1)
    lp = F.logsigmoid(act)
    w = torch.exp(lp - torch.logsumexp(lp, -1, keepdim=True))
    w = w * ROUTE_SCALE * global_scale.double()
    return w[:, :k].contiguous(), w[:, k:].contiguous(), idx.to(torch.int32)


def _gate(logits, bias, global_scale, *, use_router: bool, packed: bool = False):
    """Call the production entry point with one dispatch branch forced."""
    with envs.SGLANG_OPT_USE_ROUTER_GATE_EPILOGUE.override(use_router):
        return sigmoid_gate_topk_renorm(
            logits,
            TOPK,
            N_SHARED,
            ROUTE_SCALE,
            global_scale,
            bias,
            return_packed_topk=packed,
        )


class TestInklingGateEpilogue(CustomTestCase):
    # --- A: the dispatch predicate -------------------------------------------

    def test_dispatch_reaches_a_triton_kernel(self):
        """XPU must not select the CUDA JIT path (it asserts on `is_cuda`).

        This is the regression guard for `torch.version.hip is None` ->
        `logits.is_cuda and torch.version.hip is None`. Every JIT gate other
        than the device check is satisfied by these inputs (k == 6,
        n_shared == 2, G == 258, stride % 8 == 0, ptr % 32 == 0), so before the
        fix this raised at every token count. The ROCm half of that predicate
        cannot be covered from here; it is argued in the commit message.
        """
        for tokens in (1, 4, 8, 512):
            logits, bias, global_scale = _make_logits(tokens, seed=tokens)
            self.assertEqual(logits.stride(1), 1)
            self.assertEqual(logits.stride(0) % 8, 0)
            routed_w, indices, shared_w, packed = sigmoid_gate_topk_renorm(
                logits, TOPK, N_SHARED, ROUTE_SCALE, global_scale, bias
            )
            self.assertEqual(routed_w.shape, (tokens, TOPK))
            self.assertEqual(indices.shape, (tokens, TOPK))
            self.assertEqual(shared_w.shape, (tokens, N_SHARED))
            self.assertIsNone(packed)
            self.assertTrue(torch.isfinite(routed_w).all(), f"{tokens=}")
            self.assertTrue(torch.isfinite(shared_w).all(), f"{tokens=}")

    # --- B and C: agreement with the fp64 oracle ------------------------------

    def _check_against_oracle(self, logits, bias, global_scale, *, use_router, msg):
        routed_w, indices, shared_w, _ = _gate(
            logits, bias, global_scale, use_router=use_router
        )
        o_routed, o_shared, o_idx = _oracle(logits, bias, global_scale)

        self.assertFalse(torch.isnan(routed_w).any(), f"{msg}: NaN in routed weights")
        self.assertFalse(torch.isnan(shared_w).any(), f"{msg}: NaN in shared weights")

        # fp32 top-k selection is a genuine knife edge: two selection scores can
        # be bit-identical in fp32 and be ordered differently by two sigmoid
        # implementations. Judge the weights only where selection agrees with
        # the oracle, and require that no token disagrees on these inputs.
        agree = (indices.int() == o_idx).all(dim=-1)
        self.assertEqual(
            int((~agree).sum()), 0, f"{msg}: top-k indices differ from the fp64 oracle"
        )
        torch.testing.assert_close(
            routed_w[agree].float(),
            o_routed[agree].float(),
            rtol=2e-3,
            atol=2e-3,
            msg=f"{msg}: routed weights",
        )
        torch.testing.assert_close(
            shared_w[agree].float(),
            o_shared[agree].float(),
            rtol=2e-3,
            atol=2e-3,
            msg=f"{msg}: shared weights",
        )

    def test_matches_fp64_oracle(self):
        """Both dispatch branches must reproduce the contract.

        Token counts include 1 (decode), a non-multiple-of-BLOCK tail (37), and
        a prefill-sized batch.
        """
        for use_router in (True, False):
            branch = "router" if use_router else "tl.topk"
            for tokens in (1, 8, 37, 512):
                logits, bias, global_scale = _make_logits(tokens, seed=tokens)
                with self.subTest(branch=branch, tokens=tokens):
                    self._check_against_oracle(
                        logits,
                        bias,
                        global_scale,
                        use_router=use_router,
                        msg=f"{branch} T={tokens}",
                    )

    def test_underflow_logits_are_nan_free(self):
        """All-negative logits must not produce NaN (change B).

        fp32 sigmoid goes subnormal below x ~ -87 and flushes to zero below
        x ~ -104. At base -90 and -200 every one of the k + s active logits is
        in that region, so `sigmoid(x) / sum(sigmoid(x))` divides 0/0. Base -40
        is included as the control that was always fine.
        """
        for use_router in (True, False):
            branch = "router" if use_router else "tl.topk"
            for base in (-40.0, -90.0, -200.0):
                g = torch.Generator(device=DEVICE).manual_seed(7)
                pad = torch.full((16, G_PAD), base, device=DEVICE, dtype=torch.float32)
                pad[:, :G_COLS] += (
                    torch.randn(
                        (16, G_COLS), generator=g, device=DEVICE, dtype=torch.float32
                    )
                    * 0.5
                )
                bias = (
                    torch.randn((N_ROUTED,), generator=g, device=DEVICE) * 0.1
                ).float()
                global_scale = torch.tensor(
                    [GLOBAL_SCALE], device=DEVICE, dtype=torch.float32
                )
                with self.subTest(branch=branch, base=base):
                    self._check_against_oracle(
                        pad[:, :G_COLS],
                        bias,
                        global_scale,
                        use_router=use_router,
                        msg=f"{branch} all_negative_{int(abs(base))}",
                    )

    def test_shared_sink_outranking_every_routed_expert(self):
        """A sink column can dominate the normalizer; it still never enters top-k."""
        g = torch.Generator(device=DEVICE).manual_seed(11)
        pad = (
            torch.randn((16, G_PAD), generator=g, device=DEVICE, dtype=torch.float32)
            - 3.0
        )
        pad[:, N_ROUTED:G_COLS] = 10.0
        bias = (torch.randn((N_ROUTED,), generator=g, device=DEVICE) * 0.1).float()
        global_scale = torch.tensor([GLOBAL_SCALE], device=DEVICE, dtype=torch.float32)
        for use_router in (True, False):
            with self.subTest(branch="router" if use_router else "tl.topk"):
                self._check_against_oracle(
                    pad[:, :G_COLS],
                    bias,
                    global_scale,
                    use_router=use_router,
                    msg="shared_outranks_routed",
                )
                _, indices, _, _ = _gate(
                    pad[:, :G_COLS], bias, global_scale, use_router=use_router
                )
                # Sink columns live at N_ROUTED.., outside the routed id range.
                self.assertTrue((indices < N_ROUTED).all(), "sink id leaked into top-k")

    def test_ties_do_not_produce_nan_or_break_the_invariant(self):
        """Only 4 distinct selection values over 256 columns, bias identically 0.

        Indices are deliberately not compared: with exact ties the winning set is
        implementation-defined. What must hold is that the weights are finite and
        still normalized.
        """
        pad = torch.zeros((16, G_PAD), device=DEVICE, dtype=torch.float32)
        vals = torch.tensor([2.0, 2.0, 1.0, 0.5], device=DEVICE)
        pad[:, :N_ROUTED] = vals.repeat(N_ROUTED // 4)
        pad[:, N_ROUTED:G_COLS] = 0.25
        bias = torch.zeros(N_ROUTED, device=DEVICE, dtype=torch.float32)
        global_scale = torch.tensor([GLOBAL_SCALE], device=DEVICE, dtype=torch.float32)
        for use_router in (True, False):
            routed_w, _, shared_w, _ = _gate(
                pad[:, :G_COLS], bias, global_scale, use_router=use_router
            )
            with self.subTest(branch="router" if use_router else "tl.topk"):
                self.assertTrue(torch.isfinite(routed_w).all())
                self.assertTrue(torch.isfinite(shared_w).all())
                total = routed_w.float().sum(-1) + shared_w.float().sum(-1)
                torch.testing.assert_close(
                    total,
                    torch.full_like(total, ROUTE_SCALE * GLOBAL_SCALE),
                    rtol=2e-3,
                    atol=2e-3,
                )

    def test_weights_sum_to_route_scale_times_global_scale(self):
        """Free invariant: the k routed plus s shared weights are a normalized set."""
        for use_router in (True, False):
            for tokens in (1, 37, 512):
                logits, bias, global_scale = _make_logits(tokens, seed=tokens + 100)
                routed_w, _, shared_w, _ = _gate(
                    logits, bias, global_scale, use_router=use_router
                )
                total = routed_w.float().sum(-1) + shared_w.float().sum(-1)
                with self.subTest(
                    branch="router" if use_router else "tl.topk", tokens=tokens
                ):
                    torch.testing.assert_close(
                        total,
                        torch.full_like(total, ROUTE_SCALE * GLOBAL_SCALE),
                        rtol=2e-3,
                        atol=2e-3,
                    )

    def test_packed_matches_plain(self):
        """Packed mode is what InklingGate.emit_packed_topk requests.

        The packed int32 is `(expert_id << 16) | bf16_bits(weight)`, so it must
        carry the same ids and the bf16 rounding of the same weights.
        """
        for use_router in (True, False):
            for tokens in (1, 37, 512):
                logits, bias, global_scale = _make_logits(tokens, seed=tokens + 200)
                routed_w, indices, shared_w, packed_none = _gate(
                    logits, bias, global_scale, use_router=use_router
                )
                p_w, p_idx, p_shared, packed = _gate(
                    logits, bias, global_scale, use_router=use_router, packed=True
                )
                with self.subTest(
                    branch="router" if use_router else "tl.topk", tokens=tokens
                ):
                    self.assertIsNone(packed_none)
                    self.assertIsNone(p_w)
                    self.assertIsNone(p_idx)
                    self.assertIsNotNone(packed)
                    self.assertEqual(packed.shape, (tokens, TOPK))
                    self.assertEqual(packed.dtype, torch.int32)
                    unpacked_idx = (packed >> 16).to(torch.int32)
                    unpacked_w = (
                        (packed << 16 >> 16).to(torch.int16).view(torch.bfloat16)
                    )
                    self.assertTrue(torch.equal(unpacked_idx, indices.to(torch.int32)))
                    self.assertTrue(
                        torch.equal(unpacked_w.float(), routed_w.bfloat16().float())
                    )
                    self.assertTrue(torch.equal(p_shared, shared_w))

    # --- C: the router's default epilogue must be untouched -------------------

    def test_moe_fused_gate_default_epilogue_unchanged(self):
        """`shared_sink == 0` must keep the historical SUM_NORM behaviour.

        The Inkling epilogue is behind the `SHARED_SINK` / `EPILOGUE` constexprs,
        so every pre-existing caller must be unaffected: a 2-tuple return, and
        weights that still match `activated / sum(activated) * scale`.
        """
        for num_experts, topk, rsf, apply_scale in (
            (256, 8, 2.5, True),
            (256, 6, 8.0, True),
            (384, 8, 1.0, False),
        ):
            g = torch.Generator(device=DEVICE).manual_seed(num_experts + topk)
            scores = torch.randn(
                (64, num_experts), generator=g, device=DEVICE, dtype=torch.float32
            )
            bias = (
                torch.randn((num_experts,), generator=g, device=DEVICE) * 0.1
            ).float()

            out = moe_fused_gate(
                scores,
                bias,
                topk=topk,
                scoring_func="sigmoid",
                renormalize=True,
                routed_scaling_factor=rsf,
                apply_routed_scaling_factor_on_output=apply_scale,
            )
            with self.subTest(num_experts=num_experts, topk=topk):
                # Still a 2-tuple: the third/fourth outputs appear only when
                # shared_sink > 0.
                self.assertEqual(len(out), 2)
                weights, indices = out

                activated = torch.sigmoid(scores.double())
                ref_idx = (activated + bias.double()).topk(topk, dim=-1).indices
                sel = activated.gather(-1, ref_idx)
                ref_w = sel / sel.sum(-1, keepdim=True)
                if apply_scale:
                    ref_w = ref_w * rsf
                self.assertTrue(torch.equal(indices.int(), ref_idx.to(torch.int32)))
                torch.testing.assert_close(
                    weights.float(), ref_w.float(), rtol=2e-3, atol=2e-3
                )

    def test_shared_sink_kwargs_rejected_without_shared_sink(self):
        """The new kwargs are meaningless at `shared_sink == 0` and must not be silent."""
        g = torch.Generator(device=DEVICE).manual_seed(3)
        scores = torch.randn((8, 256), generator=g, device=DEVICE, dtype=torch.float32)
        bias = (torch.randn((256,), generator=g, device=DEVICE) * 0.1).float()
        global_scale = torch.tensor([GLOBAL_SCALE], device=DEVICE, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            moe_fused_gate(scores, bias, topk=6, global_scale=global_scale)
        with self.assertRaises(AssertionError):
            moe_fused_gate(scores, bias, topk=6, return_packed=True)
        # bias covers the routed experts only, so a [M, N] score row with
        # shared_sink=2 is a shape error rather than a silent misread.
        with self.assertRaises(AssertionError):
            moe_fused_gate(scores, bias, topk=6, shared_sink=N_SHARED)


if __name__ == "__main__":
    unittest.main()
