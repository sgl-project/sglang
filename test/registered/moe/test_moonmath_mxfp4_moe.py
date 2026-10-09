"""Accuracy tests for the gfx942 (CDNA3) moonmath MXFP4 MoE route.

`moe_runner/moonmath_mxfp4_moe.py` swaps the two MXFP4 MoE GEMMs for the
hand-written gfx942 kernels in `moonmath_amd`. It rewrites the weight layout as
well as the numerics, so both halves are checked here:

  * the repack moves bytes and changes no value -- the nibble multiset survives
    it once MXFP4's negative zero is folded onto +0, and the scales are a plain
    transpose -- which is what makes it safe to do once at load;
  * the fused MoE matches an independent fp32 reference that dequantizes the
    same MXFP4 bytes and runs the MoE in plain torch, at token counts either
    side of the tile-height crossovers so all three gate/up tiles compile;
  * expert parallelism, where experts owned by another rank map to -1 and are
    dropped by the align step -- the case where the reduce's masked load is
    load-bearing, since those rows of `down` are never written;
  * `use_moonmath_mxfp4_moe` refuses every shape the kernels are not compiled
    for, because saying yes there would rewrite the weights into a layout
    nothing can read.

The route is gfx942-only, so the shape tests skip elsewhere.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")

# Kimi-K3-shaped but small enough for a unit test. INTER must be one of the
# down kernel's compiled K values and HIDDEN a multiple of 128 (gate/up's slab).
NUM_EXPERTS = 8
HIDDEN = 256
INTER = 384
TOPK = 2
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

# The kernels accumulate in fp32 but read bf16 activations and write bf16, so
# the floor is bf16 rounding on the two reductions, not the fp4 weights (which
# are exact -- the reference dequantizes the same bytes).
REL_L2_TOL = 1e-2


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.to(torch.float32)
    ref = ref.to(torch.float32)
    return (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()


def _dequant_mxfp4(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """MXFP4 -> fp32. ``w``: uint8 [..., N, K/2] packed e2m1, low nibble = even
    index. ``s``: uint8 [..., N, K/32] e8m0 exponents (value = 2**(s-127))."""
    from sglang.srt.layers.quantization.dequantization import _FP4_E2M1_LUT

    lut = _FP4_E2M1_LUT.to(device=w.device, dtype=torch.float32)
    *batch, n, half_k = w.shape
    out = torch.empty(*batch, n, half_k * 2, dtype=torch.float32, device=w.device)
    out[..., 0::2] = lut[(w & 0xF).to(torch.int64)]
    out[..., 1::2] = lut[(w >> 4).to(torch.int64)]
    scale = torch.exp2(s.to(torch.float32) - 127.0)
    return out * scale.repeat_interleave(32, dim=-1)


def _situ_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    g = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    u = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    return g * u


def _ref_moe(x, w13, w13_s, w2, w2_s, topk_weights, topk_ids, *, num_global, ep_rank):
    """Plain-torch fp32 MoE over the dequantized STOCK-layout weights."""
    E = w13.shape[0]
    inter = w13.shape[1] // 2
    w13_d = _dequant_mxfp4(w13, w13_s)  # [E, 2I, H]
    w2_d = _dequant_mxfp4(w2, w2_s)  # [E, H,  I]
    x32 = x.to(torch.float32)
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    lo = ep_rank * E
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            local = int(topk_ids[t, k]) - (lo if num_global is not None else 0)
            if not (0 <= local < E):
                continue  # expert lives on another EP rank -> dropped
            gu = x32[t] @ w13_d[local].T
            h = _situ_ref(gu[:inter], gu[inter:])
            out[t] += (h @ w2_d[local].T) * float(topk_weights[t, k])
    return out


def _make_weights(device):
    g = torch.Generator(device="cpu").manual_seed(0)

    def q(n, k):
        w = torch.randint(
            0, 256, (NUM_EXPERTS, n, k // 2), dtype=torch.uint8, generator=g
        )
        # e8m0 exponents around 1.0 (127); a narrow spread keeps the reference
        # and the kernel in the same dynamic range without saturating bf16.
        s = torch.randint(
            125, 130, (NUM_EXPERTS, n, k // 32), dtype=torch.uint8, generator=g
        )
        return w.to(device), s.to(device)

    w13, w13_s = q(2 * INTER, HIDDEN)
    w2, w2_s = q(HIDDEN, INTER)
    return w13, w13_s, w2, w2_s


def _make_routing(num_tokens, num_global, device):
    g = torch.Generator(device="cpu").manual_seed(1)
    ids = torch.stack(
        [torch.randperm(num_global, generator=g)[:TOPK] for _ in range(num_tokens)]
    ).to(torch.int32)
    w = torch.rand(num_tokens, TOPK, generator=g)
    w = (w / w.sum(dim=1, keepdim=True)).to(torch.float32)
    return w.to(device), ids.to(device)


class _StubLayer(torch.nn.Module):
    """Just the four weight parameters `repack_moonmath_mxfp4_weights` rewrites."""

    def __init__(self, w13, w13_s, w2, w2_s):
        super().__init__()
        for name, t in (
            ("w13_weight", w13),
            ("w13_weight_scale", w13_s),
            ("w2_weight", w2),
            ("w2_weight_scale", w2_s),
        ):
            self.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))


def _skip_reason():
    if not torch.cuda.is_available():
        return "GPU required"
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx942":
        return f"gfx942-only route (running on {arch})"
    try:
        import moonmath_amd  # noqa: F401
    except ImportError as exc:
        return f"moonmath_amd unavailable: {exc}"
    return None


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestMoonmathMxfp4MoE(CustomTestCase):
    device = "cuda"
    dtype = torch.bfloat16

    def _run(self, num_tokens, num_global_experts=None, ep_rank=0):
        from sglang.srt.layers.moe.moe_runner.moonmath_mxfp4_moe import (
            fused_moe_mxfp4_moonmath,
            repack_moonmath_mxfp4_weights,
        )

        num_global = num_global_experts or NUM_EXPERTS
        w13, w13_s, w2, w2_s = _make_weights(self.device)
        topk_w, topk_ids = _make_routing(num_tokens, num_global, self.device)
        x = (
            torch.randn(
                num_tokens,
                HIDDEN,
                generator=torch.Generator(device="cpu").manual_seed(2),
            )
            .to(self.device)
            .to(self.dtype)
        )

        layer = _StubLayer(w13, w13_s, w2, w2_s)
        repack_moonmath_mxfp4_weights(layer)

        got = fused_moe_mxfp4_moonmath(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_w,
            topk_ids,
            situ_beta=SITU_BETA,
            situ_linear_beta=SITU_LINEAR_BETA,
            num_global_experts=num_global_experts,
            ep_rank=ep_rank,
        )
        ref = _ref_moe(
            x,
            w13,
            w13_s,
            w2,
            w2_s,
            topk_w,
            topk_ids,
            num_global=num_global_experts,
            ep_rank=ep_rank,
        )
        self.assertEqual(tuple(got.shape), (num_tokens, HIDDEN))
        self.assertTrue(torch.isfinite(got).all(), "output has NaN/Inf")
        err = _rel_l2(got, ref)
        self.assertLess(err, REL_L2_TOL, f"rel L2 {err:.3e}")
        return err

    def test_matches_reference(self):
        # 1 / 64 / 512 tokens at TOPK=2 over 8 experts puts rows-per-expert on
        # either side of both tile-height crossovers (12 and 29), so all three
        # gate/up tiles and both down tiles are compiled and run.
        for num_tokens in (1, 8, 64, 512):
            with self.subTest(num_tokens=num_tokens):
                self._run(num_tokens)

    def test_expert_parallel_drops_unowned_experts(self):
        # num_global = 2 * local: half the routed experts map to -1 and are
        # dropped by moe_align_block_size(ignore_invalid_expert=True). Those
        # rows of `down` are never written, so this is where the reduce's masked
        # load is load-bearing -- an unmasked sum would read uninitialised
        # memory and the finite-output assert would trip.
        for ep_rank in (0, 1):
            with self.subTest(ep_rank=ep_rank):
                self._run(64, num_global_experts=2 * NUM_EXPERTS, ep_rank=ep_rank)

    def test_repack_is_a_permutation(self):
        # The repack runs once, in place of the weights, so a value-changing bug
        # in it is unrecoverable at run time. What must hold is the nibble
        # multiset and the scale transpose -- modulo the one canonicalization
        # the repack also does: MXFP4's negative zero (0x8) comes out as 0x0.
        # Both dequantize to zero, so no value moves; fold it on the stock side
        # and require it gone from the packed side, so the check keeps teeth.
        w13, w13_s, w2, w2_s = _make_weights(self.device)
        layer = _StubLayer(w13.clone(), w13_s.clone(), w2.clone(), w2_s.clone())

        from sglang.srt.layers.moe.moe_runner.moonmath_mxfp4_moe import (
            repack_moonmath_mxfp4_weights,
        )

        repack_moonmath_mxfp4_weights(layer)

        def nibbles(t, fold_neg_zero=False):
            t = t.flatten()
            lo, hi = t & 0xF, t >> 4
            if fold_neg_zero:
                lo = torch.where(lo == 0x8, torch.zeros_like(lo), lo)
                hi = torch.where(hi == 0x8, torch.zeros_like(hi), hi)
            return torch.bincount(lo, minlength=16) + torch.bincount(hi, minlength=16)

        for stock, packed in ((w13, layer.w13_weight), (w2, layer.w2_weight)):
            self.assertEqual(packed.dim(), 4)
            self.assertEqual(packed.shape[3], 16)
            got = nibbles(packed.data)
            self.assertEqual(int(got[0x8]), 0, "repack left an MXFP4 negative zero")
            self.assertTrue(torch.equal(nibbles(stock, fold_neg_zero=True), got))
        for stock, packed in (
            (w13_s, layer.w13_weight_scale),
            (w2_s, layer.w2_weight_scale),
        ):
            self.assertTrue(torch.equal(packed.data, stock.permute(0, 2, 1)))


class TestMoonmathMxfp4Gate(CustomTestCase):
    """`use_moonmath_mxfp4_moe` decides a weight layout, not just a kernel, so a
    yes on a shape the kernels are not compiled for is unrecoverable at run
    time -- they do not raise on it, they return wrong numbers. Checked without
    a GPU."""

    def _gate(self, **kwargs):
        from sglang.srt.layers.moe.moe_runner.moonmath_mxfp4_moe import (
            use_moonmath_mxfp4_moe,
        )

        args = dict(
            hidden_size=7168,
            intermediate_size=384,
            activation="situ",
            apply_router_weight_on_input=False,
            has_bias=False,
        )
        args.update(kwargs)
        return use_moonmath_mxfp4_moe(**args)

    def test_rejects_unsupported_shapes(self):
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import moonmath_mxfp4_moe as mod

        with mock.patch.object(mod, "_route_available", return_value=True):
            self.assertTrue(self._gate())
            # SwiGLU: the gate/up kernel has no such epilogue.
            self.assertFalse(self._gate(activation="silu"))
            # No slot for a per-row routing scale, and no bias term.
            self.assertFalse(self._gate(apply_router_weight_on_input=True))
            self.assertFalse(self._gate(has_bias=True))
            # gate/up drops a K remainder; down is compiled for 384/512/768.
            self.assertFalse(self._gate(hidden_size=7000))
            self.assertFalse(self._gate(intermediate_size=256))

    def test_env_opt_out(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner import moonmath_mxfp4_moe as mod

        mod._route_available.cache_clear()
        with envs.SGLANG_USE_MOONMATH_MXFP4_MOE.override(False):
            self.assertFalse(mod._route_available())
        mod._route_available.cache_clear()


if __name__ == "__main__":
    unittest.main()
