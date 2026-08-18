"""Accuracy tests for the gfx942 (CDNA3) Triton MXFP4 MoE route.

`aiter.fused_moe` has no gfx942 a4w4 path and aborts in the activation
quantiser, so MXFP4 MoE on CDNA3 goes through aiter's *Triton* MXFP4 GEMMs
instead (`moe_runner/aiter_mxfp4_triton.py`), with a SiTU variant of the fused
gated epilogue (`moe_runner/mxfp4_situ_fused.py`).

Those two files replace a numeric path, so they are checked against an
independent fp32 reference that dequantizes the same MXFP4 bytes and runs the
MoE in plain torch:

  * both epilogues -- SwiGLU (upstream, must stay byte-equivalent in intent)
    and SiTU (Kimi-K3), including the tanh soft-clips;
  * both routed-weight placements (`apply_router_weight_on_input`);
  * expert parallelism, where experts owned by another rank map to -1 and are
    dropped by the align step -- the case where `torch.empty` + a masked reduce
    would read uninitialised memory if the mask were wrong;
  * token counts either side of the small/medium tile-config boundary, so both
    tile shapes are compiled and run.

The route is gfx942-only, so the whole module skips elsewhere.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")

# Kimi-K3-shaped but small enough for a unit test: H and I stay multiples of
# 128 (BLOCK_SIZE_K) and of 32 (the MX scale group).
NUM_EXPERTS = 8
HIDDEN = 256
INTER = 128
TOPK = 2
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

# The kernels accumulate in fp32 but read bf16 activations and write bf16, so
# the floor is bf16 rounding on a K=256 reduction, not the fp4 weights (which
# are exact -- the reference dequantizes the same bytes). Measured max rel L2
# across these shapes was ~2e-3.
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


def _act_ref(gate: torch.Tensor, up: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    g = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    u = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    return g * u


def _ref_moe(
    x,
    w13,
    w13_s,
    w2,
    w2_s,
    topk_weights,
    topk_ids,
    *,
    activation,
    num_global_experts,
    ep_rank,
    apply_router_weight_on_input,
):
    """Plain-torch fp32 MoE over the dequantized weights."""
    E = w13.shape[0]
    inter = w13.shape[1] // 2
    w13_d = _dequant_mxfp4(w13, w13_s)  # [E, 2I, H]
    w2_d = _dequant_mxfp4(w2, w2_s)  # [E, H,  I]
    x32 = x.to(torch.float32)
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    lo = ep_rank * E
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            gid = int(topk_ids[t, k])
            local = gid - lo if num_global_experts is not None else gid
            if not (0 <= local < E):
                continue  # expert lives on another EP rank -> dropped
            rw = float(topk_weights[t, k])
            a = x32[t] * rw if apply_router_weight_on_input else x32[t]
            gu = a @ w13_d[local].T
            h = _act_ref(gu[:inter], gu[inter:], activation)
            d = h @ w2_d[local].T
            out[t] += d if apply_router_weight_on_input else d * rw
    return out


def _make_weights(device, dtype):
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


def _skip_reason():
    if not torch.cuda.is_available():
        return "GPU required"
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx942":
        return f"gfx942-only route (running on {arch})"
    try:
        import aiter.ops.triton.moe.moe_op_mxfp4  # noqa: F401
    except ImportError as exc:
        return f"aiter Triton MXFP4 MoE kernels unavailable: {exc}"
    return None


@unittest.skipIf(_skip_reason() is not None, _skip_reason() or "")
class TestAiterMxfp4TritonMoE(CustomTestCase):
    device = "cuda"
    dtype = torch.bfloat16

    def _run(
        self,
        num_tokens,
        activation="situ",
        num_global_experts=None,
        ep_rank=0,
        apply_router_weight_on_input=False,
    ):
        from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
            fused_moe_mxfp4_triton,
        )

        num_global = num_global_experts or NUM_EXPERTS
        w13, w13_s, w2, w2_s = _make_weights(self.device, self.dtype)
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

        got = fused_moe_mxfp4_triton(
            x,
            w13,
            w2,
            w13_s,
            w2_s,
            topk_w,
            topk_ids,
            activation=activation,
            situ_beta=SITU_BETA,
            situ_linear_beta=SITU_LINEAR_BETA,
            num_global_experts=num_global_experts,
            ep_rank=ep_rank,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        ref = _ref_moe(
            x,
            w13,
            w13_s,
            w2,
            w2_s,
            topk_w,
            topk_ids,
            activation=activation,
            num_global_experts=num_global_experts,
            ep_rank=ep_rank,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        self.assertEqual(tuple(got.shape), (num_tokens, HIDDEN))
        self.assertTrue(torch.isfinite(got).all(), "output has NaN/Inf")
        err = _rel_l2(got, ref)
        self.assertLess(err, REL_L2_TOL, f"rel L2 {err:.3e}")
        return err

    def test_situ_matches_reference(self):
        # 16 and 512 straddle the small/medium tile-config boundary (M=256), so
        # both compiled tile shapes are exercised.
        for num_tokens in (1, 16, 512):
            with self.subTest(num_tokens=num_tokens):
                self._run(num_tokens, activation="situ")

    def test_silu_matches_reference(self):
        # The SwiGLU branch must stay equivalent to aiter's upstream epilogue:
        # ACT_SITU=False is the only difference from the kernel it derives from.
        for num_tokens in (16, 512):
            with self.subTest(num_tokens=num_tokens):
                self._run(num_tokens, activation="silu")

    def test_router_weight_on_input(self):
        # The routed weight folds into gate/up instead of down; both placements
        # must land on the same math.
        self._run(64, apply_router_weight_on_input=True)

    def test_expert_parallel_drops_unowned_experts(self):
        # num_global = 2 * local: half the routed experts map to -1 and are
        # dropped by moe_align_block_size(ignore_invalid_expert=True). Those
        # rows of `down` are never written, so this is the case where the fused
        # reduce's masked load is load-bearing -- an unmasked sum would read
        # uninitialised memory and the finite-output assert would trip.
        for ep_rank in (0, 1):
            with self.subTest(ep_rank=ep_rank):
                self._run(64, num_global_experts=2 * NUM_EXPERTS, ep_rank=ep_rank)

    def test_situ_differs_from_silu(self):
        # Guard against the activation switch silently doing nothing: at these
        # betas SiTU soft-clips both branches, so the two must not coincide.
        from sglang.srt.layers.moe.moe_runner.aiter_mxfp4_triton import (
            fused_moe_mxfp4_triton,
        )

        w13, w13_s, w2, w2_s = _make_weights(self.device, self.dtype)
        topk_w, topk_ids = _make_routing(64, NUM_EXPERTS, self.device)
        x = torch.randn(64, HIDDEN, device=self.device, dtype=self.dtype)
        outs = [
            fused_moe_mxfp4_triton(
                x,
                w13,
                w2,
                w13_s,
                w2_s,
                topk_w,
                topk_ids,
                activation=act,
                situ_beta=SITU_BETA,
                situ_linear_beta=SITU_LINEAR_BETA,
            )
            for act in ("silu", "situ")
        ]
        self.assertGreater(_rel_l2(outs[1], outs[0]), 1e-2)


class TestTritonMxfp4Gate(CustomTestCase):
    """`use_triton_mxfp4_moe()` is the only thing keeping this off every other
    arch, so its env override is checked without a GPU."""

    def test_env_override(self):
        import os
        from unittest import mock

        from sglang.srt.layers.moe.moe_runner import aiter_mxfp4_triton as mod

        for value, expected in (("0", False), ("1", True)):
            with mock.patch.dict(
                os.environ, {"SGLANG_AITER_MXFP4_TRITON": value}, clear=False
            ):
                mod.use_triton_mxfp4_moe.cache_clear()
                self.assertIs(mod.use_triton_mxfp4_moe(), expected)
        mod.use_triton_mxfp4_moe.cache_clear()


if __name__ == "__main__":
    unittest.main()
