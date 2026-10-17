"""FlyDSL GDN prefill state-matrix kernel vs. the Triton kernel it replaces.

`chunk_gated_delta_rule_fwd` optionally routes the state-matrix step to aiter's
`chunk_gated_delta_rule_fwd_h_flydsl_opt` instead of the Triton
`chunk_gated_delta_rule_fwd_h` (SGLANG_GDN_PREFILL_FLYDSL=1, HIP only). These
tests install the kernel directly rather than through the env var, so they
cover the integration on both sides of the branch in one process.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla import chunk as chunk_mod
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd")

# The kernel is compiled for the 16x16x16 bf16 MFMA tile on CDNA3/CDNA4 only.
SUPPORTED_ARCHS = ("gfx942", "gfx950")


def _flydsl_kernel():
    """The aiter kernel, or None when this platform cannot provide it."""
    if not (is_hip() and torch.cuda.is_available()):
        return None
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in SUPPORTED_ARCHS:
        return None
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
        )
    except Exception:
        return None
    return chunk_gated_delta_rule_fwd_h_flydsl_opt


FLYDSL_KERNEL = _flydsl_kernel()
SKIP_REASON = (
    f"requires ROCm {'/'.join(SUPPORTED_ARCHS)} with the aiter FlyDSL GDN kernel"
)

# Relative to the mean magnitude of the reference, which is stable across
# near-zero elements in a way that a per-element rtol is not.
REL_TOL = 5e-3
# The `h` snapshots are bf16, where a single ULP on a large element already
# exceeds REL_TOL measured this way; `test_h_snapshots_within_bf16_rounding`
# is the strict check on those.
H_REL_TOL = 2e-2


def _rel_err(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref32, got32 = ref.float(), got.float()
    return (
        ((ref32 - got32).abs().max() / ref32.abs().mean().clamp_min(1e-9)).max().item()
    )


def _build(seqs, H, Hg, K=128, V=128, pool_size=8, state_dtype=torch.float32):
    """GDN prefill inputs for a varlen batch, packed into one row."""
    T = sum(seqs)
    dev = "cuda"
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seqs).cumsum(0).tolist(), dtype=torch.int32, device=dev
    )
    q = torch.randn(1, T, Hg, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, T, Hg, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=dev)
    # Log-space decay, kept small and negative like the real gate.
    g = -torch.rand(1, T, H, dtype=torch.float32, device=dev) * 0.5
    beta = torch.rand(1, T, H, dtype=torch.bfloat16, device=dev).sigmoid()
    pool = torch.randn(pool_size, H, V, K, dtype=state_dtype, device=dev) * 0.1
    # Non-identity slots, so an indexing bug cannot pass by accident.
    slots = torch.tensor([4, 1, 6][: len(seqs)], dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, g=g, beta=beta, pool=pool, slots=slots, cu=cu_seqlens)


class TestGDNPrefillFlyDSL(CustomTestCase):
    def setUp(self):
        # The dispatch decision is cached in module globals; snapshot them so a
        # test cannot leak its override into the next one.
        self._saved = (
            chunk_mod._flydsl_fwd_h,
            chunk_mod._flydsl_probed,
            chunk_mod._GDN_PREFILL_FLYDSL,
            set(chunk_mod._flydsl_logged),
        )
        torch.manual_seed(0)

    def tearDown(self):
        (
            chunk_mod._flydsl_fwd_h,
            chunk_mod._flydsl_probed,
            chunk_mod._GDN_PREFILL_FLYDSL,
            logged,
        ) = self._saved
        chunk_mod._flydsl_logged.clear()
        chunk_mod._flydsl_logged.update(logged)

    @staticmethod
    def _install(kernel):
        chunk_mod._flydsl_probed = True
        chunk_mod._flydsl_fwd_h = kernel
        chunk_mod._flydsl_logged.clear()

    @staticmethod
    def _forward(t, inplace_update=True):
        o, _, h = chunk_gated_delta_rule(
            q=t["q"],
            k=t["k"],
            v=t["v"],
            g=t["g"],
            beta=t["beta"],
            initial_state=t["pool"],
            initial_state_indices=t["slots"],
            cu_seqlens=t["cu"],
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            inplace_update=inplace_update,
        )
        return o, h

    def _both_paths(self, t):
        """Run the same inputs down each path against a private pool copy."""
        pool = t["pool"]
        triton_pool, flydsl_pool = pool.clone(), pool.clone()

        self._install(None)
        o_tri, h_tri = self._forward({**t, "pool": triton_pool})

        self._install(FLYDSL_KERNEL)
        o_fly, h_fly = self._forward({**t, "pool": flydsl_pool})

        return (o_tri, h_tri, triton_pool), (o_fly, h_fly, flydsl_pool)

    @unittest.skipUnless(FLYDSL_KERNEL is not None, SKIP_REASON)
    def test_matches_triton_across_shapes(self):
        cases = [
            ("aligned varlen", [128, 192], 8, 8, torch.float32),
            # 129 is not a multiple of BT=64, so the tail predicate rejects rows.
            ("unaligned tail", [128, 192, 129], 8, 8, torch.float32),
            ("gqa", [256, 129], 8, 4, torch.float32),
            ("bf16 state pool", [128, 129], 8, 8, torch.bfloat16),
            ("single sequence", [64], 4, 4, torch.float32),
        ]
        for label, seqs, H, Hg, state_dtype in cases:
            with self.subTest(case=label):
                t = _build(seqs, H, Hg, state_dtype=state_dtype)
                (o_tri, h_tri, pool_tri), (o_fly, h_fly, pool_fly) = self._both_paths(t)

                # Layouts must match, not just values: the whole point of the
                # token-major option is that no transposes are needed.
                self.assertEqual(o_fly.shape, o_tri.shape)
                self.assertEqual(h_fly.shape, h_tri.shape)
                self.assertEqual(o_fly.dtype, o_tri.dtype)
                self.assertEqual(h_fly.dtype, h_tri.dtype)

                self.assertLess(_rel_err(o_tri, o_fly), REL_TOL, f"{label}: o")
                self.assertLess(_rel_err(h_tri, h_fly), H_REL_TOL, f"{label}: h")
                self.assertLess(_rel_err(pool_tri, pool_fly), REL_TOL, f"{label}: pool")

                # The final state goes back into the indexed slots only.
                untouched = torch.ones(
                    pool_tri.shape[0], dtype=torch.bool, device=pool_tri.device
                )
                untouched[t["slots"].to(torch.int64)] = False
                self.assertTrue(
                    torch.equal(pool_tri[untouched], pool_fly[untouched]),
                    f"{label}: the kernel wrote outside its own pool slots",
                )

    @unittest.skipUnless(FLYDSL_KERNEL is not None, SKIP_REASON)
    def test_h_snapshots_within_bf16_rounding(self):
        """The `h` difference must look like rounding, not a layout error.

        bf16 bit patterns are monotonic for same-sign values, so the integer
        distance between them is the ULP distance. A layout or addressing bug
        moves whole tiles and blows both bounds instantly.
        """
        t = _build([128, 192, 129], 8, 8)
        (_, h_tri, _), (_, h_fly, _) = self._both_paths(t)
        self.assertEqual(h_tri.dtype, torch.bfloat16)

        ulp = (
            h_tri.view(torch.int16).to(torch.int32)
            - h_fly.view(torch.int16).to(torch.int32)
        ).abs()
        mismatch_frac = (ulp > 0).sum().item() / ulp.numel()
        self.assertLessEqual(ulp.max().item(), 16, "h differs by more than rounding")
        self.assertLess(mismatch_frac, 1e-3, "too many h elements differ")

    @unittest.skipUnless(FLYDSL_KERNEL is not None, SKIP_REASON)
    def test_dispatch_is_logged_once(self):
        t = _build([128, 129], 8, 8)
        self._install(FLYDSL_KERNEL)

        with self.assertLogs("sglang.srt.utils.common", level="INFO") as captured:
            self._forward(t)
        hits = [m for m in captured.output if "using the aiter FlyDSL" in m]
        self.assertEqual(len(hits), 1, captured.output)

        # The decision never changes, so it must not be logged again.
        with self.assertNoLogs("sglang.srt.utils.common", level="INFO"):
            self._forward(t)

    @unittest.skipUnless(FLYDSL_KERNEL is not None, SKIP_REASON)
    def test_ineligible_call_falls_back_with_a_reason(self):
        """`inplace_update=False` has no indexed-pool equivalent in FlyDSL."""
        t = _build([128, 129], 8, 8)
        self._install(FLYDSL_KERNEL)

        with self.assertLogs("sglang.srt.utils.common", level="INFO") as captured:
            self._forward(t, inplace_update=False)
        rejects = [m for m in captured.output if "does not fit this call" in m]
        self.assertEqual(len(rejects), 1, captured.output)
        self.assertIn("inplace_update=False", rejects[0])

    def test_reject_reasons_cover_the_kernel_contract(self):
        """The host-side guard must reject what the kernel cannot compile for.

        aiter's own audit of these preconditions is behind AITER_K5_OPT_CHECK
        and off by default, so a miss here is a wrong result rather than an
        exception. Runs on any platform: it only inspects tensor metadata.
        """
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        B, T, H, Hg, K, V = 1, 128, 8, 8, 128, 128
        good = dict(
            k=torch.zeros(B, T, Hg, K, dtype=torch.bfloat16, device=dev),
            w=torch.zeros(B, T, H, K, dtype=torch.bfloat16, device=dev),
            u=torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=dev),
            g=torch.zeros(B, T, H, dtype=torch.float32, device=dev),
            initial_state=torch.zeros(4, H, V, K, dtype=torch.float32, device=dev),
            initial_state_indices=torch.zeros(1, dtype=torch.int32, device=dev),
            cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=dev),
            inplace_update=True,
        )
        self.assertIsNone(chunk_mod._flydsl_fwd_h_reject_reason(**good))

        for label, override, expected in [
            ("fp16 inputs", {"k": good["k"].to(torch.float16)}, "bf16"),
            (
                "head-major w",
                {"w": good["w"].transpose(1, 2).contiguous()},
                "token-major",
            ),
            ("2-D g", {"g": good["g"].squeeze(0)}, "g must be fp32"),
            ("no state pool", {"initial_state": None}, "indexed state pool"),
            ("no write-back", {"inplace_update": False}, "inplace_update=False"),
            (
                "wrong pool layout",
                {"initial_state": torch.zeros(4, H, K, V + 8, device=dev)},
                "state pool must be",
            ),
        ]:
            with self.subTest(case=label):
                reason = chunk_mod._flydsl_fwd_h_reject_reason(**{**good, **override})
                self.assertIsNotNone(reason, f"{label} should have been rejected")
                self.assertIn(expected, reason)


if __name__ == "__main__":
    unittest.main()
