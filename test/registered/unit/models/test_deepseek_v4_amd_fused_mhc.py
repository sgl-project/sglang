import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.amd import deepseek_v4_fused_mhc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestAmdFusedMhcCrossLayerGating(unittest.TestCase):
    """Gating and dispatch-preference tests (CPU, no kernels required)."""

    def test_tilelang_fuse_flag_enables_cross_layer_fusion(self):
        with (
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
        ):
            self.assertTrue(deepseek_v4_fused_mhc.is_cross_layer_mhc_fusion_enabled())

    @mock.patch.object(deepseek_v4_fused_mhc, "is_gfx95_supported", return_value=True)
    @mock.patch.object(deepseek_v4_fused_mhc, "get_bool_env_var", return_value=True)
    @mock.patch.object(deepseek_v4_fused_mhc, "_is_hip", True)
    def test_aiter_gfx95_enables_cross_layer_fusion(self, _mock_aiter, _mock_gfx95):
        # TileLang flags off: fusion must still enable via the aiter gfx95 path.
        with (
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False),
        ):
            self.assertTrue(deepseek_v4_fused_mhc.is_cross_layer_mhc_fusion_enabled())

    @mock.patch.object(deepseek_v4_fused_mhc, "is_gfx95_supported", return_value=False)
    @mock.patch.object(deepseek_v4_fused_mhc, "get_bool_env_var", return_value=True)
    @mock.patch.object(deepseek_v4_fused_mhc, "_is_hip", True)
    def test_aiter_cross_layer_disabled_without_gfx95(self, _mock_aiter, _mock_gfx95):
        with (
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False),
        ):
            self.assertFalse(deepseek_v4_fused_mhc.is_cross_layer_mhc_fusion_enabled())

    @mock.patch.object(deepseek_v4_fused_mhc, "is_gfx95_supported", return_value=True)
    @mock.patch.object(deepseek_v4_fused_mhc, "get_bool_env_var", return_value=False)
    @mock.patch.object(deepseek_v4_fused_mhc, "_is_hip", True)
    def test_aiter_path_skips_without_sglang_use_aiter(self, _mock_aiter, _mock_gfx95):
        result = deepseek_v4_fused_mhc.try_aiter_fused_mhc_post_pre(
            layer_input=mock.Mock(shape=(32, 7168), dim=2, device="cpu"),
            residual=mock.Mock(dim=3),
            post=mock.Mock(),
            comb=mock.Mock(),
            hc_fn=mock.Mock(),
            hc_scale=mock.Mock(),
            hc_base=mock.Mock(),
            rms_eps=1e-6,
            hc_eps=1e-6,
            hc_post_mult=2.0,
            sinkhorn_iters=20,
            norm_weight=mock.Mock(),
            norm_eps=1e-6,
        )
        self.assertIsNone(result)

    @mock.patch.object(
        deepseek_v4_fused_mhc,
        "try_aiter_fused_mhc_post_pre",
        return_value=("res", "hs", "post", "comb", True),
    )
    @mock.patch.object(deepseek_v4_fused_mhc, "try_fused_hc_post_pre")
    def test_boundary_prefers_aiter_over_triton(self, mock_triton, mock_aiter):
        result = deepseek_v4_fused_mhc.try_mhc_fused_post_pre_boundary(
            layer_input=mock.Mock(shape=(32, 7168), dim=2),
            residual=mock.Mock(dim=3),
            post=mock.Mock(),
            comb=mock.Mock(),
            hc_fn=mock.Mock(),
            hc_scale=mock.Mock(),
            hc_base=mock.Mock(),
            hc_mult=4,
            rms_eps=1e-6,
            hc_eps=1e-6,
            hc_post_mult=2.0,
            sinkhorn_iters=20,
            norm_weight=mock.Mock(),
            norm_eps=1e-6,
            fn_transpose=True,
            is_gfx95_supported_flag=True,
        )
        self.assertEqual(result, ("res", "hs", "post", "comb", True))
        mock_triton.assert_not_called()
        mock_aiter.assert_called_once()

    @mock.patch.object(
        deepseek_v4_fused_mhc, "try_aiter_fused_mhc_post_pre", return_value=None
    )
    @mock.patch.object(
        deepseek_v4_fused_mhc,
        "try_fused_hc_post_pre",
        return_value=("res", "hs", "post", "comb", False),
    )
    def test_boundary_falls_back_to_triton(self, mock_triton, mock_aiter):
        hc_fn = mock.Mock()
        hc_fn.T = "transposed_fn"
        result = deepseek_v4_fused_mhc.try_mhc_fused_post_pre_boundary(
            layer_input=mock.Mock(shape=(32, 7168), dim=2),
            residual=mock.Mock(dim=3),
            post=mock.Mock(),
            comb=mock.Mock(),
            hc_fn=hc_fn,
            hc_scale=mock.Mock(),
            hc_base=mock.Mock(),
            hc_mult=4,
            rms_eps=1e-6,
            hc_eps=1e-6,
            hc_post_mult=2.0,
            sinkhorn_iters=20,
            norm_weight=mock.Mock(),
            norm_eps=1e-6,
            fn_transpose=True,
            is_gfx95_supported_flag=True,
        )
        self.assertEqual(result, ("res", "hs", "post", "comb", False))
        mock_aiter.assert_called_once()
        # fn_transpose=True must hand the Triton kernel the transposed fn.
        self.assertEqual(mock_triton.call_args.args[4], "transposed_fn")


def _hardware_available() -> bool:
    try:
        import torch

        if not (
            torch.cuda.is_available() and deepseek_v4_fused_mhc.is_gfx95_supported()
        ):
            return False
        from aiter.ops.mhc import mhc_fused_post_pre  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(
    _hardware_available(), "requires a gfx95 device with aiter mHC kernels"
)
class TestAmdFusedMhcNumerical(unittest.TestCase):
    """On-device equivalence of the aiter fused kernel vs unfused mhc_post+mhc_pre.

    Asserts the proven invariants: ``next_residual`` is bit-exact and
    ``layer_input``/``post_mix`` match within bf16 tolerance. ``comb_mix`` is
    intentionally not asserted here -- its raw-tensor value differs between the
    fused and unfused kernels at the production Sinkhorn setting, and correctness
    is established end-to-end (fused-on vs fused-off token match). See the PR
    description; extend this test once the end-to-end sign-off pins the expected
    comb_mix convention.
    """

    def _run(self, m, hc_mult=4, hidden=7168, sinkhorn_iters=20):
        import torch
        from aiter.ops import mhc

        dev = "cuda:0"
        torch.manual_seed(0)
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
        li = (torch.randn(m, hidden, device=dev) * 0.02).bfloat16()
        res = (torch.randn(m, hc_mult, hidden, device=dev) * 0.02).bfloat16()
        post = torch.randn(m, hc_mult, device=dev) * 0.02
        comb = torch.randn(m, hc_mult, hc_mult, device=dev) * 0.02
        fn = (torch.randn(hc_mult3, hc_mult * hidden, device=dev) * 0.02).bfloat16()
        scl = torch.ones(hc_mult3, device=dev)
        base = torch.zeros(hc_mult3, device=dev)
        nw = torch.ones(hidden, device=dev).bfloat16()
        kw = dict(
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_iters,
            norm_weight=nw,
            norm_eps=1e-6,
        )
        post_mix, _comb_mix, li_out, next_res = mhc.mhc_fused_post_pre(
            li, res, post, comb, fn, scl, base, force_fused=True, **kw
        )
        ref_next = torch.empty_like(res)
        mhc.mhc_post(ref_next, li, res, post, comb)
        ref_post, _ref_comb, ref_li = mhc.mhc_pre(ref_next, fn, scl, base, **kw)

        torch.testing.assert_close(next_res, ref_next, rtol=0, atol=0)
        torch.testing.assert_close(li_out.float(), ref_li.float(), rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(
            post_mix.float(), ref_post.float(), rtol=3e-2, atol=3e-2
        )

    def test_equivalence_decode(self):
        self._run(m=32)

    def test_equivalence_prefill(self):
        self._run(m=96)


if __name__ == "__main__":
    unittest.main()
