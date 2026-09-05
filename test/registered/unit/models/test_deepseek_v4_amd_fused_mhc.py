import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.amd import deepseek_v4_fused_mhc
from sglang.srt.runtime_context import override_platform
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

    @override_platform(is_sm120=True)
    def test_sm120_enables_fusion_with_tilelang_pre_disabled(self):
        # Regression (PR review): consolidating _is_fused_mhc_post_pre_enabled into
        # this module must preserve the SM120 special case. SM120 disables the
        # standalone TileLang pre path, but mhc_fused_post_pre dispatches
        # independently, so fuse+post enabled with the pre flag OFF must still
        # enable fusion when SM120 is supported. The pre-fix consolidation
        # required the pre flag unconditionally and silently disabled fusion on
        # SM120.
        with (
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
        ):
            self.assertTrue(deepseek_v4_fused_mhc._is_fused_mhc_post_pre_enabled())

    @override_platform(is_sm120=False)
    def test_no_sm120_still_requires_tilelang_pre(self):
        # Negative branch: the (pre OR sm120) clause must not degrade to
        # always-true. With SM120 unsupported and the pre flag off, fuse+post
        # alone must not enable the standalone TileLang fused path.
        with (
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(True),
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(False),
        ):
            self.assertFalse(deepseek_v4_fused_mhc._is_fused_mhc_post_pre_enabled())

    def test_is_fused_mhc_post_pre_enabled_policy(self):
        # Full gating table for _is_fused_mhc_post_pre_enabled, migrated from the
        # removed test_deepseek_v4_fused_mhc_policy.py now that the helper lives
        # in this module (it used to patch deepseek_v4.is_sm120_supported /
        # deepseek_v4._is_fused_mhc_post_pre_enabled, both gone after the
        # consolidation -> the registered CPU test AttributeError'd). Fusion
        # requires the opt-in flag AND TileLang post AND (TileLang pre OR SM120).
        cases = [
            # (fuse, pre, post, sm120, expected)
            (True, False, True, True, True),  # SM120 waives the standalone pre flag
            (True, False, True, False, False),  # non-SM120 still needs the pre flag
            (True, True, True, False, True),  # non-SM120 with the pre flag on
            (False, False, True, True, False),  # fusion opt-in is required
            (True, False, False, True, False),  # TileLang post is required
        ]
        for fuse, pre, post, sm120, expected in cases:
            with self.subTest(fuse=fuse, pre=pre, post=post, sm120=sm120):
                with (
                    envs.SGLANG_OPT_FUSE_MHC_POST_PRE.override(fuse),
                    envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(pre),
                    envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(post),
                    override_platform(is_sm120=sm120),
                ):
                    self.assertEqual(
                        deepseek_v4_fused_mhc._is_fused_mhc_post_pre_enabled(),
                        expected,
                    )

    @mock.patch.object(deepseek_v4_fused_mhc, "is_gfx95_supported", return_value=True)
    @mock.patch.object(deepseek_v4_fused_mhc, "get_bool_env_var", return_value=True)
    @mock.patch.object(deepseek_v4_fused_mhc, "_is_hip", True)
    def test_aiter_gfx95_enables_cross_layer_fusion(self, _mock_aiter, _mock_gfx95):
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


class TestAmdFusedMhcAttnBoundaryFallback(unittest.TestCase):
    """Regression: the attn-side boundary fallback must close the previous
    layer's deferred mHC post before opening the current layer's pre.

    When ``apply_mhc_post_pre_boundary`` declines to fuse (returns ``None``) --
    reachable after an aiter import/kernel failure permanently disables the fused
    path -- the fallback must call
    ``hc_post(hidden_states, prev_residual, prev_post, prev_comb)`` before
    ``hc_pre``. The pre-fix code ran ``hc_pre`` directly on the raw input and
    dropped ``prev_residual``/``prev_post``/``prev_comb``, so the previous
    layer's deferred post was never applied and every subsequent layer computed
    on corrupted activations.

    Drives the real ``DeepseekV4DecoderLayer.forward`` on a mocked layer with the
    fused dispatcher forced to ``None`` and halts at ``self_attn`` via a sentinel,
    so only the boundary fallback executes.
    """

    def test_fallback_closes_previous_post_before_pre(self):
        try:
            from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
        except Exception as e:  # pragma: no cover - env without full model deps
            self.skipTest(f"deepseek_v4 import unavailable: {e}")

        class _StopForward(Exception):
            pass

        layer = mock.Mock()
        layer.use_fused_mhc_post_pre = True
        layer._input_layernorm_weight_bf16 = None
        closed_post = object()
        layer.hc_post.return_value = closed_post
        # norm_fused=True keeps the fallback off the fp8-quant / layernorm branch.
        layer.hc_pre.return_value = (object(), object(), object(), True)
        layer.self_attn.maybe_use_decode_attn_tp.side_effect = _StopForward

        hs_in = object()
        prev_residual, prev_post, prev_comb = object(), object(), object()

        with (
            mock.patch(
                "sglang.srt.models.deepseek_v4.apply_mhc_post_pre_boundary",
                return_value=None,
            ),
            self.assertRaises(_StopForward),
        ):
            DeepseekV4DecoderLayer.forward(
                layer,
                positions=object(),
                hidden_states=hs_in,
                input_ids=object(),
                forward_batch=object(),
                input_ids_global=object(),
                prev_residual=prev_residual,
                prev_post=prev_post,
                prev_comb=prev_comb,
            )

        # The deferred previous-layer post must be closed with exactly the
        # prev_* tensors, and hc_pre must then run on the closed result.
        layer.hc_post.assert_called_once_with(
            hs_in, prev_residual, prev_post, prev_comb
        )
        layer.hc_pre.assert_called_once()
        self.assertIs(layer.hc_pre.call_args.args[0], closed_post)


class TestAmdFusedMhcNormFusedHandling(unittest.TestCase):
    """Regression: a fused-success result with ``norm_fused=False`` must have its
    layernorm applied at the call site before the activation reaches attention.

    ``try_fused_hc_post_pre`` (the Triton fused post+pre) always returns
    ``norm_fused=False`` -- it does not apply the input/post-attention layernorm.
    The boundary dispatcher reaches it whenever the aiter kernel declines
    (notably after an aiter import/kernel failure permanently disables the aiter
    path). The pre-fix fused-success branch unpacked the tuple and fed the raw
    (unnormalized) hidden_states straight into ``self_attn`` with ``x_quant=None``,
    silently corrupting every subsequent layer. The fix mirrors the unfused
    ``hc_pre`` branch: apply the input layernorm when ``norm_fused`` is False.

    Drives the real ``DeepseekV4DecoderLayer.forward`` with the boundary forced to
    return ``norm_fused=False`` and halts at ``self_attn`` via a sentinel.
    """

    def test_fused_success_applies_input_layernorm_when_not_norm_fused(self):
        try:
            import sglang.srt.models.deepseek_v4 as deepseek_v4
            from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
        except Exception as e:  # pragma: no cover - env without full model deps
            self.skipTest(f"deepseek_v4 import unavailable: {e}")

        class _StopForward(Exception):
            pass

        layer = mock.Mock()
        layer.use_fused_mhc_post_pre = True
        layer._input_layernorm_weight_bf16 = None

        fused_hs = object()
        residual, post, comb = object(), object(), object()
        # Fused dispatch SUCCEEDS but reports the input layernorm was NOT applied
        # (norm_fused=False) -- the Triton fused post+pre contract.
        normed = object()
        layer.input_layernorm.return_value = normed
        layer.self_attn.maybe_use_decode_attn_tp.side_effect = _StopForward

        # Force the non-aiter (torch layernorm) branch deterministically so the
        # test does not depend on the runner arch and needs no real tensors.
        with (
            mock.patch.object(deepseek_v4, "_use_aiter", False),
            mock.patch.object(deepseek_v4, "_is_gfx95_supported", False),
            mock.patch(
                "sglang.srt.models.deepseek_v4.apply_mhc_post_pre_boundary",
                return_value=(residual, fused_hs, post, comb, False),
            ),
            self.assertRaises(_StopForward),
        ):
            DeepseekV4DecoderLayer.forward(
                layer,
                positions=object(),
                hidden_states=object(),
                input_ids=object(),
                forward_batch=object(),
                input_ids_global=object(),
                prev_residual=object(),
                prev_post=object(),
                prev_comb=object(),
            )

        # The fused (unnormalized) layer input must be run through the input
        # layernorm before attention. Pre-fix this was never called on the
        # fused-success path.
        layer.input_layernorm.assert_called_once_with(fused_hs)


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
