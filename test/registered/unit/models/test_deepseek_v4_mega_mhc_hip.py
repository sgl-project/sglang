import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models import deepseek_v4_platform_stage
from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _cfg(*, mult=4, hidden=5120, cp_prefill=False):
    return mhc.HcConfig(
        mult=mult,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-6,
        hidden=hidden,
        pre_from_prev=True,
        cp_prefill=cp_prefill,
    )


class _Norm:
    def __init__(self):
        self.weight = torch.ones(4)
        self.variance_epsilon = 1e-6


def _sublayer(cfg):
    return mhc.HcSubLayer(
        cfg=cfg,
        fn=torch.zeros(1),
        scale=torch.zeros(1),
        base=torch.zeros(1),
        norm=_Norm(),
    )


class TestMegaMhcGate(unittest.TestCase):
    def test_gate(self):
        # (flag, gfx950, config, expected)
        cases = [
            (False, True, _cfg(), False),
            (True, True, _cfg(), True),
            (True, True, _cfg(mult=2), False),
            (True, True, _cfg(hidden=4096), False),
            (True, True, _cfg(cp_prefill=True), False),
            (True, False, _cfg(), False),
        ]
        for flag, gfx95, cfg, expected in cases:
            with (
                self.subTest(flag=flag, gfx95=gfx95, cfg=cfg),
                envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(flag),
                mock.patch.object(mhc, "_is_hip", True),
                mock.patch.object(mhc, "_is_gfx95_supported", gfx95),
            ):
                self.assertEqual(mhc._can_use_mega_mhc(cfg), expected)


class TestHipFlydslDispatch(unittest.TestCase):
    def _boundary(self, cfg):
        hc = _sublayer(cfg)
        return mhc.HcNextBoundary(
            norm=hc.norm,
            accepts_mxfp8=False,
            norm_fusable=True,
            hc=hc,
            extend=True,
        )

    def test_calls_flydsl_bf16_and_packs_next_stats(self):
        cfg = _cfg()
        y = torch.zeros(2, 4)
        residual = torch.zeros(2, 4, 4)
        pre = torch.zeros(2, 4)
        post_mix = torch.zeros(2, 4)
        comb = torch.zeros(2, 4, 4)
        updated = torch.ones(2, 4, 4)
        normalized = torch.full((2, 4), 3.0)
        next_post = torch.full((2, 4, 1), 4.0)
        next_comb = torch.full((2, 4, 4), 5.0)
        next_pre = torch.full((2, 4), 6.0)
        seen = {}

        def fake_flydsl(*args, **kwargs):
            seen["out_dtype"] = kwargs["out_dtype"]
            seen["post_ndim"] = kwargs["post_layer_mix"].ndim
            return updated, next_post, next_comb, normalized, next_pre

        flydsl = types.ModuleType("aiter.ops.flydsl")
        flydsl.flydsl_mega_mhc = fake_flydsl
        ops = types.ModuleType("aiter.ops")
        ops.flydsl = flydsl
        aiter = types.ModuleType("aiter")
        aiter.ops = ops
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(mhc, "_is_hip", True),
            mock.patch.object(mhc, "_is_gfx95_supported", True),
            mock.patch.dict(
                sys.modules,
                {
                    "aiter": aiter,
                    "aiter.ops": ops,
                    "aiter.ops.flydsl": flydsl,
                },
            ),
        ):
            state = mhc._post_fusion(
                _sublayer(cfg), y, residual, (pre, post_mix, comb), self._boundary(cfg)
            )

        self.assertEqual(seen["out_dtype"], "bf16")
        self.assertEqual(seen["post_ndim"], 3)
        self.assertIsInstance(state.input, mhc.HcNormed)
        self.assertTrue(torch.equal(state.input.rows, normalized))
        self.assertTrue(torch.equal(state.streams, updated))
        self.assertEqual(state.stats[0].shape, (2, 4))
        self.assertEqual(state.stats[1].shape, (2, 4))
        self.assertTrue(torch.equal(state.pre, pre))


class TestHipClosedSeam(unittest.TestCase):
    def _hc(self, hidden=5120):
        norm = _Norm()
        norm.weight = torch.ones(hidden, dtype=torch.bfloat16)
        norm.cast_x_before_out_mul = False
        norm.variance_size_override = None
        return mhc.HcSubLayer(
            cfg=_cfg(),
            fn=torch.zeros(1),
            scale=torch.zeros(1),
            base=torch.zeros(1),
            norm=norm,
        )

    def _run(self, state, *, hc=None, extend=True, flag=True):
        normalized = torch.full((2, 4), 3.0)
        post_mix = torch.full((2, 4, 1), 4.0)
        comb = torch.full((2, 4, 4), 5.0)
        next_pre = torch.full((2, 4), 6.0)
        calls = []

        def fake_flydsl(residual, *args, **kwargs):
            calls.append((args, kwargs))
            return residual, post_mix, comb, normalized, next_pre

        flydsl = types.ModuleType("aiter.ops.flydsl")
        flydsl.flydsl_mega_mhc = fake_flydsl
        ops = types.ModuleType("aiter.ops")
        ops.flydsl = flydsl
        aiter = types.ModuleType("aiter")
        aiter.ops = ops
        hc = hc or self._hc()
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(flag),
            mock.patch.object(mhc, "_is_hip", True),
            mock.patch.object(mhc, "_is_gfx95_supported", True),
            mock.patch.dict(
                sys.modules,
                {"aiter": aiter, "aiter.ops": ops, "aiter.ops.flydsl": flydsl},
            ),
        ):
            out = mhc.mega_closed_seam(hc, state, extend)
        return out, calls, hc, (next_pre, comb, normalized)

    def test_engram_seam_runs_no_post_mega(self):
        residual = torch.zeros(2, 4, 5120)
        pre = torch.full((2, 4), 0.25)
        state = mhc.HcState(residual, pre)
        out, calls, hc, (next_pre, comb, normalized) = self._run(state)

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertIs(args[8], pre)
        for key in ("sublayer_out", "post_layer_mix", "comb_res_mix", "residual_out"):
            self.assertNotIn(key, kwargs)
        self.assertIs(kwargs["norm_weight"], hc.norm.weight)
        self.assertEqual(kwargs["out_dtype"], "bf16")
        self.assertIs(out.streams, residual)
        self.assertIs(out.pre, pre)
        self.assertIsInstance(out.input, mhc.HcNormed)
        self.assertTrue(torch.equal(out.input.rows, normalized))
        self.assertTrue(torch.equal(out.stats[0], next_pre))
        self.assertEqual(out.stats[1].shape, (2, 4))
        self.assertTrue(torch.equal(out.stats[2], comb))
        self.assertIsNone(state.streams)

    def test_first_layer_takes_identity_pre(self):
        out, calls, _, _ = self._run(mhc.HcState(torch.zeros(2, 4, 5120)))
        self.assertIsNone(calls[0][0][8])
        self.assertIsNone(out.pre)

    def test_other_seams_unchanged(self):
        residual = torch.zeros(2, 4, 5120)
        pre = torch.zeros(2, 4)
        cases = {
            "folded input": (
                mhc.HcState(residual, pre, mhc.HcNormed(residual[:, 0])),
                {},
            ),
            "folded stats": (
                mhc.HcState(residual, pre, None, (pre, pre, residual)),
                {},
            ),
            "decode": (mhc.HcState(residual, pre), {"extend": False}),
            "flag off": (mhc.HcState(residual, pre), {"flag": False}),
            "no rows": (mhc.HcState(torch.zeros(0, 4, 5120), pre), {}),
            "norm not fusable": (mhc.HcState(residual, pre), {"hc": self._hc(4096)}),
        }
        for name, (state, kwargs) in cases.items():
            with self.subTest(name):
                out, calls, _, _ = self._run(state, **kwargs)
                self.assertIs(out, state)
                self.assertEqual(calls, [])


class TestHipBoundaryExtend(unittest.TestCase):
    def test_prefill_boundaries_extend(self):
        layer = deepseek_v4_platform_stage.NonCudaDecoderMixin()
        layer.hc_cfg = _cfg()
        layer.hc_stats_stream = None
        layer.attn_hc = layer.ffn_hc = _sublayer(_cfg())
        layer.post_attention_layernorm = _Norm()
        layer._next_layer = mock.Mock()
        layer.self_attn = mock.MagicMock(attn_tp_size=1)
        layer.self_attn.accepts_mxfp8_swizzled_input.return_value = False
        layer._can_fuse_attn_mhc = layer._can_fuse_ffn_mhc = False
        layer.mlp = mock.Mock(tp_size=1)
        layer._run_moe_ffn_dp_sync = mock.Mock()
        seen = []

        def boundary(norm, **kwargs):
            return mhc.HcNextBoundary(
                norm=norm, accepts_mxfp8=False, norm_fusable=True, hc=kwargs["hc"]
            )

        def post(hc, y, residual, *, next, **kwargs):
            seen.append(next.extend)
            return mock.Mock()

        def closed_seam(hc, state, extend):
            seen.append(extend)
            return state

        modes = {
            ForwardMode.EXTEND: True,
            ForwardMode.MIXED: True,
            ForwardMode.DECODE: False,
            ForwardMode.TARGET_VERIFY: False,
        }
        with (
            mock.patch.object(mhc, "make_boundary", side_effect=boundary),
            mock.patch.object(mhc, "use_stats_stream", return_value=False),
            mock.patch.object(mhc, "fork_stats_stream"),
            mock.patch.object(mhc, "combine"),
            mock.patch.object(mhc, "mega_closed_seam", side_effect=closed_seam),
            mock.patch.object(mhc, "run_attn_post", side_effect=post),
            mock.patch.object(mhc, "run_moe_post", side_effect=post),
        ):
            for mode, extend in modes.items():
                with self.subTest(mode=mode.name):
                    seen.clear()
                    layer.forward_hc_pre_from_prev(
                        None,
                        mock.Mock(),
                        None,
                        types.SimpleNamespace(forward_mode=mode),
                        None,
                    )
                    # the closed-seam check, the attention seam, then the MoE seam
                    self.assertEqual(seen, [extend, extend, extend])


class TestHipDraftMega(unittest.TestCase):
    def _hc(self, hidden):
        norm = _Norm()
        norm.weight = torch.ones(hidden, dtype=torch.bfloat16)
        norm.cast_x_before_out_mul = False
        norm.variance_size_override = None
        return mhc.HcSubLayer(
            cfg=_cfg(hidden=hidden),
            fn=torch.zeros(1),
            scale=torch.zeros(1),
            base=torch.zeros(1),
            norm=norm,
        )

    def _stage(self, rows=2, hidden=5120):
        stage = deepseek_v4_platform_stage.NonCudaDSparkMixin()
        stage.hc_cfg = _cfg(hidden=hidden)
        stage.hc_stats_stream = None
        stage.attn_hc, stage.ffn_hc = self._hc(hidden), self._hc(hidden)
        stage.self_attn = mock.MagicMock()
        stage.self_attn.return_value = torch.full((rows, hidden), 7.0)
        stage._run_ffn = mock.Mock(return_value=torch.full((rows, hidden), 8.0))
        stage.hc_post = mock.Mock(return_value=torch.full((rows, 4, hidden), 9.0))
        return stage

    def _run(self, stage, residual, prev_pre, *, flag=True, prefill=True, hip=True):
        rows, hidden = residual.shape[0], residual.shape[-1]
        # (post_mix, comb, normalized, next_pre) of the no-post launch
        nopost = (
            torch.full((rows, 4, 1), 1.0),
            torch.full((rows, 4, 4), 2.0),
            torch.full((rows, hidden), 3.0),
            torch.full((rows, 4), 4.0),
        )
        # (updated, next_post, next_comb, normalized, next_pre) of the post launch
        post = (
            torch.full((rows, 4, hidden), 5.0),
            torch.full((rows, 4, 1), 6.0),
            torch.full((rows, 4, 4), 7.0),
            torch.full((rows, hidden), 8.0),
            torch.full((rows, 4), 9.0),
        )
        calls = []

        def fake_flydsl(residual, *args, **kwargs):
            calls.append((residual, args, kwargs))
            return post if "sublayer_out" in kwargs else (residual, *nopost)

        flydsl = types.ModuleType("aiter.ops.flydsl")
        flydsl.flydsl_mega_mhc = fake_flydsl
        ops = types.ModuleType("aiter.ops")
        ops.flydsl = flydsl
        aiter = types.ModuleType("aiter")
        aiter.ops = ops
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_DRAFT.override(flag),
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(prefill),
            mock.patch.object(mhc, "_is_hip", hip),
            mock.patch.object(mhc, "_is_gfx95_supported", True),
            mock.patch.dict(
                sys.modules,
                {"aiter": aiter, "aiter.ops": ops, "aiter.ops.flydsl": flydsl},
            ),
        ):
            out = stage._forward_hc_pre_from_prev(
                torch.arange(rows), residual, mock.Mock(), prev_pre
            )
        return out, calls, nopost, post

    def test_stage_runs_two_launches(self):
        stage = self._stage()
        residual = torch.zeros(2, 4, 5120)
        prev_pre = torch.full((2, 4), 0.25)
        (out, next_pre), calls, nopost, post = self._run(stage, residual, prev_pre)

        self.assertEqual(len(calls), 2)
        (r0, a0, k0), (r1, a1, k1) = calls
        # the attention seam: no post, the carried pre, the attention's weights
        self.assertIs(r0, residual)
        self.assertIs(a0[0], stage.attn_hc.fn)
        self.assertIs(a0[8], prev_pre)
        self.assertNotIn("sublayer_out", k0)
        self.assertIs(k0["norm_weight"], stage.attn_hc.norm.weight)
        self.assertTrue(torch.equal(stage.self_attn.call_args.args[1], nopost[2]))
        # the attention post with the FFN seam: the attention's mixes, the FFN's weights
        self.assertIs(r1, residual)
        self.assertIs(a1[0], stage.ffn_hc.fn)
        self.assertTrue(torch.equal(a1[8], nopost[3]))
        self.assertIs(k1["sublayer_out"], stage.self_attn.return_value)
        self.assertTrue(torch.equal(k1["post_layer_mix"], nopost[0]))
        self.assertIs(k1["comb_res_mix"], nopost[1])
        self.assertIs(k1["norm_weight"], stage.ffn_hc.norm.weight)
        # the FFN reads the second launch's input; its plain post closes the stage
        self.assertTrue(torch.equal(stage._run_ffn.call_args.args[0], post[3]))
        x, res, post_mix, comb = stage.hc_post.call_args.args
        self.assertIs(x, stage._run_ffn.return_value)
        self.assertIs(res, post[0])
        self.assertTrue(torch.equal(post_mix, post[1].squeeze(-1)))
        self.assertIs(comb, post[2])
        self.assertIs(out, stage.hc_post.return_value)
        self.assertTrue(torch.equal(next_pre, post[4]))

    def test_first_stage_takes_identity_pre(self):
        _, calls, _, _ = self._run(self._stage(), torch.zeros(2, 4, 5120), None)
        self.assertIsNone(calls[0][1][8])

    def test_plain_stage_where_mega_does_not_serve(self):
        cases = {
            "switch off": ({"flag": False}, 2, 5120, False),
            "prefill switch off": ({"prefill": False}, 2, 5120, False),
            "not HIP": ({"hip": False}, 2, 5120, False),
            "no rows": ({}, 0, 5120, False),
            "other width": ({}, 2, 4096, False),
            "FFN norm not fusable": ({}, 2, 5120, True),
        }
        for name, (kwargs, rows, hidden, ffn_cast) in cases.items():
            with self.subTest(name):
                stage = self._stage(rows, hidden)
                stage.ffn_hc.norm.cast_x_before_out_mul = ffn_cast
                combine = mock.Mock(return_value=torch.zeros(rows, hidden))
                stats = mock.Mock(
                    return_value=(
                        torch.zeros(rows, 4),
                        torch.zeros(rows, 4),
                        torch.zeros(rows, 4, 4),
                    )
                )
                with (
                    mock.patch.object(mhc, "combine", combine),
                    mock.patch.object(mhc, "mix_stats", stats),
                ):
                    _, calls, _, _ = self._run(
                        stage, torch.zeros(rows, 4, hidden), None, **kwargs
                    )
                self.assertEqual(calls, [])
                self.assertEqual(combine.call_count, 2)
                self.assertEqual(stats.call_count, 2)


class TestFusedBoundaryRouting(unittest.TestCase):
    def test_routing(self):
        layer = mock.Mock(hc_pre_from_prev_sublayer=True, hc_mult=4)
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            mock.patch.object(deepseek_v4_hip, "_is_gfx95_supported", True),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fp8_quant_eligible",
                return_value=False,
            ),
            mock.patch.object(
                deepseek_v4_hip.gfx95_dense,
                "fused_rmsnorm_fake_quant_eligible",
                return_value=False,
            ),
        ):
            deepseek_v4_hip.init_decoder_layer(layer, quant_config=None)
        # the AMD monokernel refuses a layer without it at attach
        self.assertTrue(layer.hc_boundary_fused)

        prefill = (ForwardMode.EXTEND, ForwardMode.MIXED)
        decode = (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        )
        # mega -> whether prefill and decode steps take the fused boundary
        cases = {False: (True, True), True: (False, True)}
        for mega, (fused_prefill, fused_decode) in cases.items():
            for modes, expected in ((prefill, fused_prefill), (decode, fused_decode)):
                for mode in modes:
                    with (
                        self.subTest(mega=mega, mode=mode.name),
                        envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(mega),
                    ):
                        fused = deepseek_v4_hip.fused_boundary_this_step(
                            types.SimpleNamespace(forward_mode=mode)
                        )
                        self.assertEqual(fused, expected)


if __name__ == "__main__":
    unittest.main()
