"""CPU contract and dispatch tests for GLM-5.3-Flash mHC."""

import sys
import types
import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers.communicator_mhc import MHCState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMHCDispatch(CustomTestCase):
    hc_mult = 2
    hidden_size = 4

    def setUp(self):
        mhc._AITER_MHC_RUNTIME_DISABLED = False
        mhc._AITER_MHC_IMPORT_WARNED = False
        mhc._AITER_MHC_ACTIVE_LOGGED = False

    def _inputs(self, tokens=3):
        torch.manual_seed(0)
        total = self.hc_mult * self.hidden_size
        mix_size = 2 * self.hc_mult + self.hc_mult**2
        x = torch.randn(tokens, total, dtype=torch.bfloat16) * 0.1
        fn = torch.randn(mix_size, total, dtype=torch.float32) * 0.01
        scale = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32)
        base = torch.zeros(mix_size, dtype=torch.float32)
        return x, fn, scale, base

    def _pre(self, x, fn, scale, base, sinkhorn_iters=4):
        return mhc.hc_pre(
            x,
            fn,
            scale,
            base,
            self.hc_mult,
            rms_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=sinkhorn_iters,
        )

    def test_expand_contract_and_zero_token_contracts(self):
        x = torch.randn(3, self.hidden_size, dtype=torch.bfloat16)
        expanded = mhc.hc_expand(x, self.hc_mult)
        self.assertEqual(expanded.shape, (3, self.hc_mult * self.hidden_size))
        torch.testing.assert_close(mhc.hc_contract(expanded, self.hc_mult), x)

        empty, fn, scale, base = self._inputs(tokens=0)
        layer_input, h_res, h_post, norm_fused = self._pre(empty, fn, scale, base)
        self.assertEqual(layer_input.shape, (0, self.hidden_size))
        self.assertEqual(layer_input.dtype, torch.bfloat16)
        self.assertEqual(h_res.shape, (0, self.hc_mult**2))
        self.assertEqual(h_res.dtype, torch.float32)
        self.assertEqual(h_post.shape, (0, self.hc_mult))
        self.assertEqual(h_post.dtype, torch.float32)
        self.assertFalse(norm_fused)
        post = mhc.hc_post(layer_input, empty, h_post, h_res, self.hc_mult)
        self.assertEqual(post.shape, empty.shape)
        self.assertEqual(post.dtype, empty.dtype)

    def test_torch_reference_shapes_sinkhorn_and_round_trip(self):
        x, fn, scale, base = self._inputs()
        with (
            patch.object(mhc, "_use_aiter_mhc", return_value=False),
            patch.object(mhc, "_use_tilelang_mhc_pre", return_value=False),
            patch.object(mhc, "_use_tilelang_mhc_post", return_value=False),
        ):
            layer_input, h_res, h_post, norm_fused = self._pre(
                x, fn, scale, base, sinkhorn_iters=20
            )
            out = mhc.hc_post(layer_input, x, h_post, h_res, self.hc_mult)

        self.assertEqual(layer_input.shape, (3, self.hidden_size))
        self.assertEqual(layer_input.dtype, torch.bfloat16)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(norm_fused)
        comb = h_res.view(3, self.hc_mult, self.hc_mult)
        torch.testing.assert_close(
            comb.sum(dim=-1), torch.ones(3, self.hc_mult), atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            comb.sum(dim=-2), torch.ones(3, self.hc_mult), atol=2e-5, rtol=2e-5
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_aiter_pre_forwards_arguments_and_fused_norm(self):
        x, fn, scale, base = self._inputs()
        captured = {}

        def fake_pre(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return mhc._mhc_pre_torch(
                *args[:4],
                rms_eps=args[4],
                hc_pre_eps=args[5],
                hc_sinkhorn_eps=args[6],
                hc_post_mult_value=args[7],
                sinkhorn_repeat=args[8],
            )

        fake_module = types.ModuleType("aiter.ops.mhc")
        fake_module.mhc_pre = fake_pre
        modules = {
            "aiter": types.ModuleType("aiter"),
            "aiter.ops": types.ModuleType("aiter.ops"),
            "aiter.ops.mhc": fake_module,
        }
        norm_weight = torch.ones(self.hidden_size, dtype=torch.bfloat16)
        with (
            patch.dict(sys.modules, modules),
            patch.object(mhc, "_use_aiter_mhc", return_value=True),
            patch.object(mhc, "_use_tilelang_mhc_pre", return_value=False),
        ):
            result = mhc.hc_pre(
                x,
                fn,
                scale,
                base,
                self.hc_mult,
                rms_eps=1e-6,
                hc_eps=1e-6,
                sinkhorn_iters=4,
                out_norm_weight=norm_weight,
                out_norm_eps=1e-5,
            )

        self.assertIs(captured["args"][0].untyped_storage(), x.untyped_storage())
        self.assertIs(captured["args"][1], fn)
        self.assertIs(captured["args"][2], scale)
        self.assertIs(captured["args"][3], base)
        self.assertIs(captured["kwargs"]["norm_weight"], norm_weight)
        self.assertEqual(captured["kwargs"]["norm_eps"], 1e-5)
        self.assertTrue(result[3])

    def test_aiter_import_and_runtime_failures_latch_to_torch(self):
        x, fn, scale, base = self._inputs()
        missing = types.ModuleType("aiter.ops.mhc")
        modules = {
            "aiter": types.ModuleType("aiter"),
            "aiter.ops": types.ModuleType("aiter.ops"),
            "aiter.ops.mhc": missing,
        }
        with patch.dict(sys.modules, modules):
            result = mhc._try_aiter_mhc_pre(
                x.view(3, self.hc_mult, self.hidden_size),
                fn,
                scale,
                base,
                1e-6,
                1e-6,
                1e-6,
                2.0,
                4,
                None,
                None,
            )
        self.assertIsNone(result)
        self.assertTrue(mhc._AITER_MHC_RUNTIME_DISABLED)

        mhc._AITER_MHC_RUNTIME_DISABLED = False
        failing = types.ModuleType("aiter.ops.mhc")

        def fail_post(*_args, **_kwargs):
            raise RuntimeError("synthetic failure")

        failing.mhc_post = fail_post
        modules["aiter.ops.mhc"] = failing
        layer_input, h_res, h_post, _ = self._pre(x, fn, scale, base)
        with (
            patch.dict(sys.modules, modules),
            patch.object(mhc, "_use_aiter_mhc", return_value=True),
            patch.object(mhc, "_use_tilelang_mhc_post", return_value=False),
        ):
            out = mhc.hc_post(layer_input, x, h_post, h_res, self.hc_mult)
        self.assertTrue(mhc._AITER_MHC_RUNTIME_DISABLED)
        self.assertTrue(torch.isfinite(out).all())

    def test_hip_never_selects_tilelang_or_deepgemm(self):
        x, fn, scale, base = self._inputs()
        with (
            patch.object(mhc, "is_hip", return_value=True),
            patch.object(mhc, "_use_aiter_mhc", return_value=False),
            patch.object(
                mhc.envs.SGLANG_OPT_USE_TILELANG_MHC_PRE, "get", return_value=True
            ),
            patch.object(
                mhc.envs.SGLANG_OPT_USE_TILELANG_MHC_POST, "get", return_value=True
            ),
            patch.object(
                mhc.envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM, "get", return_value=True
            ),
            patch.object(
                mhc, "_load_tilelang", side_effect=AssertionError("TileLang imported")
            ),
        ):
            self.assertFalse(mhc._use_tilelang_mhc_pre())
            self.assertFalse(mhc._use_tilelang_mhc_post())
            self.assertFalse(mhc._use_deep_gemm_hc_prenorm())
            layer_input, h_res, h_post, _ = self._pre(x, fn, scale, base)
            out = mhc.hc_post(layer_input, x, h_post, h_res, self.hc_mult)
        self.assertTrue(torch.isfinite(out).all())

    def test_mhc_state_runs_attention_to_ffn_flow_and_resets(self):
        x, attn_fn, scale, base = self._inputs()
        _, ffn_fn, _, _ = self._inputs()

        def make_pre(fn):
            return lambda states, _weight, _eps: self._pre(states, fn, scale, base)

        state = MHCState(
            hc_mult=self.hc_mult,
            hc_attn_pre=make_pre(attn_fn),
            hc_ffn_pre=make_pre(ffn_fn),
            hc_post=lambda y, residual, h_res, h_post: mhc.hc_post(
                y, residual, h_post, h_res, self.hc_mult
            ),
        )
        with (
            patch.object(mhc, "_use_aiter_mhc", return_value=False),
            patch.object(mhc, "_use_tilelang_mhc_pre", return_value=False),
            patch.object(mhc, "_use_tilelang_mhc_post", return_value=False),
        ):
            attn_input, residual = state.attn_split(x)
            attn_output = attn_input + torch.tensor(0.125, dtype=attn_input.dtype)
            ffn_input, residual = state.attn_to_mlp(attn_output, residual)
            ffn_output = ffn_input * torch.tensor(0.5, dtype=ffn_input.dtype)
            final = state.mlp_combine(ffn_output, residual)

        self.assertEqual(final.shape, x.shape)
        self.assertEqual(final.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(final).all())
        self.assertIsNotNone(state.h_res)
        self.assertIsNotNone(state.h_post)
        state.reset_aux()
        self.assertIsNone(state.h_res)
        self.assertIsNone(state.h_post)


if __name__ == "__main__":
    unittest.main()
