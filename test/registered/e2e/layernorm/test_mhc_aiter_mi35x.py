"""Isolated AITER mHC correctness tests for GLM-5.3-Flash dimensions."""

import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestAiterMHCGLM53Flash(CustomTestCase):
    hidden_size = 4096
    hc_mult = 4
    rms_eps = 1e-6
    hc_eps = 1e-6

    def setUp(self):
        mhc._AITER_MHC_RUNTIME_DISABLED = False
        mhc._AITER_MHC_IMPORT_WARNED = False
        mhc._AITER_MHC_ACTIVE_LOGGED = False
        self.env = patch.dict(os.environ, {"SGLANG_USE_AITER": "1"})
        self.env.start()
        self.allocators = (
            patch.object(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext()),
            patch.object(mhc, "is_allocation_symmetric", return_value=False),
            patch.object(mhc, "get_tp_group", return_value=None),
        )
        for item in self.allocators:
            item.start()

    def tearDown(self):
        for item in reversed(self.allocators):
            item.stop()
        self.env.stop()

    def _inputs(self, tokens: int, seed: int = 0):
        torch.manual_seed(seed)
        device = torch.device("cuda")
        mix_size = 2 * self.hc_mult + self.hc_mult**2
        residual = (
            torch.randn(
                tokens,
                self.hc_mult,
                self.hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        fn = (
            torch.randn(
                mix_size,
                self.hc_mult * self.hidden_size,
                device=device,
                dtype=torch.float32,
            )
            * 0.01
        )
        scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
        base = torch.zeros(mix_size, device=device, dtype=torch.float32)
        return residual, fn, scale, base

    def _aiter_pre(
        self,
        residual,
        fn,
        scale,
        base,
        sinkhorn_iters,
        norm_weight=None,
    ):
        result = mhc._try_aiter_mhc_pre(
            residual,
            fn,
            scale,
            base,
            self.rms_eps,
            self.hc_eps,
            self.hc_eps,
            2.0,
            sinkhorn_iters,
            norm_weight,
            self.rms_eps if norm_weight is not None else None,
        )
        self.assertIsNotNone(result, "AITER mHC pre unexpectedly fell back")
        return result

    def test_aiter_pre_post_match_torch_oracle(self):
        for tokens in (1, 8, 17, 32, 64, 128):
            for sinkhorn_iters in (2, 20):
                with self.subTest(tokens=tokens, sinkhorn_iters=sinkhorn_iters):
                    residual, fn, scale, base = self._inputs(tokens)
                    post_ref, comb_ref, layer_ref = mhc._mhc_pre_torch(
                        residual,
                        fn,
                        scale,
                        base,
                        self.rms_eps,
                        self.hc_eps,
                        self.hc_eps,
                        2.0,
                        sinkhorn_iters,
                    )
                    post_out, comb_out, layer_out = self._aiter_pre(
                        residual, fn, scale, base, sinkhorn_iters
                    )
                    torch.cuda.synchronize()

                    torch.testing.assert_close(post_out, post_ref, atol=2e-3, rtol=2e-3)
                    torch.testing.assert_close(comb_out, comb_ref, atol=2e-3, rtol=2e-3)
                    torch.testing.assert_close(
                        layer_out, layer_ref, atol=2e-2, rtol=2e-2
                    )

                    x = layer_ref * torch.tensor(
                        0.75, device=layer_ref.device, dtype=layer_ref.dtype
                    )
                    post_ref_out = mhc._mhc_post_torch(x, residual, post_ref, comb_ref)
                    post_out_actual = mhc._try_aiter_mhc_post(
                        x, residual, post_out, comb_out
                    )
                    self.assertIsNotNone(
                        post_out_actual, "AITER mHC post unexpectedly fell back"
                    )
                    torch.testing.assert_close(
                        post_out_actual, post_ref_out, atol=2e-2, rtol=2e-2
                    )
                    for tensor in (
                        post_out,
                        comb_out,
                        layer_out,
                        post_out_actual,
                    ):
                        self.assertTrue(torch.isfinite(tensor).all())

    def test_fused_norm_and_zero_token_contracts(self):
        norm_weight = torch.linspace(
            0.75,
            1.25,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for tokens in (1, 32):
            with self.subTest(tokens=tokens):
                residual, fn, scale, base = self._inputs(tokens)
                post_ref, comb_ref, layer_ref = mhc._mhc_pre_torch(
                    residual,
                    fn,
                    scale,
                    base,
                    self.rms_eps,
                    self.hc_eps,
                    self.hc_eps,
                    2.0,
                    20,
                )
                post_out, comb_out, layer_out = self._aiter_pre(
                    residual, fn, scale, base, 20, norm_weight=norm_weight
                )
                norm_ref = (
                    layer_ref.float()
                    * torch.rsqrt(
                        layer_ref.float().square().mean(dim=-1, keepdim=True)
                        + self.rms_eps
                    )
                    * norm_weight.float()
                ).bfloat16()
                torch.testing.assert_close(post_out, post_ref, atol=2e-3, rtol=2e-3)
                torch.testing.assert_close(comb_out, comb_ref, atol=2e-3, rtol=2e-3)
                torch.testing.assert_close(layer_out, norm_ref, atol=2e-2, rtol=2e-2)

        residual, fn, scale, base = self._inputs(0)
        flat = residual.reshape(0, self.hc_mult * self.hidden_size)
        layer_input, h_res, h_post, norm_fused = mhc.hc_pre(
            flat,
            fn,
            scale,
            base,
            self.hc_mult,
            self.rms_eps,
            self.hc_eps,
            20,
        )
        self.assertEqual(layer_input.shape, (0, self.hidden_size))
        self.assertEqual(h_res.shape, (0, self.hc_mult**2))
        self.assertEqual(h_post.shape, (0, self.hc_mult))
        self.assertFalse(norm_fused)

    def _run_flow(self, x, attn_fn, ffn_fn, scale, base, use_aiter):
        dispatch = (
            nullcontext()
            if use_aiter
            else patch.object(mhc, "_use_aiter_mhc", return_value=False)
        )
        with (
            dispatch,
            patch.object(mhc, "_use_tilelang_mhc_pre", return_value=False),
            patch.object(mhc, "_use_tilelang_mhc_post", return_value=False),
        ):
            attn_input, attn_res, attn_post, _ = mhc.hc_pre(
                x,
                attn_fn,
                scale,
                base,
                self.hc_mult,
                self.rms_eps,
                self.hc_eps,
                20,
            )
            attn_output = (attn_input.float() * 0.75 + 0.125).bfloat16()
            after_attn = mhc.hc_post(attn_output, x, attn_post, attn_res, self.hc_mult)
            ffn_input, ffn_res, ffn_post, _ = mhc.hc_pre(
                after_attn,
                ffn_fn,
                scale,
                base,
                self.hc_mult,
                self.rms_eps,
                self.hc_eps,
                20,
            )
            ffn_output = (ffn_input.float() * 0.5 - 0.0625).bfloat16()
            after_ffn = mhc.hc_post(
                ffn_output, after_attn, ffn_post, ffn_res, self.hc_mult
            )
            output = mhc.hc_contract(after_ffn, self.hc_mult)
        return (
            attn_input,
            attn_res,
            attn_post,
            after_attn,
            ffn_input,
            ffn_res,
            ffn_post,
            after_ffn,
            output,
        )

    def test_complete_two_sublayer_flow_matches_and_is_deterministic(self):
        for tokens in (0, 1, 17, 64, 128):
            with self.subTest(tokens=tokens):
                residual, attn_fn, scale, base = self._inputs(tokens)
                _, ffn_fn, _, _ = self._inputs(tokens, seed=1)
                x = residual.reshape(tokens, self.hc_mult * self.hidden_size)
                reference = self._run_flow(
                    x, attn_fn, ffn_fn, scale, base, use_aiter=False
                )
                actual = self._run_flow(x, attn_fn, ffn_fn, scale, base, use_aiter=True)
                repeated = self._run_flow(
                    x, attn_fn, ffn_fn, scale, base, use_aiter=True
                )

                for index, (out, ref) in enumerate(zip(actual, reference)):
                    tolerance = 3e-3 if index in (1, 2, 5, 6) else 4e-2
                    torch.testing.assert_close(out, ref, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(out, repeated[index], atol=0, rtol=0)
                    self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
