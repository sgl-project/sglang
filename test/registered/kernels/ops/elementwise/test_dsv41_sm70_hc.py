"""SM70 DSV4.1 mHC kernels vs CPU Sinkhorn. Not registered for GPU CI."""

from __future__ import annotations

import os
import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

HC = 4
HIDDEN = 5120
HC_DIM = HC * HIDDEN
MIX_HC = (2 + HC) * HC
SINKHORN_ITERS = 20
RMS_EPS = 1e-6
HC_EPS = 1e-6


def _require_sm70():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    if torch.cuda.get_device_capability()[0] != 7:
        raise unittest.SkipTest("SM70 required")


def _torch_mix_stats(x, hc_fn, rms_eps=RMS_EPS):
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    return (F.linear(x_flat, hc_fn) * rsqrt)


def _torch_combine(x, pre):
    # y[t,h] = sum_k pre[t,k] * x[t,k,h]
    return (pre.unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)


def _torch_post(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).type_as(x)


class TestDsv41Sm70Hc(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _require_sm70()
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        torch.set_num_threads(2)
        # Import after the SM70 skip so CPU CI collection does not JIT-compile.
        from sglang.kernels.ops.elementwise.sm70_dsv41_hc_mix import (
            combine,
            hc_post,
            hc_pre,
            mix_and_combine,
            mix_sinkhorn,
            mix_sinkhorn_combine,
            mix_stats,
            split_sinkhorn,
        )

        cls.mix_stats = staticmethod(mix_stats)
        cls.split_sinkhorn = staticmethod(split_sinkhorn)
        cls.combine = staticmethod(combine)
        cls.hc_post = staticmethod(hc_post)
        cls.hc_pre = staticmethod(hc_pre)
        cls.mix_and_combine = staticmethod(mix_and_combine)
        cls.mix_sinkhorn = staticmethod(mix_sinkhorn)
        cls.mix_sinkhorn_combine = staticmethod(mix_sinkhorn_combine)

    def _inputs(self, tokens: int, seed: int = 0):
        torch.manual_seed(seed)
        device = torch.device("cuda")
        x = torch.randn(tokens, HC, HIDDEN, device=device, dtype=torch.float16) * 0.1
        hc_fn = (
            torch.randn(MIX_HC, HC_DIM, device=device, dtype=torch.float32) * 0.01
        )
        hc_scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
        hc_base = torch.randn(MIX_HC, device=device, dtype=torch.float32) * 0.1
        return x, hc_fn, hc_scale, hc_base

    def test_sinkhorn_vs_cpu_golden_tokens_1_and_4(self):
        for tokens in (1, 4):
            with self.subTest(tokens=tokens):
                x, hc_fn, hc_scale, hc_base = self._inputs(tokens)
                mixes_cpu = _torch_mix_stats(x, hc_fn).detach().cpu()
                mixes_b = mixes_cpu.unsqueeze(1)  # [T, 1, 24]
                pre_ref, post_ref, comb_ref = _hc_split_sinkhorn_torch(
                    mixes_b,
                    hc_scale.cpu(),
                    hc_base.cpu(),
                    hc_mult=HC,
                    sinkhorn_iters=SINKHORN_ITERS,
                    eps=HC_EPS,
                )
                pre, post, comb = self.split_sinkhorn(
                    mixes_b.to(x.device),
                    hc_scale,
                    hc_base,
                    HC,
                    SINKHORN_ITERS,
                    HC_EPS,
                )
                torch.testing.assert_close(
                    pre.cpu(), pre_ref, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    post.cpu(), post_ref, rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    comb.cpu(), comb_ref, rtol=1e-5, atol=1e-5
                )

    def test_mix_stats_close_to_torch_linear(self):
        x, hc_fn, _, _ = self._inputs(tokens=1)
        got = self.mix_stats(x, hc_fn, RMS_EPS)
        ref = _torch_mix_stats(x, hc_fn)
        # 20480-wide CTA tree vs cublas; fp16 activations. Not bitwise.
        torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)

    def test_combine_and_post_vs_torch(self):
        x, _, _, _ = self._inputs(tokens=4)
        pre = torch.rand(4, HC, device=x.device, dtype=torch.float32)
        post = torch.rand(4, HC, device=x.device, dtype=torch.float32)
        comb = torch.rand(4, HC, HC, device=x.device, dtype=torch.float32)
        hidden = torch.randn(4, HIDDEN, device=x.device, dtype=torch.float16) * 0.1
        y = self.combine(x, pre)
        y_ref = _torch_combine(x, pre)
        torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)
        out = self.hc_post(hidden, x, post, comb)
        out_ref = _torch_post(hidden, x, post, comb)
        torch.testing.assert_close(out, out_ref, rtol=1e-3, atol=1e-3)

    def test_mix_sinkhorn_matches_unfused(self):
        for tokens in (1, 4):
            with self.subTest(tokens=tokens):
                x, hc_fn, hc_scale, hc_base = self._inputs(tokens, seed=1)
                mixes = self.mix_stats(x, hc_fn, RMS_EPS)
                pre_u, post_u, comb_u = self.split_sinkhorn(
                    mixes, hc_scale, hc_base, HC, SINKHORN_ITERS, HC_EPS
                )
                pre_f, post_f, comb_f = self.mix_sinkhorn(
                    x,
                    hc_fn,
                    hc_scale,
                    hc_base,
                    sinkhorn_iters=SINKHORN_ITERS,
                    rms_eps=RMS_EPS,
                    hc_eps=HC_EPS,
                )
                torch.testing.assert_close(pre_f, pre_u, rtol=0, atol=0)
                torch.testing.assert_close(post_f, post_u, rtol=0, atol=0)
                torch.testing.assert_close(comb_f, comb_u, rtol=0, atol=0)

                apply_pre = torch.rand(tokens, HC, device=x.device, dtype=torch.float32)
                y_u = self.combine(x, apply_pre)
                y_f, pre2, post2, comb2 = self.mix_sinkhorn_combine(
                    x,
                    hc_fn,
                    hc_scale,
                    hc_base,
                    apply_pre,
                    sinkhorn_iters=SINKHORN_ITERS,
                    rms_eps=RMS_EPS,
                    hc_eps=HC_EPS,
                    use_apply_pre=True,
                )
                torch.testing.assert_close(y_f, y_u, rtol=0, atol=0)
                torch.testing.assert_close(pre2, pre_u, rtol=0, atol=0)
                torch.testing.assert_close(post2, post_u, rtol=0, atol=0)
                torch.testing.assert_close(comb2, comb_u, rtol=0, atol=0)

                y_new_u = self.combine(x, pre_u)
                y_new_f, pre3, _, _ = self.mix_sinkhorn_combine(
                    x,
                    hc_fn,
                    hc_scale,
                    hc_base,
                    combine_pre=None,
                    sinkhorn_iters=SINKHORN_ITERS,
                    rms_eps=RMS_EPS,
                    hc_eps=HC_EPS,
                    use_apply_pre=False,
                )
                torch.testing.assert_close(pre3, pre_u, rtol=0, atol=0)
                torch.testing.assert_close(y_new_f, y_new_u, rtol=0, atol=0)

    def test_mix_and_combine_and_hc_pre_fusion_matches_unfused(self):
        x, hc_fn, hc_scale, hc_base = self._inputs(tokens=1, seed=2)
        apply_pre = torch.rand(1, HC, device=x.device, dtype=torch.float32)

        def _id_norm(t):
            return t

        with envs.SGLANG_DSV41_MHC_FUSION.override(False):
            y_u, pre_u, post_u, comb_u = self.mix_and_combine(
                x, hc_fn, hc_scale, hc_base, apply_pre, _id_norm
            )
            y_pre_u, post_pre_u, comb_pre_u, flag_u = self.hc_pre(
                x, hc_fn, hc_scale, hc_base
            )
        with envs.SGLANG_DSV41_MHC_FUSION.override(True):
            y_f, pre_f, post_f, comb_f = self.mix_and_combine(
                x, hc_fn, hc_scale, hc_base, apply_pre, _id_norm
            )
            y_pre_f, post_pre_f, comb_pre_f, flag_f = self.hc_pre(
                x, hc_fn, hc_scale, hc_base
            )
        torch.testing.assert_close(y_f, y_u, rtol=0, atol=0)
        torch.testing.assert_close(pre_f, pre_u, rtol=0, atol=0)
        torch.testing.assert_close(post_f, post_u, rtol=0, atol=0)
        torch.testing.assert_close(comb_f, comb_u, rtol=0, atol=0)
        torch.testing.assert_close(y_pre_f, y_pre_u, rtol=0, atol=0)
        torch.testing.assert_close(post_pre_f, post_pre_u, rtol=0, atol=0)
        torch.testing.assert_close(comb_pre_f, comb_pre_u, rtol=0, atol=0)
        self.assertEqual(flag_u, False)
        self.assertEqual(flag_f, False)

        # Copy-0 collapse (apply_pre is None) still runs fused mix+sinkhorn.
        with envs.SGLANG_DSV41_MHC_FUSION.override(False):
            y0_u, pre0_u, _, _ = self.mix_and_combine(
                x, hc_fn, hc_scale, hc_base, None, _id_norm
            )
        with envs.SGLANG_DSV41_MHC_FUSION.override(True):
            y0_f, pre0_f, _, _ = self.mix_and_combine(
                x, hc_fn, hc_scale, hc_base, None, _id_norm
            )
        torch.testing.assert_close(y0_f, y0_u, rtol=0, atol=0)
        torch.testing.assert_close(pre0_f, pre0_u, rtol=0, atol=0)

    def test_fused_faster_than_three_launches(self):
        x, hc_fn, hc_scale, hc_base = self._inputs(tokens=1, seed=3)
        apply_pre = torch.rand(1, HC, device=x.device, dtype=torch.float32)
        # Warm the JIT module before timing.
        self.mix_sinkhorn_combine(
            x, hc_fn, hc_scale, hc_base, apply_pre, use_apply_pre=True
        )
        torch.cuda.synchronize()

        def _unfused():
            mixes = self.mix_stats(x, hc_fn, RMS_EPS)
            pre, _, _ = self.split_sinkhorn(
                mixes, hc_scale, hc_base, HC, SINKHORN_ITERS, HC_EPS
            )
            return self.combine(x, apply_pre)

        def _fused():
            return self.mix_sinkhorn_combine(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                apply_pre,
                use_apply_pre=True,
            )

        def _timed(fn, iters=50):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters

        unfused_ms = _timed(_unfused)
        fused_ms = _timed(_fused)
        print(
            f"D6-d mHC unfused {unfused_ms:.3f} ms vs fused {fused_ms:.3f} ms "
            f"({unfused_ms / fused_ms:.2f}x; mix GEMV dominates, fusion "
            f"cuts launches/graph nodes)",
            flush=True,
        )
        # Mix GEMV is the work; extra launches are ~noise at this size. Do not
        # gate on a speedup -- correctness tests above are the contract.

    def test_qwen_sm70_hc_not_used(self):
        import sglang.kernels.ops.elementwise.sm70_dsv41_hc_mix as dsv

        self.assertFalse(hasattr(dsv, "hc_down"))
        self.assertNotIn("sm70_hc_mix", dsv.__file__)


if __name__ == "__main__":
    unittest.main()
