"""Accuracy and dispatch tests for the fused bf16->fp8 KV-cache write on the
standard aiter FP8 path. Skipped on CPU (Triton requires a GPU)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

_HAS_CUDA = torch.cuda.is_available()


def _naive_bf16_write(k, v, k_cache, v_cache, loc, k_scale, v_scale, fp8_dtype):
    """Naive reference: bf16 in-place divide -> fp8 cast -> scatter."""
    ck = k.clone()
    cv = v.clone()
    ck.div_(k_scale)
    cv.div_(v_scale)
    k_cache[loc, 0] = ck.to(fp8_dtype)
    v_cache[loc, 0] = cv.to(fp8_dtype)


def _fp32_divide_write(k, v, k_cache, v_cache, loc, k_scale, v_scale, fp8_dtype):
    """Fused-kernel reference: fp32 divide -> fp8 cast -> scatter."""
    k_cache[loc, 0] = (k.float() / k_scale.float()).to(fp8_dtype)
    v_cache[loc, 0] = (v.float() / v_scale.float()).to(fp8_dtype)


@unittest.skipUnless(_HAS_CUDA, "Triton kernels require a GPU")
class TestFusedFp8KvWrite(unittest.TestCase):
    def _run(self, num_tokens, num_heads, head_dim, total_slots=None, seed=0xC0FFEE):
        from sglang.srt.layers.attention.utils import (
            launch_reshape_and_cache_flash,
        )
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        torch.manual_seed(seed)
        dev = "cuda"
        page_size = 1
        if total_slots is None:
            total_slots = num_tokens
        assert num_tokens <= total_slots

        k = torch.randn(
            (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=dev
        )
        v = torch.randn(
            (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=dev
        )

        # Per-tensor descale = amax / fp8_max (matches the production path).
        fp8_max = torch.finfo(fp8_dtype).max
        k_scale = (k.abs().amax().float() / fp8_max).clamp_min(1e-12).view(1)
        v_scale = (v.abs().amax().float() / fp8_max).clamp_min(1e-12).view(1)

        loc = torch.randperm(total_slots, device=dev)[:num_tokens].to(torch.int64)

        # Fused path (kernel under test).
        k_cache_f = torch.zeros(
            (total_slots, page_size, num_heads, head_dim), dtype=fp8_dtype, device=dev
        )
        v_cache_f = torch.zeros_like(k_cache_f)
        launch_reshape_and_cache_flash(
            k.view(-1, num_heads, head_dim),
            v.view(-1, num_heads, head_dim),
            k_cache_f.view(-1, page_size, num_heads, head_dim),
            v_cache_f.view(-1, page_size, num_heads, head_dim),
            loc,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        return {
            "k": k,
            "v": v,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "loc": loc,
            "fp8_dtype": fp8_dtype,
            "k_cache_f": k_cache_f,
            "v_cache_f": v_cache_f,
        }

    def test_matches_fp32_divide_reference(self):
        """Fused write matches the fp32-divide reference (within fp8 rounding)."""
        r = self._run(num_tokens=32, num_heads=1, head_dim=256)
        loc, fp8_dtype = r["loc"], r["fp8_dtype"]

        k_ref = torch.zeros_like(r["k_cache_f"])
        v_ref = torch.zeros_like(r["v_cache_f"])
        _fp32_divide_write(
            r["k"], r["v"], k_ref, v_ref, loc, r["k_scale"], r["v_scale"], fp8_dtype
        )

        k_mismatch = (r["k_cache_f"] != k_ref).float().mean().item()
        v_mismatch = (r["v_cache_f"] != v_ref).float().mean().item()
        self.assertLess(k_mismatch, 1e-3, f"K vs fp32-divide ref: {k_mismatch:.4%}")
        self.assertLess(v_mismatch, 1e-3, f"V vs fp32-divide ref: {v_mismatch:.4%}")

    def test_no_precision_regression_vs_naive(self):
        """Fused write is at least as close to the fp32 reference as naive."""
        r = self._run(num_tokens=8000, num_heads=1, head_dim=256)
        loc, fp8_dtype = r["loc"], r["fp8_dtype"]

        k_naive = torch.zeros_like(r["k_cache_f"])
        v_naive = torch.zeros_like(r["v_cache_f"])
        _naive_bf16_write(
            r["k"], r["v"], k_naive, v_naive, loc, r["k_scale"], r["v_scale"], fp8_dtype
        )

        ref_k = r["k"].float() / r["k_scale"].float()
        ref_v = r["v"].float() / r["v_scale"].float()

        err_fused = (r["k_cache_f"][loc, 0].float() - ref_k).abs().mean().item() + (
            r["v_cache_f"][loc, 0].float() - ref_v
        ).abs().mean().item()
        err_naive = (k_naive[loc, 0].float() - ref_k).abs().mean().item() + (
            v_naive[loc, 0].float() - ref_v
        ).abs().mean().item()

        mismatch = (
            (r["k_cache_f"][loc, 0] != k_naive[loc, 0]).float().mean().item()
            + (r["v_cache_f"][loc, 0] != v_naive[loc, 0]).float().mean().item()
        ) / 2.0
        print(
            f"[fused_fp8_kv_write] prefill fused_vs_naive_mismatch={mismatch:.4%} "
            f"err_fused={err_fused:.5f} err_naive={err_naive:.5f}"
        )

        self.assertLessEqual(
            err_fused,
            err_naive * 1.02,
            f"fused err {err_fused:.5f} regressed vs naive {err_naive:.5f}",
        )

    def test_writes_only_target_slots(self):
        """Fused write touches only the target slots; others stay zero."""
        r = self._run(num_tokens=16, num_heads=4, head_dim=128, total_slots=64)
        loc, fp8_dtype = r["loc"], r["fp8_dtype"]

        k_ref = torch.zeros_like(r["k_cache_f"])
        v_ref = torch.zeros_like(r["v_cache_f"])
        _fp32_divide_write(
            r["k"], r["v"], k_ref, v_ref, loc, r["k_scale"], r["v_scale"], fp8_dtype
        )

        # Target slots match the fp32-divide reference.
        self.assertTrue(
            torch.equal(r["k_cache_f"][loc], k_ref[loc]), "target-slot K mismatch"
        )
        self.assertTrue(
            torch.equal(r["v_cache_f"][loc], v_ref[loc]), "target-slot V mismatch"
        )

        # Non-target slots must remain zero (no stray writes).
        untouched = torch.ones(
            r["k_cache_f"].shape[0], dtype=torch.bool, device=loc.device
        )
        untouched[loc] = False
        self.assertEqual(
            r["k_cache_f"][untouched].float().abs().sum().item(),
            0.0,
            "fused wrote into non-target K slots",
        )
        self.assertEqual(
            r["v_cache_f"][untouched].float().abs().sum().item(),
            0.0,
            "fused wrote into non-target V slots",
        )


class _StopForward(Exception):
    """Short-circuit forward_* after the KV write to skip the attention math."""


@unittest.skipUnless(_HAS_CUDA, "Triton kernels require a GPU")
class TestAiterFp8KvDispatch(unittest.TestCase):
    """Backend dispatch: scale passing (incl. v_scale fallback) and the
    head-dim-mismatch fallback."""

    def _make_backend(self, fp8_dtype, self_k_scale, self_v_scale):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        be = AiterAttnBackend.__new__(AiterAttnBackend)
        be.kv_cache_dtype = fp8_dtype
        be.k_scale = self_k_scale
        be.v_scale = self_v_scale
        be.kv_cache_is_vectorized_5d = False
        be.use_triton_unified_attention = False
        be.use_sliding_window_kv_pool = False
        be.use_mla = False
        be.page_size = 1

        class _Meta:
            swa_out_cache_loc = None

        be.forward_metadata = _Meta()
        return be

    def _make_layer(self, heads, qk_dim, v_dim):
        class _Layer:
            k_scale = None
            v_scale = None
            layer_id = 0
            tp_q_head_num = heads
            tp_k_head_num = heads
            tp_v_head_num = heads
            qk_head_dim = qk_dim
            v_head_dim = v_dim
            is_cross_attention = False

        return _Layer()

    def _make_pool(self, num_slots, heads, qk_dim, v_dim, fp8_dtype, on_set):
        kc = torch.zeros((num_slots, 1, heads, qk_dim), dtype=fp8_dtype, device="cuda")
        vc = torch.zeros((num_slots, 1, heads, v_dim), dtype=fp8_dtype, device="cuda")

        class _Pool:
            def get_kv_buffer(self, layer_id):
                return kc, vc

            def set_kv_buffer(self, *args, **kwargs):
                on_set(args, kwargs)
                raise _StopForward

        return _Pool()

    def test_decode_uses_fused_and_v_scale_falls_back_to_self(self):
        """Standard FP8 decode takes the fused path; with layer.v_scale=None,
        v_scale falls back to self.v_scale (not self.k_scale)."""
        from unittest import mock

        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        dev = "cuda"
        heads, dim, n = 2, 64, 4
        k_scale = torch.tensor([0.111], device=dev)
        v_scale = torch.tensor([0.222], device=dev)

        be = self._make_backend(fp8_dtype, k_scale, v_scale)
        be.token_to_kv_pool = self._make_pool(
            n, heads, dim, dim, fp8_dtype, on_set=lambda a, k: None
        )
        layer = self._make_layer(heads, dim, dim)  # k_scale/v_scale None -> fallback

        q = torch.randn((n, heads, dim), dtype=torch.bfloat16, device=dev)
        k = torch.randn((n, heads, dim), dtype=torch.bfloat16, device=dev)
        v = torch.randn((n, heads, dim), dtype=torch.bfloat16, device=dev)

        class _FB:
            out_cache_loc = torch.arange(n, device=dev, dtype=torch.int64)

        captured = {}

        def fake_launch(*args, **kwargs):
            captured["kwargs"] = kwargs
            raise _StopForward

        with mock.patch(
            "sglang.srt.layers.attention.aiter_backend.launch_reshape_and_cache_flash",
            fake_launch,
        ):
            try:
                be.forward_decode(q, k, v, layer, _FB(), save_kv_cache=True)
            except _StopForward:
                pass

        self.assertIn("kwargs", captured, "fused path was not taken for standard FP8")
        self.assertIs(
            captured["kwargs"]["v_scale"],
            v_scale,
            "v_scale must fall back to self.v_scale, not self.k_scale",
        )
        self.assertIs(captured["kwargs"]["k_scale"], k_scale)

    def test_decode_head_dim_mismatch_falls_back_to_set_kv_buffer(self):
        """qk_head_dim != v_head_dim falls back to set_kv_buffer (kernel reuses
        K's head_dim for V)."""
        from unittest import mock

        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        dev = "cuda"
        heads, qk_dim, v_dim, n = 2, 64, 32, 4  # qk_head_dim != v_head_dim
        k_scale = torch.tensor([0.111], device=dev)
        v_scale = torch.tensor([0.222], device=dev)
        be = self._make_backend(fp8_dtype, k_scale, v_scale)
        set_calls = []
        be.token_to_kv_pool = self._make_pool(
            n,
            heads,
            qk_dim,
            v_dim,
            fp8_dtype,
            on_set=lambda a, k: set_calls.append((a, k)),
        )
        layer = self._make_layer(heads, qk_dim, v_dim)

        q = torch.randn((n, heads, qk_dim), dtype=torch.bfloat16, device=dev)
        k = torch.randn((n, heads, qk_dim), dtype=torch.bfloat16, device=dev)
        v = torch.randn((n, heads, v_dim), dtype=torch.bfloat16, device=dev)

        class _FB:
            out_cache_loc = torch.arange(n, device=dev, dtype=torch.int64)

        launch_calls = {"n": 0}

        def fake_launch(*args, **kwargs):
            launch_calls["n"] += 1
            raise _StopForward

        with mock.patch(
            "sglang.srt.layers.attention.aiter_backend.launch_reshape_and_cache_flash",
            fake_launch,
        ):
            try:
                be.forward_decode(q, k, v, layer, _FB(), save_kv_cache=True)
            except _StopForward:
                pass

        self.assertEqual(
            launch_calls["n"], 0, "fused kernel must be skipped on head-dim mismatch"
        )
        self.assertEqual(
            len(set_calls), 1, "must fall back to set_kv_buffer on head-dim mismatch"
        )
        # Fallback must still forward the descales, else the fp8 cache is wrong.
        args, kwargs = set_calls[0]
        passed = list(args) + list(kwargs.values())
        self.assertTrue(any(a is k_scale for a in passed), "fallback dropped k_scale")
        self.assertTrue(any(a is v_scale for a in passed), "fallback dropped v_scale")

    def test_decode_mla_uses_set_kv_buffer_without_scales(self):
        """MLA decode must call the MLA pool's set_kv_buffer(layer, loc, k, v)
        without scale args (its signature takes no scales)."""
        from unittest import mock

        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        dev = "cuda"
        heads, dim, n = 2, 64, 4
        be = self._make_backend(
            fp8_dtype,
            torch.tensor([0.111], device=dev),
            torch.tensor([0.222], device=dev),
        )
        be.use_mla = True
        set_calls = []
        be.token_to_kv_pool = self._make_pool(
            n, heads, dim, dim, fp8_dtype, on_set=lambda a, k: set_calls.append((a, k))
        )
        layer = self._make_layer(heads, dim, dim)

        q = torch.randn((n, heads, dim), dtype=torch.bfloat16, device=dev)
        k = torch.randn((n, heads, dim), dtype=torch.bfloat16, device=dev)
        v = torch.randn((n, heads, dim), dtype=torch.bfloat16, device=dev)

        class _FB:
            out_cache_loc = torch.arange(n, device=dev, dtype=torch.int64)

        launch_calls = {"n": 0}

        def fake_launch(*args, **kwargs):
            launch_calls["n"] += 1
            raise _StopForward

        with mock.patch(
            "sglang.srt.layers.attention.aiter_backend.launch_reshape_and_cache_flash",
            fake_launch,
        ):
            try:
                be.forward_decode(q, k, v, layer, _FB(), save_kv_cache=True)
            except _StopForward:
                pass

        self.assertEqual(launch_calls["n"], 0, "MLA must not use the fused kernel")
        self.assertEqual(len(set_calls), 1, "MLA must use set_kv_buffer")
        args, kwargs = set_calls[0]
        # MLA pool signature is (layer, loc, k, v) — exactly 4 args, no scales.
        self.assertEqual(
            len(args), 4, f"MLA set_kv_buffer got {len(args)} args (scales leaked?)"
        )
        self.assertEqual(kwargs, {})


class TestUseFusedFp8KvWritePredicate(unittest.TestCase):
    """Unit test for the shared _use_fused_fp8_kv_write predicate."""

    def _backend(self, **overrides):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

        be = AiterAttnBackend.__new__(AiterAttnBackend)
        be.kv_cache_dtype = overrides.get("kv_cache_dtype", fp8_dtype)
        be.use_mla = overrides.get("use_mla", False)
        be.use_sliding_window_kv_pool = overrides.get(
            "use_sliding_window_kv_pool", False
        )
        return be

    def _layer(self, tp_k=2, tp_v=2, qk_dim=64, v_dim=64):
        class _Layer:
            tp_k_head_num = tp_k
            tp_v_head_num = tp_v
            qk_head_dim = qk_dim
            v_head_dim = v_dim

        return _Layer()

    def test_predicate(self):
        # Eligible: FP8 + non-MLA + non-SWA + matching K/V heads.
        self.assertTrue(self._backend()._use_fused_fp8_kv_write(self._layer()))
        self.assertFalse(
            self._backend(kv_cache_dtype=torch.bfloat16)._use_fused_fp8_kv_write(
                self._layer()
            )
        )
        self.assertFalse(
            self._backend(use_mla=True)._use_fused_fp8_kv_write(self._layer())
        )
        self.assertFalse(
            self._backend(use_sliding_window_kv_pool=True)._use_fused_fp8_kv_write(
                self._layer()
            )
        )
        self.assertFalse(
            self._backend()._use_fused_fp8_kv_write(self._layer(qk_dim=64, v_dim=32))
        )
        self.assertFalse(
            self._backend()._use_fused_fp8_kv_write(self._layer(tp_k=2, tp_v=1))
        )


if __name__ == "__main__":
    unittest.main()
