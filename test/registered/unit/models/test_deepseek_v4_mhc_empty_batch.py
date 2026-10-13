"""Mega-mHC must skip 0-row (DP-idle) batches; see ``_post_fusion``."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

try:
    from sglang.srt.models import deepseek_v4_mhc as mhc

    _SKIP_REASON = None
except ModuleNotFoundError as e:
    if e.name != "deep_gemm":
        raise
    mhc = None
    _SKIP_REASON = f"deep_gemm is not installed: {e}"

MULT = 4
HIDDEN = 5120


def _cfg():
    return mhc.HcConfig(MULT, 20, 1e-6, 1e-6, HIDDEN, True, False)


def _norm(device="cpu"):
    return SimpleNamespace(
        weight=torch.ones(HIDDEN, dtype=torch.bfloat16, device=device),
        variance_epsilon=1e-6,
        cast_x_before_out_mul=False,
        variance_size_override=None,
    )


def _sublayer(device="cpu", norm=None):
    return mhc.HcSubLayer(
        _cfg(),
        torch.zeros(MULT * 2 + MULT * MULT, MULT * HIDDEN, device=device),
        torch.ones(3, device=device),
        torch.zeros(MULT * 2 + MULT * MULT, device=device),
        norm if norm is not None else _norm(device),
    )


def _boundary(device="cpu"):
    return mhc.HcNextBoundary(
        norm=_norm(device),
        accepts_mxfp8=False,
        norm_fusable=True,
        hc=_sublayer(device),
    )


def _idle_attention_output(device="cpu"):
    return torch.empty(0, MULT, HIDDEN, dtype=torch.bfloat16, device=device)[:, 0, :]


def _inputs(rows, device="cpu", strided_empty=True):
    if rows == 0 and strided_empty:
        y = _idle_attention_output(device)
    else:
        y = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    residual = torch.randn(rows, MULT, HIDDEN, dtype=torch.bfloat16, device=device)
    coefficients = (
        torch.rand(rows, MULT, device=device),
        torch.rand(rows, MULT, device=device),
        torch.rand(rows, MULT, MULT, device=device),
    )
    return y, residual, coefficients


class _FakeMegaBoundary:
    def __init__(self):
        self.calls = 0

    def __call__(self, x, residual, pre, post, comb, *args):
        self.calls += 1
        updated = torch.empty_like(residual)
        normalized = torch.empty_like(x)
        next_pre = torch.empty((x.shape[0], MULT), dtype=torch.float32, device=x.device)
        return updated, normalized, (next_pre, torch.empty_like(next_pre), comb)


class _PostRecorder:
    def __init__(self, real_post):
        self.real_post = real_post
        self.calls = []

    def __call__(self, cfg, x, residual, post_mix, comb):
        self.calls.append(x.shape[0])
        if x.shape[0] == 0:
            return self.real_post(cfg, x, residual, post_mix, comb)
        return torch.empty_like(residual)


def _moe_post_importable():
    try:
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (  # noqa: F401
            can_fuse_all_reduce,
        )
    except Exception:
        return False
    return True


@unittest.skipIf(mhc is None, _SKIP_REASON)
class TestMegaMhcEmptyBatch(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeMegaBoundary()
        self.post = _PostRecorder(mhc.post)
        for p in (
            mock.patch.object(mhc, "mhc_mega_boundary", self.fake),
            mock.patch.object(mhc, "post", self.post),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _run(self, seam, rows, mega, strided_empty=True):
        y, residual, coefficients = _inputs(rows, strided_empty=strided_empty)
        hc, nxt = _sublayer(), _boundary()
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(mega),
            override_platform(is_sm100=True),
        ):
            if seam == "post_fusion":
                return mhc._post_fusion(hc, y, residual, coefficients, nxt)
            fn = mhc.run_attn_post if seam == "attn" else mhc.run_moe_post
            return fn(
                hc,
                y,
                residual,
                stats_stream=None,
                next=nxt,
                world_size=1,
                precomputed=coefficients,
            )

    def _seams(self):
        seams = ["post_fusion", "attn"]
        if _moe_post_importable():
            seams.append("moe")
        return seams

    def test_diagnostic_empty_like_keeps_stride_of_empty_view(self):
        y = _idle_attention_output()
        self.assertEqual(y.shape, (0, HIDDEN))
        self.assertEqual(y.stride(), (MULT * HIDDEN, 1))
        self.assertTrue(y.is_contiguous())
        if torch.empty_like(y).stride() != (MULT * HIDDEN, 1):
            self.skipTest(
                "this PyTorch no longer preserves the strided layout in empty_like"
            )
        self.assertEqual(torch.empty_like(y).contiguous().stride(), (MULT * HIDDEN, 1))

    def test_zero_rows_skip_mega_and_return_canonical_state(self):
        for seam in self._seams():
            for strided_empty in (True, False):
                with self.subTest(seam=seam, strided_empty=strided_empty):
                    self.fake.calls = 0
                    state = self._run(seam, 0, True, strided_empty)
                    self.assertEqual(self.fake.calls, 0)
                    self.assertIsNone(state.input)
                    self.assertIsNone(state.stats)
                    self.assertIsNotNone(state.pre)
                    self.assertEqual(state.residual.shape, (0, MULT, HIDDEN))
                    self.assertEqual(
                        state.residual.stride(), (MULT * HIDDEN, HIDDEN, 1)
                    )

    def test_nonempty_rows_still_use_mega(self):
        for seam in self._seams():
            for rows in (1, 2):
                with self.subTest(seam=seam, rows=rows):
                    self.fake.calls = 0
                    self.post.calls.clear()
                    state = self._run(seam, rows, True)
                    self.assertEqual(self.fake.calls, 1)
                    self.assertIsInstance(state.input, mhc.HcNormed)
                    self.assertEqual(state.input.rows.shape, (rows, HIDDEN))
                    self.assertIsNotNone(state.stats)
                    self.assertEqual(self.post.calls, [])

    def test_state_narrowed_to_zero_rows(self):
        rows = 3
        _, residual, coefficients = _inputs(rows)
        carried = mhc.HcState(
            residual,
            coefficients[0],
            mhc.HcNormed(torch.randn(rows, HIDDEN, dtype=torch.bfloat16)),
            coefficients,
        )
        empty = carried.take_rows(lambda t: t[:0])
        self.assertEqual(empty.residual.shape, (0, MULT, HIDDEN))
        self.assertIsNotNone(empty.input)
        self.assertIsNotNone(empty.stats)
        y0 = _idle_attention_output()
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            override_platform(is_sm100=True),
        ):
            state = mhc._post_fusion(
                _sublayer(), y0, empty.residual, empty.stats, _boundary()
            )
        self.assertEqual(self.fake.calls, 0)
        self.assertIsNone(state.input)
        self.assertIsNone(state.stats)
        self.assertEqual(state.residual.stride(), (MULT * HIDDEN, HIDDEN, 1))

    def test_mega_off_is_unchanged(self):
        for seam in self._seams():
            for rows in (0, 1, 2):
                with self.subTest(seam=seam, rows=rows):
                    self.fake.calls = 0
                    self.post.calls.clear()
                    state = self._run(seam, rows, False)
                    self.assertEqual(self.fake.calls, 0)
                    self.assertEqual(self.post.calls, [rows])
                    self.assertIsNone(state.input)
                    self.assertEqual(state.residual.shape, (rows, MULT, HIDDEN))


@unittest.skipIf(mhc is None, _SKIP_REASON)
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMegaMhcEmptyBatchCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        major, _ = torch.cuda.get_device_capability()
        if major != 10:
            raise unittest.SkipTest("Mega-mHC needs an sm10x GPU")
        try:
            import deep_gemm

            if not hasattr(deep_gemm, "mega_mhc"):
                raise ImportError("deep_gemm.mega_mhc missing")
        except ImportError as e:
            raise unittest.SkipTest(str(e))

    def _setup(self):
        from sglang.srt.layers.layernorm import RMSNorm

        norm = RMSNorm(HIDDEN, eps=1e-6).cuda().bfloat16()
        hc = _sublayer("cuda", norm)
        nxt = mhc.HcNextBoundary(
            norm=norm, accepts_mxfp8=False, norm_fusable=True, hc=hc
        )
        return norm, hc, nxt

    def test_zero_rows_next_combine_is_canonical(self):
        _, hc, nxt = self._setup()
        y, residual, coefficients = _inputs(0, "cuda")
        with (
            envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
            override_platform(is_sm100=True),
        ):
            state = mhc._post_fusion(hc, y, residual, coefficients, nxt)
        out = mhc.combine(hc, state)
        self.assertEqual(out.shape, (0, HIDDEN))
        self.assertEqual(out.stride(), (HIDDEN, 1))

    def test_nonempty_rows_match_direct_kernel_call(self):
        norm, hc, nxt = self._setup()
        for rows in (1, 2):
            with self.subTest(rows=rows):
                torch.manual_seed(rows)
                y, residual, coefficients = _inputs(rows, "cuda")
                pre, post_mix, comb = coefficients
                updated, normalized, stats = mhc.mhc_mega_boundary(
                    y,
                    residual,
                    pre,
                    post_mix,
                    comb,
                    hc.fn,
                    hc.scale,
                    hc.base,
                    norm.weight,
                    hc.cfg.rms_eps,
                    hc.cfg.eps,
                    norm.variance_epsilon,
                    hc.cfg.sinkhorn_iters,
                )
                with (
                    envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True),
                    override_platform(is_sm100=True),
                ):
                    state = mhc._post_fusion(hc, y, residual, coefficients, nxt)
                self.assertIsInstance(state.input, mhc.HcNormed)
                torch.testing.assert_close(state.residual, updated, rtol=0, atol=0)
                torch.testing.assert_close(state.input.rows, normalized, rtol=0, atol=0)
                for got, want in zip(state.stats, stats):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
