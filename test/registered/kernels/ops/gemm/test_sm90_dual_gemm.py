"""Exercise WGMMA pipeline reuse, tails, and graph replay against independent math."""

import unittest

import torch

from sglang.kernels.ops.gemm import sm90_dual_gemm_swiglu
from sglang.kernels.ops.gemm.sm90_dual_gemm import (
    _COMPILED,
    SM90_DUAL_GEMM_TACTICS,
    _pick_sm90_tactic,
)
from sglang.kernels.selector import clear_cache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=360, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _is_sm90():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _reference(x, w):
    # Independent FP64 projections preserve the explicit low-precision GEMM
    # boundary; up/gate use different weights to catch accidental aliasing.
    gate, up = (x.double() @ w.double().T).to(x.dtype).float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up).to(x.dtype)


@unittest.skipUnless(_is_sm90(), "requires an SM90 CUDA device")
class TestSM90DualGemm(CustomTestCase):
    def _inputs(self, m, n, k, dtype):
        generator = torch.Generator(device="cuda").manual_seed(17)
        x = torch.randn(m, k, device="cuda", dtype=dtype, generator=generator) * 0.25
        w = torch.randn(2 * n, k, device="cuda", dtype=dtype, generator=generator) * 0.1
        return x, w

    def _check(self, x, w, actual=None, tactic=-1):
        if actual is None:
            actual = sm90_dual_gemm_swiglu(x, w, tactic)
        ref = _reference(x, w)
        self.assertEqual(actual.shape, ref.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertTrue(torch.isfinite(actual).all().item())
        # One-ULP projection rounding can amplify near cancellation in SwiGLU;
        # use a mixed tolerance and also constrain the normalized RMS error.
        atol = 0.006 if x.dtype == torch.bfloat16 else 0.001
        torch.testing.assert_close(actual, ref, atol=atol, rtol=0.03)
        rms = ((actual.float() - ref.float()).square().mean()).sqrt()
        ref_rms = ref.float().square().mean().sqrt().clamp_min(1e-6)
        self.assertLess((rms / ref_rms).item(), 0.015)

    def test_short_pipeline_and_wraparound_with_tails(self):
        for dtype in (torch.bfloat16, torch.float16):
            for tactic in SM90_DUAL_GEMM_TACTICS:
                for m, n, k in (
                    (1, 1, 64),  # single stage, padded K for K=128 tactics
                    (1, 64, 128),  # singleton X row must retain a 2D TMA map
                    (1, 65, 128),  # reproduces the reported TMA illegal instruction
                    (8, 1, 128),  # singleton weight rows must also retain 2D maps
                    (8, 64, 128),
                    (9, 65, 192),  # token, feature and reduction tails
                    (16, 127, 384),  # stages fill; feature tail
                    (16, 129, 1024),  # ring buffer wraps for every tactic
                ):
                    with self.subTest(dtype=dtype, tactic=tactic, m=m, n=n, k=k):
                        self._check(*self._inputs(m, n, k, dtype), tactic=tactic)

    def test_graph_replay_observes_changed_inputs_and_weights(self):
        x, w = self._inputs(9, 131, 256, torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._check(x, w)  # compile before capture on a nondefault stream
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = sm90_dual_gemm_swiglu(x, w)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(3):
            x.normal_(std=0.25)
            w.normal_(std=0.1)
            graph.replay()
            self._check(x, w, actual)

    def test_storage_contract_and_empty_batch(self):
        x, w = self._inputs(2, 65, 128, torch.bfloat16)
        self.assertEqual(sm90_dual_gemm_swiglu(x[:0], w).shape, (0, 65))
        misaligned_x = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[
            1:
        ].view_as(x)
        for invalid_x, invalid_w in (
            (x[:, :96].contiguous(), w[:, :96].contiguous()),  # partial K tile
            (x[:, ::2], w[:, ::2]),  # noncontiguous
            (x, w.float()),  # mismatched storage type
            (x, w[:, :64].contiguous()),  # mismatched K
            (misaligned_x, w),  # contiguous, but not 16-byte aligned
        ):
            with self.subTest(shape=invalid_x.shape, dtype=invalid_w.dtype):
                with self.assertRaises(ValueError):
                    sm90_dual_gemm_swiglu(invalid_x, invalid_w)

    def test_decode_limit_and_tactic_contract(self):
        x, w = self._inputs(17, 128, 128, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "at most 16"):
            sm90_dual_gemm_swiglu(x, w)
        for tactic in (-2, 0, 3, 4):
            with self.assertRaisesRegex(ValueError, "tactic"):
                sm90_dual_gemm_swiglu(x[:1], w, tactic)

    def test_auto_dispatch_shape_boundaries(self):
        # Preserve measured winners and exercise the inferred region at
        # reduction endpoints, interpolated widths and the one-wave boundary.
        pairs = (
            (2048, 7168),
            (4096, 3584),
            (4096, 7168),
            (2048, 3584),
            (3072, 4096),
            (4096, 7169),
            (4096, 7296),
        )
        for k, n in pairs:
            for dtype in (torch.bfloat16, torch.float16):
                for m in range(1, 17):
                    with self.subTest(m=m, k=k, n=n, dtype=dtype):
                        self.assertEqual(_pick_sm90_tactic(m, k, n, 114, dtype), 2)
                for m in (0, 17):
                    self.assertEqual(_pick_sm90_tactic(m, k, n, 114, dtype), 1)
                for sms in (78, 132):
                    self.assertEqual(_pick_sm90_tactic(8, k, n, sms, dtype), 1)
            self.assertEqual(_pick_sm90_tactic(8, k, n, 114, torch.float32), 1)
        for k, n in (
            (4096, 14336),
            (4096, 18944),
            (8192, 14336),
            (1984, 7168),
            (4160, 7168),
            (4096, 7297),
            (4096, 0),
            (8192, 7168),
        ):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(k=k, n=n, dtype=dtype):
                    self.assertEqual(_pick_sm90_tactic(8, k, n, 114, dtype), 1)

    def test_auto_dispatch_reuses_explicit_specialization(self):
        # Auto and explicit selection must share a specialization across
        # both token tiles, dtypes and the measured K/I pairs.
        device = torch.device("cuda", torch.cuda.current_device())
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        for k, n in ((2048, 7168), (4096, 3584), (4096, 7168)):
            for dtype in (torch.bfloat16, torch.float16):
                for m in (1, 9):
                    with self.subTest(m=m, k=k, n=n, dtype=dtype):
                        x, w = self._inputs(m, n, k, dtype)
                        chosen = _pick_sm90_tactic(m, k, n, sms, x.dtype)
                        for tactic in SM90_DUAL_GEMM_TACTICS:
                            _COMPILED.pop((device, x.dtype, m, n, k, tactic), None)
                        self._check(x, w)
                        key = (device, x.dtype, m, n, k, chosen)
                        specialization = _COMPILED[key]
                        explicit = sm90_dual_gemm_swiglu(x, w, chosen)
                        self._check(x, w, explicit)
                        self.assertIs(_COMPILED[key], specialization)
                        self.assertNotIn((device, x.dtype, m, n, k, -1), _COMPILED)

    def test_torch_compile_uses_custom_op_fake_contract(self):
        x, w = self._inputs(8, 3584, 4096, torch.bfloat16)
        # Warm the device specialization before tracing/capture.
        self._check(x, w)
        # Force lazy backend resolution during tracing, even if previous
        # numerical tests populated the selector cache.
        clear_cache()
        compiled = torch.compile(sm90_dual_gemm_swiglu, fullgraph=True)
        self._check(x, w, compiled(x, w))


if __name__ == "__main__":
    unittest.main()
