"""Block-FP8 linear dispatch by weight block size and the MXFP8 view helpers (CPU)."""

import functools
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization import fp8, fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    Mxfp8DenseGemmBackend,
    block_fp8_scale_to_mxfp8_e8m0,
    can_serve_block_fp8_as_mxfp8,
    dispatch_w8a8_block_fp8_linear,
    resolve_block_fp8_mxfp8_backend,
    triton_w8a8_block_fp8_linear,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _backend(name: str):
    return patch.object(
        fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend(name)
    )


class TestBlockSizeDispatch(unittest.TestCase):
    def test_128_wide_k_blocks_keep_the_backend_choice(self):
        for name in ("triton", "deep_gemm"):
            with self.subTest(backend=name), _backend(name):
                default = dispatch_w8a8_block_fp8_linear()
                self.assertIs(dispatch_w8a8_block_fp8_linear([128, 128]), default)
                self.assertIs(dispatch_w8a8_block_fp8_linear([1, 128]), default)

    def test_other_block_widths_go_to_triton(self):
        for name in ("triton", "deep_gemm"):
            with self.subTest(backend=name), _backend(name):
                fn = dispatch_w8a8_block_fp8_linear([32, 32], act_scale_ue8m0=True)
                self.assertIsInstance(fn, functools.partial)
                self.assertIs(fn.func, triton_w8a8_block_fp8_linear)
                self.assertEqual(fn.keywords, {"act_scale_ue8m0": True})
                fn = dispatch_w8a8_block_fp8_linear([32, 32])
                self.assertEqual(fn.keywords, {"act_scale_ue8m0": False})

    def test_method_dispatches_on_the_effective_block_size(self):
        # An MXFP8 checkpoint converted to block-fp8 at load time is a [128, 128]
        # weight; dispatching on the pre-conversion [1, 32] would pick Triton.
        with (
            _backend("triton"),
            patch.object(fp8, "_mxfp8_to_block_fp8_required", True),
        ):
            config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                use_mxfp8=True,
                weight_block_size=[1, 32],
                scale_fmt="ue8m0",
            )
            method = Fp8LinearMethod(config)
            self.assertTrue(method.convert_mxfp8_to_block)
            self.assertIs(method.w8a8_block_fp8_linear, triton_w8a8_block_fp8_linear)
            self.assertFalse(method.block_fp8_as_mxfp8)

    def test_block32_method_uses_triton_with_ue8m0_activations(self):
        with _backend("triton"):
            method = Fp8LinearMethod(
                Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    weight_block_size=[32, 32],
                    scale_fmt="ue8m0",
                )
            )
            fn = method.w8a8_block_fp8_linear
            self.assertIs(fn.func, triton_w8a8_block_fp8_linear)
            self.assertEqual(fn.keywords, {"act_scale_ue8m0": True})
            self.assertFalse(method.block_fp8_as_mxfp8)
            self.assertIsNone(method.w8a8_mxfp8_linear)


class TestBlockFp8ScaleToE8m0(unittest.TestCase):
    def test_matches_reference_including_trailing_rows(self):
        gen = torch.Generator().manual_seed(0)
        n, k, block_n = 100, 256, 32  # 100 rows: 4 scale rows, last one partial
        exps = torch.randint(-20, 21, (4, k // 32), generator=gen)
        scale = torch.exp2(exps.float())
        got = block_fp8_scale_to_mxfp8_e8m0(scale, (n, k), [block_n, 32])
        ref = (exps + 127).to(torch.uint8).repeat_interleave(block_n, dim=0)[:n]
        self.assertEqual(got.dtype, torch.uint8)
        self.assertTrue(torch.equal(got, ref))

    def test_rejects_non_mxfp8_inputs(self):
        scale = torch.ones(2, 8)
        with self.assertRaises(ValueError):  # 128-wide K block
            block_fp8_scale_to_mxfp8_e8m0(scale, (64, 1024), [32, 128])
        with self.assertRaises(ValueError):  # K not a multiple of 32
            block_fp8_scale_to_mxfp8_e8m0(scale, (64, 250), [32, 32])
        with self.assertRaises(ValueError):  # scale shape does not match the weight
            block_fp8_scale_to_mxfp8_e8m0(scale, (64, 512), [32, 32])
        for bad in (1.5, 0.0, -2.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                block_fp8_scale_to_mxfp8_e8m0(
                    torch.full((2, 8), bad), (64, 256), [32, 32]
                )


class TestCanServeBlockFp8AsMxfp8(unittest.TestCase):
    def _blackwell(self, *, cuda=True, blackwell=True, flashinfer=True):
        platform = MagicMock()
        platform.is_blackwell = blackwell
        cutedsl = next(b for b in Mxfp8DenseGemmBackend if b.is_flashinfer_cutedsl())
        return (
            patch.object(fp8_utils, "_is_cuda", cuda),
            patch.object(fp8_utils, "get_platform", return_value=platform),
            patch.object(fp8_utils, "is_flashinfer_available", return_value=flashinfer),
            patch.object(
                fp8_utils, "resolve_mxfp8_dense_gemm_backend", return_value=cutedsl
            ),
        )

    def _with(self, patches):
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_shape_and_scale_format_gate(self):
        self._with(self._blackwell())
        with _backend("flashinfer_cutedsl"):
            self.assertFalse(can_serve_block_fp8_as_mxfp8(None, "ue8m0"))
            self.assertFalse(can_serve_block_fp8_as_mxfp8([128, 128], "ue8m0"))
            self.assertFalse(can_serve_block_fp8_as_mxfp8([32, 32], None))
            self.assertTrue(can_serve_block_fp8_as_mxfp8([32, 32], "ue8m0"))
            self.assertTrue(resolve_block_fp8_mxfp8_backend().is_flashinfer_cutedsl())

    def test_backend_gate(self):
        self._with(self._blackwell())
        for name in ("auto", "triton", "deep_gemm", "flashinfer_trtllm"):
            with self.subTest(backend=name), _backend(name):
                self.assertFalse(can_serve_block_fp8_as_mxfp8([32, 32], "ue8m0"))
                self.assertTrue(resolve_block_fp8_mxfp8_backend().is_unsupported())
        for name in ("flashinfer_cutlass", "flashinfer_cutedsl"):
            with self.subTest(backend=name), _backend(name):
                self.assertTrue(can_serve_block_fp8_as_mxfp8([32, 32], "ue8m0"))

    def test_platform_gate(self):
        for kwargs in (
            dict(blackwell=False),
            dict(cuda=False),
            dict(flashinfer=False),
        ):
            with self.subTest(**kwargs):
                patches = self._blackwell(**kwargs)
                for p in patches:
                    p.start()
                try:
                    with _backend("flashinfer_cutedsl"):
                        self.assertFalse(
                            can_serve_block_fp8_as_mxfp8([32, 32], "ue8m0")
                        )
                finally:
                    for p in patches:
                        p.stop()


if __name__ == "__main__":
    unittest.main()
