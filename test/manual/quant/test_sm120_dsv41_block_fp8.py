"""Manual numerical coverage of the device-named DeepSeek-V4.1 decode configs.

Run on an otherwise idle RTX PRO 6000 Blackwell Server Edition:
    python test/manual/quant/test_sm120_dsv41_block_fp8.py -v
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


class TestSM120DSV41BlockFP8(CustomTestCase):
    DEVICE_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    SHAPES = (
        (1792, 5120),
        (4096, 1280),
        (5120, 1024),
        (576, 5120),
        (5120, 288),
        (25600, 6144),
    )
    MS = (1, 3, 4, 5, 12, 16, 24, 32, 33, 64)
    DTYPES = (torch.bfloat16, torch.float16, torch.float32)
    LAYOUTS = ("contiguous", "transposed", "strided")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_available() or not torch.version.cuda:
            raise unittest.SkipTest("Requires CUDA")
        cls.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_name(
            cls.device
        ) != cls.DEVICE_NAME or torch.cuda.get_device_capability(cls.device) != (12, 0):
            raise unittest.SkipTest(f"Requires {cls.DEVICE_NAME} with SM120")

    def setUp(self):
        super().setUp()
        # Import only after the device gate. No dispatcher/config monkeypatches:
        # the wrapper must find the checked-in device-exact JSON files itself.
        from sglang.kernels.ops.quantization.fp8_kernel import (
            w8a8_block_fp8_matmul_triton,
        )

        self.matmul = w8a8_block_fp8_matmul_triton
        self.generator = torch.Generator(device=self.device).manual_seed(20260916)
        precision = torch.get_float32_matmul_precision()
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32

        def restore_precision():
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.set_float32_matmul_precision(precision)

        self.addCleanup(restore_precision)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False

    def _fp8(self, rows, columns):
        # Chunk initialization rather than keeping a full FP32 random weight.
        # The largest weight is 150 MiB FP8 plus 600 MiB for the FP32 oracle.
        result = torch.empty(
            (rows, columns), device=self.device, dtype=torch.float8_e4m3fn
        )
        for start in range(0, rows, 1024):
            chunk = torch.randn(
                (min(1024, rows - start), columns),
                device=self.device,
                dtype=torch.float32,
                generator=self.generator,
            )
            result[start : start + 1024].copy_(chunk)
        return result

    def _scales(self, rows, columns):
        # Exact powers of two isolate scale addressing from scale-rounding noise.
        exponents = torch.randint(
            -3,
            2,
            (rows, columns),
            device=self.device,
            generator=self.generator,
        )
        return torch.pow(2.0, exponents.float())

    def _layout(self, scales, layout):
        if layout == "contiguous":
            return scales
        if layout == "transposed":
            return scales.t().contiguous().t()
        # Both axes have non-unit strides and a nonzero storage offset.
        rows, columns = scales.shape
        storage = torch.full(
            (2 * rows, 2 * columns + 1),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )
        view = storage[::2, 1::2]
        view.copy_(scales)
        return view

    @staticmethod
    def _dequant_weight(B, Bs):
        N, K = B.shape
        weight = B.float()
        weight.view(N // 32, 32, K // 32, 32).mul_(Bs[:, None, :, None])
        return weight

    @staticmethod
    def _reference(A, As, weight):
        M, K = A.shape
        activation = A.float()
        activation.view(M, K // 32, 32).mul_(As[:, :, None])
        return activation @ weight.t()

    def _assert_result(self, actual, reference, dtype, *, generic_fallback=False):
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(actual.shape, reference.shape)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue(torch.isfinite(reference).all().item())
        rms = reference.square().mean().sqrt().item()
        self.assertGreater(rms, 0.0)
        error = actual.float() - reference
        # FP16/BF16 allow one final rounding (typical relative RMS ~2e-4/2e-3).
        # FP32 is much tighter: rounding split partials through either 16-bit
        # format must fail, as must a missing split or a misaddressed scale.
        # The untouched generic fallback historically rounds FP32 outputs via
        # FP16. Do not demand the tuned path's precision outside its domain.
        tolerance_dtype = (
            torch.float16 if generic_fallback and dtype == torch.float32 else dtype
        )
        relative_rms_limit = {
            torch.bfloat16: 3e-3,
            torch.float16: 4e-4,
            torch.float32: 2e-5,
        }[tolerance_dtype]
        # Also catch localized corruption without dividing by near-zero outputs.
        peak_limit = {
            torch.bfloat16: 4e-2,
            torch.float16: 5e-3,
            torch.float32: 2e-4,
        }[tolerance_dtype]
        self.assertLess(error.square().mean().sqrt().item() / rms, relative_rms_limit)
        self.assertLess(error.abs().max().item() / rms, peak_limit)

    def _check_shape(self, N, K):
        B = self._fp8(N, K)
        A = self._fp8(max(self.MS), K)
        As = self._scales(max(self.MS), K // 32)
        Bs = self._scales(N // 32, K // 32)
        weight = self._dequant_weight(B, Bs)
        reference = self._reference(A, As, weight)
        del weight
        # One weight and one FP32 oracle per shape. Rotate dtype/layout pairing
        # to cover every dtype and layout at each M without a Cartesian product.
        # The output projection switches from its M=4 tuning to the standard
        # M=16 entry at the nearest-M boundary between M=10 and M=11.
        ms = self.MS + ((10, 11) if (N, K) == (25600, 6144) else ())
        for layout_index, layout in enumerate(self.LAYOUTS):
            a_scales = self._layout(As, layout)
            b_scales = self._layout(Bs, layout)
            for m_index, M in enumerate(ms):
                dtype = self.DTYPES[(layout_index + m_index) % len(self.DTYPES)]
                with self.subTest(N=N, K=K, M=M, dtype=dtype, layout=layout):
                    actual = self.matmul(
                        A[:M], B, a_scales[:M], b_scales, [32, 32], dtype
                    )
                    fallback = M > 32 or ((N, K) == (25600, 6144) and M >= 11)
                    self._assert_result(
                        actual, reference[:M], dtype, generic_fallback=fallback
                    )

    def test_decode_configs_and_fallback_boundaries(self):
        # M=3/5/12/24 exercise nearest-M lookup, including ties; M=33/64
        # exercise the generic guard. The 25600x6144 shape falls back at M=11.
        for N, K in self.SHAPES:
            with self.subTest(N=N, K=K):
                self._check_shape(N, K)

    def _check_graph(self, B, M, dtype):
        N, K = B.shape
        A = self._fp8(M, K)
        As = self._layout(self._scales(M, K // 32), "strided")
        Bs = self._layout(self._scales(N // 32, K // 32), "transposed")
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.matmul(A, B, As, Bs, [32, 32], dtype)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = self.matmul(A, B, As, Bs, [32, 32], dtype)

        for replay in range(3):
            with self.subTest(replay=replay):
                if replay == 1:
                    # Change values, not addresses: captured partial buffers must
                    # be overwritten and both scale strides must remain valid.
                    A.copy_(self._fp8(M, K))
                    As.copy_(self._scales(M, K // 32))
                    Bs.copy_(self._scales(N // 32, K // 32))
                elif replay == 2:
                    As.zero_()
                graph.replay()
                if replay == 2:
                    # A nonzero-to-zero transition exposes stale/accumulated
                    # partials that relative-error checks on random data miss.
                    self.assertTrue(torch.isfinite(actual).all().item())
                    self.assertEqual(actual.abs().max().item(), 0.0)
                else:
                    weight = self._dequant_weight(B, Bs)
                    reference = self._reference(A, As, weight)
                    self._assert_result(actual, reference, dtype)

    def test_cuda_graph_replay_mutated_inputs_and_scales(self):
        # 9 K tiles over 8 splits include empty splits; the skinny-N case uses
        # non-swapped split-K. Both execute the production config-selected path.
        for N, K, M in ((5120, 288, 1), (576, 5120, 3)):
            B = self._fp8(N, K)
            for dtype in self.DTYPES:
                with self.subTest(N=N, K=K, M=M, dtype=dtype):
                    self._check_graph(B, M, dtype)


if __name__ == "__main__":
    unittest.main()
