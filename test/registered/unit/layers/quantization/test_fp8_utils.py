import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization import fp8_utils, modelopt_quant
from sglang.srt.layers.quantization.fp8 import (
    Fp8MoEMethod,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
)
from sglang.srt.layers.quantization.fp8_utils import (
    Fp8GemmRunnerBackend,
    Mxfp8DenseGemmBackend,
    block_fp8_scale_to_mxfp8_e8m0,
    can_serve_block_fp8_as_mxfp8,
    inverse_transform_scale_ue8m0,
    quant_weight_ue8m0,
    resolve_block_fp8_mxfp8_backend,
    transform_scale_ue8m0,
)
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=14, stage="base-b", runner_config="1-gpu-large")


class TestFlashinferBmmPrequantContract(unittest.TestCase):
    def test_e4m3_rows_bypass_quantization_and_restore_producer_dtype(self):
        for output_dtype in (torch.bfloat16, torch.float16):
            with self.subTest(output_dtype=output_dtype):
                self._assert_prequantized_dtype(output_dtype)

    def _assert_prequantized_dtype(self, output_dtype):
        qinput = torch.empty((2, 4), dtype=torch.float8_e4m3fn, device="cpu")
        weight = torch.empty((4, 3), dtype=torch.float8_e4m3fn, device="cpu")
        input_scale = torch.tensor([0.02], device="cpu")
        weight_scale = torch.tensor([0.01], device="cpu")
        expected = torch.empty((2, 3), dtype=output_dtype, device="cpu")

        with (
            patch.object(fp8_utils, "static_quant_fp8") as quantize,
            patch.object(
                fp8_utils,
                "flashinfer_bmm_fp8",
                return_value=expected,
                create=True,
            ) as bmm,
        ):
            actual = fp8_utils.apply_fp8_linear_bmm_flashinfer(
                qinput,
                weight,
                weight_scale,
                input_scale,
                pre_quant_output_dtype=output_dtype,
            )

        quantize.assert_not_called()
        bmm.assert_called_once()
        got_input, got_weight, got_input_scale, got_weight_scale, got_dtype = (
            bmm.call_args.args
        )
        self.assertEqual(got_input.data_ptr(), qinput.data_ptr())
        self.assertEqual(tuple(got_input.shape), tuple(qinput.shape))
        self.assertIs(got_weight, weight)
        self.assertIs(got_input_scale, input_scale)
        self.assertIs(got_weight_scale, weight_scale)
        self.assertIs(got_dtype, output_dtype)
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertEqual(tuple(actual.shape), (2, 3))

    def test_e4m3_rows_require_an_explicit_output_dtype(self):
        qinput = torch.empty((2, 4), dtype=torch.float8_e4m3fn, device="cpu")
        weight = torch.empty((4, 3), dtype=torch.float8_e4m3fn, device="cpu")
        with self.assertRaisesRegex(ValueError, "explicit BF16 or FP16"):
            fp8_utils.apply_fp8_linear_bmm_flashinfer(
                qinput,
                weight,
                torch.tensor([0.01]),
                torch.tensor([0.02]),
            )


class TestModelOptPrequantOutputDtype(unittest.TestCase):
    def test_model_dtype_reaches_both_static_fp8_backends(self):
        for use_flashinfer_bmm in (False, True):
            for output_dtype in (torch.bfloat16, torch.float16):
                with self.subTest(
                    use_flashinfer_bmm=use_flashinfer_bmm,
                    output_dtype=output_dtype,
                ):
                    self._assert_model_dtype_forwarded(
                        use_flashinfer_bmm=use_flashinfer_bmm,
                        output_dtype=output_dtype,
                    )

    def _assert_model_dtype_forwarded(self, *, use_flashinfer_bmm, output_dtype):
        method = modelopt_quant.ModelOptFp8LinearMethod.__new__(
            modelopt_quant.ModelOptFp8LinearMethod
        )
        method.use_marlin = False
        method.use_sm120_fp8 = False
        method.cutlass_fp8_supported = False
        layer = SimpleNamespace(
            use_flashinfer_bmm=use_flashinfer_bmm,
            orig_dtype=output_dtype,
            weight=object(),
            weight_scale=object(),
            input_scale=object(),
        )
        x = torch.empty((2, 4), dtype=torch.float8_e4m3fn, device="cpu")
        expected = object()
        function_name = (
            "apply_fp8_linear_bmm_flashinfer"
            if use_flashinfer_bmm
            else "apply_fp8_linear"
        )
        with patch.object(
            modelopt_quant,
            function_name,
            return_value=expected,
        ) as apply:
            actual = method.apply(layer, x)

        self.assertIs(actual, expected)
        self.assertIs(apply.call_args.kwargs["pre_quant_output_dtype"], output_dtype)


class TestMxfp8MoeScaleLayout(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not (
            (_is_cuda and get_platform().is_sm100) or (_is_hip and _is_gfx95_supported)
        ):
            raise unittest.SkipTest(
                "MXFP8 MoE quantization requires SM100 or ROCm gfx95"
            )

    def test_cutlass_serialized_scales_remain_expert_first(self):
        class CutlassBackend:
            def is_cutlass(self):
                return True

            def is_flashinfer_trtllm(self):
                return False

            def is_flashinfer_trtllm_routed(self):
                return False

            def is_deep_gemm(self):
                return False

        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((2, 64, 32), dtype=torch.float8_e4m3fn, device="cuda")
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((2, 32, 32), dtype=torch.float8_e4m3fn, device="cuda")
            ),
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.zeros((2, 64, 1), dtype=torch.uint8, device="cuda"),
                requires_grad=False,
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.zeros((2, 32, 1), dtype=torch.uint8, device="cuda"),
                requires_grad=False,
            ),
        )
        method = object.__new__(Fp8MoEMethod)

        with patch(
            "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
            return_value=CutlassBackend(),
        ):
            method._process_mxfp8_moe_weights(layer, quantize=False)

        self.assertEqual(tuple(layer.w13_weight_scale_inv.shape), (2, 64, 1))
        self.assertEqual(tuple(layer.w2_weight_scale_inv.shape), (2, 32, 1))


class TestInverseTransformScaleUe8m0(CustomTestCase):
    def test_round_trip(self):
        for _ in range(100):
            weight_bf16 = torch.randn(
                # DeepSeek V3 kv_b_proj
                (32768, 512),
                dtype=torch.bfloat16,
                device="cuda",
            )

            weight_block_size = [128, 128]

            qweight, sf_fp32_original = quant_weight_ue8m0(
                weight_bf16, weight_block_size=weight_block_size
            )
            mn = qweight.shape[-2]

            sf_packed_original = transform_scale_ue8m0(sf_fp32_original, mn=mn)
            sf_fp32_recreated = inverse_transform_scale_ue8m0(sf_packed_original, mn=mn)

            sf_packed_recreated = transform_scale_ue8m0(sf_fp32_recreated, mn=mn)

            assert torch.all(sf_packed_original == sf_packed_recreated), (
                f"{sf_packed_original=} {sf_packed_recreated}"
            )
            assert torch.all(sf_fp32_original == sf_fp32_recreated), (
                f"{sf_fp32_original=} {sf_fp32_recreated}"
            )


class TestBlockFp8AsMxfp8(CustomTestCase):
    def test_block_scale_to_e8m0_matches_reference(self):
        # Sibling classes leave torch's default device on cuda; stay on cpu.
        gen = torch.Generator().manual_seed(0)
        n, k, block_n = 100, 256, 32  # 4 scale rows, the last one partial
        exps = torch.randint(-20, 21, (4, k // 32), generator=gen, device="cpu")
        got = block_fp8_scale_to_mxfp8_e8m0(
            torch.exp2(exps.float()), (n, k), [block_n, 32]
        )
        ref = (exps + 127).to(torch.uint8).repeat_interleave(block_n, dim=0)[:n]
        self.assertTrue(torch.equal(got, ref))
        with self.assertRaises(ValueError):  # 128-wide K block is not MXFP8
            block_fp8_scale_to_mxfp8_e8m0(
                torch.ones(2, 8, device="cpu"), (64, 1024), [32, 128]
            )
        with self.assertRaises(ValueError):  # not a power of two
            block_fp8_scale_to_mxfp8_e8m0(
                torch.full((2, 8), 1.5, device="cpu"), (64, 256), [32, 32]
            )

    def test_serve_gate(self):
        platform = MagicMock()
        platform.is_blackwell = True
        cutedsl = next(b for b in Mxfp8DenseGemmBackend if b.is_flashinfer_cutedsl())
        with (
            patch.object(fp8_utils, "_is_cuda", True),
            patch.object(fp8_utils, "get_platform", return_value=platform),
            patch.object(fp8_utils, "is_flashinfer_available", return_value=True),
            patch.object(
                fp8_utils, "resolve_mxfp8_dense_gemm_backend", return_value=cutedsl
            ),
        ):
            for name, expected in (
                ("flashinfer_cutedsl", True),
                ("flashinfer_cutlass", True),
                ("flashinfer_trtllm", False),
                ("triton", False),
                ("auto", False),
            ):
                with (
                    self.subTest(backend=name),
                    patch.object(
                        fp8_utils, "FP8_GEMM_RUNNER_BACKEND", Fp8GemmRunnerBackend(name)
                    ),
                ):
                    self.assertEqual(
                        can_serve_block_fp8_as_mxfp8([32, 32], "ue8m0"), expected
                    )
                    self.assertEqual(
                        resolve_block_fp8_mxfp8_backend().is_unsupported(), not expected
                    )
            with patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL,
            ):
                self.assertFalse(can_serve_block_fp8_as_mxfp8([128, 128], "ue8m0"))
                self.assertFalse(can_serve_block_fp8_as_mxfp8([32, 32], None))
                platform.is_blackwell = False
                self.assertFalse(can_serve_block_fp8_as_mxfp8([32, 32], "ue8m0"))


class TestApplyFp8LinearScaleDispatch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    @staticmethod
    def _make_inputs(dtype=torch.bfloat16):
        M, K, N = 8, 16, 32
        input = torch.randn(M, K, dtype=dtype)
        qinput = input.to(torch.float8_e4m3fn)
        weight = torch.randn(N, K).to(torch.float8_e4m3fn).t()
        input_scale = torch.tensor([0.05], dtype=torch.float32)
        weight_scale = torch.linspace(0.01, 0.03, N, dtype=torch.float32)
        return input, qinput, weight, input_scale, weight_scale

    def test_native_scalar_a_static_prequant_and_dynamic_scale_shapes(self):
        import sglang.srt.layers.quantization.fp8_utils as fp8_utils

        exec_config = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(tc_compiler="none")
                )
            )
        )
        for capability in (
            "is_sm90",
            "is_sm100",
            "is_sm120",
        ):
            with self.subTest(capability=capability):
                input, qinput, weight, input_scale, weight_scale = self._make_inputs()
                seen_scales = []

                def fake_fp8_scaled_mm(
                    mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None
                ):
                    seen_scales.append(scales_a)
                    return torch.empty(
                        (mat_a.shape[0], mat_b.shape[1]),
                        dtype=out_dtype,
                        device=mat_a.device,
                    )

                capabilities = {
                    "is_sm90": False,
                    "is_sm100": False,
                    "is_sm120": False,
                }
                capabilities[capability] = True
                with (
                    patch.object(
                        fp8_utils,
                        "get_platform",
                        return_value=SimpleNamespace(**capabilities),
                    ),
                    patch.object(
                        fp8_utils, "fp8_scaled_mm", side_effect=fake_fp8_scaled_mm
                    ),
                    patch.object(fp8_utils, "get_exec", return_value=exec_config),
                ):
                    fp8_utils.apply_fp8_linear(
                        input,
                        weight,
                        weight_scale,
                        input_scale=input_scale,
                        cutlass_fp8_supported=True,
                    )
                    fp8_utils.apply_fp8_linear(
                        input,
                        weight,
                        weight_scale,
                        input_scale=input_scale,
                        cutlass_fp8_supported=True,
                        use_per_token_if_dynamic=True,
                        compressed_tensor_quant=True,
                    )
                    fp8_utils.apply_fp8_linear(
                        qinput,
                        weight,
                        weight_scale,
                        input_scale=input_scale,
                        cutlass_fp8_supported=True,
                        pre_quant_output_dtype=input.dtype,
                    )
                    fp8_utils.apply_fp8_linear(
                        input,
                        weight,
                        weight_scale,
                        input_scale=None,
                        cutlass_fp8_supported=True,
                        use_per_token_if_dynamic=True,
                        compressed_tensor_quant=True,
                    )

                self.assertEqual(seen_scales[0].numel(), 1)
                self.assertEqual(seen_scales[1].numel(), 1)
                self.assertIs(seen_scales[2], input_scale)
                self.assertEqual(tuple(seen_scales[3].shape), (input.shape[0], 1))

    def test_without_native_scalar_a_static_scale_is_repeated(self):
        import sglang.srt.layers.quantization.fp8_utils as fp8_utils

        input, qinput, weight, input_scale, weight_scale = self._make_inputs()
        seen_scales = []

        def fake_fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
            seen_scales.append(scales_a)
            return torch.empty(
                (mat_a.shape[0], mat_b.shape[1]), dtype=out_dtype, device=mat_a.device
            )

        with (
            patch.object(
                fp8_utils,
                "get_platform",
                return_value=SimpleNamespace(
                    is_sm90=False,
                    is_sm100=False,
                    is_sm120=False,
                ),
            ),
            patch.object(fp8_utils, "fp8_scaled_mm", side_effect=fake_fp8_scaled_mm),
        ):
            fp8_utils.apply_fp8_linear(
                input,
                weight,
                weight_scale,
                input_scale=input_scale,
                cutlass_fp8_supported=True,
            )
            fp8_utils.apply_fp8_linear(
                qinput,
                weight,
                weight_scale,
                input_scale=input_scale,
                cutlass_fp8_supported=True,
                pre_quant_output_dtype=input.dtype,
            )

        self.assertEqual(tuple(seen_scales[0].shape), (input.shape[0], 1))
        self.assertEqual(tuple(seen_scales[1].shape), (input.shape[0], 1))

    def test_linear_methods_forward_fused_scalar_tuple(self):
        import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 as compressed_fp8
        import sglang.srt.layers.quantization.fp8 as native_fp8

        input, qinput, weight, input_scale, weight_scale = self._make_inputs(
            torch.float16
        )

        class Layer:
            pass

        layer = Layer()
        layer.weight = weight
        layer.weight_scale = weight_scale
        layer.input_scale = input_scale

        native_method = native_fp8.Fp8LinearMethod.__new__(native_fp8.Fp8LinearMethod)
        native_method.use_marlin = False
        native_method.use_mxfp8 = False
        native_method.block_fp8_as_mxfp8 = False
        native_method.block_quant = False
        native_method.cutlass_fp8_supported = True
        native_method.use_per_token_if_dynamic = False

        compressed_method = compressed_fp8.CompressedTensorsW8A8Fp8.__new__(
            compressed_fp8.CompressedTensorsW8A8Fp8
        )
        compressed_method.weight_block_size = None

        fused_input = (qinput, input_scale, input.dtype)
        with patch.object(native_fp8, "apply_fp8_linear") as native_apply:
            native_apply.return_value = torch.empty(
                (qinput.shape[0], weight.shape[1]), dtype=input.dtype
            )
            native_method.apply(layer, fused_input)
            self.assertIs(native_apply.call_args.kwargs["input_scale"], input_scale)
            self.assertEqual(
                native_apply.call_args.kwargs["pre_quant_output_dtype"], input.dtype
            )

        with patch.object(compressed_fp8, "apply_fp8_linear") as compressed_apply:
            compressed_apply.return_value = torch.empty(
                (qinput.shape[0], weight.shape[1]), dtype=input.dtype
            )
            compressed_method.apply_weights(layer, fused_input)
            self.assertIs(compressed_apply.call_args.kwargs["input_scale"], input_scale)
            self.assertEqual(
                compressed_apply.call_args.kwargs["pre_quant_output_dtype"],
                input.dtype,
            )


class TestApplyFp8LinearPrequantOutputDtype(CustomTestCase):
    """apply_fp8_linear with a pre-quantized fp8 activation must emit the
    caller-supplied ``pre_quant_output_dtype`` (the model's activation dtype),
    not the fp8 input dtype. Regression test for FP16 models where hardcoding
    bf16 caused a query/key dtype mismatch in attention."""

    DTYPES = [torch.float16, torch.bfloat16]
    FP8_DTYPE = torch.float8_e4m3fn

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run(self, dtype):
        from sglang.srt.layers.quantization.fp8_utils import (
            apply_fp8_linear,
            cutlass_fp8_supported,
        )

        torch.manual_seed(0)
        M, K, N = 33, 512, 256
        cf = cutlass_fp8_supported()
        fp8_info = torch.finfo(self.FP8_DTYPE)

        normed = torch.randn(M, K, dtype=dtype)
        input_scale = torch.tensor([0.05], dtype=torch.float32)
        # Per-channel fp8 weight in column-major (K, N) layout.
        w = torch.randn(N, K, dtype=dtype) * 0.05
        w_scale = (w.abs().amax(dim=1) / fp8_info.max).float()
        weight = (
            (w.float() / w_scale[:, None])
            .clamp(fp8_info.min, fp8_info.max)
            .to(self.FP8_DTYPE)
            .t()
        )

        # Reference: non-pre-quantized input -> output dtype == input dtype.
        ref = apply_fp8_linear(
            input=normed,
            weight=weight,
            weight_scale=w_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=cf,
        )
        self.assertEqual(ref.dtype, dtype)

        qinput = (
            (normed.float() * input_scale.reciprocal())
            .clamp(fp8_info.min, fp8_info.max)
            .to(self.FP8_DTYPE)
        )

        # Pre-quantized input with the dtype propagated -> output matches dtype.
        out = apply_fp8_linear(
            input=qinput,
            weight=weight,
            weight_scale=w_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=cf,
            pre_quant_output_dtype=dtype,
        )
        self.assertEqual(out.dtype, dtype)
        self.assertTrue(torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2))

        # Without the dtype hint, the pre-quantized path falls back to bf16.
        out_default = apply_fp8_linear(
            input=qinput,
            weight=weight,
            weight_scale=w_scale,
            input_scale=input_scale,
            cutlass_fp8_supported=cf,
        )
        self.assertEqual(out_default.dtype, torch.bfloat16)

    def test_prequant_output_dtype(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                self._run(dtype)


if __name__ == "__main__":
    unittest.main()
