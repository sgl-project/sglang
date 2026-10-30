import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import (
    Fp8LinearMethod,
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

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


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


class TestBlock32Ue8m0DeepGemmSelection(CustomTestCase):
    """Which [32, 32] block-FP8 linears run DeepGEMM, and which scales get prepacked.

    The platform matrix is checked without a GPU: dispatch only picks a callable,
    and the packer's own gate is tested against a stubbed deep_gemm.
    """

    @staticmethod
    def _platform(sm90=False, sm100=False, sm120=False):
        platform = MagicMock()
        platform.is_sm90 = sm90
        platform.is_sm100 = sm100
        platform.is_sm120 = sm120
        return platform

    @staticmethod
    def _serves_deepgemm(linear) -> bool:
        target = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        return getattr(linear, "func", linear) is target

    @staticmethod
    def _cpu_block32_layer(scale, runner):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.zeros(96, 288, dtype=torch.float8_e4m3fn, device="cpu"),
            requires_grad=False,
        )
        layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        layer.weight_scale_inv.format_ue8m0 = False
        layer.orig_dtype = torch.bfloat16
        layer.deepgemm_mxfp8_input = False
        method = object.__new__(Fp8LinearMethod)
        method.block_quant = True
        method.weight_block_size = [32, 32]
        method.quant_config = SimpleNamespace(weight_block_size=[32, 32])
        method.convert_mxfp8_to_block = False
        method.use_mxfp8 = False
        method.use_marlin = False
        method.use_npu_block_fp8 = False
        method.block_fp8_as_mxfp8 = False
        method.w8a8_block_fp8_linear = runner
        layer.quant_method = method
        return layer

    def test_unpack_mn_major_degenerate_scale_layouts(self):
        cases = (
            ((8, 1), (1, 8), (8, 32), [32, 32], (1, 1)),
            ((1, 3), (1, 4), (128, 1152), [128, 128], (1, 9)),
        )
        for shape, stride, weight_shape, block_size, expected_shape in cases:
            with self.subTest(shape=shape, stride=stride):
                packed = torch.empty_strided(
                    shape, stride, dtype=torch.int32, device="cpu"
                )
                packed.fill_(0x7F7F7F7F)
                unpacked = fp8_utils._unpack_ue8m0_scale_for_triton(
                    packed, weight_shape, block_size
                )
                torch.testing.assert_close(
                    unpacked, torch.ones(expected_shape, device="cpu"), rtol=0, atol=0
                )

    def test_subnormal_scale_refuses_packing_and_native_execution(self):
        from sglang.srt.layers.quantization import fp8 as fp8_module

        runner = fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback
        platform = self._platform(sm120=True)
        for exponent, eligible in (
            (-149, False),
            (-127, False),
            (-126, True),
            (-125, True),
        ):
            with self.subTest(exponent=exponent):
                scale = torch.full(
                    (3, 9), 2.0**exponent, dtype=torch.float32, device="cpu"
                )
                layer = self._cpu_block32_layer(scale, runner)
                packed = torch.zeros(96, 3, dtype=torch.int32, device="cpu")
                transform = MagicMock(return_value=packed)
                with (
                    patch.object(fp8_module, "_is_cuda", True),
                    patch.object(fp8_module, "_is_cpu", False),
                    patch.object(fp8_module, "_is_fp8_fnuz", False),
                    patch.object(fp8_module, "get_platform", return_value=platform),
                    patch.object(fp8_utils, "get_platform", return_value=platform),
                    patch.object(
                        fp8_utils.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", True
                    ),
                    patch.dict(
                        sys.modules,
                        {
                            "deep_gemm": SimpleNamespace(
                                transform_sf_into_required_layout=transform
                            )
                        },
                    ),
                ):
                    layer.quant_method.process_weights_after_loading(layer)
                    self.assertEqual(layer.weight_scale_inv.power_of_two, eligible)
                    self.assertEqual(
                        fp8_utils._ue8m0_deepgemm_skips_block32(
                            [32, 32], layer.weight_scale_inv, None
                        ),
                        not eligible,
                    )
                self.assertEqual(transform.call_count, int(eligible))
                self.assertEqual(
                    hasattr(layer, "_deepgemm_block32_weight_scale"), eligible
                )
                torch.testing.assert_close(
                    layer.weight_scale_inv, scale, rtol=0, atol=0
                )

    def test_sharded_load_processes_native_scales_after_checkpoint_copy(self):
        import tempfile

        from sglang.srt.layers.quantization import fp8 as fp8_module
        from sglang.srt.model_loader import loader as model_loader

        platform = self._platform(sm120=True)
        scale = torch.full((3, 9), 0.25, dtype=torch.float32, device="cpu")
        packed = torch.zeros(96, 3, dtype=torch.int32, device="cpu")
        transform = MagicMock(return_value=packed)
        load_config = SimpleNamespace(model_loader_extra_config=None)
        for runner, deferred in (
            (fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback, True),
            (fp8_utils.triton_w8a8_block_fp8_linear, False),
        ):
            with (
                self.subTest(deferred=deferred),
                tempfile.TemporaryDirectory() as directory,
            ):
                process_scales = []
                transform.reset_mock()

                def make_model(*args):
                    model = torch.nn.Module()
                    model.linear = self._cpu_block32_layer(
                        torch.full(
                            (3, 9),
                            torch.finfo(torch.float32).min,
                            dtype=torch.float32,
                            device="cpu",
                        ),
                        runner,
                    )
                    method = model.linear.quant_method

                    def process(layer):
                        process_scales.append(layer.weight_scale_inv.detach().clone())
                        Fp8LinearMethod.process_weights_after_loading(method, layer)

                    method.process_weights_after_loading = process
                    return model

                with (
                    patch.object(fp8_module, "_is_cuda", True),
                    patch.object(fp8_module, "_is_cpu", False),
                    patch.object(fp8_module, "_is_fp8_fnuz", False),
                    patch.object(fp8_module, "get_platform", return_value=platform),
                    patch.object(fp8_utils, "get_platform", return_value=platform),
                    patch.object(
                        fp8_utils.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", True
                    ),
                    patch.dict(
                        sys.modules,
                        {
                            "deep_gemm": SimpleNamespace(
                                transform_sf_into_required_layout=transform
                            )
                        },
                    ),
                    patch.object(
                        model_loader,
                        "get_parallel",
                        return_value=SimpleNamespace(tp_rank=0),
                    ),
                    patch.object(
                        model_loader, "_get_quantization_config", return_value=None
                    ),
                    patch.object(
                        model_loader, "_initialize_model", side_effect=make_model
                    ),
                ):
                    source = torch.nn.Module()
                    source.linear = self._cpu_block32_layer(scale.clone(), runner)
                    source.linear.quant_method.process_weights_after_loading(
                        source.linear
                    )
                    model_loader.ShardedStateLoader.save_model(source, directory)
                    restored = model_loader.ShardedStateLoader(load_config).load_model(
                        model_config=SimpleNamespace(
                            model_path=directory, revision=None, dtype=torch.bfloat16
                        ),
                        device_config=SimpleNamespace(device="cpu"),
                    )
                self.assertEqual(len(process_scales), 1)
                self.assertEqual(transform.call_count, 2 if deferred else 0)
                expected = (
                    scale
                    if deferred
                    else torch.full_like(scale, torch.finfo(torch.float32).min)
                )
                torch.testing.assert_close(process_scales[0], expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    restored.linear.weight_scale_inv, scale, rtol=0, atol=0
                )
                self.assertEqual(
                    hasattr(restored.linear, "_deepgemm_block32_weight_scale"), deferred
                )
                self.assertNotIn(
                    "linear._deepgemm_block32_weight_scale", restored.state_dict()
                )
                if deferred:
                    self.assertTrue(restored.linear.weight_scale_inv.power_of_two)
                    torch.testing.assert_close(
                        restored.linear._deepgemm_block32_weight_scale,
                        packed,
                        rtol=0,
                        atol=0,
                    )

    def test_block32_route_gates(self):
        """Dispatch and prepacking must both admit a layer for the native route."""
        platforms = {
            "sm90": self._platform(sm90=True),
            "sm100": self._platform(sm100=True),
            "sm120": self._platform(sm120=True),
        }
        dispatch_cases = [
            # (platform, backend, JIT enabled, act_scale_ue8m0, deepgemm expected)
            ("sm90", "auto", True, False, True),
            ("sm90", "deep_gemm", True, False, True),
            ("sm90", "triton", True, False, False),
            ("sm100", "deep_gemm", True, True, True),
            ("sm100", "deep_gemm", True, False, False),
            ("sm100", "auto", True, True, False),
            ("sm120", "deep_gemm", True, True, True),
            ("sm120", "deep_gemm", True, False, False),
            ("sm120", "deep_gemm", False, True, False),
            ("sm120", "auto", True, True, False),
            ("sm120", "triton", True, True, False),
        ]
        for name, backend, jit, act_scale_ue8m0, expected in dispatch_cases:
            with (
                self.subTest(
                    platform=name, backend=backend, jit=jit, ue8m0=act_scale_ue8m0
                ),
                patch.object(fp8_utils, "get_platform", return_value=platforms[name]),
                patch.object(fp8_utils.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", jit),
                patch.object(
                    fp8_utils,
                    "FP8_GEMM_RUNNER_BACKEND",
                    Fp8GemmRunnerBackend(backend),
                ),
            ):
                linear = fp8_utils.dispatch_w8a8_block_fp8_linear(
                    [32, 32], act_scale_ue8m0=act_scale_ue8m0
                )
                self.assertEqual(self._serves_deepgemm(linear), expected)

        packed = torch.zeros(96, 3, dtype=torch.int32)
        calls = []

        def transform(sf, **kwargs):
            calls.append(kwargs)
            return packed

        fake_deep_gemm = SimpleNamespace(transform_sf_into_required_layout=transform)

        def make_weight(n=96, k=320):
            return torch.nn.Parameter(
                torch.zeros(n, k, dtype=torch.float8_e4m3fn), requires_grad=False
            )

        def make_scale(**flags):
            parameter = torch.nn.Parameter(
                torch.full((3, 10), 0.25, dtype=torch.float32), requires_grad=False
            )
            for key, value in flags.items():
                setattr(parameter, key, value)
            return parameter

        def packable(**overrides):
            args = dict(
                weight=make_weight(),
                weight_scale=make_scale(power_of_two=True),
                weight_block_size=[32, 32],
                use_deepgemm_runner=True,
            )
            args.update(overrides)
            return args

        prepack_cases = [
            # (case, platform, kwargs, packed expected)
            ("SM120 power-of-two scales", "sm120", packable(), True),
            (
                "not the DeepGEMM runner",
                "sm120",
                packable(use_deepgemm_runner=False),
                False,
            ),
            (
                "not [32, 32] blocks",
                "sm120",
                packable(weight_block_size=[128, 128]),
                False,
            ),
            ("SM100", "sm100", packable(), False),
            ("SM90", "sm90", packable(), False),
            (
                "scales not proven powers of two",
                "sm120",
                packable(weight_scale=make_scale()),
                False,
            ),
            (
                "already packed",
                "sm120",
                packable(weight_scale=make_scale(power_of_two=True, format_ue8m0=True)),
                False,
            ),
            (
                "output is not bf16",
                "sm120",
                packable(output_dtype=torch.float16),
                False,
            ),
            (
                "N not a multiple of 8",
                "sm120",
                packable(weight=make_weight(n=10)),
                False,
            ),
            (
                "K not a multiple of 32",
                "sm120",
                packable(weight=make_weight(k=33)),
                False,
            ),
        ]
        for case, name, args, expected in prepack_cases:
            with (
                self.subTest(prepack=case),
                patch.object(fp8_utils, "get_platform", return_value=platforms[name]),
                patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}),
            ):
                weight_scale = args["weight_scale"]
                before = weight_scale.detach().clone()
                before_ptr = weight_scale.data_ptr()
                before_format = getattr(weight_scale, "format_ue8m0", False)
                result = fp8_utils.pack_block32_scale_ue8m0_for_deepgemm(**args)
                self.assertIs(result, packed if expected else None)
                self.assertEqual(weight_scale.data_ptr(), before_ptr)
                torch.testing.assert_close(weight_scale, before, rtol=0, atol=0)
                self.assertEqual(
                    getattr(weight_scale, "format_ue8m0", False), before_format
                )
        self.assertEqual(
            calls, [{"mn": 96, "k": 320, "recipe": (1, 32, 32), "is_sfa": False}]
        )

    def test_skips_block32_without_proven_power_of_two_scales(self):
        def scale(**flags):
            parameter = torch.nn.Parameter(torch.ones(2, 2))
            for key, value in flags.items():
                setattr(parameter, key, value)
            return parameter

        packed_input = torch.zeros(2, 1, dtype=torch.int32)
        packed_weight = torch.zeros(64, 1, dtype=torch.int32)
        cases = [
            ([32, 32], scale(), None, True),
            ([32, 32], scale(power_of_two=True), None, False),
            ([32, 32], packed_weight, None, False),
            ([32, 32], scale(power_of_two=True), packed_input, True),
            ([128, 128], scale(power_of_two=True), packed_input, False),
        ]
        for scale_ue8m0 in (True, False):
            with patch.object(
                fp8_utils.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", scale_ue8m0
            ):
                for block_size, weight_scale, input_scale, expected in cases:
                    with self.subTest(
                        scale_ue8m0=scale_ue8m0,
                        block_size=block_size,
                        input_scale=input_scale is not None,
                    ):
                        self.assertEqual(
                            fp8_utils._ue8m0_deepgemm_skips_block32(
                                block_size, weight_scale, input_scale
                            ),
                            scale_ue8m0 and expected,
                        )


if __name__ == "__main__":
    unittest.main()
