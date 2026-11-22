"""Which [32, 32] block-FP8 linears run DeepGEMM, and which scales get prepacked.

Split out of test_fp8_utils.py: this matrix needs no accelerator (dispatch only
picks a callable, and the packer's gate is tested against a stubbed deep_gemm),
so it runs on the CPU stage instead of riding a 1-gpu-large registration.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
                        return_value=SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0),
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
