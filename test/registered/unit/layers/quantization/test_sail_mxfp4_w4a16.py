"""W4A16 must reject incompatible backends before the weights are repacked, and
must preserve the E8M0 format marker."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.arg_groups.platform_hook import handle_sail_backends
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.sail.quantization import mxfp4 as sail_mxfp4
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import mxfp4
from sglang.srt.runtime_context import get_flags, override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSailMxfp4W4a16(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(False))
        stack.enter_context(envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(False))

    @staticmethod
    def _method(mma=False):
        method = object.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_mega_moe = False
        method.use_deep_gemm = False
        method.use_deepgemm_mxfp4_w4a16_valu = not mma
        method.use_deepgemm_mxfp4_w4a16_mma = mma
        method.moe_runner_config = SimpleNamespace(gemm1_alpha=None)
        return method

    def test_quant_method_rejects_incompatible_runtime(self):
        cases = (
            (MoeRunnerBackend.AUTO, MoeA2ABackend.NONE, True, ValueError),
            (MoeRunnerBackend.MARLIN, MoeA2ABackend.NONE, True, ValueError),
            (MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.MOONCAKE, True, ValueError),
            (MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.NONE, False, RuntimeError),
        )
        for runner, a2a, jit, error in cases:
            with (
                self.subTest(runner=runner, a2a=a2a, jit=jit),
                patch.object(mxfp4, "_is_sail", True),
                patch.object(sail_mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", jit),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(True),
                get_flags().moe.override(runner_backend=runner, a2a_backend=a2a),
                self.assertRaises(error),
            ):
                mxfp4.Mxfp4MoEMethod("experts")

    def test_platform_validation_reads_resolved_backend(self):
        seed = SimpleNamespace(moe_runner_backend="auto", moe_a2a_backend="deepep")
        declare_resolution(
            seed, "test", moe_runner_backend="deep_gemm", moe_a2a_backend="none"
        )
        with (
            override_platform(is_sail=True),
            envs.SGLANG_SAIL_DEEPGEMM_DENSE.override(False),
            envs.SGLANG_SAIL_DEEPGEMM_MOE.override(False),
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(True),
            patch(
                "sglang.srt.arg_groups.platform_hook.use_mla_backend",
                return_value=False,
            ),
        ):
            handle_sail_backends(seed)
            declare_resolution(seed, "test", moe_a2a_backend="deepep")
            handle_sail_backends(seed)
            declare_resolution(seed, "test", moe_a2a_backend="mooncake")
            with self.assertRaisesRegex(ValueError, "none or deepep"):
                handle_sail_backends(seed)
            self.assertEqual(seed.moe_runner_backend, "auto")
            self.assertEqual(seed.moe_a2a_backend, "deepep")

    def test_platform_hook_leaves_the_hardware_half_to_the_quant_method(self):
        """The argument hook only sees the arguments: resolving a W4A16 MMA start
        must not look at the device or the DeepGEMM build. Both probes are armed
        to fail if they run, which is also what lets this pass on CPU."""
        seed = SimpleNamespace(moe_runner_backend="auto", moe_a2a_backend="none")
        declare_resolution(
            seed, "test", moe_runner_backend="deep_gemm", moe_a2a_backend="none"
        )
        with (
            override_platform(is_sail=True),
            envs.SGLANG_SAIL_DEEPGEMM_DENSE.override(False),
            envs.SGLANG_SAIL_DEEPGEMM_MOE.override(False),
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(True),
            patch(
                "sglang.srt.arg_groups.platform_hook.use_mla_backend",
                return_value=False,
            ),
            patch.object(
                sail_mxfp4,
                "get_device_sm",
                side_effect=AssertionError("the hook must not probe the device"),
            ),
            # Without an available DeepGEMM the availability check would raise.
            patch.object(sail_mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False),
        ):
            handle_sail_backends(seed)

    def test_opt_in_does_not_change_other_platforms_or_native_default(self):
        for sail, valu, mma in (
            (False, True, False),
            (False, False, True),
            (False, True, True),
            (True, False, False),
            (True, True, False),
            (True, False, True),
        ):
            with (
                self.subTest(sail=sail, valu=valu, mma=mma),
                patch.object(mxfp4, "_is_sail", sail),
                patch.object(sail_mxfp4, "get_device_sm", return_value=89),
                patch.object(sail_mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
                get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.DEEP_GEMM,
                    a2a_backend=MoeA2ABackend.NONE,
                ),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(valu),
                envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(mma),
                patch.object(
                    mxfp4,
                    "get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
                    ),
                ),
            ):
                method = mxfp4.Mxfp4MoEMethod("experts")
                self.assertEqual(method.use_deepgemm_mxfp4_w4a16_valu, sail and valu)
                self.assertEqual(method.use_deepgemm_mxfp4_w4a16_mma, sail and mma)
                # The runner is DeepGEMM on both platforms; only NVIDIA's W4A8
                # weight/scales layout rewrite is community-only.
                self.assertTrue(method.use_deep_gemm)
                self.assertEqual(method.use_deep_gemm_w4a8, not sail)

    def test_apply_routes_each_deep_gemm_flavor(self):
        """One DeepGEMM branch, three flavors: SAIL W4A16, SAIL FP4xFF4 and
        NVIDIA's W4A8 recipe must each reach the runner with their own flags."""
        layer = SimpleNamespace(
            w13_weight=torch.zeros(1, 2, 2, dtype=torch.uint8),
            w2_weight=torch.zeros(1, 2, 2, dtype=torch.uint8),
            w13_weight_scale=torch.zeros(1, 2, 1, dtype=torch.uint8),
            w2_weight_scale=torch.zeros(1, 2, 1, dtype=torch.uint8),
            w13_weight_bias=torch.zeros(1, 2, dtype=torch.bfloat16),
            w2_weight_bias=torch.zeros(1, 2, dtype=torch.bfloat16),
        )
        seen = []
        for sail, valu, bias, expected in (
            (True, True, False, {"use_mxfp4_w4a16": True, "block_shape": [0, 32]}),
            (True, False, True, {"use_mxfp4": True, "block_shape": [0, 16]}),
            (
                False,
                False,
                False,
                {
                    "use_fp8": True,
                    "is_fp4_experts": True,
                    "block_shape": [128, 128],
                },
            ),
        ):
            method = self._method(mma=False)
            method.use_deep_gemm = True
            method.use_deepgemm_mxfp4_w4a16_valu = valu
            method.with_bias = bias
            method.runner = SimpleNamespace(run=lambda _out, info: seen.append(info))
            seen.clear()
            with (
                self.subTest(sail=sail, valu=valu),
                patch.object(mxfp4, "_is_sail", sail),
            ):
                self.assertIsNone(method.apply(layer, object()))
            (quant_info,) = seen
            for field, value in expected.items():
                self.assertEqual(getattr(quant_info, field), value)
            # Only the FP4xFF4 flavor fuses the expert bias into the GEMM.
            self.assertEqual(quant_info.b13 is not None, bias and not valu)

    def test_sail_nopad_gemm_keeps_checkpoint_shapes(self):
        """The SAIL grouped GEMM is the "nopad" variant, so K and N must stay at
        the checkpoint's sizes even when triton_kernels is importable."""
        method = self._method()
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deep_gemm = True
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        layer.hidden_size = 65
        layer.intermediate_size_per_partition = 129
        with (
            patch.object(mxfp4, "_is_sail", True),
            patch.object(mxfp4, "has_triton_kernels", True),
        ):
            method.create_weights(layer, 2, 65, 129, torch.bfloat16)
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 258, 32))
        self.assertEqual(tuple(layer.w2_weight.shape), (2, 65, 64))

    def test_sail_fp4_runner_does_not_alias_shared_expert_input(self):
        """The SAIL runner combine must use a separate output; it must not overwrite
        the input of a bypassed shared expert."""
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

        for sail in (False, True):
            method = self._method()
            method.use_deepgemm_mxfp4_w4a16_valu = False
            method.use_triton_kernels = False
            config = MoeRunnerConfig(inplace=True)
            with (
                self.subTest(sail=sail),
                patch.object(mxfp4, "_is_sail", sail),
                get_flags().moe.override(
                    runner_backend=MoeRunnerBackend.DEEP_GEMM,
                    a2a_backend=MoeA2ABackend.NONE,
                ),
            ):
                method.create_moe_runner(torch.nn.Module(), config)
            # Check the runner that actually holds the config; checking the
            # create_moe_runner call arguments alone is not enough.
            self.assertEqual(method.runner.config.inplace, not sail)
            self.assertIs(method.runner.runner_core.config, method.runner.config)
            self.assertEqual(method.runner.runner_backend, MoeRunnerBackend.DEEP_GEMM)

    def test_post_load_preserves_e8m0_bytes(self):
        # Repacking is an external GPU boundary; this test checks that the repack
        # output is not numerically converted on its way into the vendor ABI.
        payload = torch.tensor([127, 128, 129, 126], dtype=torch.uint8).reshape(1, 2, 2)
        layer = torch.nn.Module()

        def prepared_scales(target):
            for name in ("w13_weight_scale", "w2_weight_scale"):
                target.register_parameter(
                    name,
                    torch.nn.Parameter(
                        payload.clone().view(torch.float8_e8m0fnu), requires_grad=False
                    ),
                )

        method = self._method()
        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
            patch(
                "sglang.srt.layers.quantization.marlin_utils.check_moe_marlin_supports_layer",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.marlin_utils_fp4.prepare_moe_mxfp4_layer_for_marlin",
                side_effect=prepared_scales,
            ),
            patch.object(torch.cuda, "empty_cache"),
        ):
            method.process_weights_after_loading(layer)
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name)
            self.assertEqual(scale.dtype, torch.uint8)
            self.assertFalse(scale.requires_grad)
            torch.testing.assert_close(scale, payload, rtol=0, atol=0)

    def test_mode_guards_and_deepep_allowlist(self):
        """The weight ABI choice must first reject mutually exclusive modes, old
        architectures and the wrong communication backend."""
        with (
            patch.object(sail_mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.DEEP_GEMM,
                a2a_backend=MoeA2ABackend.DEEPEP,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                sail_mxfp4.validate_w4a16_modes(True, True)
            for sm, mma in ((80, False), (89, True), (100, True)):
                with (
                    self.subTest(sm=sm, mma=mma),
                    patch.object(sail_mxfp4, "get_device_sm", return_value=sm),
                ):
                    sail_mxfp4.validate_w4a16_modes(not mma, mma)
            with patch.object(sail_mxfp4, "get_device_sm", return_value=80):
                with self.assertRaisesRegex(ValueError, "SM89"):
                    sail_mxfp4.validate_w4a16_modes(False, True)

    def test_mma_creates_direct_weights_without_bias(self):
        """64-alignment must not misuse Marlin's 128/256 padding or create an
        unsupported bias."""
        method = self._method(mma=True)
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        layer.hidden_size = 65
        layer.intermediate_size_per_partition = 129
        method.create_weights(layer, 2, 65, 129, torch.bfloat16)
        self.assertEqual((method.hidden_pad, method.intermediate_pad), (63, 63))
        expected_shapes = {
            "w13_weight": (2, 384, 64),
            "w2_weight": (2, 128, 96),
            "w13_weight_scale": (2, 384, 4),
            "w2_weight_scale": (2, 128, 6),
        }
        for name, shape in expected_shapes.items():
            tensor = getattr(layer, name)
            self.assertEqual(tuple(tensor.shape), shape)
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertFalse(tensor.requires_grad)
            if name.endswith("scale"):
                torch.testing.assert_close(tensor, torch.full_like(tensor, 127))
        self.assertFalse(hasattr(layer, "w13_weight_bias"))
        self.assertFalse(hasattr(layer, "w2_weight_bias"))
        with self.assertRaisesRegex(ValueError, "bias"):
            method.create_weights(torch.nn.Module(), 2, 64, 128, torch.bfloat16, True)

    def test_mma_scale_tiles_preserve_e8m0_bit_patterns(self):
        """Independent vendor tile indexing checks the byte positions across experts,
        N tiles and K tiles."""
        from sglang.kernels.ops.quantization.sail_mxfp4 import (
            preprocess_mxfp4_w4a16_mma_scales,
        )

        raw = (torch.arange(2 * 128 * 4).reshape(2, 128, 4) % 256).to(torch.uint8)
        expected = torch.empty((2, 2, 256), dtype=torch.uint8)
        # Every 128-byte tile holds 8 row pairs; adjacent scales are paired and the
        # rows are interleaved with a stride of 8.
        for expert in range(2):
            for nt in range(2):
                for kt in range(2):
                    for row_pair in range(8):
                        for row_stride in range(8):
                            for half in range(2):
                                dst = kt * 128 + row_pair * 16 + row_stride * 2 + half
                                row = nt * 64 + row_pair + row_stride * 8
                                expected[expert, nt, dst] = raw[
                                    expert, row, kt * 2 + half
                                ]
        for source in (raw, raw.view(torch.float8_e8m0fnu)):
            actual = preprocess_mxfp4_w4a16_mma_scales(source)
            self.assertEqual(actual.dtype, torch.uint8)
            self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for bad in (raw[0], raw[:, :63], raw[:, :, :3]):
            with self.assertRaises(ValueError):
                preprocess_mxfp4_w4a16_mma_scales(bad)
        with self.assertRaises(TypeError):
            preprocess_mxfp4_w4a16_mma_scales(raw.float())

    def test_mma_post_load_keeps_packed_weights_and_transforms_scales(self):
        """MMA must not call the Marlin repack or reinterpret FP4 data as INT8."""
        from sglang.kernels.ops.quantization.sail_mxfp4 import (
            preprocess_mxfp4_w4a16_mma_scales,
        )

        layer = torch.nn.Module()
        originals = {}
        for name in ("w13", "w2"):
            weight = torch.nn.Parameter(
                torch.full((2, 128, 64), 0xA2, dtype=torch.uint8), requires_grad=False
            )
            scale = (torch.arange(2 * 128 * 4).reshape(2, 128, 4) % 256).to(torch.uint8)
            layer.register_parameter(name + "_weight", weight)
            layer.register_parameter(
                name + "_weight_scale", torch.nn.Parameter(scale, False)
            )
            originals[name] = (weight, preprocess_mxfp4_w4a16_mma_scales(scale))
        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
            patch.object(torch.cuda, "empty_cache"),
        ):
            self._method(mma=True).process_weights_after_loading(layer)
        self.assertEqual(layer._mxfp4_backend, "deep_gemm")
        for name, (weight, scales) in originals.items():
            self.assertIs(getattr(layer, name + "_weight"), weight)
            torch.testing.assert_close(getattr(layer, name + "_weight_scale"), scales)
            self.assertFalse(getattr(layer, name + "_weight_scale").requires_grad)

    def test_dispatcher_dtype_matches_activation_not_packed_weights(self):
        """Both W4A16 modes use BF16 dispatch; only W4A4 may use uint8 dispatch."""
        from sglang.srt.hardware_backend.sail.moe.dispatcher_quant import (
            build_early_dispatcher_quant_config,
        )
        from sglang.srt.layers.moe.fused_moe_triton import layer as layer_module

        layer = object.__new__(layer_module.FusedMoE)
        layer.quant_config = None
        for valu, mma, expected in (
            (False, False, "uint8"),
            (True, False, "bf16"),
            (False, True, "bf16"),
        ):
            method = self._method(mma=mma)
            method.use_deepgemm_mxfp4_w4a16_valu = valu
            layer.quant_method = method
            with self.subTest(valu=valu, mma=mma):
                self.assertEqual(
                    build_early_dispatcher_quant_config(layer),
                    {"dispatcher_output_dtype": expected, "is_channel_quant": False},
                )


if __name__ == "__main__":
    unittest.main()
