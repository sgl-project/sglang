import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import torch

from sglang.srt.layers.deep_gemm_wrapper import entrypoint as deep_gemm_entrypoint
from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import mxfp4
from sglang.srt.layers.quantization.ppu_mxfp4_utils import (
    preprocess_mxfp4_w4a16_mma_scales,
)
from sglang.srt.models.minimax_m3 import MiniMaxM3SparseForCausalLM
from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMiniMaxM3PackedModulesMapping(CustomTestCase):
    def test_index_qkv_mapping_omits_disabled_value_projection(self):
        for model_class in (
            MiniMaxM3SparseForCausalLM,
            MiniMaxM3SparseForConditionalGeneration,
        ):
            with self.subTest(model_class=model_class.__name__):
                self.assertEqual(
                    model_class.packed_modules_mapping["index_qkv_proj"],
                    ["index_q_proj", "index_k_proj"],
                )


class TestPpuMxfp4W4A16(unittest.TestCase):
    @staticmethod
    def _exec():
        return SimpleNamespace(
            moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
        )

    def test_ppu_env_selects_valu_w4a16(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=89),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=False,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MOE, "get", return_value=False
            ),
            patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        ):
            method = mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")
            self.assertTrue(method.use_deepgemm_mxfp4_w4a16_valu)
            self.assertFalse(method.use_deepgemm_mxfp4_w4a16_mma)
            self.assertFalse(method.use_marlin)
            self.assertTrue(method.is_deepgemm_moe_runner_backend_enabled())

    def test_ppu_env_selects_direct_mma_w4a16(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=89),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=False,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MOE, "get", return_value=False
            ),
            patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        ):
            method = mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

        self.assertFalse(method.use_deepgemm_mxfp4_w4a16_valu)
        self.assertTrue(method.use_deepgemm_mxfp4_w4a16_mma)

    def test_ppu_mma_rejects_sm80(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=80),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=False,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "requires SM89 or newer"),
        ):
            mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

    def test_ppu_valu_allows_sm80(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=80),
            patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        ):
            mxfp4.Mxfp4MoEMethod._validate_ppu_mxfp4_w4a16(True, False)

    def test_ppu_valu_and_mma_selectors_are_mutually_exclusive(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "mutually exclusive"),
        ):
            mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

    def test_ppu_selectors_do_not_affect_other_backends(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "_is_ppu", False),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=True,
            ),
        ):
            method = mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

        self.assertFalse(method.use_deepgemm_mxfp4_w4a16_valu)
        self.assertFalse(method.use_deepgemm_mxfp4_w4a16_mma)

    def test_ppu_w4a16_accepts_deepep_a2a(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(
                mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
            ),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=89),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=False,
            ),
            patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        ):
            method = mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

        self.assertTrue(method.use_deepgemm_mxfp4_w4a16_valu)
        self.assertFalse(method.use_deepgemm_mxfp4_w4a16_mma)

    def test_ppu_w4a16_rejects_unsupported_a2a(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(
                mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.MOONCAKE
            ),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "none or deepep"):
                mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

    def test_ppu_w4a16_requires_explicit_deepgemm_backend(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=True,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "moe-runner-backend deep_gemm"):
                mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")

    def test_native_mxfp4_w4a4_is_not_gated_by_w4a16_jit_check(self):
        with (
            patch.object(
                mxfp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(mxfp4, "get_exec", return_value=self._exec()),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "get_device_sm", return_value=89),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16,
                "get",
                return_value=False,
            ),
            patch.object(
                mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA,
                "get",
                return_value=False,
            ),
            patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False),
        ):
            method = mxfp4.Mxfp4MoEMethod("model.layers.0.mlp.experts")
            self.assertFalse(method.use_deepgemm_mxfp4_w4a16_valu)
            self.assertFalse(method.use_deepgemm_mxfp4_w4a16_mma)
            self.assertTrue(method.is_deepgemm_moe_runner_backend_enabled())

    def test_ppu_reorders_e8m0_scales_for_mma(self):
        scales = torch.arange(128, dtype=torch.uint8).reshape(1, 64, 2)
        scale_perm = torch.tensor(
            [
                2 * i + j * 16 + offset
                for i in range(8)
                for j in range(8)
                for offset in (0, 1)
            ]
        )

        actual = preprocess_mxfp4_w4a16_mma_scales(scales)

        self.assertEqual(actual.shape, (1, 1, 128))
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual.flatten(), scales.flatten()[scale_perm]))

    def test_ppu_mma_scale_preprocess_accepts_e8m0_view(self):
        scales = torch.arange(128, dtype=torch.uint8).reshape(1, 64, 2)

        actual = preprocess_mxfp4_w4a16_mma_scales(scales.view(torch.float8_e8m0fnu))

        self.assertEqual(actual.dtype, torch.uint8)
        self.assertEqual(actual.shape, (1, 1, 128))

    def test_ppu_mma_scale_preprocess_preserves_nk_tile_order(self):
        scales = (
            torch.arange(2 * 128 * 4, dtype=torch.int64)
            .remainder(251)
            .to(torch.uint8)
            .reshape(2, 128, 4)
        )
        scale_perm = torch.tensor(
            [
                2 * i + j * 16 + offset
                for i in range(8)
                for j in range(8)
                for offset in (0, 1)
            ]
        )
        expected = torch.empty((2, 2, 256), dtype=torch.uint8)
        for expert in range(2):
            for n_tile in range(2):
                for k_tile in range(2):
                    block = scales[
                        expert,
                        n_tile * 64 : (n_tile + 1) * 64,
                        k_tile * 2 : (k_tile + 1) * 2,
                    ].flatten()
                    expected[expert, n_tile, k_tile * 128 : (k_tile + 1) * 128] = block[
                        scale_perm
                    ]

        actual = preprocess_mxfp4_w4a16_mma_scales(scales)

        self.assertTrue(torch.equal(actual, expected))

    def test_ppu_mma_scale_preprocess_rejects_unaligned_shape(self):
        with self.assertRaisesRegex(ValueError, "multiples of 64"):
            preprocess_mxfp4_w4a16_mma_scales(
                torch.empty((1, 63, 2), dtype=torch.uint8)
            )
        with self.assertRaisesRegex(ValueError, "multiples of 64"):
            preprocess_mxfp4_w4a16_mma_scales(
                torch.empty((1, 64, 1), dtype=torch.uint8)
            )

    def test_ppu_process_keeps_direct_weights_and_preprocesses_scales(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deepgemm_mxfp4_w4a16_mma = True
        method.moe_runner_config = SimpleNamespace(gemm1_alpha=None)
        w13_weight = torch.nn.Parameter(
            torch.empty((1, 64, 32), dtype=torch.uint8), requires_grad=False
        )
        w2_weight = torch.nn.Parameter(
            torch.empty((1, 64, 32), dtype=torch.uint8), requires_grad=False
        )
        raw_scale = torch.arange(128, dtype=torch.uint8).reshape(1, 64, 2)
        layer = SimpleNamespace(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_scale=torch.nn.Parameter(
                raw_scale.view(torch.float8_e8m0fnu), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(raw_scale, requires_grad=False),
        )
        with (
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(torch.cuda, "empty_cache"),
        ):
            method.process_weights_after_loading(layer)

        self.assertIs(layer.w13_weight, w13_weight)
        self.assertIs(layer.w2_weight, w2_weight)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale.shape, (1, 1, 128))
        self.assertEqual(layer.w2_weight_scale.shape, (1, 1, 128))

    def test_ppu_valu_preserves_marlin_e8m0_scales_as_uint8(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deepgemm_mxfp4_w4a16_valu = True
        method.use_deepgemm_mxfp4_w4a16_mma = False
        method.moe_runner_config = SimpleNamespace(gemm1_alpha=None)
        raw_scale = torch.tensor([127, 128], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

        def prepare_marlin_scales(layer):
            layer.w13_weight_scale = torch.nn.Parameter(
                raw_scale.reshape(1, 1, 2), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                raw_scale.reshape(1, 1, 2), requires_grad=False
            )

        layer = SimpleNamespace()
        with (
            patch(
                "sglang.srt.layers.quantization.marlin_utils."
                "check_moe_marlin_supports_layer",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.marlin_utils_fp4."
                "prepare_moe_mxfp4_layer_for_marlin",
                side_effect=prepare_marlin_scales,
            ),
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(torch.cuda, "empty_cache"),
        ):
            method.process_weights_after_loading(layer)

        expected = torch.tensor([[[127, 128]]], dtype=torch.uint8)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.uint8)
        self.assertEqual(layer._mxfp4_backend, "marlin")
        self.assertTrue(torch.equal(layer.w13_weight_scale, expected))
        self.assertTrue(torch.equal(layer.w2_weight_scale, expected))

    def test_ppu_valu_weight_creation_keeps_legacy_padding(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deepgemm_mxfp4_w4a16_valu = True
        method.use_deepgemm_mxfp4_w4a16_mma = False
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        layer.hidden_size = 64
        layer.intermediate_size_per_partition = 64

        with (
            patch.object(mxfp4, "_is_ppu", True),
            patch.object(mxfp4, "_is_hip", False),
        ):
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=64,
                params_dtype=torch.bfloat16,
            )

        self.assertEqual(layer.w13_weight.shape, (2, 256, 128))
        self.assertEqual(layer.w13_weight_scale.shape, (2, 256, 8))
        self.assertEqual(layer.w2_weight.shape, (2, 256, 64))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 256, 4))
        self.assertTrue(hasattr(layer, "w13_weight_bias"))
        self.assertTrue(hasattr(layer, "w2_weight_bias"))

    def test_ppu_mma_weight_creation_uses_direct_layout_without_bias(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deepgemm_mxfp4_w4a16_mma = True
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        layer.hidden_size = 64
        layer.intermediate_size_per_partition = 64

        with patch.object(mxfp4, "_is_ppu", True):
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=64,
                params_dtype=torch.bfloat16,
            )

        self.assertEqual(layer.w13_weight.shape, (2, 128, 32))
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale.shape, (2, 128, 2))
        self.assertEqual(layer.w2_weight.shape, (2, 64, 32))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 64, 2))
        self.assertFalse(hasattr(layer, "w13_weight_bias"))
        self.assertFalse(hasattr(layer, "w2_weight_bias"))

    def test_ppu_mma_rejects_expert_bias(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.use_deepgemm_mxfp4_w4a16_mma = True
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        layer.hidden_size = 64
        layer.intermediate_size_per_partition = 64

        with (
            patch.object(mxfp4, "_is_ppu", True),
            self.assertRaisesRegex(ValueError, "does not support expert bias"),
        ):
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=64,
                params_dtype=torch.bfloat16,
                with_bias=True,
            )

    def test_w4a16_weight_shape_supports_mma_and_legacy_abis(self):
        self.assertEqual(
            deep_gemm_entrypoint._get_w4a16_weight_shape(
                torch.empty((4, 128, 32), dtype=torch.uint8)
            ),
            (4, 128, 64),
        )
        self.assertEqual(
            deep_gemm_entrypoint._get_w4a16_weight_shape(
                torch.empty((4, 4, 256), dtype=torch.int32)
            ),
            (4, 128, 64),
        )

    def test_w4a16_nopad_dispatches_direct_weights_to_mma_kernel_type(self):
        lhs = torch.empty((3, 64), dtype=torch.bfloat16)
        weight = torch.empty((4, 128, 32), dtype=torch.uint8)
        scales = torch.empty((4, 2, 128), dtype=torch.uint8)
        out = torch.empty((3, 128), dtype=torch.bfloat16)
        m_indices = torch.zeros((3,), dtype=torch.int32)
        tuner = Mock()
        tuner.get_deep_gemm_config.return_value = sentinel.config
        deep_gemm = Mock()

        with (
            patch.object(deep_gemm_entrypoint, "_is_ppu", True),
            patch.object(deep_gemm_entrypoint, "tuner", tuner, create=True),
            patch.object(deep_gemm_entrypoint, "deep_gemm", deep_gemm, create=True),
            patch.object(
                deep_gemm_entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ) as execution_hook,
        ):
            deep_gemm_entrypoint.grouped_gemm_nt_bf16i4bf16_nopad(
                lhs, (weight, scales), out, m_indices
            )

        kernel_type = deep_gemm_entrypoint.compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD
        execution_hook.assert_called_once_with(
            3,
            128,
            64,
            4,
            kernel_type,
        )
        deep_gemm.m_grouped_gemm_w4a16_nopad.assert_called_once_with(
            lhs,
            (weight, scales),
            out,
            m_indices,
            None,
            sentinel.config,
        )

    def test_w4a16_fused_dispatches_direct_weights_to_mma_kernel_type(self):
        lhs = torch.empty((3, 64), dtype=torch.bfloat16)
        weight = torch.empty((4, 128, 32), dtype=torch.uint8)
        scales = torch.empty((4, 2, 128), dtype=torch.uint8)
        out = torch.empty((24, 128), dtype=torch.bfloat16)
        metadata = [torch.empty((1,), dtype=torch.int32) for _ in range(4)]
        deep_gemm = Mock()

        with (
            patch.object(deep_gemm_entrypoint, "_is_ppu", True),
            patch.object(deep_gemm_entrypoint, "deep_gemm", deep_gemm, create=True),
            patch.object(
                deep_gemm_entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ) as execution_hook,
        ):
            deep_gemm_entrypoint.grouped_gemm_nt_bf16i4bf16_fused(
                lhs,
                (weight, scales),
                out,
                *metadata,
                sentinel.config,
            )

        kernel_type = deep_gemm_entrypoint.compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED
        execution_hook.assert_called_once_with(
            3,
            128,
            64,
            4,
            kernel_type,
        )
        deep_gemm.m_grouped_gemm_w4a16_fused.assert_called_once_with(
            lhs,
            (weight, scales),
            out,
            *metadata,
            sentinel.config,
        )

    def test_w4a16_fused_dispatches_legacy_weights_to_legacy_kernel_type(self):
        lhs = torch.empty((3, 64), dtype=torch.bfloat16)
        weight = torch.empty((4, 4, 256), dtype=torch.int32)
        scales = torch.empty((4, 2, 128), dtype=torch.bfloat16)
        out = torch.empty((24, 128), dtype=torch.bfloat16)
        metadata = [torch.empty((1,), dtype=torch.int32) for _ in range(4)]
        deep_gemm = Mock()

        with (
            patch.object(deep_gemm_entrypoint, "_is_ppu", True),
            patch.object(deep_gemm_entrypoint, "deep_gemm", deep_gemm, create=True),
            patch.object(
                deep_gemm_entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ) as execution_hook,
        ):
            deep_gemm_entrypoint.grouped_gemm_nt_bf16i4bf16_fused(
                lhs,
                (weight, scales),
                out,
                *metadata,
                sentinel.config,
            )

        kernel_type = deep_gemm_entrypoint.compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED
        execution_hook.assert_called_once_with(
            3,
            128,
            64,
            4,
            kernel_type,
        )
        deep_gemm.m_grouped_gemm_w4a16_fused.assert_called_once_with(
            lhs,
            (weight, scales),
            out,
            *metadata,
            sentinel.config,
        )

    def test_w4a16_fused_dispatches_valu_scales_to_valu_kernel_type(self):
        lhs = torch.empty((3, 64), dtype=torch.bfloat16)
        weight = torch.empty((4, 4, 256), dtype=torch.int32)
        scales = torch.empty((4, 2, 128), dtype=torch.uint8)
        out = torch.empty((24, 128), dtype=torch.bfloat16)
        metadata = [torch.empty((1,), dtype=torch.int32) for _ in range(4)]
        deep_gemm = Mock()

        with (
            patch.object(deep_gemm_entrypoint, "_is_ppu", True),
            patch.object(deep_gemm_entrypoint, "deep_gemm", deep_gemm, create=True),
            patch.object(
                deep_gemm_entrypoint.compile_utils,
                "deep_gemm_execution_hook",
                return_value=nullcontext(),
            ) as execution_hook,
        ):
            deep_gemm_entrypoint.grouped_gemm_nt_bf16i4bf16_fused(
                lhs,
                (weight, scales),
                out,
                *metadata,
                sentinel.config,
            )

        kernel_type = deep_gemm_entrypoint.compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED
        execution_hook.assert_called_once_with(
            3,
            128,
            64,
            4,
            kernel_type,
        )
        deep_gemm.m_grouped_gemm_w4a16_fused.assert_called_once_with(
            lhs,
            (weight, scales),
            out,
            *metadata,
            sentinel.config,
        )

    def test_moe_align_uses_int4_tuning_for_mma_weights(self):
        lhs = torch.empty((3, 64), dtype=torch.bfloat16)
        weight = torch.empty((4, 128, 32), dtype=torch.uint8)
        topk_ids = torch.zeros((3, 2), dtype=torch.int32)
        tuner = Mock()
        tuner.get_deep_gemm_config.return_value = sentinel.config
        deep_gemm = Mock()
        deep_gemm.moe_align_block_size.return_value = sentinel.metadata

        with (
            patch.object(deep_gemm_entrypoint, "_is_ppu", True),
            patch.object(deep_gemm_entrypoint, "tuner", tuner, create=True),
            patch.object(deep_gemm_entrypoint, "deep_gemm", deep_gemm, create=True),
        ):
            actual = deep_gemm_entrypoint.moe_align_block_size(lhs, weight, topk_ids)

        self.assertIs(actual, sentinel.metadata)
        tuner.get_deep_gemm_config.assert_called_once_with(
            3, 128, 64, num_groups=4, nopad=True, dtype="int4"
        )
        deep_gemm.moe_align_block_size.assert_called_once_with(
            lhs,
            weight,
            topk_ids,
            perchannel_quant=False,
            config=sentinel.config,
        )

    def test_moe_align_uses_int4_tuning_for_legacy_weights(self):
        lhs = torch.empty((3, 64), dtype=torch.bfloat16)
        weight = torch.empty((4, 4, 256), dtype=torch.int32)
        topk_ids = torch.zeros((3, 2), dtype=torch.int32)
        tuner = Mock()
        tuner.get_deep_gemm_config.return_value = sentinel.config
        deep_gemm = Mock()
        deep_gemm.moe_align_block_size.return_value = sentinel.metadata

        with (
            patch.object(deep_gemm_entrypoint, "_is_ppu", True),
            patch.object(deep_gemm_entrypoint, "tuner", tuner, create=True),
            patch.object(deep_gemm_entrypoint, "deep_gemm", deep_gemm, create=True),
        ):
            actual = deep_gemm_entrypoint.moe_align_block_size(lhs, weight, topk_ids)

        self.assertIs(actual, sentinel.metadata)
        tuner.get_deep_gemm_config.assert_called_once_with(
            3, 128, 64, num_groups=4, nopad=True, dtype="int4"
        )
        deep_gemm.moe_align_block_size.assert_called_once_with(
            lhs,
            weight,
            topk_ids,
            perchannel_quant=False,
            config=sentinel.config,
        )

    def test_ppu_w4a16_uses_e8m0_scales_in_runner(self):
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_deepgemm_mxfp4_w4a16_valu = True
        method.use_deepgemm_mxfp4_w4a16_mma = False
        method.runner = Mock()
        method.runner.run.return_value = sentinel.combine_input
        layer = SimpleNamespace(
            w13_weight=sentinel.w13_weight,
            w2_weight=sentinel.w2_weight,
            w13_weight_scale=torch.ones(1, dtype=torch.uint8),
            w2_weight_scale=torch.ones(1, dtype=torch.uint8),
        )

        with patch.object(mxfp4, "_is_ppu", True):
            self.assertIs(
                method.apply(layer, sentinel.dispatch_output), sentinel.combine_input
            )
        quant_info = method.runner.run.call_args.args[1]
        self.assertTrue(quant_info.use_mxfp4_w4a16)
        self.assertFalse(quant_info.use_int4_w4a16)
        self.assertEqual(quant_info.w13_scale.dtype, torch.uint8)
        self.assertEqual(quant_info.w2_scale.dtype, torch.uint8)

    def test_ppu_w4a16_dispatches_bf16_activations(self):
        from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_deepgemm_mxfp4_w4a16_valu = True
        method.use_deepgemm_mxfp4_w4a16_mma = False
        layer = fused_moe_layer.FusedMoE.__new__(fused_moe_layer.FusedMoE)
        object.__setattr__(layer, "quant_method", method)
        object.__setattr__(layer, "quant_config", None)

        with patch.object(fused_moe_layer, "_is_ppu", True):
            quant_config = layer._build_dispatcher_quant_config()
            method.use_deepgemm_mxfp4_w4a16_valu = False
            native_quant_config = layer._build_dispatcher_quant_config()

        self.assertEqual(quant_config["dispatcher_output_dtype"], "bf16")
        self.assertEqual(native_quant_config["dispatcher_output_dtype"], "uint8")


if __name__ == "__main__":
    unittest.main()
