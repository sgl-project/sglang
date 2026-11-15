"""Unit tests for modelopt_fp4 checkpoints that quantize Qwen3.5 attention.

Covers four things:
  1. ModelOptFp4Config.is_layer_excluded() decides per prefix whether attention is
     quantized or kept in BF16.
  2. RadixAttention registers k_scale/v_scale when built with a quant_config that
     declares kv_cache_quant_algo; without them, baked KV scales have nowhere to
     load into and silently fall back to 1.0.
  3. QWEN3_5_KV_SCALE_MAPPER remaps the checkpoint's baked KV-scale names onto the
     RadixAttention parameter names.
  4. The dense-MLP SiLU+FP4 fusion turns on for NVFP4 W4A4 and stays off where
     its (fp4, scale) tuple would be wrong or unusable: NVFP4 AWQ, the Marlin FP4
     GEMM backend, K not a multiple of 64, sequence parallelism, a LoRA-wrapped
     down_proj.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import QWEN3_5_KV_SCALE_MAPPER
from sglang.srt.models.utils import (
    can_run_silu_fp4_quant_fusion,
    maybe_enable_silu_fp4_quant_fusion,
    silu_and_mul_fp4_quant,
)
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestModelOptFp4AttentionExclusion(CustomTestCase):
    def test_moe_only_checkpoint_excludes_attention(self):
        # NVIDIA's Qwen3.5 NVFP4 checkpoints: attention and lm_head are excluded,
        # MoE experts are not.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=["*self_attn*", "lm_head"],
        )

        self.assertTrue(cfg.is_layer_excluded("model.layers.0.self_attn.qkv_proj"))
        self.assertTrue(cfg.is_layer_excluded("lm_head"))
        self.assertFalse(
            cfg.is_layer_excluded("model.layers.0.mlp.experts.3.gate_up_proj")
        )

    def test_uniform_w4a4_checkpoint_quantizes_attention(self):
        # Uniform W4A4 checkpoint: only lm_head is excluded, attention is quantized.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=["lm_head"],
        )

        self.assertFalse(cfg.is_layer_excluded("model.layers.0.self_attn.qkv_proj"))
        self.assertFalse(
            cfg.is_layer_excluded("model.layers.0.linear_attn.in_proj_qkvz")
        )
        self.assertTrue(cfg.is_layer_excluded("lm_head"))


class TestRadixAttentionKvScaleRegistration(CustomTestCase):
    def _make_attn(self, quant_config):
        return RadixAttention(
            num_heads=2,
            head_dim=8,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=0,
            quant_config=quant_config,
            prefix="model.layers.0.attn",
        )

    def test_with_fp8_kv_quant_config_registers_scale_params(self):
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo="FP8",
            group_size=16,
            exclude_modules=[],
        )
        attn = self._make_attn(cfg)

        self.assertIsInstance(attn.k_scale, torch.nn.Parameter)
        self.assertIsInstance(attn.v_scale, torch.nn.Parameter)
        # create_weights seeds -1.0, the sentinel for "checkpoint had no scale".
        self.assertEqual(attn.k_scale.item(), -1.0)
        self.assertEqual(attn.v_scale.item(), -1.0)

    def test_without_quant_config_has_no_scale_params(self):
        attn = self._make_attn(None)

        self.assertIsNone(attn.k_scale)
        self.assertIsNone(attn.v_scale)

    def test_quant_config_without_kv_cache_algo_has_no_scale_params(self):
        # Registration is gated on kv_cache_quant_algo, not on quant_config alone.
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            group_size=16,
            exclude_modules=[],
        )
        attn = self._make_attn(cfg)

        self.assertIsNone(attn.k_scale)
        self.assertIsNone(attn.v_scale)


class TestQwen3_5KvScaleMapper(CustomTestCase):
    def test_maps_baked_kv_scale_names_onto_radix_attention(self):
        # Source names come from ModelOpt's export format, target names from the
        # sglang module tree; a typo on either side silently zeroes the scales.
        weights = [
            ("model.layers.3.self_attn.k_proj.k_scale", torch.tensor(0.0347)),
            ("model.layers.3.self_attn.v_proj.v_scale", torch.tensor(0.0128)),
        ]

        mapped = list(QWEN3_5_KV_SCALE_MAPPER.apply(weights))

        self.assertEqual(
            [name for name, _ in mapped],
            ["model.layers.3.attn.k_scale", "model.layers.3.attn.v_scale"],
        )
        torch.testing.assert_close(mapped[0][1], torch.tensor(0.0347))
        torch.testing.assert_close(mapped[1][1], torch.tensor(0.0128))

    def test_all_other_names_pass_through_unchanged(self):
        # A mapping key that is too broad would corrupt regular weight loading.
        names = [
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.k_proj.input_scale",
            "model.layers.3.self_attn.k_proj.weight_scale",
            "model.layers.2.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.mlp.experts.5.down_proj.weight",
            "lm_head.weight",
        ]
        weights = [(name, torch.zeros(1)) for name in names]

        mapped = list(QWEN3_5_KV_SCALE_MAPPER.apply(weights))

        self.assertEqual([name for name, _ in mapped], names)

    def test_mapped_scale_loads_via_default_weight_loader(self):
        # The scale params carry no weight_loader, so load_weights' fallback uses
        # default_weight_loader; its scalar path is what tolerates the 0-dim param
        # vs the shape-[1] checkpoint tensor.
        scale_param = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        loaded_weight = torch.tensor([0.0347], dtype=torch.float32)

        weight_loader = getattr(scale_param, "weight_loader", default_weight_loader)
        weight_loader(scale_param, loaded_weight)

        self.assertAlmostEqual(scale_param.item(), 0.0347, places=6)


class TestSiluFp4QuantFusionGates(CustomTestCase):
    @staticmethod
    def _enabled_mlp(down_proj) -> SimpleNamespace:
        return SimpleNamespace(_enable_silu_fp4_quant_fusion=True, down_proj=down_proj)

    def test_runs_for_a_plain_down_proj(self):
        mlp = self._enabled_mlp(SimpleNamespace(params_dtype=torch.float32))
        self.assertTrue(can_run_silu_fp4_quant_fusion(mlp, torch.empty(2, 32)))

    def test_skips_a_lora_wrapped_down_proj(self):
        # Like RowParallelLinearWithLoRA: a base_layer, no params_dtype.
        mlp = self._enabled_mlp(SimpleNamespace(base_layer=SimpleNamespace()))
        self.assertFalse(can_run_silu_fp4_quant_fusion(mlp, torch.empty(2, 32)))

    def test_skips_an_activation_of_another_dtype(self):
        """down_proj would output its BF16 params_dtype for an FP16 activation."""
        mlp = self._enabled_mlp(SimpleNamespace(params_dtype=torch.bfloat16))
        x = torch.empty(2, 32, dtype=torch.float16)
        self.assertFalse(can_run_silu_fp4_quant_fusion(mlp, x))
        self.assertTrue(can_run_silu_fp4_quant_fusion(mlp, x.bfloat16()))

    def test_skips_inputs_with_leading_dims(self):
        mlp = self._enabled_mlp(SimpleNamespace(params_dtype=torch.float32))
        self.assertFalse(can_run_silu_fp4_quant_fusion(mlp, torch.empty(2, 3, 32)))

    def test_skips_under_sequence_parallelism(self):
        mlp = self._enabled_mlp(SimpleNamespace(params_dtype=torch.float32))
        with get_forward().scoped(sp_active=True):
            self.assertFalse(can_run_silu_fp4_quant_fusion(mlp, torch.empty(2, 32)))

    @staticmethod
    def _fusion_turns_on(
        *, is_awq: bool, backend: Fp4GemmRunnerBackend, down_proj_k: int = 128
    ) -> bool:
        method = MagicMock(spec=ModelOptFp4LinearMethod, quant_mode="w4a4")
        method.quant_config = SimpleNamespace(is_awq=is_awq)
        mlp = SimpleNamespace(
            _enable_silu_fp4_quant_fusion=False,
            gate_up_proj=SimpleNamespace(quant_method=method),
            down_proj=SimpleNamespace(
                quant_method=method, input_size_per_partition=down_proj_k
            ),
        )
        # A stand-in FlashInfer keeps the result independent of the installed one.
        flashinfer = SimpleNamespace(silu_and_mul_scaled_nvfp4_experts_quantize=None)
        with (
            patch.dict(sys.modules, {"flashinfer": flashinfer}),
            patch.object(fp4_utils, "FP4_GEMM_RUNNER_BACKEND", backend),
        ):
            maybe_enable_silu_fp4_quant_fusion(mlp)
        return mlp._enable_silu_fp4_quant_fusion

    def test_turns_on_for_w4a4_on_a_native_fp4_backend(self):
        self.assertTrue(
            self._fusion_turns_on(
                is_awq=False, backend=Fp4GemmRunnerBackend.FLASHINFER_CUTLASS
            )
        )

    def test_stays_off_for_nvfp4_awq(self):
        """The fused producer would skip AWQ's per-input-channel pre_quant_scale."""
        self.assertFalse(
            self._fusion_turns_on(
                is_awq=True, backend=Fp4GemmRunnerBackend.FLASHINFER_CUTLASS
            )
        )

    def test_stays_off_when_k_is_not_a_multiple_of_64(self):
        """K = 1376 pads 86 scale columns to 88, which the fused kernel never writes;
        a NaN byte left there makes the whole GEMM output NaN."""
        self.assertFalse(
            self._fusion_turns_on(
                is_awq=False,
                backend=Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
                down_proj_k=1376,
            )
        )

    def test_stays_off_for_the_marlin_backend(self):
        """Marlin, the SM80-SM90 default, takes a floating-point input; the
        (fp4, scale) tuple crashes its first forward."""
        self.assertFalse(
            self._fusion_turns_on(is_awq=False, backend=Fp4GemmRunnerBackend.MARLIN)
        )


class TestSiluFp4QuantFusionCompressedTensors(CustomTestCase):
    """Compressed-tensors W4A4 NVFP4 MLPs take the same fused path as ModelOpt."""

    @staticmethod
    def _mlp(gate_up_scheme, down_scheme, down_proj_k: int = 128) -> SimpleNamespace:
        return SimpleNamespace(
            _enable_silu_fp4_quant_fusion=False,
            _masked_m_cache={},
            _down_input_scale_inv_1d=None,
            gate_up_proj=SimpleNamespace(quant_method=object(), scheme=gate_up_scheme),
            down_proj=SimpleNamespace(
                quant_method=object(),
                scheme=down_scheme,
                input_size_per_partition=down_proj_k,
                _accepts_prequantized_fp4=False,
            ),
        )

    @staticmethod
    def _enable(mlp, flashinfer=None) -> None:
        flashinfer = flashinfer or SimpleNamespace(
            silu_and_mul_scaled_nvfp4_experts_quantize=None
        )
        with (
            patch.dict(sys.modules, {"flashinfer": flashinfer}),
            patch.object(
                fp4_utils,
                "FP4_GEMM_RUNNER_BACKEND",
                Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
            ),
        ):
            maybe_enable_silu_fp4_quant_fusion(mlp)

    def test_turns_on_for_w4a4_nvfp4_projections(self):
        mlp = self._mlp(CompressedTensorsW4A4Fp4(), CompressedTensorsW4A4Fp4())
        self._enable(mlp)
        self.assertTrue(mlp._enable_silu_fp4_quant_fusion)
        self.assertTrue(mlp.down_proj._accepts_prequantized_fp4)

    def test_stays_off_unless_both_projections_are_w4a4_nvfp4(self):
        for gate_up, down in (
            (CompressedTensorsW4A4Fp4(), None),
            (None, CompressedTensorsW4A4Fp4()),
            (None, None),
        ):
            mlp = self._mlp(gate_up, down)
            self._enable(mlp)
            self.assertFalse(mlp._enable_silu_fp4_quant_fusion)
            self.assertFalse(mlp.down_proj._accepts_prequantized_fp4)

    def test_kill_switch(self):
        mlp = self._mlp(CompressedTensorsW4A4Fp4(), CompressedTensorsW4A4Fp4())
        with envs.SGLANG_DISABLE_SILU_FP4_QUANT_FUSION.override(True):
            self._enable(mlp)
        self.assertFalse(mlp._enable_silu_fp4_quant_fusion)

    def test_quantizes_with_the_down_proj_input_global_scale(self):
        mlp = self._mlp(CompressedTensorsW4A4Fp4(), CompressedTensorsW4A4Fp4())
        mlp.down_proj.input_global_scale = torch.tensor(42.0)
        self.assertEqual(self._fused_scale(mlp), [42.0])

    def test_modelopt_quantizes_with_the_down_proj_input_scale_inv(self):
        mlp = self._mlp(None, None)
        mlp.down_proj.input_scale_inv = torch.tensor([7.0])
        self.assertEqual(self._fused_scale(mlp), [7.0])

    @staticmethod
    def _fused_scale(mlp) -> list:
        seen = {}

        def fake_kernel(x, masked_m, global_scale):
            seen["scale"] = global_scale
            m, k = x.shape[1], x.shape[2] // 2
            # The kernel's expert-grouped layouts for one expert of m <= 128 rows.
            return (
                torch.zeros(m, k // 2, 1, dtype=torch.uint8),
                torch.zeros(32, 4, 1, 4, k // 64, 1, dtype=torch.uint8),
            )

        flashinfer = SimpleNamespace(
            silu_and_mul_scaled_nvfp4_experts_quantize=fake_kernel
        )
        with patch.dict(sys.modules, {"flashinfer": flashinfer}):
            x_fp4, x_sf = silu_and_mul_fp4_quant(
                mlp, torch.zeros(8, 256, dtype=torch.bfloat16)
            )
        assert tuple(x_fp4.shape) == (8, 64) and tuple(x_sf.shape) == (128, 8)
        return seen["scale"].tolist()


if __name__ == "__main__":
    unittest.main()
