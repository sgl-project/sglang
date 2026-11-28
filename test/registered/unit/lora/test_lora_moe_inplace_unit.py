"""LoRA MoE must not overwrite inputs shared experts read on another stream.

In-place reduction races with that read during decode graph replay. FusedMoE passes
``inplace=True`` explicitly, so changing the config default does not fix it:
the real LoRA wrapper must disable it on the shared runner config object.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types
import unittest
import unittest.mock as mock

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase


def _make_base_layer() -> types.SimpleNamespace:
    """The attribute surface FusedMoEWithLoRA.__init__ reads off its base FusedMoE."""
    config = MoeRunnerConfig(
        num_experts=8,
        num_local_experts=8,
        hidden_size=64,
        intermediate_size_per_partition=32,
        layer_id=0,
        top_k=2,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        inplace=True,
    )
    # Let AUTO use Triton; quant-info getters remain mocked.
    quant_method = mock.MagicMock()
    quant_method.runner = None
    return types.SimpleNamespace(
        quant_method=quant_method,
        moe_runner_config=config,
        dispatcher=None,
        num_local_experts=8,
        should_fuse_routed_scaling_factor_in_topk=False,
        moe_tp_size=1,
        moe_tp_rank=0,
        intermediate_size_per_partition=32,
        runner=None,
    )


class FusedMoEWithLoRAInplaceTest(CustomTestCase):
    def test_constructor_forces_inplace_off(self):
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        base_layer = _make_base_layer()
        lora_backend = types.SimpleNamespace(name="triton")
        self.assertTrue(base_layer.moe_runner_config.inplace)
        layer = FusedMoEWithLoRA(base_layer, lora_backend)
        self.assertIs(layer.moe_runner_config, base_layer.moe_runner_config)
        self.assertFalse(layer.moe_runner_config.inplace)

    def test_compressed_fp8_initializes_with_real_quant_info(self):
        # The quant method delegates to the scheme, which previously lacked the getter.
        from compressed_tensors.quantization import QuantizationArgs

        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        base_layer = _make_base_layer()
        base_layer.quant_method = CompressedTensorsFusedMoEMethod(None)
        base_layer.scheme = CompressedTensorsW8A8Fp8MoE(
            weight_quant=QuantizationArgs(
                type="float", strategy="channel", dynamic=False
            ),
            input_quant=QuantizationArgs(type="float", strategy="token", dynamic=True),
        )
        base_layer.w13_weight = torch.zeros(8, 64, 64, dtype=torch.float8_e4m3fn)
        base_layer.w2_weight = torch.zeros(8, 64, 32, dtype=torch.float8_e4m3fn)
        base_layer.w13_weight_scale = torch.ones(8, 64, 1)
        base_layer.w2_weight_scale = torch.ones(8, 64, 1)
        base_layer.w13_input_scale = base_layer.w2_input_scale = None

        layer = FusedMoEWithLoRA(base_layer, types.SimpleNamespace(name="triton"))

        self.assertTrue(layer._quant_info.use_fp8_w8a8)
        self.assertTrue(layer._quant_info.per_channel_quant)
        self.assertIsNone(layer._quant_info.block_shape)
        self.assertIs(layer._quant_info.w13_weight, base_layer.w13_weight)
        self.assertIs(layer._quant_info.w2_weight, base_layer.w2_weight)
        self.assertIs(layer._quant_info.w13_scale, base_layer.w13_weight_scale)
        self.assertIs(layer._quant_info.w2_scale, base_layer.w2_weight_scale)
        self.assertFalse(layer.moe_runner_config.inplace)

    def test_online_weight_updates_follow_the_provider_cache(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
        from sglang.srt.lora.moe.runner import MoeLoraRunner
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            _unsupported_derived_weight_cache_error,
        )

        for family, vendor, backend, has_cache in (
            ("nvfp4", "cutedsl", MoeRunnerBackend.LORA_CUTEDSL, True),
            ("fp8", "marlin", MoeRunnerBackend.LORA_MARLIN, True),
            ("fp8", "cutedsl", MoeRunnerBackend.LORA_CUTEDSL, False),
            ("bf16", "triton", MoeRunnerBackend.LORA_TRITON, False),
        ):
            with self.subTest(family=family, vendor=vendor):
                runner = object.__new__(MoeLoraRunner)
                provider_cls = select_provider_cls("expert_major", family, vendor)
                runner.providers = {"expert_major": object.__new__(provider_cls)}
                base = _make_base_layer()
                base.run_moe_core = mock.Mock()
                with (
                    mock.patch(
                        "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                        return_value=backend,
                    ),
                    mock.patch.object(MoeLoraRunner, "from_layer", return_value=runner),
                    mock.patch(
                        "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
                        return_value=False,
                    ),
                ):
                    layer = FusedMoEWithLoRA(
                        base,
                        types.SimpleNamespace(
                            name="triton_v2", lora_workspace=object()
                        ),
                        max_lora_rank=16,
                    )
                    error = _unsupported_derived_weight_cache_error(layer)
                if has_cache:
                    self.assertIn("Online weight updates are not supported", error)
                else:
                    self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
