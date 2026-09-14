"""Compare NVFP4 MLP fusion selection with the same unfused computation."""

import unittest
from unittest import mock

import torch
from flashinfer import fp4_quantize

from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.models.qwen3_5 import _maybe_enable_silu_fp4_quant_fusion
from sglang.srt.runtime_context import get_context, get_flags, get_platform
from sglang.srt.utils import get_device_sm
from sglang.srt.utils.common import calc_diff
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist, load_linear_weights
from sglang.test.quant_ref_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
    quantize_nvfp4_shard,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class Nvfp4MlpFusionCases:
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    @mock.patch.dict("os.environ", {"SGLANG_DISABLE_SILU_FP4_QUANT_FUSION": "0"})
    @mock.patch.object(
        fp4_utils, "FP4_GEMM_RUNNER_BACKEND", Fp4GemmRunnerBackend.FLASHINFER_CUTLASS
    )
    @torch.inference_mode()
    def _check_output(self, is_awq, gate_channel_scale, down_channel_scale):
        torch.manual_seed(8107)
        hidden_size, intermediate_size = 128, 256
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            is_awq=is_awq,
            exclude_modules=["model.layers.0.mlp.experts"],
            packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
        )
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            candidate = self._build_mlp(cfg, hidden_size, intermediate_size).cuda()
            unfused = self._build_mlp(cfg, hidden_size, intermediate_size).cuda()
        finally:
            torch.set_default_dtype(previous_dtype)

        # Disable only the reference copy's fusion, before weight preparation:
        # DeepSeek's fused layout frees the ordinary gate/up weight storage.
        unfused._enable_silu_fp4_quant_fusion = False
        unfused._enable_nvfp4_gemm_swiglu_fusion = False
        unfused.gate_up_proj._interleave_for_swiglu_fusion = False
        unfused.down_proj._accepts_prequantized_fp4 = False

        gate_scale = torch.ones(hidden_size, device="cuda", dtype=torch.bfloat16)
        down_scale = torch.ones(intermediate_size, device="cuda", dtype=torch.bfloat16)
        if gate_channel_scale:
            gate_scale = torch.linspace(
                0.25, 4, hidden_size, device="cuda", dtype=torch.bfloat16
            )
        if down_channel_scale:
            down_scale = torch.linspace(
                0.25, 4, intermediate_size, device="cuda", dtype=torch.bfloat16
            )

        # Both copies load exactly the same quantized checkpoint weights.
        # Compensate each weight's input channels for its activation prescale.
        shards = (
            torch.randn(
                2, intermediate_size, hidden_size, device="cuda", dtype=torch.bfloat16
            )
            / 10
            / gate_scale
        )
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / shards.abs().max().float()
        for shard_id, weight in enumerate(shards):
            packed, scales, gs, _ = quantize_nvfp4_shard(weight, gs=global_scale)
            for mlp in (candidate, unfused):
                load_linear_weights(
                    mlp.gate_up_proj,
                    shard_id=shard_id,
                    weight=packed,
                    weight_scale=scales,
                    weight_scale_2=1 / gs,
                    input_scale=torch.tensor(1 / 256, device="cuda"),
                )

        weight = (
            torch.randn(
                hidden_size, intermediate_size, device="cuda", dtype=torch.bfloat16
            )
            / 10
        )
        packed, scales, gs, weight_ref = quantize_nvfp4_shard(weight / down_scale)
        for mlp in (candidate, unfused):
            load_linear_weights(
                mlp.down_proj,
                weight=packed,
                weight_scale=scales,
                weight_scale_2=1 / gs,
                input_scale=torch.tensor(1 / 256, device="cuda"),
            )
            if is_awq:
                # These are per-input-channel vectors, shared by gate/up shards.
                default_weight_loader(mlp.gate_up_proj.pre_quant_scale, gate_scale)
                default_weight_loader(mlp.down_proj.pre_quant_scale, down_scale)
            for layer in (mlp.gate_up_proj, mlp.down_proj):
                layer.quant_method.process_weights_after_loading(layer)

        for num_tokens in (1, 33):
            with self.subTest(num_tokens=num_tokens):
                x = (
                    torch.randn(
                        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
                    )
                    / 10
                )
                actual = candidate(x).float()
                expected = unfused(x).float()

                # Retain a separate down-projection oracle so parity alone
                # cannot hide both paths omitting the same prescale.
                gate_up, _ = unfused.gate_up_proj(x)
                gate, up = gate_up.float().chunk(2, dim=-1)
                activation = (torch.nn.functional.silu(gate) * up).bfloat16()
                q, sf = fp4_quantize(
                    activation * down_scale, unfused.down_proj.input_scale_inv
                )
                reference = (
                    dequantize_nvfp4_to_dtype(
                        q, sf, unfused.down_proj.input_scale_inv, torch.float32
                    )
                    @ weight_ref.T
                )
                oracle_error = (expected - reference).norm() / reference.norm()
                self.assertLess(oracle_error.item(), 0.03)
                self._assert_fusion_close(actual, expected)

    def _assert_fusion_close(self, actual, expected):
        relative_error = (actual - expected).norm() / expected.norm()
        self.assertLess(relative_error.item(), 0.03)

    def test_awq_channel_prescale_output(self):
        self._check_output(True, False, True)

    def test_awq_gate_prescale_output(self):
        self._check_output(True, True, False)

    def test_awq_both_prescales_output(self):
        self._check_output(True, True, True)

    def test_awq_identity_prescale_output(self):
        self._check_output(True, False, False)

    def test_plain_nvfp4_output(self):
        self._check_output(False, False, False)


@unittest.skipIf(get_device_sm() < 100, "NVFP4 fusion requires Blackwell")
class TestQwenMlpNvfp4Fusion(Nvfp4MlpFusionCases, CustomTestCase):
    def _build_mlp(self, cfg, hidden_size, intermediate_size):
        mlp = Qwen2MoeMLP(
            hidden_size,
            intermediate_size,
            "silu",
            quant_config=cfg,
            prefix="model.layers.0.mlp",
            tp_rank=0,
            tp_size=1,
        )
        _maybe_enable_silu_fp4_quant_fusion(mlp)
        return mlp


@unittest.skipUnless(get_platform().is_sm100, "DeepSeek's fused kernel requires SM100")
class TestDeepseekSharedExpertNvfp4Fusion(Nvfp4MlpFusionCases, CustomTestCase):
    def _assert_fusion_close(self, actual, expected):
        # The fused epilogue keeps FP32 accumulators through SwiGLU and FP4
        # quantization; the unfused path materializes BF16 intermediates.
        # Match the metric and bound used for this same kernel in
        # test_flux2_fused_nvfp4_swiglu_quant_matches_unfused.
        self.assertLess(calc_diff(actual, expected).item(), 0.02)

    def _build_mlp(self, cfg, hidden_size, intermediate_size):
        from transformers import DeepseekV3Config

        from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        config = DeepseekV3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_intermediate_size=intermediate_size,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            num_hidden_layers=1,
            num_attention_heads=4,
            topk_method="noaux_tc",
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
        )
        # Keep routed experts unquantized and the shared expert separate.
        # Construct the real owning module so its fusion selection is exercised.
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
                disable_shared_experts_fusion=True,
            ),
        ):
            moe = DeepseekV2MoE(
                config, 0, quant_config=cfg, prefix="model.layers.0.mlp"
            )
        return moe.shared_experts


if __name__ == "__main__":
    unittest.main()
