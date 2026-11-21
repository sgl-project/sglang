"""Unit tests for blockwise FP8 SiLU-and-mul + quant fusion gating in Qwen2MoeMLP."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    deepgemm_w8a8_block_fp8_linear_with_fallback,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_mlp(
    *,
    hidden_size: int = 2048,
    intermediate_size: int = 512,
    block_quant: bool = True,
    use_marlin: bool = False,
    use_mxfp8: bool = False,
    block_fp8_as_mxfp8: bool = False,
    use_npu_block_fp8: bool = False,
    weight_block_size: list[int] | None = None,
    scale_fmt: str | None = None,
    linear_fn=deepgemm_w8a8_block_fp8_linear_with_fallback,
) -> Qwen2MoeMLP:
    mlp = object.__new__(Qwen2MoeMLP)
    qm = object.__new__(Fp8LinearMethod)
    qm.block_quant = block_quant
    qm.use_marlin = use_marlin
    qm.use_mxfp8 = use_mxfp8
    qm.block_fp8_as_mxfp8 = block_fp8_as_mxfp8
    qm.use_npu_block_fp8 = use_npu_block_fp8
    qm.weight_block_size = (
        [128, 128] if weight_block_size is None else weight_block_size
    )
    qm.w8a8_block_fp8_linear = linear_fn
    qm.quant_config = (
        Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        if scale_fmt is None
        else SimpleNamespace(scale_fmt=scale_fmt)
    )
    if isinstance(qm.quant_config, Fp8Config) and scale_fmt is not None:
        qm.quant_config.scale_fmt = scale_fmt
    mlp.down_proj = SimpleNamespace(
        quant_method=qm,
        weight=torch.empty((hidden_size, intermediate_size), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.ones(
            (hidden_size // 128, intermediate_size // 128), dtype=torch.float32
        ),
    )
    return mlp


class TestQwen2MoeMlpBlockFp8FusionGating(CustomTestCase):
    def test_eligibility_gating(self):
        gate_up = torch.empty((4, 1024), dtype=torch.bfloat16)

        # Valid SM90 block-FP8 case
        mlp = _make_mlp(intermediate_size=512)
        with (
            patch("sglang.srt.models.qwen2_moe._is_cuda", True),
            patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0",
                False,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils._deepgemm_block_fp8_supported",
                return_value=True,
            ),
        ):
            self.assertTrue(mlp._can_use_silu_block_fp8_quant_fusion(gate_up))

        # Rejects kill switch, deterministic/RL/LoRA/SP modes, batch-invariant,
        # M=0, non-bf16, unsupported widths, base_layer, and MXFP8/Marlin.
        def _run_gate(
            mod,
            tensor,
            *,
            bi=False,
            disable_env=False,
            det=False,
            rl_target=None,
            lora=False,
            sp=False,
        ):
            exec_ctx = SimpleNamespace(
                deterministic=SimpleNamespace(
                    enable_deterministic_inference=det,
                    rl_on_policy_target=rl_target,
                )
            )
            lora_ctx = SimpleNamespace(enable_lora=lora)
            fwd_ctx = SimpleNamespace(sp_active=sp)
            with (
                patch("sglang.srt.models.qwen2_moe._is_cuda", True),
                patch(
                    "sglang.srt.models.qwen2_moe.envs.SGLANG_DISABLE_SILU_BLOCK_FP8_QUANT_FUSION.get",
                    return_value=disable_env,
                ),
                patch(
                    "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                    return_value=bi,
                ),
                patch(
                    "sglang.srt.models.qwen2_moe.get_exec",
                    return_value=exec_ctx,
                ),
                patch(
                    "sglang.srt.runtime_context.get_lora",
                    return_value=lora_ctx,
                ),
                patch(
                    "sglang.srt.runtime_context.get_forward",
                    return_value=fwd_ctx,
                ),
                patch(
                    "sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0",
                    False,
                ),
                patch(
                    "sglang.srt.layers.quantization.fp8_utils._deepgemm_block_fp8_supported",
                    return_value=True,
                ),
            ):
                return mod._can_use_silu_block_fp8_quant_fusion(tensor)

        lora_wrapped = _make_mlp()
        lora_wrapped.down_proj.base_layer = object()

        for desc, mod, tensor, kw in [
            ("kill_switch", _make_mlp(), gate_up, {"disable_env": True}),
            ("batch_invariant", _make_mlp(), gate_up, {"bi": True}),
            ("deterministic", _make_mlp(), gate_up, {"det": True}),
            ("rl_on_policy", _make_mlp(), gate_up, {"rl_target": "ref"}),
            ("lora_enabled", _make_mlp(), gate_up, {"lora": True}),
            ("lora_base_layer", lora_wrapped, gate_up, {}),
            ("sp_active", _make_mlp(), gate_up, {"sp": True}),
            (
                "empty_tokens",
                _make_mlp(),
                torch.empty((0, 1024), dtype=torch.bfloat16),
                {},
            ),
            (
                "fp16_input",
                _make_mlp(),
                torch.empty((4, 1024), dtype=torch.float16),
                {},
            ),
            (
                "width_192",
                _make_mlp(intermediate_size=192),
                torch.empty((4, 384), dtype=torch.bfloat16),
                {},
            ),
            ("mxfp8", _make_mlp(use_mxfp8=True), gate_up, {}),
            ("marlin", _make_mlp(use_marlin=True), gate_up, {}),
            ("block32", _make_mlp(weight_block_size=[32, 32]), gate_up, {}),
        ]:
            with self.subTest(case=desc):
                self.assertFalse(_run_gate(mod, tensor, **kw))

    def test_sm100_ue8m0_requires_four_group_alignment(self):
        # K=512 (4 groups) is eligible on SM100 UE8M0; K=768 (6 groups) falls back.
        for intermediate_size, expected in [(512, True), (768, False), (1024, True)]:
            with self.subTest(intermediate_size=intermediate_size):
                mlp = _make_mlp(intermediate_size=intermediate_size)
                gate_up = torch.empty((4, intermediate_size * 2), dtype=torch.bfloat16)
                with (
                    patch("sglang.srt.models.qwen2_moe._is_cuda", True),
                    patch(
                        "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                        return_value=False,
                    ),
                    patch(
                        "sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0",
                        True,
                    ),
                    patch(
                        "sglang.srt.layers.quantization.fp8_utils._deepgemm_block_fp8_supported",
                        return_value=True,
                    ),
                ):
                    self.assertEqual(
                        mlp._can_use_silu_block_fp8_quant_fusion(gate_up), expected
                    )

    def test_scale_layout_selection(self):
        gate_up = torch.empty((8, 1024), dtype=torch.bfloat16)
        cases = [
            # (use_gemv, deepgemm_ok, dg_ue8m0, act_ue8m0) -> (exp_ue8m0, exp_transposed)
            (True, True, False, False, False, False),
            (False, True, False, False, False, True),
            (False, False, False, False, False, False),
            (False, True, True, False, True, True),
        ]
        for (
            use_gemv,
            dg_ok,
            dg_ue8m0,
            act_ue8m0,
            exp_ue8m0,
            exp_transposed,
        ) in cases:
            with self.subTest(
                use_gemv=use_gemv,
                dg_ok=dg_ok,
                dg_ue8m0=dg_ue8m0,
                act_ue8m0=act_ue8m0,
            ):
                mlp = _make_mlp(intermediate_size=512)
                mlp._block_fp8_down_act_ue8m0 = act_ue8m0
                mlp._block_fp8_down_deepgemm_ok = dg_ok
                captured = {}

                def _fake_quant(**kwargs):
                    captured.update(kwargs)

                with (
                    patch(
                        "sglang.srt.layers.quantization.fp8_utils._use_fp8_block_gemv",
                        return_value=use_gemv,
                    ),
                    patch(
                        "sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0",
                        dg_ue8m0,
                    ),
                    patch(
                        "sglang.kernels.ops.activation.silu_and_mul_clamp.silu_and_mul_contig_post_quant",
                        side_effect=_fake_quant,
                    ),
                ):
                    mlp._silu_and_mul_block_fp8_quant(gate_up)

                self.assertEqual(captured["scale_ue8m0"], exp_ue8m0)
                self.assertEqual(captured["transposed"], exp_transposed)
                self.assertEqual(captured["quant_group_size"], 128)
                self.assertTrue(captured["round_bf16"])


if __name__ == "__main__":
    unittest.main()
