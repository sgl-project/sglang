"""CPU contract tests for the FlashInfer AlphaMoE SGLang adapter.

The GPU kernels are covered by FlashInfer's own SM100/SM103 tests.  These tests
pin SGLang's load-time layout and strict W8A8 admission contract without mocks
or a device dependency.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.moe.moe_runner.base import FusedOpPool, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_alphamoe import (
    deinterleave_alphamoe_gated_rows,
    fused_experts_none_to_flashinfer_alphamoe,
    restore_alphamoe_fp8_weights_for_loading,
    validate_alphamoe_runner_contract,
    validate_alphamoe_w8a8_weights,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.test_utils import CustomTestCase


def _valid_tensors(
    *, num_experts: int = 2, hidden_size: int = 128, intermediate_size: int = 128
):
    return (
        torch.empty(
            (num_experts, 2 * intermediate_size, hidden_size),
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        torch.empty(
            (num_experts, hidden_size, intermediate_size),
            dtype=torch.float8_e4m3fn,
            device="meta",
        ),
        torch.empty(
            (num_experts, 2 * intermediate_size // 128, hidden_size // 128),
            dtype=torch.float32,
            device="meta",
        ),
        torch.empty(
            (num_experts, hidden_size // 128, intermediate_size // 128),
            dtype=torch.float32,
            device="meta",
        ),
    )


class TestFlashInferAlphaMoeContract(CustomTestCase):
    def _valid_runner_kwargs(self):
        return {
            "tp_size": 4,
            "ep_size": 1,
            "a2a_is_none": True,
            "num_fused_shared_experts": 0,
            "with_bias": False,
            "is_gated": True,
            "activation": "silu",
            "apply_router_weight_on_input": False,
            "no_combine": False,
            "gemm1_alpha": None,
            "gemm1_clamp_limit": None,
            "swiglu_limit": None,
            "params_dtype": torch.bfloat16,
            "top_k": 10,
            "num_experts": 512,
        }

    def test_backend_and_fused_entry_are_registered(self):
        self.assertEqual(
            MoeRunnerBackend("flashinfer_alphamoe"),
            MoeRunnerBackend.FLASHINFER_ALPHAMOE,
        )
        self.assertIsNotNone(FusedOpPool.get_fused_func("none", "flashinfer_alphamoe"))

    def test_weight_and_scale_deinterleave_restore_checkpoint_order(self):
        # Weight rows arrive as 8 gate, 8 up, 8 gate, 8 up.
        interleaved_weight_rows = list(range(0, 8))
        interleaved_weight_rows += list(range(16, 24))
        interleaved_weight_rows += list(range(8, 16))
        interleaved_weight_rows += list(range(24, 32))
        weight = torch.tensor(interleaved_weight_rows).reshape(1, 32, 1)

        # One scale row covers 128 weight rows, so scale rows alternate singly.
        scale = torch.tensor([0, 2, 1, 3]).reshape(1, 4, 1)
        restore_alphamoe_fp8_weights_for_loading(weight, scale)

        torch.testing.assert_close(weight.flatten(), torch.arange(32))
        torch.testing.assert_close(scale.flatten(), torch.arange(4))

    def test_empty_batch_skips_bypassed_router_and_flashinfer(self):
        # Qwen's idle-rank path deliberately supplies StandardTopKOutput, not
        # BypassedTopKOutput. M=0 must return before validating or importing FI.
        hidden_states = torch.empty((0, 128), dtype=torch.bfloat16)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.empty((0, 2), dtype=torch.float32),
                topk_ids=torch.empty((0, 2), dtype=torch.int32),
                router_logits=torch.empty((0, 2), dtype=torch.float32),
            ),
        )

        result = fused_experts_none_to_flashinfer_alphamoe(
            dispatch_output, object(), MoeRunnerConfig()
        )
        self.assertEqual(result.hidden_states.shape, (0, 128))
        self.assertEqual(result.hidden_states.dtype, torch.bfloat16)

    def test_runner_contract_rejects_no_combine(self):
        kwargs = self._valid_runner_kwargs()
        kwargs["no_combine"] = True
        with self.assertRaisesRegex(ValueError, "combined"):
            validate_alphamoe_runner_contract(**kwargs)

    def test_runner_contract_rejects_fused_shared_expert(self):
        kwargs = self._valid_runner_kwargs()
        kwargs["num_fused_shared_experts"] = 1
        with self.assertRaisesRegex(ValueError, "shared-expert fusion"):
            validate_alphamoe_runner_contract(**kwargs)

    def test_runner_contract_rejects_non_tp4(self):
        kwargs = self._valid_runner_kwargs()
        kwargs["tp_size"] = 2
        with self.assertRaisesRegex(ValueError, "moe_tp_size=4"):
            validate_alphamoe_runner_contract(**kwargs)

    def test_deinterleave_rejects_nonrepresentable_row_count(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            deinterleave_alphamoe_gated_rows(torch.empty((1, 24, 1)), rows_per_chunk=8)

    def test_target_tp4_shapes_are_admitted(self):
        # Qwen3-Next-80B-A3B FP8: E=512, hidden=2048, inter=512/TP4=128.
        tensors = _valid_tensors(
            num_experts=512, hidden_size=2048, intermediate_size=128
        )
        result = validate_alphamoe_w8a8_weights(
            *tensors,
            block_shape=[128, 128],
            top_k=10,
            use_mxfp8=False,
            is_fp4_expert=False,
        )
        self.assertEqual(result, (512, 2048, 128))

    def test_known_long_k_geometry_is_not_admitted(self):
        # The exported W8A8 kernel's K=7168 source coordinate fails the strict
        # FP8 error gate; keep model-facing dispatch on the verified Qwen TP4
        # geometry until that kernel issue is fixed and independently retested.
        tensors = _valid_tensors(
            num_experts=512, hidden_size=7168, intermediate_size=128
        )
        with self.assertRaisesRegex(ValueError, "long-K dispatch remains disabled"):
            validate_alphamoe_w8a8_weights(
                *tensors,
                block_shape=[128, 128],
                top_k=10,
                use_mxfp8=False,
                is_fp4_expert=False,
            )

    def test_non_qwen_routing_geometry_is_not_admitted(self):
        kwargs = self._valid_runner_kwargs()
        kwargs["top_k"] = 8
        with self.assertRaisesRegex(ValueError, "Qwen3-Next TP4 routing geometry"):
            validate_alphamoe_runner_contract(**kwargs)

    def test_fused_shared_expert_slot_is_rejected_at_513(self):
        tensors = _valid_tensors(num_experts=513)
        with self.assertRaisesRegex(ValueError, "at most 512 experts"):
            validate_alphamoe_w8a8_weights(
                *tensors,
                block_shape=[128, 128],
                top_k=10,
                use_mxfp8=False,
                is_fp4_expert=False,
            )

    def test_nvfp4_and_mxfp8_are_explicitly_rejected(self):
        tensors = _valid_tensors()
        for flags in (
            {"use_mxfp8": True, "is_fp4_expert": False},
            {"use_mxfp8": False, "is_fp4_expert": True},
        ):
            with self.subTest(**flags), self.assertRaisesRegex(
                ValueError, "NVFP4/ModelOpt"
            ):
                validate_alphamoe_w8a8_weights(
                    *tensors,
                    block_shape=[128, 128],
                    top_k=2,
                    **flags,
                )

    def test_non_fp32_block_scales_are_rejected(self):
        w13, w2, w13_scale, w2_scale = _valid_tensors()
        w13_scale = torch.empty_like(w13_scale, dtype=torch.float16)
        with self.assertRaisesRegex(TypeError, "FP32"):
            validate_alphamoe_w8a8_weights(
                w13,
                w2,
                w13_scale,
                w2_scale,
                block_shape=[128, 128],
                top_k=2,
                use_mxfp8=False,
                is_fp4_expert=False,
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
