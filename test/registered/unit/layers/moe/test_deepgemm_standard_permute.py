"""CPU guards for the standard-dispatch DeepGEMM permute contract."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_gemm_runner
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    pre_permute_standard_to_deep_gemm,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.test.test_utils import CustomTestCase


class TestDeepGemmStandardPermute(CustomTestCase):
    def test_packed_scale_buffer_layout(self):
        row_major = deep_gemm_runner._allocate_packed_scale_buffer(
            384,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            column_major=False,
        )
        column_major = deep_gemm_runner._allocate_packed_scale_buffer(
            384,
            2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            column_major=True,
        )

        self.assertEqual(row_major.shape, (384, 2))
        self.assertEqual(row_major.stride(), (2, 1))
        self.assertEqual(column_major.shape, (384, 2))
        self.assertEqual(column_major.stride(), (1, 384))
        deep_gemm_runner._assert_packed_ue8m0_scale_layout(
            column_major.to(torch.int32),
            rows=384,
            width=2,
        )

    def _run_pre_permute(self, inplace: bool):
        hidden_states = torch.zeros((3, 128), dtype=torch.bfloat16)
        topk_weights = torch.ones((3, 2), dtype=torch.float32)
        # This is the standard-dispatch TEP representation: global experts
        # owned by another EP rank have already been mapped to -1.
        topk_ids = torch.tensor([[0, -1], [1, -1], [0, -1]], dtype=torch.int32)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=(topk_weights, topk_ids, None),
        )
        quant_info = SimpleNamespace(
            w13_weight=torch.zeros((2, 64, 128), dtype=torch.bfloat16),
            block_shape=[128, 128],
            use_mxfp8=False,
        )
        runner_config = SimpleNamespace(
            num_experts=2,
            num_local_experts=2,
            top_k=2,
            inplace=inplace,
        )
        expert_counts = torch.tensor([2, 1], dtype=torch.int32)
        unused_masked_dst = torch.zeros((6,), dtype=torch.int32)
        running_state = {}

        with (
            patch(
                "sglang.kernels.ops.moe.ep_moe_kernels." "fused_moe_dispatch_index",
                return_value=(expert_counts, unused_masked_dst),
            ) as dispatch_index,
            patch("sglang.kernels.ops.moe.ep_moe_kernels.ep_scatter") as scatter,
            patch.object(deep_gemm_runner, "dispose_tensor") as dispose,
        ):
            result = pre_permute_standard_to_deep_gemm(
                dispatch_output,
                quant_info,
                runner_config,
                running_state=running_state,
            )
        return (
            hidden_states,
            topk_ids,
            dispose,
            dispatch_index,
            scatter,
            result,
            running_state,
        )

    def test_teps_nonlocal_experts_keep_graph_static_compact_capacity(self):
        (
            hidden_states,
            topk_ids,
            dispose,
            dispatch_index,
            scatter,
            result,
            running_state,
        ) = self._run_pre_permute(inplace=False)

        self.assertTrue(
            all(call.args[0] is not hidden_states for call in dispose.call_args_list)
        )
        dispatch_index.assert_called_once_with(topk_ids, 2, 1)
        self.assertIs(scatter.call_args.args[2], topk_ids)
        self.assertEqual(scatter.call_args.args[3].tolist(), [128, 256])
        self.assertEqual(scatter.call_args.args[7].shape, (384,))
        self.assertEqual(scatter.call_args.args[8].shape, topk_ids.shape)
        self.assertFalse(result.use_masked_gemm)
        self.assertEqual(result.hidden_states.shape, (384, 128))
        self.assertEqual(result.m_indices.shape, (384,))
        self.assertEqual(running_state["all_tokens"], 384)
        self.assertIs(running_state["src2dst"], scatter.call_args.args[8])

    def test_disposes_input_only_when_runner_is_inplace(self):
        hidden_states, _, dispose, _, _, _, _ = self._run_pre_permute(inplace=True)

        self.assertTrue(
            any(call.args[0] is hidden_states for call in dispose.call_args_list)
        )

    def test_small_local_expert_ep_uses_masked_layout_and_global_expected_m(self):
        hidden_states = torch.zeros((3, 128), dtype=torch.bfloat16)
        topk_weights = torch.ones((3, 2), dtype=torch.float32)
        topk_ids = torch.tensor([[0, -1], [1, -1], [0, -1]], dtype=torch.int32)
        dispatch_output = StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=(topk_weights, topk_ids, None),
        )
        quant_info = SimpleNamespace(
            w13_weight=torch.zeros((2, 64, 128), dtype=torch.bfloat16),
            block_shape=[128, 128],
            use_mxfp8=False,
        )
        runner_config = SimpleNamespace(
            num_experts=8,
            num_local_experts=2,
            top_k=2,
            inplace=False,
        )
        masked_m = torch.tensor([2, 1], dtype=torch.int32)
        src2dst = torch.zeros_like(topk_ids, dtype=torch.int32)
        grouped_hidden = torch.zeros((2, 256, 128), dtype=torch.bfloat16)
        grouped_scale = torch.empty((2, 256, 1), dtype=torch.float32)

        with (
            patch(
                "sglang.kernels.ops.moe.ep_moe_kernels." "moe_ep_deepgemm_preprocess",
                return_value=(
                    masked_m,
                    3,
                    src2dst,
                    grouped_hidden,
                    grouped_scale,
                ),
            ) as preprocess,
            patch.object(deep_gemm_runner, "dispose_tensor") as dispose,
        ):
            result = pre_permute_standard_to_deep_gemm(
                dispatch_output,
                quant_info,
                runner_config,
                running_state={},
            )

        preprocess.assert_called_once()
        self.assertTrue(result.use_masked_gemm)
        self.assertIs(result.masked_m, masked_m)
        self.assertEqual(result.expected_m, 1)
        self.assertIs(result.hidden_states, grouped_hidden)
        self.assertFalse(
            any(call.args[0] is hidden_states for call in dispose.call_args_list)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
