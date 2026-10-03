import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch

import sglang.srt.layers.moe.moe_runner.triton as triton_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, PermuteMethodPool
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferCombineInput,
    FlashinferDispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _flashinfer_output(topk_ids: torch.Tensor) -> FlashinferDispatchOutput:
    return FlashinferDispatchOutput(
        hidden_states=torch.zeros(topk_ids.shape[0], 8, dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
            topk_ids=topk_ids,
            router_logits=None,
        ),
    )


class TestFlashInferTritonDispatch(CustomTestCase):
    def test_runner_paths_are_registered(self):
        self.assertIn(("flashinfer", "triton"), PermuteMethodPool._pre_permute_methods)
        self.assertIn(("triton", "flashinfer"), PermuteMethodPool._post_permute_methods)

    def test_global_ids_are_mapped_and_invalid_routes_are_dropped(self):
        config = MoeRunnerConfig(num_experts=8, num_local_experts=4)
        topk_ids = torch.tensor([[0, 4, 7, 8], [3, 5, -1, 2]], dtype=torch.int32)
        parallel = SimpleNamespace(moe_ep_size=2, moe_ep_rank=1)

        with patch("sglang.srt.runtime_context.get_parallel", return_value=parallel):
            mapped = triton_runner._map_flashinfer_expert_ids_to_local(topk_ids, config)

        torch.testing.assert_close(
            mapped,
            torch.tensor([[-1, 0, 3, -1], [-1, 1, -1, -1]], dtype=torch.int32),
        )

    def test_partition_contract_rejects_non_contiguous_layout(self):
        config = MoeRunnerConfig(num_experts=10, num_local_experts=4)
        parallel = SimpleNamespace(moe_ep_size=2, moe_ep_rank=0)

        with patch(
            "sglang.srt.runtime_context.get_parallel", return_value=parallel
        ), self.assertRaisesRegex(
            NotImplementedError, "equal contiguous expert partition"
        ):
            triton_runner._map_flashinfer_expert_ids_to_local(
                torch.zeros(1, 1, dtype=torch.int32), config
            )

    def test_fused_shared_experts_are_rejected(self):
        config = MoeRunnerConfig(
            num_experts=8, num_local_experts=4, num_fused_shared_experts=1
        )
        parallel = SimpleNamespace(moe_ep_size=2, moe_ep_rank=0)

        with patch(
            "sglang.srt.runtime_context.get_parallel", return_value=parallel
        ), self.assertRaisesRegex(NotImplementedError, "fused shared experts"):
            triton_runner._map_flashinfer_expert_ids_to_local(
                torch.zeros(1, 1, dtype=torch.int32), config
            )

    def test_decode_output_uses_standard_triton_alignment(self):
        config = MoeRunnerConfig(num_experts=8, num_local_experts=4)
        dispatch_output = _flashinfer_output(torch.tensor([[4, 7]], dtype=torch.int32))
        expected_input = object()
        parallel = SimpleNamespace(moe_ep_size=2, moe_ep_rank=1)

        with patch(
            "sglang.srt.runtime_context.get_parallel", return_value=parallel
        ), patch.object(
            triton_runner,
            "pre_permute_standard_to_triton",
            return_value=expected_input,
        ) as standard_adapter:
            result = triton_runner.pre_permute_flashinfer_to_triton(
                dispatch_output,
                SimpleNamespace(),
                config,
                {},
            )

        self.assertIs(result, expected_input)
        standard_output = standard_adapter.call_args.args[0]
        self.assertIsInstance(standard_output, StandardDispatchOutput)
        torch.testing.assert_close(
            standard_output.topk_output.topk_ids,
            torch.tensor([[0, 3]], dtype=torch.int32),
        )

    def test_prefill_standard_output_keeps_existing_path(self):
        dispatch_output = StandardDispatchOutput(
            hidden_states=torch.zeros(1, 8, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.ones(1, 1),
                topk_ids=torch.zeros(1, 1, dtype=torch.int32),
                router_logits=None,
            ),
        )
        expected_input = object()

        with patch.object(
            triton_runner,
            "pre_permute_standard_to_triton",
            return_value=expected_input,
        ) as standard_adapter:
            result = triton_runner.pre_permute_flashinfer_to_triton(
                dispatch_output,
                SimpleNamespace(),
                MoeRunnerConfig(),
                {},
            )

        self.assertIs(result, expected_input)
        standard_adapter.assert_called_once_with(dispatch_output, ANY, ANY, {})

    def test_scaled_flashinfer_payload_is_not_silently_ignored(self):
        dispatch_output = _flashinfer_output(
            torch.zeros(1, 1, dtype=torch.int32)
        )._replace(hidden_states_scale=torch.ones(1, 1))

        with self.assertRaisesRegex(NotImplementedError, "scaled payloads"):
            triton_runner.pre_permute_flashinfer_to_triton(
                dispatch_output,
                SimpleNamespace(),
                MoeRunnerConfig(num_experts=1, num_local_experts=1),
                {},
            )

    def test_triton_output_is_wrapped_for_flashinfer_combine(self):
        output = triton_runner.TritonRunnerOutput(
            hidden_states=torch.zeros(2, 8, dtype=torch.bfloat16)
        )
        result = triton_runner.post_permute_triton_to_flashinfer(
            output, SimpleNamespace(), MoeRunnerConfig(), {}
        )

        self.assertIsInstance(result, FlashinferCombineInput)
        self.assertIs(result.hidden_states, output.hidden_states)


if __name__ == "__main__":
    unittest.main()
