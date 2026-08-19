"""Unit tests for LLaDA2 model validation and routing metadata."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.models.llada2 import (
    LLaDA2MoeSparseMoeBlock,
    _get_effective_moe_runner_backend,
    _make_block_routing_triton_output,
    _require_block_routing_ep1,
    _require_block_routing_runner_compatibility,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLLaDA2BlockRoutingValidation(CustomTestCase):
    def test_block_routing_accepts_ep1(self):
        _require_block_routing_ep1(1)

    def test_block_routing_rejects_expert_parallelism(self):
        config = SimpleNamespace(
            num_experts_per_tok=8,
            norm_topk_prob=True,
            hidden_size=1024,
            num_shared_experts=1,
            expert_capacity=48,
        )

        with get_parallel().override(tp_size=4, moe_ep_size=4):
            with self.assertRaisesRegex(
                ValueError,
                r"does not support expert parallelism.*moe_ep_size=4",
            ):
                LLaDA2MoeSparseMoeBlock(layer_id=0, config=config)

    def test_block_routing_uses_five_field_triton_output(self):
        ragged_metadata = object()
        combine_indx = torch.tensor([5, 0, 7, 2], dtype=torch.int32)
        gate_scal = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)

        result = _make_block_routing_triton_output(
            ragged_metadata,
            combine_indx,
            gate_scal,
            n_expts_act=2,
        )

        self.assertIs(result.a_ragged_metadata, ragged_metadata)
        torch.testing.assert_close(
            result.gather_indx,
            torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        )
        self.assertIs(result.scatter_indx, combine_indx)
        self.assertIs(result.gate_scal, gate_scal)
        self.assertEqual(result.n_expts_act, 2)

    @patch("sglang.srt.layers.quantization.modelopt_quant.MoeRunner")
    @patch(
        "sglang.srt.layers.quantization.modelopt_quant.get_device_capability",
        return_value=(10, 0),
    )
    @patch("sglang.srt.layers.quantization.modelopt_quant.is_cuda", return_value=True)
    @patch("sglang.srt.layers.quantization.modelopt_quant.get_moe_runner_backend")
    def test_block_routing_rejects_auto_selected_plain_flashinfer(
        self,
        backend,
        _is_cuda,
        _capability,
        runner,
    ):
        backend.return_value = MoeRunnerBackend.AUTO
        quant_method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        quant_method.create_moe_runner(
            SimpleNamespace(),
            SimpleNamespace(),
        )
        experts = SimpleNamespace(quant_method=quant_method)

        self.assertIs(
            _get_effective_moe_runner_backend(experts),
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        )
        runner.assert_called_once_with(
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            quant_method.moe_runner_config,
        )
        with self.assertRaisesRegex(ValueError, r"does not support.*flashinfer_trtllm"):
            _require_block_routing_runner_compatibility(experts)

    @patch("sglang.srt.models.llada2.get_moe_runner_backend")
    def test_fused_scaling_contract_applies_scale_once(self, backend):
        backend.return_value.is_triton_kernels.return_value = False
        backend.return_value.is_aiter.return_value = False

        block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.routed_scaling_factor = 2.5
        block.experts = SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=True)

        router_logits = torch.zeros((1, 4), dtype=torch.float32)
        topk_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)

        result = block._make_block_topk_output(
            router_logits,
            topk_weights,
            topk_ids,
        )

        torch.testing.assert_close(
            result.topk_weights,
            topk_weights * block.routed_scaling_factor,
        )
        torch.testing.assert_close(topk_weights, torch.tensor([[0.25, 0.75]]))
        self.assertIs(result.topk_ids, topk_ids)
        self.assertIs(result.router_logits, router_logits)

    def test_aiter_block_routing_applies_scale_once(self):
        selected_backend = SimpleNamespace(
            is_triton_kernels=lambda: False,
            is_aiter=lambda: True,
        )
        block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.routed_scaling_factor = 2.5
        block.experts = SimpleNamespace(
            quant_method=SimpleNamespace(_moe_runner_backend=selected_backend),
            should_fuse_routed_scaling_factor_in_topk=False,
        )

        router_logits = torch.zeros((1, 4), dtype=torch.float32)
        topk_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)

        result = block._make_block_topk_output(
            router_logits,
            topk_weights,
            topk_ids,
        )

        torch.testing.assert_close(
            result.topk_weights,
            topk_weights * block.routed_scaling_factor,
        )
        torch.testing.assert_close(topk_weights, torch.tensor([[0.25, 0.75]]))
        self.assertIs(result.topk_ids, topk_ids)
        self.assertIs(result.router_logits, router_logits)


if __name__ == "__main__":
    unittest.main()
