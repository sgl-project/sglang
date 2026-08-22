"""Unit tests for LLaDA2 model validation and routing metadata."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
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
        block.layer_id = 0
        block.routed_scaling_factor = 2.5
        block.topk = SimpleNamespace(
            topk_config=SimpleNamespace(allow_routed_experts_capture=False)
        )
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
        with (
            patch(
                "sglang.srt.layers.quantization.unquant._use_aiter",
                True,
            ),
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            patch(
                "sglang.srt.layers.quantization.unquant.get_moe_a2a_backend"
            ) as get_a2a_backend,
            patch(
                "sglang.srt.layers.quantization.unquant.MoeRunner",
                side_effect=lambda backend, config: SimpleNamespace(
                    runner_backend=backend
                ),
            ),
        ):
            get_a2a_backend.return_value.supports_aiter.return_value = True
            quant_method = UnquantizedFusedMoEMethod()
            quant_method.create_moe_runner(
                SimpleNamespace(intermediate_size_per_partition=256),
                SimpleNamespace(),
            )

        self.assertIs(
            quant_method.runner.runner_backend,
            MoeRunnerBackend.TRITON,
        )
        self.assertIs(
            quant_method._aiter_runner.runner_backend,
            MoeRunnerBackend.AITER,
        )

        block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.layer_id = 0
        block.routed_scaling_factor = 2.5
        block.topk = SimpleNamespace(
            topk_config=SimpleNamespace(allow_routed_experts_capture=False)
        )
        block.experts = SimpleNamespace(
            quant_method=quant_method,
            runner=quant_method.runner,
            should_fuse_routed_scaling_factor_in_topk=False,
        )

        self.assertIs(
            _get_effective_moe_runner_backend(block.experts),
            MoeRunnerBackend.AITER,
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

    @patch("sglang.srt.models.llada2.get_moe_runner_backend")
    @patch("sglang.srt.models.llada2.get_global_expert_distribution_recorder")
    @patch("sglang.srt.models.llada2.capture_routed_experts_if_allowed")
    def test_block_routing_runs_precomputed_route_hooks(
        self,
        capture_routed_experts,
        get_recorder,
        backend,
    ):
        backend.return_value.is_triton_kernels.return_value = False
        backend.return_value.is_aiter.return_value = False

        block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.layer_id = 7
        block.routed_scaling_factor = 1.0
        block.topk = SimpleNamespace(topk_config=object())
        block.experts = SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=False)
        topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)

        block._make_block_topk_output(
            torch.zeros((1, 4)),
            torch.tensor([[0.4, 0.6]]),
            topk_ids,
        )

        capture_routed_experts.assert_called_once_with(
            block.topk.topk_config, block.layer_id, topk_ids
        )
        get_recorder.return_value.on_select_experts.assert_called_once_with(
            topk_ids=topk_ids
        )


class TestLLaDA2DllmBlockSizeValidation(CustomTestCase):
    @staticmethod
    def _server_args(algorithm_config=None):
        return SimpleNamespace(
            dllm_algorithm="LowConfidence",
            dllm_algorithm_config=algorithm_config,
            max_running_requests=1,
            model_path="unused",
            revision=None,
            dllm_fdfo=False,
        )

    @staticmethod
    def _model_config():
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LLaDA2MoeModelLM"],
                block_size=32,
                expert_capacity=48,
            )
        )

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_block_routing_accepts_matching_dllm_block_size(self, from_server_args):
        from_server_args.return_value = self._model_config()

        config = DllmConfig.from_server_args(self._server_args())

        self.assertEqual(config.block_size, 32)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_block_routing_rejects_mismatched_dllm_block_size(self, from_server_args):
        from_server_args.return_value = self._model_config()
        with TemporaryDirectory() as tmpdir:
            config_path = f"{tmpdir}/dllm.yaml"
            with open(config_path, "w") as config_file:
                config_file.write("block_size: 16\n")

            with self.assertRaisesRegex(
                ValueError,
                r"requires the dLLM block size to match.*\(32\), got 16",
            ):
                DllmConfig.from_server_args(self._server_args(config_path))


if __name__ == "__main__":
    unittest.main()
