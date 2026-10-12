"""Unit tests for LLaDA2 model validation and routing metadata."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.models.llada2 import (
    LLaDA2MoeGate,
    LLaDA2MoeSparseMoeBlock,
    _get_effective_moe_runner_backend,
    _make_block_routing_triton_output,
    _prepare_llada2_language_weights,
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

    def test_block_routing_rejects_uniform_expert_simulation(self):
        with (
            get_parallel().override(tp_size=1, moe_ep_size=1),
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(True),
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(False),
            self.assertRaisesRegex(
                ValueError, r"does not support SGLANG_SIMULATE_UNIFORM_EXPERTS"
            ),
        ):
            LLaDA2MoeSparseMoeBlock(
                layer_id=0,
                config=SimpleNamespace(
                    num_experts_per_tok=8,
                    norm_topk_prob=True,
                    hidden_size=1024,
                    num_shared_experts=1,
                    expert_capacity=48,
                ),
            )

    def test_block_routing_rejects_round_robin_expert_simulation(self):
        with (
            get_parallel().override(tp_size=1, moe_ep_size=1),
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(False),
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(True),
            self.assertRaisesRegex(
                ValueError, r"does not support SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS"
            ),
        ):
            LLaDA2MoeSparseMoeBlock(
                layer_id=0,
                config=SimpleNamespace(
                    num_experts_per_tok=8,
                    norm_topk_prob=True,
                    hidden_size=1024,
                    num_shared_experts=1,
                    expert_capacity=48,
                ),
            )

    def test_block_routing_rejects_conflicting_expert_simulations(self):
        with (
            get_parallel().override(tp_size=1, moe_ep_size=1),
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(True),
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(True),
            self.assertRaisesRegex(ValueError, r"mutually exclusive"),
        ):
            LLaDA2MoeSparseMoeBlock(
                layer_id=0,
                config=SimpleNamespace(
                    num_experts_per_tok=8,
                    norm_topk_prob=True,
                    hidden_size=1024,
                    num_shared_experts=1,
                    expert_capacity=48,
                ),
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


class TestLLaDA2LanguageModel(CustomTestCase):
    def test_fp32_router_logits_are_not_rounded_to_the_activation_dtype(self):
        """An fp32 router must score bf16 activations in fp32, as the reference does."""
        config = SimpleNamespace(
            num_experts=4, hidden_size=8, moe_router_enable_expert_bias=True
        )
        gate = LLaDA2MoeGate(config, params_dtype=torch.float32)
        with torch.no_grad():
            gate.weight.copy_(torch.arange(32).reshape(4, 8) / 37)
        hidden = (torch.arange(24).reshape(3, 8) / 19).to(torch.bfloat16)

        logits = gate(hidden)

        self.assertEqual(logits.dtype, torch.float32)
        expected = F.linear(hidden.float(), gate.weight)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_language_checkpoint_layout_is_normalized(self):
        """Prefixed names and fused expert tensors load as the native layout."""
        fused = torch.arange(24).reshape(2, 3, 4)
        lm_head = torch.randn(4, 3)

        expanded = list(
            _prepare_llada2_language_weights(
                [
                    ("model.language_model.layers.1.mlp.experts.gate_proj", fused),
                    ("model.lm_head.weight", lm_head),
                ],
                num_experts=2,
            )
        )

        self.assertEqual(
            [name for name, _ in expanded],
            [
                "model.layers.1.mlp.experts.0.gate_proj.weight",
                "model.layers.1.mlp.experts.1.gate_proj.weight",
                "lm_head.weight",
            ],
        )
        torch.testing.assert_close(expanded[0][1], fused[0])
        torch.testing.assert_close(expanded[1][1], fused[1])
        torch.testing.assert_close(expanded[2][1], lm_head)
        with self.assertRaisesRegex(ValueError, "expected first dimension 2"):
            list(
                _prepare_llada2_language_weights(
                    [("model.layers.1.mlp.experts.down_proj", torch.empty(3, 4, 5))],
                    num_experts=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
