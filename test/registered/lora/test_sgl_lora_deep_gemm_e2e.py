"""Synthetic end-to-end test for typed SGL LoRA on masked BF16 DeepGEMM."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.sgl_lora.bf16 import Bf16MoeLaunchConfig
from sglang.srt.lora.sgl_lora.deep_gemm_bf16 import DeepGemmBf16LoRAHookBuilder
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")


class TestSglLoraDeepGemmEndToEnd(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest(
                "masked BF16 DeepGEMM requires an SM90-or-newer NVIDIA GPU"
            )
        try:
            import deep_gemm  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(
                "the DeepGEMM Python package is required"
            ) from error

        torch.cuda.set_device(0)
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        destroy_distributed_environment()

    @staticmethod
    def _bf16_linear(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input_.float(), weight.float().T).to(torch.bfloat16)

    def _reference(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        token_lora_mapping: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        pair_outputs = []
        for token_id in range(hidden_states.shape[0]):
            token_outputs = []
            adapter_id = int(token_lora_mapping[token_id].item())
            for slot in range(topk_ids.shape[1]):
                expert_id = int(topk_ids[token_id, slot].item())
                gate_up = self._bf16_linear(
                    hidden_states[token_id], self.w13_weight[expert_id]
                )
                if adapter_id >= 0:
                    gate_a = self._bf16_linear(
                        hidden_states[token_id],
                        self.gate_a[adapter_id, expert_id],
                    )
                    gate_delta = torch.cat(
                        (
                            self._bf16_linear(
                                gate_a[: self.rank],
                                self.gate_b[adapter_id, expert_id, : self.intermediate],
                            ),
                            self._bf16_linear(
                                gate_a[self.rank :],
                                self.gate_b[adapter_id, expert_id, self.intermediate :],
                            ),
                        )
                    )
                    gate_up = (gate_up.float() + gate_delta.float()).to(torch.bfloat16)

                gate = gate_up[: self.intermediate].float()
                up = gate_up[self.intermediate :].float()
                activation = (up * gate / (1 + torch.exp(-gate))).to(torch.bfloat16)
                down = self._bf16_linear(activation, self.w2_weight[expert_id])
                if adapter_id >= 0:
                    down_a = self._bf16_linear(
                        activation, self.down_a[adapter_id, expert_id]
                    )
                    down_delta = self._bf16_linear(
                        down_a, self.down_b[adapter_id, expert_id]
                    )
                    down = (down.float() + down_delta.float()).to(torch.bfloat16)
                token_outputs.append(down)
            pair_outputs.append(token_outputs)

        pairs = torch.stack(
            [torch.stack(token_outputs) for token_outputs in pair_outputs]
        )
        combined = torch.zeros(
            hidden_states.shape,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for slot in range(topk_ids.shape[1]):
            combined += pairs[:, slot].float() * topk_weights[:, slot, None]
        return (combined * self.routed_scale).to(output_dtype)

    def setUp(self):
        torch.manual_seed(47)
        self.device = torch.device("cuda")
        self.num_experts = 2
        self.num_adapters = 2
        self.hidden_size = 128
        self.intermediate = 128
        self.rank = 16
        self.topk = 2
        self.routed_scale = 1.75

        self.hidden_states = (
            torch.randn(4, self.hidden_size, device=self.device) * 0.2
        ).to(torch.bfloat16)
        self.topk_ids = torch.tensor(
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            dtype=torch.int32,
            device=self.device,
        )
        self.topk_weights = torch.tensor(
            [[0.7, 0.3], [0.2, 0.8], [0.55, 0.45], [0.35, 0.65]],
            dtype=torch.float32,
            device=self.device,
        )
        self.token_lora_mapping = torch.full(
            (4,), -1, dtype=torch.int32, device=self.device
        )
        self.w13_weight = (
            torch.randn(
                self.num_experts,
                2 * self.intermediate,
                self.hidden_size,
                device=self.device,
            )
            * 0.1
        ).to(torch.bfloat16)
        self.w2_weight = (
            torch.randn(
                self.num_experts,
                self.hidden_size,
                self.intermediate,
                device=self.device,
            )
            * 0.1
        ).to(torch.bfloat16)
        self.gate_a = (
            torch.randn(
                self.num_adapters,
                self.num_experts,
                2 * self.rank,
                self.hidden_size,
                device=self.device,
            )
            * 0.1
        ).to(torch.bfloat16)
        self.gate_b = (
            torch.randn(
                self.num_adapters,
                self.num_experts,
                2 * self.intermediate,
                self.rank,
                device=self.device,
            )
            * 0.1
        ).to(torch.bfloat16)
        self.down_a = (
            torch.randn(
                self.num_adapters,
                self.num_experts,
                self.rank,
                self.intermediate,
                device=self.device,
            )
            * 0.1
        ).to(torch.bfloat16)
        self.down_b = (
            torch.randn(
                self.num_adapters,
                self.num_experts,
                self.hidden_size,
                self.rank,
                device=self.device,
            )
            * 0.1
        ).to(torch.bfloat16)

        self.config = MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate,
            top_k=self.topk,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            no_combine=False,
            routed_scaling_factor=self.routed_scale,
        )
        launch_config = Bf16MoeLaunchConfig(
            routing_block_size=16,
            lora_a={
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 2,
            },
            lora_b={
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
                "num_stages": 2,
            },
        )
        self.runner = MoeRunner(
            MoeRunnerBackend.DEEP_GEMM,
            self.config,
            lora_hook_builder=DeepGemmBf16LoRAHookBuilder(launch_config),
        )
        self.quant_info = DeepGemmMoeQuantInfo(
            w13_weight=self.w13_weight,
            w2_weight=self.w2_weight,
            use_fp8=False,
        )
        self.dispatcher = StandardDispatcher(self.config)
        self.lora_info = SimpleNamespace(
            gate_up_lora_a_weights=self.gate_a,
            gate_up_lora_b_weights=self.gate_b,
            down_lora_a_weights=self.down_a,
            down_lora_b_weights=self.down_b,
            token_lora_mapping=self.token_lora_mapping,
            experts_shared_outer_loras=False,
        )

    def test_base_mixed_and_active_match_staged_reference(self):
        mappings = {
            "base": (-1, -1, -1, -1),
            "mixed": (0, -1, 1, -1),
            "active": (0, 1, 0, 1),
        }
        for output_dtype in (torch.bfloat16, torch.float32):
            outputs = {}
            references = {}
            for name, mapping in mappings.items():
                with self.subTest(output_dtype=output_dtype, mapping=name):
                    self.token_lora_mapping.copy_(
                        torch.tensor(mapping, dtype=torch.int32, device=self.device)
                    )
                    topk_output = StandardTopKOutput(
                        topk_weights=self.topk_weights,
                        topk_ids=self.topk_ids,
                        router_logits=None,
                    )
                    dispatch_output = self.dispatcher.dispatch(
                        self.hidden_states.clone(), topk_output
                    )
                    combine_input = self.runner.run(
                        dispatch_output,
                        self.quant_info,
                        lora_info=self.lora_info,
                        output_dtype=output_dtype,
                    )
                    output = self.dispatcher.combine(combine_input)
                    expected = self._reference(
                        self.hidden_states,
                        self.topk_ids,
                        self.topk_weights,
                        self.token_lora_mapping,
                        output_dtype,
                    )
                    torch.testing.assert_close(
                        output,
                        expected,
                        rtol=2e-2,
                        atol=2e-2,
                    )
                    outputs[name] = output
                    references[name] = expected
            self.assertGreater(
                (references["active"] - references["base"]).abs().max().item(),
                5e-2,
            )
            self.assertGreater(
                (outputs["active"] - outputs["base"]).abs().max().item(),
                5e-2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
