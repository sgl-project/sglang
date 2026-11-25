"""Muse Spark's custom experts: the TRT-LLM dispatch contract and weight loads."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.token_dispatcher import flashinfer
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType
from sglang.srt.layers.muse_spark import muse_spark_bf16_moe as bf16
from sglang.srt.layers.muse_spark import muse_spark_fp4_moe as fp4
from sglang.srt.layers.muse_spark.fused import glue
from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.runtime_context import get_context, get_flags, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _CpuCommunication:
    def __init__(self, _bootstrap, params, **_kwargs):
        self.params = params

    def dispatch(self, hidden_states, topk_ids, topk_weights, **_kwargs):
        return SimpleNamespace(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )


class TestMuseSparkRunnerDispatch(CustomTestCase):
    def test_custom_expert_config_selects_trtllm_receive_sentinel(self):
        """Unoccupied SP receive slots must use the direct TRT-LLM consumer's
        sentinel even though these experts do not construct a generic runner.
        """
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                a2a_backend=MoeA2ABackend.NONE,
            ),
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_ep_size=2,
                moe_ep_rank=1,
            ),
            get_forward().scoped(sp_active=True),
            patch.object(
                modelopt_quant,
                "get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
            # The TRT-LLM FP4 experts refuse a pre-Blackwell GPU; fake SM100 so
            # the case also runs on a CUDA host.
            patch.object(fp4, "get_device_capability", return_value=(10, 0)),
        ):
            experts = (
                MuseSpark1xBf16Experts(
                    num_experts=4,
                    num_local_experts=2,
                    local_expert_offset=2,
                    expert_hidden_size=128,
                    intermediate_size=128,
                    top_k=2,
                ),
                MuseSpark1xFp4Experts(
                    num_experts=4,
                    num_local_experts=2,
                    local_expert_offset=2,
                    hidden_size=128,
                    intermediate_size=128,
                    top_k=2,
                    quant_config=ModelOptFp4Config(
                        is_checkpoint_nvfp4_serialized=True, group_size=16
                    ),
                ),
            )
            # Replace FlashInfer's distributed workspace allocation, not the
            # SGLang constructor or its runner/sentinel selection.
            with ExitStack() as stack:
                for name, value in {
                    "use_flashinfer": True,
                    "BootstrapConfig": lambda **_: None,
                    "MoEEpCommParams": SimpleNamespace,
                    "NVLinkOneSidedConfig": lambda **_: None,
                    "TorchDistributedCommBackend": lambda _: None,
                    "create_communication": _CpuCommunication,
                    "get_flashinfer_a2a_dispatch_type": lambda: (
                        FlashinferA2ADispatchType.BF16
                    ),
                    "get_is_extend_in_batch": lambda: True,
                    "get_dp_global_num_tokens": lambda: None,
                    "is_dp_attention_enabled": lambda: False,
                }.items():
                    stack.enter_context(
                        patch.object(flashinfer, name, value, create=True)
                    )
                for expert in experts:
                    with self.subTest(expert=type(expert).__name__):
                        dispatcher = flashinfer.FlashinferDispatcher(
                            group=SimpleNamespace(size=lambda: 2, rank=lambda: 1),
                            router_topk=2,
                            num_experts=4,
                            num_local_experts=2,
                            hidden_size=128,
                            moe_runner_config=expert.moe_runner_config,
                            enable_sp_prefill_a2a=True,
                        )
                        topk = StandardTopKOutput(
                            torch.full((2, 2), 0.5),
                            torch.tensor([[0, 3], [2, 1]], dtype=torch.int32),
                            None,
                        )
                        output = dispatcher.dispatch(
                            torch.zeros((2, 128), dtype=torch.bfloat16), topk
                        )
                        self.assertEqual(
                            dispatcher.sp_prefill_moe_a2a.params.invalid_expert_id, -1
                        )
                        torch.testing.assert_close(
                            output.topk_output.topk_ids, topk.topk_ids
                        )


class TestMuseSparkBf16ExpertLoads(CustomTestCase):
    def test_loads_after_interleave_keep_the_interleaved_layout(self):
        """Once the Hopper Triton path interleaves gate / up rows, a later checkpoint
        load (weight update) must not write the old [gate; up] halves."""
        mod = bf16.MuseSpark1xBf16Experts(
            num_experts=4,
            num_local_experts=2,
            local_expert_offset=2,
            expert_hidden_size=8,
            intermediate_size=4,
            top_k=2,
        )
        mod.gate_up_proj.data.copy_(torch.randn(2, 8, 8))
        with (
            patch.object(bf16, "_trtllm_bf16_moe_supported", return_value=False),
            patch.object(bf16, "_triton_bf16_moe_supported", return_value=True),
            patch.object(glue, "hopper_kernels_enabled", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            mod.prepare_trtllm_weights(None)
        gate, up = torch.randn(4, 8), torch.randn(4, 8)
        param = mod.gate_up_proj
        mod.weight_loader(param, gate, "gate_up_proj", shard_id="w1", expert_id=3)
        mod.weight_loader(param, up, "gate_up_proj", shard_id="w3", expert_id=3)
        want = torch.stack([gate, up], 1).flatten(0, 1)
        self.assertTrue(torch.equal(param.data[1], want))
        stack = torch.randn(2, 8, 8)
        got = bf16.expert_stack_in_layout(param, stack)
        self.assertTrue(
            torch.equal(got[0], stack[0].view(2, 4, 8).transpose(0, 1).flatten(0, 1))
        )
        # a direct copy skips both loaders
        self.assertIsNotNone(mod._direct_weight_update_error)


if __name__ == "__main__":
    unittest.main()
