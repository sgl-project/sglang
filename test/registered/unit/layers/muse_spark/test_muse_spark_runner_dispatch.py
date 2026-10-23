"""Muse Spark's custom experts must declare the TRT-LLM dispatch contract."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.token_dispatcher import flashinfer
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import FlashinferA2ADispatchType
from sglang.srt.layers.muse_spark.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.runtime_context import get_context, get_flags, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _CpuMoeAlltoAll:
    def __init__(self, **_kwargs):
        self.invalid_token_expert_id = None

    def dispatch(self, _topk_ids, payloads, _max_tokens, **kwargs):
        self.invalid_token_expert_id = kwargs["invalid_token_expert_id"]
        return payloads


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
                    "moe_a2a_get_workspace_size_per_rank": lambda **_: 128,
                    "Mapping": lambda **_: None,
                    "MnnvlConfig": lambda **_: None,
                    "TorchDistributedCommBackend": lambda _: None,
                    "MoeAlltoAll": _CpuMoeAlltoAll,
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
                            dispatcher.sp_prefill_moe_a2a.invalid_token_expert_id, -1
                        )
                        torch.testing.assert_close(
                            output.topk_output.topk_ids, topk.topk_ids
                        )


if __name__ == "__main__":
    unittest.main()
