"""Muse Spark's custom experts must declare the TRT-LLM dispatch contract."""

import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
    def test_custom_nvfp4_rejects_per_token_phase(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe import (
            MuseSparkNvFp4FusedMoEMethod,
        )

        per_token = envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION
        w4a16 = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
        with (
            per_token.override(False),
            w4a16.override(False),
            patch.object(
                modelopt_quant.ModelOptNvFp4FusedMoEMethod,
                "__init__",
                return_value=None,
            ) as parent_init,
        ):
            for phase in ("prefill", "decode"):
                with self.subTest(per_token=phase), per_token.override(phase):
                    config = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True)
                    with self.assertRaisesRegex(
                        ValueError, "Muse Spark.*phase-specific"
                    ):
                        MuseSparkNvFp4FusedMoEMethod(config)
                    parent_init.assert_not_called()
            for value in (False, True, "all"):
                with self.subTest(per_token=value), per_token.override(value):
                    config = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True)
                    MuseSparkNvFp4FusedMoEMethod(config)
                    parent_init.assert_called_with(config)

    def test_w4a16_flag_is_noop_for_custom_trtllm_experts(self):
        from sglang.srt.environ import envs

        hidden = torch.zeros(2, 128, dtype=torch.bfloat16)
        packed = torch.zeros(2, 64, dtype=torch.uint8)
        scales = torch.zeros(2, 8, dtype=torch.uint8)
        token_scales = torch.ones(2, dtype=torch.float32)
        static_quant = Mock(return_value=(packed, scales))
        dynamic_quant = Mock(return_value=(packed, scales, token_scales))
        kernels = SimpleNamespace(
            fp4_quantize=static_quant,
            nvfp4_quantize=dynamic_quant,
            SfLayout=SimpleNamespace(layout_linear=object()),
        )
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                a2a_backend=MoeA2ABackend.NONE,
            ),
            patch.object(
                modelopt_quant,
                "get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
            patch.dict(sys.modules, {"flashinfer": kernels}),
        ):
            for setting in (False, "0", True, "1", "all", "prefill", "decode"):
                for per_token in (False, True):
                    with (
                        self.subTest(w4a16=setting, per_token=per_token),
                        envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(setting),
                        envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.override(
                            per_token
                        ),
                    ):
                        # Construct the real Muse expert and quantization method;
                        # only platform detection and quantization kernels are stubbed.
                        experts = MuseSpark1xFp4Experts(
                            num_experts=4,
                            num_local_experts=2,
                            local_expert_offset=2,
                            hidden_size=128,
                            intermediate_size=128,
                            top_k=2,
                            quant_config=ModelOptFp4Config(
                                is_checkpoint_nvfp4_serialized=True, group_size=16
                            ),
                        )
                        experts.w13_input_scale_quant = torch.tensor(1.0)
                        static_quant.reset_mock()
                        dynamic_quant.reset_mock()
                        prepared = experts.prepare_input(hidden)
                        self.assertEqual(prepared[0].data_ptr(), packed.data_ptr())
                        self.assertEqual(prepared[0].dtype, torch.uint8)
                        self.assertIs(prepared[2], token_scales if per_token else None)
                        self.assertTrue(
                            experts.quant_method.enable_flashinfer_trtllm_moe
                        )
                        self.assertFalse(experts.use_marlin)
                        (
                            dynamic_quant if per_token else static_quant
                        ).assert_called_once()
                        (
                            static_quant if per_token else dynamic_quant
                        ).assert_not_called()

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
