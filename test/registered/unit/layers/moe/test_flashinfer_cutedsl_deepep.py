"""Host-side contracts for CuTe DSL W4A16 with DeepEP normal dispatch."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import flashinfer_cutedsl as cutedsl
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
    CuteDslFp4MoeQuantInfo,
    fused_experts_deepep_to_flashinfer_cutedsl_fp4,
)
from sglang.srt.layers.moe.token_dispatcher import deepep as deepep_dispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingWrapper:
    def __init__(self):
        self.kwargs = None

    def run(self, **kwargs):
        self.kwargs = kwargs
        return kwargs["x"] + 1


class TestFlashinferCuteDslW4A16DeepEPNormal(CustomTestCase):
    def test_reuses_route_wrapper_with_local_expert_ids(self):
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16)
        topk_ids = torch.tensor(
            [[0, -1], [3, 1], [-1, 0]],
            dtype=torch.int64,
        )
        topk_weights = torch.rand(3, 2, dtype=torch.float32)
        dispatch_output = DeepEPNormalDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_recv_tokens_per_expert=[2, 1, 0, 1],
        )
        wrapper = _RecordingWrapper()
        placeholder = torch.empty(0)
        quant_info = CuteDslFp4MoeQuantInfo(
            w13_weight=placeholder,
            w2_weight=placeholder,
            w13_weight_sf=placeholder,
            w2_weight_sf=placeholder,
            w1_alpha=placeholder,
            w2_alpha=placeholder,
            a1_scale=placeholder,
            a2_scale=placeholder,
            wrapper=wrapper,
            quant_mode="w4a16",
        )

        combine_input = fused_experts_deepep_to_flashinfer_cutedsl_fp4(
            dispatch_output,
            quant_info,
            MoeRunnerConfig(activation="silu"),
        )

        self.assertIsInstance(combine_input, DeepEPNormalCombineInput)
        self.assertTrue(torch.equal(combine_input.hidden_states, hidden_states + 1))
        self.assertIs(combine_input.topk_ids, topk_ids)
        self.assertIs(combine_input.topk_weights, topk_weights)
        self.assertIs(wrapper.kwargs["x"], hidden_states)
        self.assertIsNone(wrapper.kwargs["x_sf"])
        self.assertEqual(wrapper.kwargs["token_selected_experts"].dtype, torch.int32)
        self.assertTrue(
            torch.equal(
                wrapper.kwargs["token_selected_experts"],
                topk_ids.to(torch.int32),
            )
        )
        self.assertIs(wrapper.kwargs["token_final_scales"], topk_weights)
        self.assertIsNone(wrapper.kwargs["fc2_input_scale"])
        self.assertIsNone(wrapper.kwargs["per_token_scale"])

    def test_deepep_wrapper_uses_rank_local_expert_namespace(self):
        created = {}
        fake_flashinfer = ModuleType("flashinfer")
        fake_flashinfer.ActivationType = SimpleNamespace(  # type: ignore[attr-defined]
            Swiglu="swiglu",
            Relu2="relu2",
        )

        def fake_wrapper(**kwargs):
            created.update(kwargs)
            return SimpleNamespace(quant_mode=kwargs["quant_mode"])

        fake_flashinfer.CuteDslMoEWrapper = fake_wrapper  # type: ignore[attr-defined]
        layer = SimpleNamespace(
            _cutedsl_wrapper=None,
            w13_weight=torch.empty(0),
            intermediate_size_per_partition=64,
            dispatcher=SimpleNamespace(),
            top_k=2,
            moe_runner_config=MoeRunnerConfig(
                activation="silu",
                params_dtype=torch.bfloat16,
            ),
            num_experts=16,
            num_local_experts=4,
            moe_ep_rank=2,
            hidden_size=128,
            quant_config=SimpleNamespace(use_per_token_activation=False),
        )
        one = torch.ones(1, dtype=torch.float32)
        with (
            patch.dict(sys.modules, {"flashinfer": fake_flashinfer}),
            envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(True),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            patch.object(cutedsl, "cuda_graph_fully_disabled", return_value=True),
            patch.object(cutedsl, "cutedsl_moe_max_num_tokens", return_value=8),
            patch.object(
                cutedsl,
                "get_parallel",
                return_value=SimpleNamespace(dp_size=1),
            ),
            patch.object(
                cutedsl,
                "resolve_cutedsl_standard_scales",
                return_value=(one, one, one, one),
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=MoeA2ABackend.DEEPEP,
            ),
        ):
            cutedsl.ensure_cutedsl_wrapper(layer)

        self.assertEqual(created["quant_mode"], "w4a16")
        self.assertEqual(created["num_experts"], 16)
        self.assertEqual(created["num_local_experts"], 4)
        self.assertEqual(created["local_expert_offset"], 0)

    def test_deepep_normal_combine_accepts_cutedsl_without_deep_gemm(self):
        impl = deepep_dispatcher._DeepEPDispatcherImplNormal.__new__(
            deepep_dispatcher._DeepEPDispatcherImplNormal
        )
        impl.async_finish = False
        hidden_states = torch.randn(3, 8, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, -1], [3, 1], [-1, 0]], dtype=torch.int64)
        topk_weights = torch.rand(3, 2, dtype=torch.float32)
        with (
            patch.object(
                deepep_dispatcher.deep_gemm_wrapper,
                "ENABLE_JIT_DEEPGEMM",
                False,
            ),
            patch.object(
                deepep_dispatcher,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_CUTEDSL,
            ),
            patch.object(deepep_dispatcher, "_use_aiter", False),
            patch.object(deepep_dispatcher, "_is_npu", False),
        ):
            output, event = impl.combine_a(hidden_states, topk_ids, topk_weights)

        self.assertIs(output, hidden_states)
        self.assertIsNone(event)


if __name__ == "__main__":
    unittest.main()
