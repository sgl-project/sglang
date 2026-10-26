"""CUDA dispatch, fake shapes and replay for the uncompleted dual-stream outputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import TopKOutputFormat
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.models import deepseek_v2
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class DualStreamOperand:
    """Small CUDA computations isolate the custom-op boundary from expert kernels."""

    _can_dual_stream_graph = deepseek_v2.DeepseekV2MoE._can_dual_stream_graph

    def __init__(self, replicated):
        self.layer_id = 0
        self._shared_expert_tp1 = replicated
        self.alt_stream = torch.cuda.Stream()
        self.num_fused_shared_experts = 0
        self._enable_a2a_moe = self._fuse_shared_experts_inside_sbo = False
        self.shared_experts = object()
        self.is_hash = False
        self.gate = SimpleNamespace(e_score_correction_bias_vl=None)
        self.topk = SimpleNamespace(
            topk_config=SimpleNamespace(output_format=TopKOutputFormat.BYPASSED)
        )
        self.experts = SimpleNamespace(
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
            use_flashinfer_trtllm_moe=True,
            quant_method=None,
            should_fuse_routed_scaling_factor_in_topk=False,
        )
        self.calls = 0

    def forward_normal_dual_stream(self, value, *, leave_output_sum):
        assert leave_output_sum
        assert get_forward().flashinfer_trtllm_bypass
        assert not get_forward().defer_moe_finalize
        self.calls += 1
        current = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current)
        routed = torch.sin(value) * 0.25
        with torch.cuda.stream(self.alt_stream):
            shared = torch.cos(value) * 0.75
        current.wait_stream(self.alt_stream)
        return deepseek_v2.MoEOutput(
            routed, shared, self.experts, 1.0, self._shared_expert_tp1
        )


class TestDualStreamOutputs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_dispatch_compile_and_replay_preserve_separate_replicated_shared(self):
        for replicated in (False, True):
            with self.subTest(replicated=replicated):
                operand = DualStreamOperand(replicated)
                value = torch.linspace(-0.5, 0.5, 128, device="cuda").reshape(2, 64)
                context = SimpleNamespace(moe_fusions={0: operand})
                with (
                    patch.object(deepseek_v2, "_enable_pcg_dsv2_dual_stream", True),
                    patch.object(
                        deepseek_v2, "is_in_breakable_cuda_graph", return_value=True
                    ),
                    patch.object(
                        deepseek_v2,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
                    ),
                    patch.object(
                        deepseek_v2,
                        "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(
                                enable_eplb=False, ep_dispatch_algorithm="static"
                            )
                        ),
                    ),
                    patch.object(
                        deepseek_v2,
                        "get_tc_piecewise_forward_context",
                        return_value=context,
                    ),
                    patch(
                        "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
                        return_value=False,
                    ),
                ):
                    selected = deepseek_v2.DeepseekV2MoE.forward(
                        operand, value, leave_output_sum=True
                    )
                    self.assertIsInstance(selected, deepseek_v2.MergedMoEOutput)
                    self.assertEqual(operand.calls, 1)
                    if replicated:
                        torch.testing.assert_close(
                            selected.routed, torch.sin(value) * 0.25
                        )
                        torch.testing.assert_close(
                            selected.shared, torch.cos(value) * 0.75
                        )
                    else:
                        self.assertIsNone(selected.shared)
                        torch.testing.assert_close(
                            selected.routed,
                            torch.sin(value) * 0.25 + torch.cos(value) * 0.75,
                        )

                    def invoke(x):
                        return deepseek_v2.dsv4_flashinfer_moe_dual_stream_outputs(
                            x, 0, False, replicated
                        )

                    # opcheck validates the registered op, not its eager wrapper
                    torch.library.opcheck(
                        torch.ops.sglang.dsv4_flashinfer_moe_dual_stream_outputs.default,
                        (value, 0, False, replicated),
                        test_utils=("test_schema", "test_faketensor"),
                    )
                    compiled = torch.compile(invoke, fullgraph=True)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            compiled(value)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        routed, shared = compiled(value)
                    count_after_capture = operand.calls
                    for offset in (0.2, -0.1, 0.7):
                        value.add_(offset)
                        graph.replay()
                        expected_routed, expected_shared = (
                            torch.sin(value) * 0.25,
                            torch.cos(value) * 0.75,
                        )
                        self.assertEqual(operand.calls, count_after_capture)
                        if replicated:
                            torch.testing.assert_close(routed, expected_routed)
                            torch.testing.assert_close(shared, expected_shared)
                        else:
                            self.assertEqual(shared.numel(), 0)
                            torch.testing.assert_close(
                                routed, expected_routed + expected_shared
                            )


if __name__ == "__main__":
    unittest.main()
