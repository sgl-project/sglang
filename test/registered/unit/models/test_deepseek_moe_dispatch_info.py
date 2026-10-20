"""Recorded expert placement must affect both normal forwards without enable_eplb;
NextN and configurations without a dispatch algorithm must retain logical ids.
"""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.eplb import expert_location_dispatch
from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Experts:
    moe_runner_config = SimpleNamespace(inplace=False)
    quant_method = None

    def __call__(self, hidden_states, topk_output):
        # Physical slots contain distinguishable expert outputs.
        return torch.tensor([[10.0], [20.0]])[topk_output.topk_ids[:, 0]]


class _MoE(SimpleNamespace):
    _normal_path_dispatch_info = deepseek_v2.DeepseekV2MoE._normal_path_dispatch_info


def _forward(*, dual_stream, algorithm, is_nextn=False):
    logical_ids = torch.tensor([[0]], dtype=torch.int32)

    def topk(hidden_states, router_logits, *, expert_location_dispatch_info, **kwargs):
        return SimpleNamespace(
            topk_ids=expert_location_dispatch.topk_ids_logical_to_physical(
                logical_ids, expert_location_dispatch_info
            )
        )

    moe = _MoE(
        is_nextn=is_nextn,
        layer_id=0,
        experts=_Experts(),
        gate=lambda *args: torch.zeros((1, 2)),
        topk=topk,
        _maybe_quant_moe_input_once=lambda x: None,
        _should_quant_routed_input_mxfp8=lambda x: False,
        _forward_shared_experts=lambda *args, **kwargs: None,
        _fuse_shared_experts_inside_sbo=False,
        _shared_expert_tp1=False,
        num_fused_shared_experts=1,
        routed_scaling_factor=1.0,
        reduce_results=False,
        alt_stream=mock.Mock(),
    )
    metadata = SimpleNamespace(
        logical_to_rank_dispatch_physical_map=torch.tensor([[1, 0]]),
        logical_to_all_physical_map=torch.tensor([[[1], [0]]]),
        logical_to_all_physical_map_num_valid=torch.ones((1, 2), dtype=torch.int32),
        num_physical_experts=2,
    )
    execution = SimpleNamespace(
        moe=SimpleNamespace(
            enable_eplb=False, ep_dispatch_algorithm=algorithm, moe_a2a_backend="none"
        )
    )
    method = (
        deepseek_v2.DeepseekV2MoE.forward_normal_dual_stream
        if dual_stream
        else deepseek_v2.DeepseekV2MoE.forward_normal
    )
    with ExitStack() as stack:
        for module in (deepseek_v2, expert_location_dispatch):
            stack.enter_context(
                mock.patch.object(module, "get_exec", return_value=execution)
            )
        stack.enter_context(
            mock.patch.object(
                expert_location_dispatch,
                "get_global_expert_location_metadata",
                return_value=metadata,
            )
        )
        stack.enter_context(mock.patch.object(deepseek_v2, "_is_hip", False))
        stack.enter_context(
            mock.patch.object(
                deepseek_v2,
                "get_forward",
                return_value=SimpleNamespace(flashinfer_trtllm_bypass=False),
            )
        )
        stack.enter_context(mock.patch.object(torch.cuda, "current_stream"))
        stack.enter_context(
            mock.patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext())
        )
        return method(moe, torch.ones((1, 1)))


class TestNormalPathDispatchInfo(CustomTestCase):
    def test_remaps_without_enable_eplb(self):
        for dual_stream in (False, True):
            with self.subTest(dual_stream=dual_stream):
                self.assertEqual(
                    _forward(dual_stream=dual_stream, algorithm="static").item(), 20.0
                )

    def test_no_dispatch_algorithm_keeps_logical_ids(self):
        for dual_stream in (False, True):
            with self.subTest(dual_stream=dual_stream):
                self.assertEqual(
                    _forward(dual_stream=dual_stream, algorithm=None).item(), 10.0
                )

    def test_nextn_layers_never_remap(self):
        for dual_stream in (False, True):
            with self.subTest(dual_stream=dual_stream):
                self.assertEqual(
                    _forward(
                        dual_stream=dual_stream, algorithm="static", is_nextn=True
                    ).item(),
                    10.0,
                )


if __name__ == "__main__":
    unittest.main()
