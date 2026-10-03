"""CP routes local rows, then returns full-row partial expert output."""

import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.distributed import communication_op
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.models import qwen2_moe as qwen
from sglang.srt.runtime_context import ForwardFlags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GATE_WEIGHT = torch.tensor([[1.0, -0.5], [-0.25, 0.75], [0.5, 0.25]])
SHARED_WEIGHT = torch.tensor([[0.5, -0.25]])
EXPERT_SCALE = torch.tensor([-2.0, 0.5, 3.0, 4.0])


def expert_output(hidden, topk):
    return hidden * (EXPERT_SCALE[topk.topk_ids.long()] * topk.topk_weights).sum(
        -1, keepdim=True
    )


class TestQwen2MoeCp(CustomTestCase):
    def setUp(self):
        self.parallel = SimpleNamespace(
            tp_size=2, moe_tp_size=2, moe_ep_size=1, moe_dp_size=1, dwdp_size=1
        )
        self.moe = SimpleNamespace(
            enable_eplb=False,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            expert_distribution_recorder_mode=None,
        )
        self.features = SimpleNamespace(enable_return_routed_experts=False)
        self.flags = ForwardFlags()
        self.recorder = Mock()
        self.all_reduce = Mock(
            side_effect=AssertionError("CP reduction belongs to the boundary")
        )
        for module, name, value in [
            (qwen, "_is_cuda", True),
            (qwen, "get_parallel", lambda: self.parallel),
            (
                qwen,
                "get_exec",
                lambda: SimpleNamespace(moe=self.moe, features=self.features),
            ),
            (qwen, "get_forward", lambda: self.flags),
            (qwen, "get_moe_a2a_backend", lambda: moe_utils.MoeA2ABackend.NONE),
            (topk_module, "_is_cuda", True),
            (
                topk_module,
                "get_moe_runner_backend",
                lambda: moe_utils.MoeRunnerBackend.TRITON,
            ),
            (topk_module, "get_global_experts_capturer", lambda: None),
            (
                topk_module,
                "get_global_expert_distribution_recorder",
                lambda: self.recorder,
            ),
            (topk_module, "fused_topk_native", topk_module.fused_topk_torch_native),
            (topk_module, "grouped_topk", topk_module.grouped_topk_gpu.__wrapped__),
            (
                topk_module,
                "biased_grouped_topk",
                topk_module.biased_grouped_topk_impl.__wrapped__,
            ),
            (
                topk_module,
                "_biased_grouped_topk_postprocess",
                topk_module._biased_grouped_topk_postprocess.__wrapped__,
            ),
            (communication_op, "tensor_model_parallel_all_reduce", self.all_reduce),
        ]:
            self.enterContext(patch.object(module, name, value))

    def block(self):
        block = qwen.Qwen2MoeSparseMoeBlock.__new__(qwen.Qwen2MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.layer_id, block.num_experts = 3, 3
        block.num_physical_routed_experts = 3
        block.topk = SimpleNamespace(
            topk_config=TopKConfig(top_k=2, torch_native=True), enable_waterfill=False
        )
        block.gate = Mock(side_effect=lambda value: (value @ GATE_WEIGHT.T, None))
        block.experts = Mock(side_effect=expert_output)
        block.enable_shared_expert_fusion = False
        block.shared_expert = block.shared_expert_gate = None
        return block

    @staticmethod
    def reference(block, hidden):
        return topk_module.select_experts(
            hidden, hidden @ GATE_WEIGHT.T, block.topk.topk_config, layer_id=3
        )

    def test_local_selection_and_full_partial_output_ignore_reduction_flags(self):
        local = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
        gathered = torch.cat((local, torch.tensor([[3.0, -1.0], [-1.0, -2.0]])))
        for rs, ar in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(reduce_scatter=rs, fuse_allreduce=ar):
                block = self.block()
                expected = self.reference(block, gathered)
                self.recorder.reset_mock()

                def gather(hidden, weights, ids):
                    torch.testing.assert_close(hidden, local)
                    torch.testing.assert_close(weights, expected.topk_weights[:2])
                    torch.testing.assert_close(
                        ids, expected.topk_ids[:2].to(torch.int32)
                    )
                    self.recorder.on_select_experts.assert_called_once()
                    self.assertEqual(
                        self.recorder.on_select_experts.call_args.kwargs[
                            "topk_ids"
                        ].shape[0],
                        len(local),
                    )
                    return [
                        gathered,
                        expected.topk_weights,
                        expected.topk_ids.to(torch.int32),
                    ]

                with (
                    self.flags.scoped(mlp_reduce_scatter=rs, fuse_mlp_allreduce=ar),
                    patch.object(
                        qwen, "select_experts", wraps=qwen.select_experts
                    ) as selector,
                ):
                    output = block.forward_cp(
                        local, all_gather_rows=gather, symmetric_memory=nullcontext
                    )
                selector.assert_called_once()
                torch.testing.assert_close(selector.call_args.args[0], local)
                torch.testing.assert_close(block.gate.call_args.args[0], local)
                torch.testing.assert_close(block.experts.call_args.args[0], gathered)
                self.assertIsNone(block.experts.call_args.args[1].router_logits)
                torch.testing.assert_close(output, expert_output(gathered, expected))
                self.all_reduce.assert_not_called()

    def test_rowwise_policies_match_the_existing_full_batch_selector(self):
        hidden = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]])
        cases = [
            (
                TopKConfig(
                    top_k=2,
                    torch_native=True,
                    scoring_func="sigmoid",
                    correction_bias=torch.tensor([0.3, -0.1, 0.2]),
                ),
                moe_utils.MoeRunnerBackend.TRITON,
            ),
            (
                TopKConfig(
                    top_k=2,
                    use_grouped_topk=True,
                    num_expert_group=1,
                    topk_group=1,
                    correction_bias=torch.tensor([0.3, -0.1, 0.2]),
                    scoring_func="sigmoid",
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=True,
                ),
                moe_utils.MoeRunnerBackend.TRITON,
            ),
            (
                TopKConfig(top_k=2, renormalize=False),
                moe_utils.MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            ),
        ]
        for cfg, backend in cases:
            with (
                self.subTest(cfg=cfg),
                patch.object(
                    topk_module, "get_moe_runner_backend", return_value=backend
                ),
            ):
                block = self.block()
                block.topk.topk_config = cfg
                expected = self.reference(block, hidden)

                def gather(local, weights, ids):
                    torch.testing.assert_close(weights, expected.topk_weights[:1])
                    torch.testing.assert_close(
                        ids, expected.topk_ids[:1].to(torch.int32)
                    )
                    return [
                        hidden,
                        expected.topk_weights,
                        expected.topk_ids.to(torch.int32),
                    ]

                output = block.forward_cp(
                    hidden[:1], all_gather_rows=gather, symmetric_memory=nullcontext
                )
                torch.testing.assert_close(output, expert_output(hidden, expected))
                torch.testing.assert_close(block.gate.call_args.args[0], hidden[:1])
        self.all_reduce.assert_not_called()

    def test_unsupported_configuration_fails_before_gate_or_gather(self):
        block = self.block()
        cases = [
            (self.parallel, "moe_ep_size", 2),
            (self.moe, "enable_eplb", True),
            (self.moe, "expert_distribution_recorder_mode", "stat"),
            (self.features, "enable_return_routed_experts", True),
            (block.topk, "enable_waterfill", True),
            (block.topk.topk_config, "custom_routing_function", Mock()),
        ]
        for target, name, value in cases:
            with self.subTest(setting=name), patch.object(target, name, value):
                self.assert_rejected(block)
        with patch.object(
            qwen, "get_moe_a2a_backend", return_value=moe_utils.MoeA2ABackend.DEEPEP
        ):
            self.assert_rejected(self.block())
        with self.flags.scoped(defer_moe_finalize=True):
            self.assert_rejected(self.block())

    def assert_rejected(self, block):
        gather = Mock()
        with self.assertRaises(NotImplementedError):
            block.forward_cp(
                torch.ones(1, 2), all_gather_rows=gather, symmetric_memory=nullcontext
            )
        block.gate.assert_not_called()
        gather.assert_not_called()
        block.experts.assert_not_called()

    def test_capture_opt_out_allows_local_routing(self):
        self.features.enable_return_routed_experts = True
        block = self.block()
        block.topk.topk_config.allow_routed_experts_capture = False
        hidden = torch.tensor([[1.0, 2.0]])
        output = block.forward_cp(
            hidden,
            all_gather_rows=lambda *values: list(values),
            symmetric_memory=nullcontext,
        )
        torch.testing.assert_close(
            output, expert_output(hidden, self.reference(block, hidden))
        )

    def test_shared_expert_and_gate_contributions_remain_unreduced(self):
        hidden = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
        for fusion in ("none", "gate"):
            with self.subTest(fusion=fusion), ExitStack() as stack:
                block = self.block()
                block.shared_expert_gate = nn.Linear(2, 1, bias=False)
                block.shared_expert_gate.weight.data.copy_(SHARED_WEIGHT)
                block.shared_expert = Mock(side_effect=lambda value: value * 4)

                def fused_add(value, gate, shared, output):
                    output.add_(shared * (value @ gate).sigmoid().unsqueeze(-1))

                stack.enter_context(
                    patch.object(
                        block, "_use_fused_shared_gate", return_value=fusion == "gate"
                    )
                )
                stack.enter_context(
                    patch.object(
                        qwen, "fused_gate_sigmoid_mul_add", side_effect=fused_add
                    )
                )
                expected = (
                    expert_output(hidden, self.reference(block, hidden))
                    + hidden * 4 * (hidden @ SHARED_WEIGHT.T).sigmoid()
                )
                output = block.forward_cp(
                    hidden,
                    all_gather_rows=lambda *values: list(values),
                    symmetric_memory=nullcontext,
                )
                torch.testing.assert_close(output, expected)
                self.all_reduce.assert_not_called()

    def test_empty_rank_still_gathers_and_skips_only_empty_compute(self):
        local = torch.empty(0, 2)
        for remote in (torch.empty(0, 2), torch.tensor([[1.0, 2.0]])):
            with self.subTest(remote_rows=len(remote)):
                block = self.block()
                expected = self.reference(block, remote)

                def gather(hidden, weights, ids):
                    self.assertEqual(tuple(hidden.shape), (0, 2))
                    self.assertEqual(tuple(weights.shape), (0, 2))
                    self.assertEqual(tuple(ids.shape), (0, 2))
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(ids.dtype, torch.int32)
                    return [
                        remote,
                        expected.topk_weights,
                        expected.topk_ids.to(torch.int32),
                    ]

                with patch.object(
                    qwen, "select_experts", wraps=qwen.select_experts
                ) as selector:
                    output = block.forward_cp(
                        local, all_gather_rows=gather, symmetric_memory=nullcontext
                    )
                block.gate.assert_not_called()
                selector.assert_not_called()
                if len(remote):
                    block.experts.assert_called_once()
                    torch.testing.assert_close(output, expert_output(remote, expected))
                else:
                    block.experts.assert_not_called()
                    self.assertIs(output, remote)
                self.all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
