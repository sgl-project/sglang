"""Local CP routing must preserve row order and obey the FFN reduction scope.

The gate, top-k kernel and expert GEMMs run as small CPU references; the model
method, shared-expert handling and post-expert reduction run unchanged.
"""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
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


def route(logits, renormalize=True):
    weights, ids = logits.float().softmax(-1).topk(2, dim=-1)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights, ids.to(torch.int32)


def cpu_topk(weights, ids, logits, renormalize):
    expected_weights, expected_ids = route(logits, renormalize)
    weights.copy_(expected_weights)
    ids.copy_(expected_ids)


def expert_output(hidden, topk):
    scale = (EXPERT_SCALE[topk.topk_ids.long()] * topk.topk_weights).sum(
        -1, keepdim=True
    )
    return hidden * scale


class TestQwen2MoeCp(CustomTestCase):
    def setUp(self):
        self.parallel = SimpleNamespace(
            tp_size=2, moe_ep_size=1, moe_dp_size=1, dwdp_size=1
        )
        self.moe = SimpleNamespace(
            enable_eplb=False,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
        )
        self.flags = ForwardFlags()
        self.recorder = Mock()
        self.capture = Mock()
        self.kernel = Mock(side_effect=cpu_topk)
        self.all_reduce = Mock(side_effect=lambda value: value * 2)
        self.stack = self.enterContext(ExitStack())
        for module, name, value in [
            (qwen, "get_parallel", lambda: self.parallel),
            (qwen, "get_exec", lambda: SimpleNamespace(moe=self.moe)),
            (qwen, "get_forward", lambda: self.flags),
            (qwen, "get_moe_a2a_backend", lambda: moe_utils.MoeA2ABackend.NONE),
            (moe_utils, "get_parallel", lambda: self.parallel),
            (moe_utils, "get_forward", lambda: self.flags),
            (moe_utils, "get_moe_a2a_backend", lambda: moe_utils.MoeA2ABackend.NONE),
            (
                moe_utils,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                lambda: False,
            ),
            (topk_module, "_is_cuda", True),
            (topk_module, "get_global_experts_capturer", lambda: self.capture),
            (
                topk_module,
                "get_global_expert_distribution_recorder",
                lambda: self.recorder,
            ),
            (communication_op, "tensor_model_parallel_all_reduce", self.all_reduce),
        ]:
            self.stack.enter_context(patch.object(module, name, value))
        # The CUDA-only import is absent on CPU CI, but the method still runs
        # with the reference implementation in place of that one kernel.
        self.stack.enter_context(
            patch.object(qwen, "topk_softmax", self.kernel, create=True)
        )

    def block(self):
        block = qwen.Qwen2MoeSparseMoeBlock.__new__(qwen.Qwen2MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.layer_id = 3
        block.num_experts = 3
        block.topk = SimpleNamespace(
            topk_config=TopKConfig(top_k=2), enable_waterfill=False
        )
        block.gate = Mock(side_effect=lambda value: (value @ GATE_WEIGHT.T, None))
        block.experts = Mock(side_effect=expert_output)
        block.enable_shared_expert_fusion = False
        block.shared_expert = None
        block.shared_expert_gate = None
        return block

    def test_local_payload_is_gathered_before_experts_and_reduced_in_scope(self):
        local = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
        remote = torch.tensor([[3.0, -1.0], [-1.0, -2.0]])
        gathered = torch.cat((local, remote))
        global_weights, global_ids = route(gathered @ GATE_WEIGHT.T)
        expected = expert_output(
            gathered,
            SimpleNamespace(topk_weights=global_weights, topk_ids=global_ids),
        )
        for reduce_scatter in (False, True):
            with self.subTest(reduce_scatter=reduce_scatter):
                block = self.block()
                self.all_reduce.reset_mock()

                def gather(hidden, weights, ids):
                    torch.testing.assert_close(hidden, local)
                    torch.testing.assert_close(weights, global_weights[:2])
                    torch.testing.assert_close(ids, global_ids[:2])
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(ids.dtype, torch.int32)
                    return [gathered, global_weights, global_ids]

                with self.flags.scoped(mlp_reduce_scatter=reduce_scatter):
                    output = block.forward_cp(
                        local, all_gather_rows=gather, symmetric_memory=nullcontext
                    )
                self.assertFalse(self.flags.mlp_reduce_scatter)
                block.gate.assert_called_once()
                torch.testing.assert_close(block.gate.call_args.args[0], local)
                torch.testing.assert_close(block.experts.call_args.args[0], gathered)
                self.assertIsNone(block.experts.call_args.args[1].router_logits)
                torch.testing.assert_close(
                    output, expected if reduce_scatter else expected * 2
                )
                self.assertEqual(self.all_reduce.call_count, int(not reduce_scatter))

    def test_router_allocates_payload_in_scope_and_preserves_capture_hooks(self):
        block = self.block()
        logits = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.25, 0.0]])
        allocations = []
        original_empty = torch.empty
        active = False

        @contextmanager
        def symmetric_memory():
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def allocate(*args, **kwargs):
            self.assertTrue(active)
            allocations.append(kwargs["dtype"])
            return original_empty(*args, **kwargs)

        for renormalize in (False, True):
            with self.subTest(renormalize=renormalize):
                block.topk.topk_config.renormalize = renormalize
                with patch.object(qwen.torch, "empty", side_effect=allocate):
                    output = block._cp_router(logits, symmetric_memory)
                weights, ids = route(logits, renormalize)
                torch.testing.assert_close(output.topk_weights, weights)
                torch.testing.assert_close(output.topk_ids, ids)
                torch.testing.assert_close(
                    self.capture.capture.call_args.kwargs["topk_indices"], ids
                )
                self.assertEqual(self.capture.capture.call_args.kwargs["layer_id"], 3)
                torch.testing.assert_close(
                    self.recorder.on_select_experts.call_args.kwargs["topk_ids"], ids
                )
        self.assertEqual(allocations, [torch.float32, torch.int32] * 2)

    def test_router_rejects_routing_that_needs_extra_processing(self):
        for name, value in [
            ("correction_bias", torch.ones(3)),
            ("use_grouped_topk", True),
            ("scoring_func", "sigmoid"),
            ("custom_routing_function", Mock()),
            ("routed_scaling_factor", 2.0),
            ("num_fused_shared_experts", 1),
            ("enable_waterfill", True),
        ]:
            with self.subTest(setting=name):
                block = self.block()
                target = (
                    block.topk if name == "enable_waterfill" else block.topk.topk_config
                )
                setattr(target, name, value)
                with self.assertRaisesRegex(
                    NotImplementedError, "unmodified CUDA softmax"
                ):
                    block._cp_router(torch.ones(1, 3), nullcontext)
        self.kernel.assert_not_called()

    def test_forward_rejects_incompatible_expert_layouts_and_deferred_finalize(self):
        cases = [
            (self.parallel, "moe_ep_size", 2, "TP-only"),
            (self.parallel, "moe_dp_size", 2, "TP-only"),
            (self.parallel, "dwdp_size", 2, "TP-only"),
            (self.moe, "enable_eplb", True, "remapping"),
            (self.moe, "init_expert_location", "random", "remapping"),
            (self.moe, "ep_num_redundant_experts", 1, "remapping"),
        ]
        block = self.block()
        gather = Mock()
        for obj, name, value, message in cases:
            with self.subTest(setting=name), patch.object(obj, name, value):
                with self.assertRaisesRegex(NotImplementedError, message):
                    block.forward_cp(
                        torch.ones(1, 2),
                        all_gather_rows=gather,
                        symmetric_memory=nullcontext,
                    )
        with patch.object(
            qwen, "get_moe_a2a_backend", return_value=moe_utils.MoeA2ABackend.DEEPEP
        ):
            with self.assertRaisesRegex(NotImplementedError, "TP-only"):
                block.forward_cp(
                    torch.ones(1, 2),
                    all_gather_rows=gather,
                    symmetric_memory=nullcontext,
                )
        with self.flags.scoped(defer_moe_finalize=True):
            with self.assertRaisesRegex(NotImplementedError, "cannot defer"):
                block.forward_cp(
                    torch.ones(1, 2),
                    all_gather_rows=gather,
                    symmetric_memory=nullcontext,
                )
        block.gate.assert_not_called()
        gather.assert_not_called()

    def test_shared_experts_are_added_before_the_single_output_reduction(self):
        hidden = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
        weights, ids = route(hidden @ GATE_WEIGHT.T)
        expected = expert_output(
            hidden, SimpleNamespace(topk_weights=weights, topk_ids=ids)
        )
        expected += hidden * 4 * (hidden @ SHARED_WEIGHT.T).sigmoid()

        def fused_gate_add(value, gate, shared, output):
            output.add_(shared * (value @ gate).sigmoid().unsqueeze(-1))

        def append(ids, weights, shared_weights, num_shared, N):
            self.assertEqual(num_shared, 1)
            return (
                torch.cat((ids, torch.full((len(ids), 1), N, dtype=ids.dtype)), dim=1),
                torch.cat((weights, shared_weights), dim=1),
            )

        from sglang.kernels.ops.moe import fused_moe_triton_kernels

        for fusion in ("none", "gate", "expert"):
            with self.subTest(fusion=fusion), ExitStack() as stack:
                block = self.block()
                block.shared_expert_gate = nn.Linear(2, 1, bias=False)
                block.shared_expert_gate.weight.data.copy_(SHARED_WEIGHT)
                block.shared_expert = Mock(side_effect=lambda value: value * 4)
                if fusion == "expert":
                    block.enable_shared_expert_fusion = True
                    block.num_fused_shared_experts = 1
                    block.shared_expert = None
                stack.enter_context(
                    patch.object(
                        block, "_use_fused_shared_gate", return_value=fusion == "gate"
                    )
                )
                stack.enter_context(
                    patch.object(
                        qwen, "fused_gate_sigmoid_mul_add", side_effect=fused_gate_add
                    )
                )
                stack.enter_context(
                    patch.object(
                        fused_moe_triton_kernels,
                        "fused_append_shared_experts_with_weights",
                        side_effect=append,
                    )
                )
                self.all_reduce.reset_mock()
                output = block.forward_cp(
                    hidden,
                    all_gather_rows=lambda *tensors: list(tensors),
                    symmetric_memory=nullcontext,
                )
                torch.testing.assert_close(output, expected * 2)
                self.all_reduce.assert_called_once()
                torch.testing.assert_close(self.all_reduce.call_args.args[0], expected)
                self.assertEqual(
                    block.experts.call_args.args[1].topk_ids.shape[1],
                    3 if fusion == "expert" else 2,
                )

    def test_empty_local_rows_still_gather_and_only_skip_empty_compute(self):
        local = torch.empty(0, 2)
        for remote in (torch.empty(0, 2), torch.tensor([[1.0, 2.0]])):
            with self.subTest(remote_rows=len(remote)):
                block = self.block()
                self.kernel.reset_mock()
                self.all_reduce.reset_mock()
                weights, ids = route(remote @ GATE_WEIGHT.T)

                def gather(hidden, local_weights, local_ids):
                    self.assertEqual(tuple(hidden.shape), (0, 2))
                    self.assertEqual(tuple(local_weights.shape), (0, 2))
                    self.assertEqual(tuple(local_ids.shape), (0, 2))
                    return [remote, weights, ids]

                gather_mock = Mock(side_effect=gather)
                output = block.forward_cp(
                    local, all_gather_rows=gather_mock, symmetric_memory=nullcontext
                )
                block.gate.assert_not_called()
                self.kernel.assert_not_called()
                gather_mock.assert_called_once()
                if len(remote):
                    block.experts.assert_called_once()
                    expected = expert_output(
                        remote, SimpleNamespace(topk_weights=weights, topk_ids=ids)
                    )
                    torch.testing.assert_close(output, expected * 2)
                else:
                    self.assertIs(output, remote)
                    block.experts.assert_not_called()
                    self.all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
