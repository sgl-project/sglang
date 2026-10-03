"""Local CP routing must preserve row order and obey the FFN reduction scope.

The gate, top-k kernel and expert GEMMs run as small CPU references; the model
method and shared-expert handling run unchanged. CP compute always returns a
partial sum for the boundary to complete.
"""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.distributed import communication_op
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
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
            expert_distribution_recorder_mode=None,
        )
        self.flags = ForwardFlags()
        self.recorder = Mock()
        self.capture = Mock()
        self.kernel = Mock(side_effect=cpu_topk)
        self.all_reduce = Mock(
            side_effect=AssertionError("CP output must be reduced by the boundary")
        )
        self.stack = self.enterContext(ExitStack())
        for module, name, value in [
            (qwen, "_is_cuda", True),
            (qwen, "get_moe_runner_backend", lambda: moe_utils.MoeRunnerBackend.TRITON),
            (
                topk_module,
                "get_moe_runner_backend",
                lambda: moe_utils.MoeRunnerBackend.TRITON,
            ),
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
        block.num_physical_routed_experts = 3
        block.is_nextn = False
        block.topk = SimpleNamespace(
            topk_config=TopKConfig(top_k=2), enable_waterfill=False
        )
        block.gate = Mock(side_effect=lambda value: (value @ GATE_WEIGHT.T, None))
        block.experts = Mock(side_effect=expert_output)
        block.enable_shared_expert_fusion = False
        block.shared_expert = None
        block.shared_expert_gate = None
        return block

    def test_local_payload_is_gathered_and_always_returns_partial_output(self):
        local = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
        remote = torch.tensor([[3.0, -1.0], [-1.0, -2.0]])
        gathered = torch.cat((local, remote))
        global_weights, global_ids = route(gathered @ GATE_WEIGHT.T)
        expected = expert_output(
            gathered,
            SimpleNamespace(topk_weights=global_weights, topk_ids=global_ids),
        )
        for reduce_scatter, fuse_allreduce in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                reduce_scatter=reduce_scatter, fuse_allreduce=fuse_allreduce
            ):
                block = self.block()
                self.all_reduce.reset_mock()
                self.capture.reset_mock()
                self.recorder.reset_mock()

                def gather(hidden, weights, ids):
                    torch.testing.assert_close(hidden, local)
                    torch.testing.assert_close(weights, global_weights[:2])
                    torch.testing.assert_close(ids, global_ids[:2])
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertEqual(ids.dtype, torch.int32)
                    self.capture.capture.assert_not_called()
                    self.recorder.on_select_experts.assert_not_called()
                    return [gathered, global_weights, global_ids]

                with self.flags.scoped(
                    mlp_reduce_scatter=reduce_scatter, fuse_mlp_allreduce=fuse_allreduce
                ):
                    output = block.forward_cp(
                        local, all_gather_rows=gather, symmetric_memory=nullcontext
                    )
                self.assertFalse(self.flags.mlp_reduce_scatter)
                block.gate.assert_called_once()
                torch.testing.assert_close(block.gate.call_args.args[0], local)
                torch.testing.assert_close(block.experts.call_args.args[0], gathered)
                self.assertIsNone(block.experts.call_args.args[1].router_logits)
                torch.testing.assert_close(output, expected)
                self.all_reduce.assert_not_called()
                self.capture.capture.assert_called_once()
                self.recorder.on_select_experts.assert_called_once()
                torch.testing.assert_close(
                    self.capture.capture.call_args.kwargs["topk_indices"], global_ids
                )
                torch.testing.assert_close(
                    self.recorder.on_select_experts.call_args.kwargs["topk_ids"],
                    global_ids,
                )

    def test_local_router_allocates_payload_without_running_global_hooks(self):
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
                    output = block._cp_router(
                        torch.ones(2, 2), logits, symmetric_memory
                    )
                weights, ids = route(logits, renormalize)
                torch.testing.assert_close(output.topk_weights, weights)
                torch.testing.assert_close(output.topk_ids, ids)
                self.capture.capture.assert_not_called()
                self.recorder.on_select_experts.assert_not_called()
        self.assertEqual(allocations, [torch.float32, torch.int32] * 2)

    def test_tokenwise_policies_match_full_batch_reference(self):
        hidden = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]])
        logits = torch.tensor(
            [[0.1, 0.9, -1.0, 2.0], [1.0, -0.5, 0.4, 0.8], [-1.0, 2.0, 0.7, 0.2]]
        )
        configs = [
            TopKConfig(
                top_k=2,
                torch_native=True,
                correction_bias=torch.tensor([2.0, 0.0, 0.1, -1.0]),
            ),
            TopKConfig(top_k=2, torch_native=True, scoring_func="sigmoid"),
            TopKConfig(
                top_k=2, use_grouped_topk=True, num_expert_group=2, topk_group=1
            ),
            TopKConfig(
                top_k=2,
                use_grouped_topk=True,
                num_expert_group=2,
                topk_group=1,
                scoring_func="sigmoid",
                correction_bias=torch.tensor([0.3, -0.1, 0.2, 0.0]),
                routed_scaling_factor=2.5,
                apply_routed_scaling_factor_on_output=True,
            ),
        ]
        # Exercise select_experts itself, replacing compiled CPU/GPU kernel
        # dispatch with the existing eager PyTorch reference implementations.
        with (
            patch.object(
                topk_module, "fused_topk_native", topk_module.fused_topk_torch_native
            ),
            patch.object(
                topk_module, "grouped_topk", topk_module.grouped_topk_gpu.__wrapped__
            ),
            patch.object(
                topk_module,
                "biased_grouped_topk",
                topk_module.biased_grouped_topk_impl.__wrapped__,
            ),
        ):
            for cfg in configs:
                with self.subTest(cfg=cfg):
                    block = self.block()
                    block.topk.topk_config = cfg
                    expected = topk_module.select_experts(
                        hidden, logits, cfg, layer_id=3, defer_postprocessing=True
                    )
                    selected = [
                        block._cp_router(hidden[:1], logits[:1], nullcontext),
                        block._cp_router(hidden[1:], logits[1:], nullcontext),
                    ]
                    torch.testing.assert_close(
                        torch.cat([value.topk_weights for value in selected]),
                        expected.topk_weights,
                    )
                    torch.testing.assert_close(
                        torch.cat([value.topk_ids for value in selected]),
                        expected.topk_ids.to(torch.int32),
                    )
        self.kernel.assert_not_called()
        self.capture.capture.assert_not_called()
        self.recorder.on_select_experts.assert_not_called()

    def test_routed_backend_preserves_raw_logit_weights(self):
        local = torch.tensor([[1.0, 2.0]])
        remote = torch.tensor([[3.0, -1.0]])
        gathered = torch.cat((local, remote))
        block = self.block()
        block.topk.topk_config.renormalize = False
        backend = moe_utils.MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
        with (
            patch.object(qwen, "get_moe_runner_backend", return_value=backend),
            patch.object(topk_module, "get_moe_runner_backend", return_value=backend),
        ):
            expected_topk = topk_module.select_experts(
                gathered,
                gathered @ GATE_WEIGHT.T,
                block.topk.topk_config,
                layer_id=3,
                defer_postprocessing=True,
            )

            def gather(hidden, weights, ids):
                torch.testing.assert_close(hidden, local)
                torch.testing.assert_close(weights, expected_topk.topk_weights[:1])
                torch.testing.assert_close(ids, expected_topk.topk_ids[:1])
                return [gathered, expected_topk.topk_weights, expected_topk.topk_ids]

            output = block.forward_cp(
                local, all_gather_rows=gather, symmetric_memory=nullcontext
            )
        torch.testing.assert_close(output, expert_output(gathered, expected_topk))
        torch.testing.assert_close(block.gate.call_args.args[0], local)
        self.kernel.assert_not_called()

    def test_forward_rejects_incompatible_expert_layouts_and_deferred_finalize(self):
        cases = [
            (self.parallel, "moe_dp_size", 2, "MoE DP=1"),
            (self.parallel, "dwdp_size", 2, "MoE DP=1"),
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
            with self.assertRaisesRegex(NotImplementedError, "MoE DP=1"):
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

    def test_ep_remaps_gathered_rows_once_and_excludes_collective_padding(self):
        local = torch.tensor([[1.0, 2.0]])
        remote = torch.tensor([[3.0, -1.0], [-1.0, -2.0]])
        gathered = torch.cat((local, torch.zeros_like(local), remote))
        valid = torch.tensor([[1], [0], [1], [1]], dtype=torch.int32)
        weights, logical_ids = route(gathered @ GATE_WEIGHT.T)
        # An actual row gather pads the weight/ID payload with zero, independently
        # of hidden-state padding. Three replicas make the second rank's global
        # row offset select a different replica than its local row index would.
        weights[1] = 0
        logical_ids[1] = 0
        mappings = {
            "permuted": torch.tensor([[2], [0], [1]], dtype=torch.int64),
            "redundant": torch.tensor(
                [[0, 3, 6], [1, 4, 7], [2, 5, 8]], dtype=torch.int64
            ),
        }
        for placement, mapping in mappings.items():
            with self.subTest(placement=placement):
                block = self.block()
                self.parallel.moe_ep_size = 2
                self.moe.enable_eplb = placement == "redundant"
                self.moe.init_expert_location = "random"
                self.moe.ep_num_redundant_experts = mapping.numel() - 3
                info = ExpertLocationDispatchInfo(
                    ep_dispatch_algorithm="dynamic",
                    partial_logical_to_rank_dispatch_physical_map=mapping[:, 0],
                    partial_logical_to_all_physical_map=mapping,
                    partial_logical_to_all_physical_map_num_valid=torch.full(
                        (3,), mapping.shape[1]
                    ),
                    num_physical_experts=mapping.numel(),
                    rank_invariant=True,
                )
                row_slots = torch.arange(len(gathered)) % mapping.shape[1]
                physical_ids = mapping[logical_ids.long(), row_slots[:, None]].to(
                    torch.int32
                )
                physical_scale = torch.empty(mapping.numel())
                for logical_id, replicas in enumerate(mapping):
                    physical_scale[replicas] = EXPERT_SCALE[logical_id]

                def experts(hidden, topk):
                    execution_ids = physical_ids.clone()
                    execution_ids[1] = 0
                    torch.testing.assert_close(topk.topk_ids, execution_ids)
                    torch.testing.assert_close(topk.topk_weights[1], torch.zeros(2))
                    return hidden * (
                        physical_scale[topk.topk_ids.long()] * topk.topk_weights
                    ).sum(-1, keepdim=True)

                block.experts = Mock(side_effect=experts)
                self.recorder.reset_mock()
                self.capture.reset_mock()

                def gather(hidden, local_weights, local_ids, local_valid):
                    torch.testing.assert_close(hidden, local)
                    torch.testing.assert_close(local_weights, weights[:1])
                    torch.testing.assert_close(local_ids, logical_ids[:1])
                    torch.testing.assert_close(local_valid, valid[:1])
                    self.capture.capture.assert_not_called()
                    self.recorder.on_select_experts.assert_not_called()
                    return [gathered, weights, logical_ids, valid]

                with (
                    patch.object(
                        qwen.ExpertLocationDispatchInfo, "init_new", return_value=info
                    ),
                    patch.object(
                        topk_module,
                        "_biased_grouped_topk_postprocess",
                        topk_module._biased_grouped_topk_postprocess.__wrapped__,
                    ),
                ):
                    output = block.forward_cp(
                        local, all_gather_rows=gather, symmetric_memory=nullcontext
                    )
                expected = expert_output(
                    gathered,
                    SimpleNamespace(topk_weights=weights, topk_ids=logical_ids),
                )
                torch.testing.assert_close(output, expected)
                self.capture.capture.assert_called_once()
                torch.testing.assert_close(
                    self.capture.capture.call_args.kwargs["topk_indices"], logical_ids
                )
                self.recorder.on_select_experts.assert_called_once()
                recorded = physical_ids.clone()
                recorded[1] = -1
                torch.testing.assert_close(
                    self.recorder.on_select_experts.call_args.kwargs["topk_ids"],
                    recorded,
                )

    def test_remapping_rejects_rank_dependent_dispatch(self):
        self.moe.enable_eplb = True
        for algorithm, invariant in (
            ("static", True),
            ("lp", True),
            ("dynamic", False),
        ):
            with self.subTest(algorithm=algorithm, rank_invariant=invariant):
                block = self.block()
                gather = Mock()
                info = SimpleNamespace(
                    ep_dispatch_algorithm=algorithm, rank_invariant=invariant
                )
                with patch.object(
                    qwen.ExpertLocationDispatchInfo, "init_new", return_value=info
                ):
                    with self.assertRaisesRegex(ValueError, "rank-invariant"):
                        block.forward_cp(
                            torch.ones(1, 2),
                            all_gather_rows=gather,
                            symmetric_memory=nullcontext,
                        )
                block.gate.assert_not_called()
                gather.assert_not_called()

    def test_recording_without_remapping_excludes_padding(self):
        self.moe.expert_distribution_recorder_mode = "stat"
        block = self.block()
        local = torch.tensor([[1.0, 2.0]])
        gathered = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0], [-1.0, -2.0]])
        weights, ids = route(gathered @ GATE_WEIGHT.T)
        weights[1] = 0
        ids[1] = 0
        valid = torch.tensor([[1], [0], [1], [1]], dtype=torch.int32)

        def gather(hidden, local_weights, local_ids, local_valid):
            torch.testing.assert_close(local_valid, valid[:1])
            return [gathered, weights, ids, valid]

        with patch.object(
            topk_module,
            "_biased_grouped_topk_postprocess",
            topk_module._biased_grouped_topk_postprocess.__wrapped__,
        ):
            output = block.forward_cp(
                local, all_gather_rows=gather, symmetric_memory=nullcontext
            )
        recorded = ids.clone()
        recorded[1] = -1
        torch.testing.assert_close(
            self.recorder.on_select_experts.call_args.kwargs["topk_ids"], recorded
        )
        torch.testing.assert_close(
            output,
            expert_output(
                gathered, SimpleNamespace(topk_weights=weights, topk_ids=ids)
            ),
        )

    def test_redundant_experts_with_trivial_placement_keep_logical_ids(self):
        self.moe.ep_num_redundant_experts = 2
        block = self.block()
        block.num_physical_routed_experts = 5
        hidden = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
        weights, ids = route(hidden @ GATE_WEIGHT.T)
        gather = Mock(side_effect=lambda *payload: list(payload))
        with patch.object(
            qwen.ExpertLocationDispatchInfo, "init_new", return_value=None
        ):
            output = block.forward_cp(
                hidden, all_gather_rows=gather, symmetric_memory=nullcontext
            )
        self.assertEqual(len(gather.call_args.args), 3)
        torch.testing.assert_close(block.experts.call_args.args[1].topk_ids, ids)
        torch.testing.assert_close(
            output,
            expert_output(hidden, SimpleNamespace(topk_weights=weights, topk_ids=ids)),
        )

    def test_missing_expert_placement_fails_before_routing(self):
        self.moe.enable_eplb = True
        block = self.block()
        gather = Mock()
        with patch.object(
            qwen.ExpertLocationDispatchInfo, "init_new", return_value=None
        ):
            with self.assertRaisesRegex(ValueError, "dispatch metadata"):
                block.forward_cp(
                    torch.ones(1, 2),
                    all_gather_rows=gather,
                    symmetric_memory=nullcontext,
                )
        block.gate.assert_not_called()
        gather.assert_not_called()

    def test_batch_dependent_callback_routes_after_gather_with_local_gate(self):
        local = torch.tensor([[1.0, 2.0]])
        remote = torch.tensor([[-8.0, -4.0], [-3.0, -2.0]])
        gathered = torch.cat((local, remote))
        block = self.block()
        calls = []

        def callback(hidden_states, gating_output, topk, renormalize):
            calls.append(len(hidden_states))
            # This intentionally chooses different routes for the local shard
            # and the gathered batch, exposing a premature callback invocation.
            first = 0 if hidden_states.mean() > 0 else 1
            ids = torch.tensor([first, first + 1], dtype=torch.int32).expand(
                len(hidden_states), -1
            )
            return torch.full(ids.shape, 0.5), ids

        block.topk.topk_config.custom_routing_function = callback
        logits = gathered @ GATE_WEIGHT.T

        def gather(hidden, local_logits):
            torch.testing.assert_close(hidden, local)
            torch.testing.assert_close(local_logits, logits[:1])
            self.assertEqual(calls, [])
            return [gathered, logits]

        output = block.forward_cp(
            local, all_gather_rows=gather, symmetric_memory=nullcontext
        )
        self.assertEqual(calls, [len(gathered)])
        torch.testing.assert_close(block.gate.call_args.args[0], local)
        torch.testing.assert_close(
            output, gathered * (EXPERT_SCALE[1] + EXPERT_SCALE[2]) * 0.5
        )
        self.kernel.assert_not_called()

    def test_waterfill_requires_compatible_shared_expert_layout(self):
        block = self.block()
        block.topk.enable_waterfill = True
        gather = Mock()
        with self.assertRaisesRegex(
            NotImplementedError, "Waterfill.*shared-expert layout"
        ):
            block.forward_cp(
                torch.ones(1, 2),
                all_gather_rows=gather,
                symmetric_memory=nullcontext,
            )
        block.gate.assert_not_called()
        gather.assert_not_called()
        block.experts.assert_not_called()

    def test_shared_expert_slot_follows_redundant_routed_experts(self):
        from sglang.kernels.ops.moe import fused_moe_triton_kernels

        block = self.block()
        block.num_physical_routed_experts = 5
        block.enable_shared_expert_fusion = True
        block.num_fused_shared_experts = 1
        block.shared_expert_gate = nn.Linear(2, 1, bias=False)
        block.shared_expert_gate.weight.data.copy_(SHARED_WEIGHT)
        hidden = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
        weights, ids = route(hidden @ GATE_WEIGHT.T)

        def append(ids, weights, shared_weights, num_shared, N):
            self.assertEqual(N, 5)
            return torch.cat(
                (ids, torch.full((len(ids), 1), N, dtype=ids.dtype)), 1
            ), torch.cat((weights, shared_weights), 1)

        with patch.object(
            fused_moe_triton_kernels,
            "fused_append_shared_experts_with_weights",
            side_effect=append,
        ):
            topk = block._append_shared_to_topk_output(
                topk_module.StandardTopKOutput(weights, ids, None), hidden
            )
        torch.testing.assert_close(
            topk.topk_ids[:, -1], torch.full((2,), 5, dtype=torch.int32)
        )
        torch.testing.assert_close(
            topk.topk_weights[:, -1], (hidden @ SHARED_WEIGHT.T).sigmoid().squeeze(-1)
        )

    def test_shared_experts_are_included_in_unreduced_output(self):
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
                torch.testing.assert_close(output, expected)
                self.all_reduce.assert_not_called()
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
                self.all_reduce.assert_not_called()
                if len(remote):
                    block.experts.assert_called_once()
                    expected = expert_output(
                        remote, SimpleNamespace(topk_weights=weights, topk_ids=ids)
                    )
                    torch.testing.assert_close(output, expected)
                else:
                    self.assertIs(output, remote)
                    block.experts.assert_not_called()
                    self.all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
