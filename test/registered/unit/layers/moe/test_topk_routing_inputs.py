"""Routing follows the owning layer and active batch across target/draft and TBO."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.operations import (
    ExecutionOperation,
    _StageExecutor,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.hardware_backend.npu.moe.topk import fused_topk_npu
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe import hash_topk as hash_topk_module
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import (
    StandardTopKOutputPacked,
    TopK,
    TopKOutputFormat,
    fused_topk_torch_native,
)
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    draft_model_build_scope,
    initialize_moe_config,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
    has_forward_context,
)
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_parallel,
    get_resources,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _batch(real):
    empty = torch.empty(0, dtype=torch.int32)
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=4,
        input_ids=torch.arange(4),
        req_pool_indices=empty,
        seq_lens=empty,
        out_cache_loc=empty,
        seq_lens_sum=0,
    )
    batch.num_token_non_padded = torch.tensor(real, dtype=torch.int32)
    batch.global_num_token_non_padded = batch.num_token_non_padded
    batch.dp_padding_mode = DpPaddingMode.SUM_LEN
    batch.global_dp_buffer_len = batch.tbo_padded_len = 4
    return batch


def _placement():
    logical_to_physical = torch.tensor([[2, 0, 3, 1]], dtype=torch.int64)
    physical_to_logical = logical_to_physical.argsort(dim=-1)
    all_physical = torch.full((1, 4, 4), -1, dtype=torch.int64)
    all_physical[:, :, 0] = logical_to_physical
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical,
        physical_to_logical_map_cpu=physical_to_logical.clone(),
        logical_to_all_physical_map=all_physical,
        logical_to_all_physical_map_cpu=all_physical.clone(),
        logical_to_all_physical_map_num_valid=torch.ones((1, 4), dtype=torch.int64),
        ep_size=1,
        logical_to_rank_dispatch_physical_map=logical_to_physical,
    )


class TestTopKRoutingInputs(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = get_context().override_server_args(
            ep_dispatch_algorithm="static", enable_eplb=True
        )
        config.install()
        self.addCleanup(config.restore)
        initialize_moe_config()
        topology = get_parallel().override(
            attn_dp_size=1, attn_cp_size=1, attn_tp_size=1, attn_tp_rank=0
        )
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        for name in ("_is_cuda", "_is_hip", "_use_aiter"):
            scope = patch.object(topk_module, name, False)
            scope.start()
            self.addCleanup(scope.stop)
        self.metadata = _placement()
        get_resources().expert_location_metadata = self.metadata
        self.addCleanup(setattr, get_resources(), "expert_location_metadata", None)
        self.hidden = torch.ones((4, 2))
        self.logits = torch.tensor([[9.0, 8.0, 1.0, 0.0]]).repeat(4, 1)

    def _router(self, **kwargs):
        return TopK(
            top_k=2,
            layer_id=0,
            custom_routing_function=fused_topk_torch_native,
            **kwargs,
        )

    def test_target_and_draft_remap_only_the_target_and_keep_mask(self):
        """Draft routes must not use the target placement or lose every live row."""
        target = self._router()
        with draft_model_build_scope():
            draft = self._router()
        batch = _batch(2)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=batch
        ):
            target_ids = target.forward_native(self.hidden, self.logits).topk_ids
            draft_ids = draft.forward_native(self.hidden, self.logits).topk_ids
        self.assertEqual(target_ids.tolist(), [[2, 0], [2, 0], [-1, -1], [-1, -1]])
        self.assertEqual(draft_ids.tolist(), [[0, 1], [0, 1], [-1, -1], [-1, -1]])
        self.assertFalse(draft.topk_config.allow_routed_experts_capture)
        self.assertTrue(target.topk_config.allow_routed_experts_capture)

    def test_late_preparation_and_in_place_rebalance(self):
        """Forward preparation and rebalance must be observed after scope entry."""
        router = self._router()
        batch = _batch(1)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=batch
        ):
            before = router.forward_native(self.hidden, self.logits).topk_ids
            batch.num_token_non_padded = torch.tensor(3)
            self.metadata.logical_to_rank_dispatch_physical_map.copy_(
                torch.tensor([[1, 3, 0, 2]])
            )
            after = router.forward_native(self.hidden, self.logits).topk_ids
        self.assertEqual(before[0].tolist(), [2, 0])
        self.assertEqual(before[1].tolist(), [-1, -1])
        self.assertEqual(after.tolist(), [[1, 3], [1, 3], [1, 3], [-1, -1]])

    def test_nested_scope_restores_and_retains_replay_buffer_identity(self):
        router = self._router()
        parent, child = _batch(3), _batch(1)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=parent
        ):
            original = get_forward_context()
            bound, _ = router.routing_inputs()
            self.assertIs(bound, parent.num_token_non_padded)
            with self.assertRaisesRegex(RuntimeError, "child failed"):
                with forward_context(
                    ForwardContext(attn_backend=object()), forward_batch=child
                ):
                    self.assertIs(
                        router.routing_inputs()[0], child.num_token_non_padded
                    )
                    raise RuntimeError("child failed")
            self.assertIs(get_forward_context(), original)
            parent.num_token_non_padded.fill_(2)
            self.assertIs(router.routing_inputs()[0], bound)
            self.assertEqual(
                router.forward_native(self.hidden, self.logits).topk_ids[-2:].tolist(),
                [[-1, -1], [-1, -1]],
            )
        self.assertFalse(has_forward_context())

    def test_tbo_children_bind_bounds_without_tbo_attention_backend(self):
        router = self._router()
        seen = []

        def read_bound(state, forward_batch):
            seen.append(int(router.routing_inputs()[0]))
            return {"forward_batch": forward_batch}

        stages = [[ExecutionOperation("route", read_bound)]] * 2
        parent, child_a, child_b = _batch(4), _batch(1), _batch(3)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=parent
        ):
            a = _StageExecutor("a", stages, {"forward_batch": child_a})
            b = _StageExecutor("b", stages, {"forward_batch": child_b})
            a.next()
            b.next()
            a.next()
            b.next()
            self.assertEqual(int(router.routing_inputs()[0]), 4)
        self.assertEqual(seen, [1, 3, 1, 3])

    def test_hash_router_uses_same_placement_and_bound(self):
        router = HashTopK(2, 4, 0, 4, scoring_func="softmax", layer_id=0)
        with envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = router(self.hidden, self.logits, torch.arange(4))
        self.assertEqual(output.topk_ids.tolist(), [[2, 0], [0, 3], [-1, -1], [-1, -1]])

    def test_vision_router_maps_routed_columns_and_masks_shared_rows_last(self):
        router = self._router(
            num_fused_shared_experts=1,
            routed_scaling_factor=2.0,
            fused_shared_experts_scaling_factor=0.25,
        )
        moe = SimpleNamespace(
            topk=router,
            layer_id=0,
            config=SimpleNamespace(image_token_id=99),
            gate=SimpleNamespace(
                e_score_correction_bias=torch.zeros(4),
                e_score_correction_bias_vl=torch.zeros(4),
            ),
        )
        with patch(
            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False
        ):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = vision_topk(moe, self.logits, torch.tensor([1, 99, 1, 99]))
        self.assertEqual(output.topk_ids.tolist(), [[2, 4], [2, 4], [-1, -1], [-1, -1]])
        torch.testing.assert_close(output.topk_weights[:2, -1], torch.full((2,), 0.125))
        torch.testing.assert_close(output.topk_weights[2:], torch.zeros((2, 2)))

    def test_vision_packed_output_follows_built_runner(self):
        """A requested MXFP4 fallback must not emit an unused packed-ID buffer."""
        with get_context().override_server_args(moe_runner_backend="flashinfer_mxfp4"):
            initialize_moe_config()
            for backend in (
                None,
                MoeRunnerBackend.TRITON,
                MoeRunnerBackend.FLASHINFER_MXFP4,
            ):
                with self.subTest(backend=backend):
                    router = self._router(is_draft=True)
                    router.bind_runner_backend(backend)
                    moe = SimpleNamespace(
                        topk=router,
                        layer_id=0,
                        config=SimpleNamespace(image_token_id=99),
                        gate=SimpleNamespace(
                            e_score_correction_bias=torch.zeros(4),
                            e_score_correction_bias_vl=torch.zeros(4),
                        ),
                    )
                    with (
                        patch(
                            "sglang.srt.multimodal.dsv41.vl_routing.is_cuda",
                            return_value=True,
                        ),
                        patch(
                            "sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate",
                            return_value=(
                                torch.full((4, 2), 0.5),
                                torch.tensor([[0, 1]], dtype=torch.int32).repeat(4, 1),
                            ),
                        ) as gate,
                    ):
                        output = vision_topk(moe, self.logits, input_ids=None)
                    packed = gate.call_args.kwargs["packed_out"]
                    expected_packed = backend == MoeRunnerBackend.FLASHINFER_MXFP4
                    self.assertEqual(packed is not None, expected_packed)
                    self.assertEqual(
                        isinstance(output, StandardTopKOutputPacked), expected_packed
                    )

    def test_vision_fake_algorithm_neutralizes_both_biases(self):
        with get_exec().moe.override(ep_dispatch_algorithm="fake"):
            router = self._router()
            moe = SimpleNamespace(
                topk=router,
                layer_id=0,
                config=SimpleNamespace(image_token_id=99),
                gate=SimpleNamespace(
                    e_score_correction_bias=torch.tensor([0.0, 0.0, 1000.0, 0.0]),
                    e_score_correction_bias_vl=torch.tensor([0.0, 0.0, 0.0, 1000.0]),
                ),
            )
            with patch(
                "sglang.srt.multimodal.dsv41.vl_routing.is_cuda", return_value=False
            ):
                with forward_context(
                    ForwardContext(attn_backend=object()), forward_batch=_batch(4)
                ):
                    logits = self.logits.clone()
                    output = vision_topk(moe, logits, torch.tensor([1, 99, 1, 99]))
            logical = logits.topk(2, dim=-1).indices
            expected = self.metadata.logical_to_all_physical_map[0, logical, 0]
            self.assertTrue(torch.equal(output.topk_ids.long(), expected))

    def test_draft_without_input_ids_retains_prepared_live_width(self):
        """Deferred draft IDs must not turn a live draft's padding bound into zero."""
        batch = _batch(0)
        batch.input_ids = None
        batch.spec_info = EagleDraftInput(num_tokens_per_req=2)
        with get_parallel().override(tp_size=4, attn_tp_size=4, moe_ep_size=4):
            batch._init_num_token_non_padded(device="cpu", is_draft_worker=True)
        self.assertEqual(batch.global_num_token_non_padded_cpu, 8)
        self.assertEqual(int(batch.global_num_token_non_padded), 8)
        batch.forward_mode = ForwardMode.IDLE
        with get_parallel().override(tp_size=4, attn_tp_size=4, moe_ep_size=4):
            batch._init_num_token_non_padded(device="cpu", is_draft_worker=True)
        self.assertEqual(int(batch.global_num_token_non_padded), 0)

    def test_npu_routes_keep_valid_ids_and_zero_only_padded_contributions(self):
        """NPU padding must stay inert without inventing invalid or duplicate routes."""
        router = self._router(num_fused_shared_experts=1)
        router.topk_config.scoring_func = "sqrtsoftplus"
        logical = torch.tensor([[0, 4], [1, 4], [2, 4], [3, 4]], dtype=torch.int32)
        weights = torch.tensor([[0.75, 0.25]]).repeat(4, 1)

        class Capture:
            def capture(self, *, layer_id, topk_indices):
                self.ids = topk_indices.clone()

        capture = Capture()
        get_resources().experts_capturer = capture
        self.addCleanup(setattr, get_resources(), "experts_capturer", None)
        _, info = router.routing_inputs()
        with patch.object(
            torch.ops.custom,
            "npu_moe_gating_top_k",
            create=True,
            return_value=(weights, logical, None),
        ):
            output = fused_topk_npu(
                hidden_states=self.hidden,
                router_logits=self.logits,
                topk_config=router.topk_config,
                num_token_non_padded=torch.tensor(2),
                expert_location_dispatch_info=info,
                layer_id=0,
            )
        self.assertEqual(output.topk_ids.tolist(), [[2, 4], [0, 4], [3, 4], [1, 4]])
        torch.testing.assert_close(
            output.topk_weights[:2], torch.tensor([[0.75, 0.25]]).repeat(2, 1)
        )
        torch.testing.assert_close(output.topk_weights[2:], torch.zeros((2, 2)))
        self.assertEqual(capture.ids.tolist(), [[0], [1], [2], [3]])

    def test_in_kernel_formats_do_not_silently_ignore_expert_placement(self):
        for output_format in (
            TopKOutputFormat.TRITON_KERNEL,
            TopKOutputFormat.BYPASSED,
        ):
            with self.subTest(output_format=output_format):
                router = self._router(output_format=output_format)
                with forward_context(
                    ForwardContext(attn_backend=object()), forward_batch=_batch(4)
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError, "placement remapping requires"
                    ):
                        router.forward_cuda(self.hidden, self.logits)

    def test_identity_placement_keeps_in_kernel_routing_available(self):
        identity = torch.arange(4).view(1, 4)
        self.metadata.logical_to_rank_dispatch_physical_map.copy_(identity)
        self.metadata.logical_to_all_physical_map[:, :, 0].copy_(identity)
        self.metadata.physical_to_logical_map.copy_(identity)
        with get_exec().moe.override(
            enable_eplb=False,
            init_expert_location="trivial",
            ep_num_redundant_experts=0,
            ep_dispatch_algorithm="dynamic",
        ):
            for output_format in (
                TopKOutputFormat.BYPASSED,
                TopKOutputFormat.TRITON_KERNEL,
            ):
                with self.subTest(output_format=output_format):
                    router = self._router(output_format=output_format)
                    with patch.object(
                        topk_module, "routing", return_value=(None, None, None, None, 2)
                    ):
                        output = router.forward_cuda(self.hidden, self.logits)
                    self.assertEqual(output.format, output_format)
            for algorithm in ("fake", "lp"):
                with (
                    self.subTest(algorithm=algorithm),
                    get_exec().moe.override(ep_dispatch_algorithm=algorithm),
                ):
                    router = self._router(output_format=TopKOutputFormat.BYPASSED)
                    with self.assertRaisesRegex(
                        NotImplementedError, "placement remapping requires"
                    ):
                        router.forward_cuda(self.hidden, self.logits)

    def test_hash_npu_keeps_valid_ids_and_zero_padded_weights(self):
        router = HashTopK(2, 4, 0, 4, scoring_func="softmax", layer_id=0)
        with (
            patch.object(hash_topk_module, "_is_npu", True),
            envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False),
        ):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = router(self.hidden, self.logits, torch.arange(4))
        self.assertEqual(output.topk_ids.tolist(), [[2, 0], [0, 3], [3, 1], [1, 2]])
        torch.testing.assert_close(output.topk_weights[2:], torch.zeros((2, 2)))

    def test_hash_global_shared_column_stays_outside_expert_map(self):
        router = HashTopK(
            3,
            4,
            1,
            4,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=2.0,
            layer_id=0,
        )
        with envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(False):
            with forward_context(
                ForwardContext(attn_backend=object()), forward_batch=_batch(2)
            ):
                output = router(self.hidden, self.logits, torch.arange(4))
        self.assertEqual(
            output.topk_ids.tolist(), [[2, 0, 4], [0, 3, 4], [-1, -1, -1], [-1, -1, -1]]
        )

    def test_empty_target_participates_in_lp_without_caller_layer_id(self):
        class Solver:
            def __init__(self):
                self.participations = []

            def solve(self, ids):
                self.participations.append(tuple(ids.shape))

        solver = Solver()
        get_resources().lplb_solvers[0] = solver
        self.addCleanup(get_resources().lplb_solvers.pop, 0)
        target = self._router()
        with draft_model_build_scope():
            draft = self._router()
        with get_parallel().override(tp_group=SimpleNamespace(world_size=1)):
            output = target.empty_topk_output(torch.device("cpu"))
            draft_output = draft.empty_topk_output(torch.device("cpu"))
        self.assertEqual(output.topk_ids.shape, (0, 2))
        self.assertEqual(draft_output.topk_ids.shape, (0, 2))
        self.assertEqual(solver.participations, [(0, 2)])

    def test_compile_reads_current_buffer(self):
        router = self._router(is_draft=True)
        batch = _batch(3)

        def mask_rows(x):
            bound, _ = router.routing_inputs()
            return x * (torch.arange(x.shape[0]) < bound)

        compiled = torch.compile(mask_rows, backend="eager", fullgraph=True)
        with forward_context(
            ForwardContext(attn_backend=object()), forward_batch=batch
        ):
            self.assertEqual(compiled(torch.ones(4)).tolist(), [1.0, 1.0, 1.0, 0.0])
            batch.num_token_non_padded.fill_(1)
            self.assertEqual(compiled(torch.ones(4)).tolist(), [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
