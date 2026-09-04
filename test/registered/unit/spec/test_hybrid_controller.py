"""CPU contracts for HYBRID speculative runtime-state routing."""

import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.overrides import (
    max_speculative_num_draft_tokens as max_speculative_num_draft_tokens_before_publish,
)
from sglang.srt.arg_groups.overrides import (
    resolving_view,
)
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode
from sglang.srt.runtime_context import (
    get_context,
    max_speculative_num_draft_tokens,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.hybrid_controller import (
    HybridController,
    HybridRuntimeState,
)
from sglang.srt.speculative.hybrid_info import HybridVerifyInput
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hybrid_config(*, adaptive=False):
    neural = {
        "algorithm": "EAGLE3",
        "speculative_num_steps": 3,
        "speculative_num_draft_tokens": 4,
        "speculative_eagle_topk": 1,
    }
    if adaptive:
        neural["speculative_adaptive"] = True
    return json.dumps(
        {
            "retrieval": {
                "algorithm": "NGRAM",
                "speculative_num_draft_tokens": 6,
            },
            "neural": neural,
            "min_continuation_ratio": 1.0,
            "min_matching_ratio": 0.5,
        }
    )


def _runtime_state(name, *, steps, width):
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=width,
        draft_attn_backend=f"draft-{name}",
        cuda_graph_runner=f"draft-graph-{name}",
        target_attn_backend=f"target-{name}",
        target_graph_runner=f"target-graph-{name}",
        draft_extend_attn_backend=f"extend-{name}",
        cuda_graph_runner_for_draft_extend=f"extend-graph-{name}",
    )


class TestHybridServerArgs(CustomTestCase):
    def test_max_width_includes_widest_role_and_controls_overlap_reserve(self):
        with get_context().override_server_args(
            speculative_algorithm="HYBRID",
            speculative_hybrid_config=_hybrid_config(),
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=6,
            page_size=1,
        ):
            self.assertEqual(max_speculative_num_draft_tokens(), 6)
            self.assertEqual(get_alloc_reserve_per_decode(), 12)

    def test_top_level_adaptive_is_owned_by_neural_role(self):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "HYBRID"
        args.speculative_hybrid_config = _hybrid_config(adaptive=True)
        args.speculative_adaptive = True
        args.device = "cuda"

        handle_speculative_decoding(args)

        cfg = resolving_view(args)
        self.assertTrue(cfg.speculative_adaptive)
        self.assertEqual(cfg.speculative_num_steps, 3)
        self.assertEqual(cfg.speculative_num_draft_tokens, 8)
        self.assertEqual(max_speculative_num_draft_tokens_before_publish(args), 8)

    def test_routing_thresholds_are_required(self):
        full = json.loads(_hybrid_config())
        for missing in ("min_continuation_ratio", "min_matching_ratio"):
            with self.assertRaisesRegex(ValueError, rf"requires {missing}"):
                HybridController._load_config(
                    json.dumps({k: v for k, v in full.items() if k != missing})
                )
        for name in ("min_continuation_ratio", "min_matching_ratio"):
            with self.assertRaisesRegex(ValueError, rf"{name} must be in \(0, 1\]"):
                HybridController._load_config(json.dumps({**full, name: 1.5}))


class TestHybridRuntimeSwitch(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._context_override = get_context().override_server_args(
            speculative_num_steps=3,
            speculative_num_draft_tokens=6,
        )
        self._context_override.install()
        self.addCleanup(self._context_override.restore)

        draft_worker = SimpleNamespace(
            draft_attn_backend=None,
            draft_runner=SimpleNamespace(
                draft_attn_backend=None,
                attn_backend=None,
            ),
            cuda_graph_runner=None,
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=None,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            _rebuild_topk1_chain_buffers=MagicMock(),
        )
        neural_worker = SimpleNamespace(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            server_args=SimpleNamespace(
                speculative_num_steps=3,
                speculative_num_draft_tokens=4,
            ),
            draft_worker=draft_worker,
            _active_runtime_state=None,
        )
        target_runner = SimpleNamespace(
            server_args=SimpleNamespace(
                speculative_num_steps=3,
                speculative_num_draft_tokens=6,
            ),
            attn_backend=None,
            decode_cuda_graph_runner=None,
        )
        controller = object.__new__(HybridController)
        controller.neural_worker = neural_worker
        controller._target_worker = SimpleNamespace(model_runner=target_runner)
        controller._active_runtime_state = None
        controller._target_prefill_attn_backend = "target-prefill"

        def apply_runtime_state(state):
            if neural_worker._active_runtime_state is state:
                return
            neural_worker.speculative_num_steps = state.speculative_num_steps
            neural_worker.speculative_num_draft_tokens = (
                state.speculative_num_draft_tokens
            )
            neural_worker._active_runtime_state = state
            draft_worker.draft_attn_backend = state.draft_attn_backend
            draft_worker.draft_runner.draft_attn_backend = state.draft_attn_backend
            draft_worker.cuda_graph_runner = state.cuda_graph_runner
            draft_worker.draft_extend_attn_backend = state.draft_extend_attn_backend
            draft_worker.draft_runner.attn_backend = state.draft_extend_attn_backend
            draft_worker.cuda_graph_runner_for_draft_extend = (
                state.cuda_graph_runner_for_draft_extend
            )
            draft_worker._rebuild_topk1_chain_buffers()
            target_runner.attn_backend = state.target_attn_backend
            target_runner.decode_cuda_graph_runner = state.target_graph_runner

        neural_worker.apply_runtime_state = MagicMock(side_effect=apply_runtime_state)

        self.controller = controller
        self.neural_worker = neural_worker
        self.draft_worker = draft_worker
        self.target_runner = target_runner

    def test_same_state_restores_verify_backend_without_reapplying_worker(self):
        spec_state = _runtime_state("neural", steps=3, width=4)
        state = HybridRuntimeState("neural", self.neural_worker, spec_state)
        self.controller._runtime_states = {("neural", 3): state}

        self.controller._apply_runtime_state("neural")
        self.controller._apply_target_prefill_backend()
        self.assertEqual(self.target_runner.attn_backend, "target-prefill")
        self.controller._apply_runtime_state("neural")

        self.assertEqual(self.target_runner.attn_backend, "target-neural")
        self.assertEqual(
            self.target_runner.decode_cuda_graph_runner, "target-graph-neural"
        )
        self.neural_worker.apply_runtime_state.assert_called_once_with(spec_state)
        self.draft_worker._rebuild_topk1_chain_buffers.assert_called_once_with()

    def test_neural_route_uses_current_adaptive_step(self):
        state3 = HybridRuntimeState(
            "neural", self.neural_worker, _runtime_state("step3", steps=3, width=4)
        )
        state7 = HybridRuntimeState(
            "neural", self.neural_worker, _runtime_state("step7", steps=7, width=8)
        )
        self.controller._runtime_states = {
            ("neural", 3): state3,
            ("neural", 7): state7,
        }
        self.neural_worker.speculative_num_steps = 7

        self.controller._apply_runtime_state("neural")

        self.assertEqual(self.target_runner.attn_backend, "target-step7")
        self.assertEqual(
            self.target_runner.decode_cuda_graph_runner, "target-graph-step7"
        )
        self.assertEqual(self.draft_worker.cuda_graph_runner, "draft-graph-step7")
        self.assertEqual(
            self.draft_worker.cuda_graph_runner_for_draft_extend,
            "extend-graph-step7",
        )
        self.assertEqual(self.neural_worker.speculative_num_draft_tokens, 8)

    def test_retrieval_route_does_not_change_neural_step(self):
        retrieval_state = HybridRuntimeState(
            "retrieval",
            SimpleNamespace(),
            _runtime_state("retrieval", steps=5, width=6),
        )
        self.controller._runtime_states = {("retrieval", None): retrieval_state}
        self.neural_worker.speculative_num_steps = 7

        self.controller._apply_runtime_state("retrieval")

        self.assertEqual(self.neural_worker.speculative_num_steps, 7)
        self.assertEqual(self.target_runner.attn_backend, "target-retrieval")
        self.assertEqual(
            self.target_runner.decode_cuda_graph_runner, "target-graph-retrieval"
        )
        self.assertEqual(
            self.draft_worker.cuda_graph_runner_for_draft_extend,
            "extend-graph-retrieval",
        )

    def test_retrieval_to_neural_restores_target_verify_resources(self):
        neural_spec_state = _runtime_state("neural", steps=3, width=4)
        self.controller._runtime_states = {
            ("neural", 3): HybridRuntimeState(
                "neural", self.neural_worker, neural_spec_state
            ),
            ("retrieval", None): HybridRuntimeState(
                "retrieval",
                SimpleNamespace(),
                _runtime_state("retrieval", steps=5, width=6),
            ),
        }

        self.controller._apply_runtime_state("neural")
        self.controller._apply_runtime_state("retrieval")
        self.controller._apply_runtime_state("neural")

        # EAGLE's second apply is a no-op because it still considers the neural
        # state active. The outer controller must restore target verify itself.
        self.assertEqual(self.target_runner.attn_backend, "target-neural")
        self.assertEqual(
            self.target_runner.decode_cuda_graph_runner, "target-graph-neural"
        )


class TestHybridGraphResources(CustomTestCase):
    @patch(
        "sglang.srt.speculative.hybrid_controller.speculative_moe_a2a_backend_context",
        return_value=nullcontext(),
    )
    @patch(
        "sglang.srt.speculative.hybrid_controller.speculative_moe_backend_context",
        return_value=nullcontext(),
    )
    def test_draft_extend_resources_do_not_inject_primary_buffers(self, _moe, _moe_a2a):
        draft_worker = SimpleNamespace(
            draft_extend_attn_backend="extend-4",
            cuda_graph_runner_for_draft_extend="extend-graph-4",
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_tp_context=MagicMock(return_value=nullcontext()),
            build_draft_extend_runtime_resource=MagicMock(
                return_value=("extend-6", "extend-graph-6")
            ),
        )
        controller = object.__new__(HybridController)
        controller.neural_worker = SimpleNamespace(
            speculative_num_draft_tokens=4,
            adaptive_runtime_states={},
            draft_worker=draft_worker,
        )
        controller._runtime_widths = (4, 6)

        controller._init_draft_extend_resources()

        draft_worker.build_draft_extend_runtime_resource.assert_called_once_with(
            num_tokens_per_bs=6
        )
        self.assertEqual(
            controller._draft_extend_resources,
            {
                4: ("extend-4", "extend-graph-4"),
                6: ("extend-6", "extend-graph-6"),
            },
        )

    def test_target_verify_resources_use_width_local_graph_construction(self):
        target_runner = SimpleNamespace(
            init_new_workspace=False,
            _get_attention_backend=MagicMock(return_value="target-4"),
        )
        graph4 = object()
        bootstrap_graph = object()
        controller = object.__new__(HybridController)
        controller._target_worker = SimpleNamespace(model_runner=target_runner)
        controller.neural_worker = SimpleNamespace(
            adaptive_runtime_states={},
            _override_worker_state=MagicMock(return_value=nullcontext()),
        )
        controller._runtime_widths = (4, 6)

        with patch(
            "sglang.srt.model_executor.runner.DecodeCudaGraphRunner",
            return_value=graph4,
        ) as graph_runner:
            controller._init_target_verify_resources(
                bootstrap_width=6,
                bootstrap_graph_runner=bootstrap_graph,
                bootstrap_attn_backend="target-6",
            )

        graph_runner.assert_called_once_with(
            target_runner,
            attn_backend="target-4",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )
        self.assertIs(controller._target_verify_graph_runners[4], graph4)
        self.assertIs(controller._target_verify_graph_runners[6], bootstrap_graph)
        self.assertFalse(hasattr(target_runner, "hybrid_target_verify_graph_runners"))
        self.assertFalse(hasattr(target_runner, "hybrid_target_verify_attn_backends"))


class TestHybridDraftInputRelay(CustomTestCase):
    def test_ngram_dummy_verify_keeps_native_input(self):
        with get_context().override_server_args(
            speculative_algorithm="NGRAM",
            speculative_num_steps=5,
            speculative_num_draft_tokens=6,
            speculative_eagle_topk=-1,
        ):
            verify_input = create_dummy_verify_input(
                SpeculativeAlgorithm.NGRAM,
                custom_mask=torch.ones(1, dtype=torch.bool),
                num_tokens_per_req=6,
                is_draft_worker=False,
            )

        self.assertIsInstance(verify_input, NgramVerifyInput)
        self.assertEqual(verify_input.draft_token_num, 6)

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
        return_value=nullcontext(),
    )
    @patch(
        "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
        return_value=nullcontext(),
    )
    @patch(
        "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
        return_value=nullcontext(),
    )
    def test_retrieval_result_becomes_eagle_input_after_draft_extend(
        self, _span, _moe, _moe_a2a
    ):
        draft_extend = MagicMock()
        draft_worker = SimpleNamespace(
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_tp_context=MagicMock(return_value=nullcontext()),
            _draft_extend_for_decode=draft_extend,
        )
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = draft_worker
        worker.spec_stage_span_prefix = "neural"

        accept_tokens = torch.tensor([[10, 11, 0, 0, 0, 0], [20, 21, 22, 0, 0, 0]])
        source_input = NgramVerifyInput(
            draft_token_num=6,
            new_seq_lens=torch.tensor([100, 200]),
            accept_tokens=accept_tokens.flatten(),
            accept_lens=torch.tensor([2, 3]),
        )
        result = SimpleNamespace(
            speculative_num_draft_tokens=6,
            next_draft_input=source_input,
            accept_lens=source_input.accept_lens,
        )
        batch = SimpleNamespace(reqs=[SimpleNamespace(), SimpleNamespace()])

        eagle_input = worker.sync_hybrid_state(batch, result)

        self.assertIsInstance(result.next_draft_input, EagleDraftInput)
        self.assertIs(eagle_input, result.next_draft_input)
        self.assertEqual(eagle_input.bonus_tokens.tolist(), [11, 22])
        self.assertEqual(
            source_input.accept_tokens.tolist(), accept_tokens.flatten().tolist()
        )
        draft_extend.assert_called_once_with(batch, result, source_draft_token_num=6)

    def test_hybrid_filter_merge_reuses_native_inputs(self):
        eagle_first = EagleDraftInput(
            topk_p=torch.tensor([[0.1]]),
            topk_index=torch.tensor([[1]]),
            hidden_states=torch.tensor([[10.0]]),
            bonus_tokens=torch.tensor([4]),
        )
        eagle_second = EagleDraftInput(
            topk_p=torch.tensor([[0.2]]),
            topk_index=torch.tensor([[2]]),
            hidden_states=torch.tensor([[20.0]]),
            bonus_tokens=torch.tensor([10]),
        )
        ngram_first = NgramVerifyInput(
            draft_token_num=6,
            new_seq_lens=torch.tensor([100]),
            accept_tokens=torch.tensor([1, 2, 3, 4, 0, 0]),
            accept_lens=torch.tensor([4]),
        )
        ngram_second = NgramVerifyInput(
            draft_token_num=6,
            new_seq_lens=torch.tensor([200]),
            accept_tokens=torch.tensor([5, 6, 7, 8, 9, 10]),
            accept_lens=torch.tensor([6]),
        )

        first = HybridVerifyInput(eagle_first, ngram_first)
        second = HybridVerifyInput(eagle_second, ngram_second)
        first.merge_batch(second)
        selected = torch.tensor([1])
        first.filter_batch(selected)

        self.assertEqual(first.eagle_draft_input.bonus_tokens.tolist(), [10])
        self.assertEqual(first.eagle_draft_input.topk_index.tolist(), [[2]])
        self.assertEqual(
            first.ngram_verify_input.accept_tokens.tolist(), [5, 6, 7, 8, 9, 10]
        )
        self.assertEqual(first.ngram_verify_input.accept_lens.tolist(), [6])

    def test_hybrid_future_indices_are_shared_by_both_native_inputs(self):
        hybrid_input = HybridVerifyInput(
            EagleDraftInput(),
            NgramVerifyInput(draft_token_num=6, new_seq_lens=torch.tensor([11, 12])),
        )

        hybrid_input.future_indices = torch.tensor([3, 7])
        hybrid_input.filter_batch(torch.tensor([1]))

        self.assertEqual(hybrid_input.eagle_draft_input.future_indices.tolist(), [7])
        self.assertEqual(hybrid_input.ngram_verify_input.future_indices.tolist(), [7])

    def test_hybrid_overlap_reuses_both_native_relay_paths(self):
        eagle_input = EagleDraftInput(
            topk_p=torch.tensor([[0.25]]),
            topk_index=torch.tensor([[3]]),
            hidden_states=torch.tensor([[2.0]]),
            bonus_tokens=torch.tensor([9]),
        )
        ngram_input = NgramVerifyInput(
            draft_token_num=6,
            new_seq_lens=torch.tensor([12]),
            accept_tokens=torch.tensor([7, 8, 9, 0, 0, 0]),
            accept_lens=torch.tensor([3]),
        )
        hybrid_input = HybridVerifyInput(eagle_input, ngram_input)
        payload = RelayPayload.from_hybrid(hybrid_input)

        self.assertIs(payload.bonus_tokens, eagle_input.bonus_tokens)
        self.assertIs(payload.topk_p, eagle_input.topk_p)
        self.assertEqual(payload.accept_tokens.tolist(), [[7, 8, 9, 0, 0, 0]])
        self.assertIs(payload.accept_lens, ngram_input.accept_lens)

        future_map = object.__new__(FutureMap)
        future_map.spec_algo = SpeculativeAlgorithm.HYBRID
        future_map._stash_ngram = MagicMock()
        future_map._stash_draft = MagicMock()
        indices = torch.tensor([4])
        future_map.stash(indices, payload)
        future_map._stash_ngram.assert_called_once_with(indices, payload)
        future_map._stash_draft.assert_called_once_with(indices, payload)

        future_map._resolve_ngram_input = MagicMock()
        future_map._resolve_draft_input = MagicMock()
        future_map._resolve_spec_extras(SimpleNamespace(spec_info=hybrid_input))
        future_map._resolve_ngram_input.assert_called_once_with(ngram_input)
        future_map._resolve_draft_input.assert_called_once_with(eagle_input)

    def test_ngram_relay_accepts_route_dtype_changes(self):
        future_map = object.__new__(FutureMap)
        future_map.accept_tokens_buf = torch.zeros((8, 6), dtype=torch.int64)
        future_map.accept_lens_buf = torch.zeros(8, dtype=torch.int32)
        future_map._maybe_init_ngram_bufs = MagicMock()
        payload = RelayPayload(
            bonus_tokens=None,
            accept_tokens=torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.int32),
            accept_lens=torch.tensor([2], dtype=torch.int64),
        )

        future_map._stash_ngram(torch.tensor([3]), payload)

        self.assertEqual(future_map.accept_tokens_buf[3].tolist(), [1, 2, 0, 0, 0, 0])
        self.assertEqual(future_map.accept_lens_buf[3].item(), 2)


class TestHybridForwardDispatch(CustomTestCase):
    def test_extend_installs_target_prefill_backend(self):
        result = SimpleNamespace(
            next_draft_input=EagleDraftInput(),
            next_token_ids=torch.tensor([7]),
            accept_lens=None,
            new_seq_lens=torch.tensor([11]),
        )
        retrieval_worker = SimpleNamespace(sync_hybrid_state=MagicMock())
        neural_worker = SimpleNamespace(
            speculative_num_draft_tokens=4,
            activate_step_by_batch=MagicMock(),
            forward_batch_generation=MagicMock(return_value=result),
        )
        controller = object.__new__(HybridController)
        controller.retrieval_worker = retrieval_worker
        controller.neural_worker = neural_worker
        controller._apply_runtime_state = MagicMock()
        controller._apply_target_prefill_backend = MagicMock()
        controller._update_route_stats = MagicMock()
        controller._runtime_widths = (4, 6)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=MagicMock(return_value=False)),
            is_extend_in_batch=False,
            seq_lens=torch.ones(1, dtype=torch.int32),
            reqs=[SimpleNamespace()],
            spec_info=None,
        )

        controller.forward_batch_generation(batch)

        controller._apply_runtime_state.assert_called_once_with("neural")
        controller._apply_target_prefill_backend.assert_called_once_with()
        neural_worker.forward_batch_generation.assert_called_once()
        retrieval_worker.sync_hybrid_state.assert_called_once_with(batch, result)
        self.assertIsInstance(result.next_draft_input, HybridVerifyInput)

    def test_forwards_scheduler_callbacks_to_selected_worker(self):
        for route in ("retrieval", "neural"):
            with self.subTest(route=route):
                if route == "retrieval":
                    native_next_input = NgramVerifyInput(
                        draft_token_num=6,
                        new_seq_lens=torch.tensor([11]),
                        accept_tokens=torch.tensor([7, 0, 0, 0, 0, 0]),
                        accept_lens=torch.tensor([1]),
                    )
                else:
                    native_next_input = EagleDraftInput()
                result = SimpleNamespace(
                    next_draft_input=native_next_input,
                    next_token_ids=torch.tensor([7]),
                    accept_lens=torch.tensor([1]),
                    new_seq_lens=torch.tensor([11]),
                    speculative_num_draft_tokens=6 if route == "retrieval" else 1,
                )
                initial_eagle_input = EagleDraftInput()
                initial_ngram_input = NgramVerifyInput(
                    draft_token_num=6, new_seq_lens=torch.tensor([10])
                )
                initial_hybrid_input = HybridVerifyInput(
                    initial_eagle_input, initial_ngram_input
                )

                def forward_selected(batch, **_kwargs):
                    expected_input = (
                        initial_ngram_input
                        if route == "retrieval"
                        else initial_eagle_input
                    )
                    self.assertIs(batch.spec_info, expected_input)
                    return result

                retrieval_worker = SimpleNamespace(
                    speculative_num_draft_tokens=6,
                    forward_batch_generation=MagicMock(side_effect=forward_selected),
                    sync_hybrid_state=MagicMock(),
                )
                neural_worker = SimpleNamespace(
                    speculative_num_draft_tokens=4,
                    forward_batch_generation=MagicMock(side_effect=forward_selected),
                    sync_hybrid_state=MagicMock(return_value=EagleDraftInput()),
                    activate_step_by_batch=MagicMock(),
                )
                controller = object.__new__(HybridController)
                controller.retrieval_worker = retrieval_worker
                controller.neural_worker = neural_worker
                controller._match_continuation_lengths = MagicMock(
                    return_value=([1], [1])
                )
                controller._should_use_retrieval = MagicMock(
                    return_value=route == "retrieval"
                )
                controller._apply_runtime_state = MagicMock()
                controller._update_route_stats = MagicMock()
                controller._runtime_widths = (4, 6)

                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(
                        is_decode=MagicMock(return_value=True)
                    ),
                    is_extend_in_batch=False,
                    seq_lens=torch.ones(1, dtype=torch.int32),
                    reqs=[SimpleNamespace()],
                    spec_info=initial_hybrid_input,
                )
                on_publish = MagicMock()
                grammar_barrier = MagicMock()
                pp_proxy_tensors = object()

                actual = controller.forward_batch_generation(
                    batch,
                    on_publish=on_publish,
                    grammar_barrier=grammar_barrier,
                    pp_proxy_tensors=pp_proxy_tensors,
                )

                selected_worker = (
                    retrieval_worker if route == "retrieval" else neural_worker
                )
                other_worker = (
                    neural_worker if route == "retrieval" else retrieval_worker
                )
                selected_worker.forward_batch_generation.assert_called_once_with(
                    batch,
                    on_publish=on_publish,
                    grammar_barrier=grammar_barrier,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                other_worker.forward_batch_generation.assert_not_called()
                other_worker.sync_hybrid_state.assert_called_once_with(batch, result)
                self.assertIs(actual, result)
                self.assertEqual(result.spec_route, route)
                self.assertIsInstance(result.next_draft_input, HybridVerifyInput)
                self.assertIsInstance(
                    result.next_draft_input.eagle_draft_input, EagleDraftInput
                )
                self.assertIsInstance(
                    result.next_draft_input.ngram_verify_input, NgramVerifyInput
                )


if __name__ == "__main__":
    unittest.main()
