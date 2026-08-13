"""Equivalence tests for the EagleDraftWorker topk=1 chain fast path.

For topk=1 the draft tree degenerates to a chain, so `draft_forward` skips the
cat/topk/sort/gather of the slow path and returns pre-allocated constants. These
tests check that the pre-allocated `parent_list` / `top_scores_index` match the
slow path (`organize_draft_results`) for num_steps in {1, 2, 3, 4}.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
    _aiter_draft_topk1,
    _aiter_draft_topk1_postprocess,
    _prune_draft_extend_logits,
    _use_aiter_draft_topk1,
    _use_draft_topk1_postprocess,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fake_server_args(**fields):
    """server_args stand-in: carries fields and the override() entry point."""
    ns = SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_chain_lists(num_steps: int, bs: int):
    """Build the (score, token, parents) lists a topk=1 chain produces.

    Shapes/values mirror `select_top_k_tokens` for topk=1: each step yields one
    token; the first step's parents are [-1, 0], later steps' parents are [i].
    """
    score_list, token_list, parents_list = [], [], []
    for i in range(num_steps):
        # Strictly decreasing scores, as a real chain produces (cumulative probs).
        score_list.append(torch.full((bs, 1, 1), float(num_steps - i), device=DEVICE))
        token_list.append(
            torch.arange(i * bs, (i + 1) * bs, device=DEVICE).unsqueeze(1)
        )
        if i == 0:
            parents_list.append(
                torch.tensor([-1, 0], dtype=torch.long, device=DEVICE).repeat(bs, 1)
            )
        else:
            parents_list.append(torch.full((bs, 1), i, dtype=torch.long, device=DEVICE))
    return score_list, token_list, parents_list


def _make_worker(num_steps: int, num_draft_tokens: int):
    worker = object.__new__(EagleDraftWorker)
    worker.topk = 1
    worker.device = DEVICE
    worker.speculative_num_steps = num_steps
    worker.speculative_num_draft_tokens = num_draft_tokens
    worker.server_args = _fake_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=8)),
        max_running_requests=8,
    )
    return worker


def _make_backend_factory(decode_backend, draft_extend_backend, captured_kwargs=None):
    class FakeDraftBackendFactory:
        def __init__(self, *args, **kwargs):
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)

        def create_decode_backend(self):
            return decode_backend

        def create_draft_extend_backend(self):
            return draft_extend_backend

    return FakeDraftBackendFactory


class TestEagleWorkerV2Topk1FastPath(CustomTestCase):
    def setUp(self):
        # _rebuild_topk1_chain_buffers sizes its preallocation from the
        # published config: get_exec().graph.cuda_graph_config stays None on
        # the dummy-boundary publish (no resolution), so
        # get_schedule().max_running_requests alone sizes the buffers.
        override = get_context().override_server_args(max_running_requests=8)
        override.install()
        self.addCleanup(override.restore)

    def test_fast_path_matches_slow_path(self):
        bs = 3
        for num_steps in (1, 2, 3, 4):
            with self.subTest(num_steps=num_steps):
                num_draft_tokens = num_steps + 1
                worker = _make_worker(num_steps, num_draft_tokens)
                worker._rebuild_topk1_chain_buffers()

                score_list, token_list, parents_list = _make_chain_lists(num_steps, bs)
                ref_parent, ref_index, ref_tokens = organize_draft_results(
                    score_list, token_list, parents_list, num_draft_tokens
                )

                fast_parent = worker._topk1_parents_prealloc[:bs]
                fast_index = worker._topk1_score_indices_prealloc[:bs]
                fast_tokens = torch.cat(token_list, dim=1)

                self.assertEqual(fast_parent.shape, ref_parent.shape)
                self.assertEqual(fast_parent.tolist(), ref_parent.long().tolist())
                self.assertEqual(fast_index.tolist(), ref_index.long().tolist())
                self.assertEqual(fast_tokens.tolist(), ref_tokens.tolist())

                # The kernel reads these via data_ptr() as contiguous int64.
                self.assertEqual(fast_parent.dtype, torch.long)
                self.assertEqual(fast_index.dtype, torch.long)
                self.assertTrue(fast_parent.is_contiguous())
                self.assertTrue(fast_index.is_contiguous())

    def test_assert_on_inconsistent_steps_and_draft_tokens(self):
        # num_draft_tokens must equal num_steps + 1 for topk=1.
        worker = _make_worker(num_steps=3, num_draft_tokens=3)
        with self.assertRaises(AssertionError):
            worker._rebuild_topk1_chain_buffers()

    def test_raw_logits_postprocess_tracks_local_backend(self):
        with patch("sglang.srt.speculative.eagle_worker_v2._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_worker_v2._use_aiter", True
        ):
            self.assertTrue(_use_draft_topk1_postprocess())

        with patch("sglang.srt.speculative.eagle_worker_v2._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_worker_v2._use_aiter", False
        ):
            self.assertFalse(_use_draft_topk1_postprocess())

    def test_cuda_raw_logits_postprocess_remains_enabled(self):
        with patch("sglang.srt.speculative.eagle_worker_v2._is_cuda", True), patch(
            "sglang.srt.speculative.eagle_worker_v2._use_aiter", False
        ):
            self.assertTrue(_use_draft_topk1_postprocess())

    def test_aiter_postprocess_updates_indices_positions_and_draft_chain(self):
        logits = torch.zeros((3, 16), dtype=torch.float32, device=DEVICE)
        positions = torch.tensor([10, 20, 30], dtype=torch.long, device=DEVICE)
        draft_tokens = torch.full((3, 4), -1, dtype=torch.long, device=DEVICE)

        def fake_greedy_sample(output, _logits):
            output.copy_(
                torch.tensor([2, 5, 11], dtype=torch.int32, device=output.device)
            )

        with patch(
            "sglang.srt.speculative.eagle_worker_v2._aiter_greedy_sample",
            side_effect=fake_greedy_sample,
        ):
            topk_p, topk_index = _aiter_draft_topk1_postprocess(
                logits, positions, draft_tokens, draft_token_column=2
            )

        torch.testing.assert_close(
            topk_index,
            torch.tensor([[2], [5], [11]], dtype=torch.long, device=DEVICE),
        )
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p))
        torch.testing.assert_close(
            positions, torch.tensor([11, 21, 31], dtype=torch.long, device=DEVICE)
        )
        torch.testing.assert_close(
            draft_tokens[:, 2],
            torch.tensor([2, 5, 11], dtype=torch.long, device=DEVICE),
        )
        torch.testing.assert_close(
            draft_tokens[:, (0, 1, 3)],
            torch.full((3, 3), -1, dtype=torch.long, device=DEVICE),
        )

    def test_aiter_raw_selector_delegates_tie_and_nonfinite_inputs_unchanged(self):
        logits = torch.tensor(
            [
                [1.0, 5.0, 5.0, 0.0],
                [float("nan"), 2.0, 1.0, 0.0],
                [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                [0.0, 1.0, float("inf"), 2.0],
            ],
            device=DEVICE,
        )
        original = logits.clone()

        def fake_greedy_sample(output, raw_logits):
            torch.testing.assert_close(raw_logits, original, equal_nan=True)
            # Model the established fallback's softmax + fast_topk(topk=1),
            # whose index operation is torch.max.
            reference_index = torch.max(
                torch.softmax(raw_logits, dim=-1), dim=-1
            ).indices
            output.copy_(reference_index.to(torch.int32))

        with patch(
            "sglang.srt.speculative.eagle_worker_v2._aiter_greedy_sample",
            side_effect=fake_greedy_sample,
        ):
            topk_p, topk_index = _aiter_draft_topk1(logits)

        torch.testing.assert_close(logits, original, equal_nan=True)
        expected_index = torch.max(
            torch.softmax(original, dim=-1), dim=-1, keepdim=True
        ).indices
        torch.testing.assert_close(topk_index, expected_index)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p))

    def test_aiter_route_preserves_all_fallback_gates(self):
        with patch("sglang.srt.speculative.eagle_worker_v2._is_hip", True), patch(
            "sglang.srt.speculative.eagle_worker_v2._use_aiter", True
        ):
            self.assertTrue(_use_aiter_draft_topk1(1, None, False))
            self.assertFalse(_use_aiter_draft_topk1(2, None, False))
            self.assertFalse(
                _use_aiter_draft_topk1(
                    1, torch.tensor([0], dtype=torch.long, device=DEVICE), False
                )
            )
            self.assertFalse(_use_aiter_draft_topk1(1, None, True))

        with patch("sglang.srt.speculative.eagle_worker_v2._is_hip", True), patch(
            "sglang.srt.speculative.eagle_worker_v2._use_aiter", False
        ):
            self.assertFalse(_use_aiter_draft_topk1(1, None, False))

    def test_draft_extend_row_pruning_keeps_full_hidden_capture(self):
        hidden_states = torch.arange(24, device=DEVICE).reshape(6, 4)
        select_index = torch.tensor([1, 5], dtype=torch.long, device=DEVICE)
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            draft_extend_select_index=select_index,
        )

        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            _,
            _,
        ) = LogitsProcessor._get_pruned_states(
            None, hidden_states, None, None, metadata
        )
        stored_hidden_states = LogitsProcessor._get_hidden_states_to_store(
            None,
            hidden_states,
            None,
            None,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            metadata,
        )

        torch.testing.assert_close(pruned_states, hidden_states[select_index])
        self.assertEqual(pruned_states.shape, (2, 4))
        self.assertIs(stored_hidden_states, hidden_states)
        self.assertEqual(stored_hidden_states.shape, (6, 4))

    def test_draft_extend_pruning_and_graph_row_count_gates(self):
        server_args = object()
        with patch(
            "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner._is_hip",
            True,
        ), patch(
            "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner.require_gathered_buffer",
            return_value=False,
        ):
            self.assertTrue(_prune_draft_extend_logits(server_args))
        with patch(
            "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner._is_hip",
            True,
        ), patch(
            "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner.require_gathered_buffer",
            return_value=True,
        ):
            self.assertFalse(_prune_draft_extend_logits(server_args))
        with patch(
            "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner._is_hip",
            False,
        ), patch(
            "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner.require_gathered_buffer",
            return_value=False,
        ):
            self.assertFalse(_prune_draft_extend_logits(server_args))

        runner = EAGLEDraftExtendCudaGraphRunner.__new__(
            EAGLEDraftExtendCudaGraphRunner
        )
        runner.prune_draft_extend_logits = True
        self.assertEqual(runner._num_logit_rows(batch_size=3, num_tokens=12), 3)
        runner.prune_draft_extend_logits = False
        self.assertEqual(runner._num_logit_rows(batch_size=3, num_tokens=12), 12)


class TestEagleWorkerV2BackendFallback(CustomTestCase):
    def setUp(self):
        # The adaptive state-machine paths write live spec switches through
        # get_context().override, which needs a published config.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def test_missing_seed_cuda_graph_fallback(self):
        graph_result = (
            [],
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            None,
        )
        tree_result = (
            torch.empty((0,), dtype=torch.bool, device=DEVICE),
            torch.zeros((1,), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((2,), dtype=torch.long, device=DEVICE),
        )

        for seed_enabled, seed_present, expect_graph in (
            (True, False, False),
            (True, True, True),
            (False, False, True),
        ):
            with self.subTest(
                seed_enabled=seed_enabled,
                seed_present=seed_present,
            ):
                worker = object.__new__(EagleDraftWorker)
                worker.req_to_token_pool = None
                worker.cuda_graph_runner = SimpleNamespace(
                    execute=MagicMock(return_value=graph_result)
                )
                worker.draft_runner = SimpleNamespace(canary_manager=None)
                worker.topk = 1
                worker.speculative_num_steps = 1
                worker.speculative_num_draft_tokens = 2
                worker.device = DEVICE
                worker.tree_mask_mode = None
                worker.seed_dsa_topk_from_draft_extend = seed_enabled
                worker.index_share_for_mtp_iteration = True
                forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                worker.draft_forward = MagicMock(return_value=graph_result)
                attn_backend = SimpleNamespace(
                    verify_mask=None,
                    max_context_len=1,
                )
                worker.target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(attn_backend=attn_backend)
                )
                draft_input = SimpleNamespace(
                    bonus_tokens=torch.zeros((1,), dtype=torch.long, device=DEVICE),
                    dsa_topk_indices=(
                        torch.ones((1, 1), dtype=torch.int32, device=DEVICE)
                        if seed_present
                        else None
                    ),
                )
                batch = SimpleNamespace(
                    spec_info=draft_input,
                    forward_mode=ForwardMode.DECODE,
                    seq_lens_sum=1,
                    seq_lens=torch.ones((1,), dtype=torch.int32, device=DEVICE),
                )

                with patch(
                    "sglang.srt.speculative.eagle_worker_common.build_tree_kernel_efficient",
                    return_value=tree_result,
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft",
                    return_value=(forward_batch, True),
                ):
                    worker.draft(batch)

                self.assertEqual(worker.cuda_graph_runner.execute.called, expect_graph)
                self.assertEqual(worker.draft_forward.called, not expect_graph)

    def test_preserves_initialized_backend_when_draft_extend_backend_is_unset(self):
        worker = object.__new__(EagleDraftWorker)
        existing_backend = object()
        decode_backend = object()
        worker.server_args = _fake_server_args()
        worker.draft_runner = SimpleNamespace(attn_backend=existing_backend)
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.seed_dsa_topk_from_draft_extend = False

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _make_backend_factory(decode_backend, None),
        ):
            worker.init_attention_backend()

        self.assertIs(worker.draft_attn_backend, decode_backend)
        self.assertIsNone(worker.draft_extend_attn_backend)
        self.assertIs(worker.draft_runner.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_runner.attn_backend, existing_backend)

    def test_uses_draft_extend_backend_when_available(self):
        worker = object.__new__(EagleDraftWorker)
        existing_backend = object()
        decode_backend = object()
        draft_extend_backend = object()
        worker.server_args = _fake_server_args()
        worker.draft_runner = SimpleNamespace(attn_backend=existing_backend)
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.seed_dsa_topk_from_draft_extend = True
        factory_kwargs = {}

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _make_backend_factory(
                decode_backend, draft_extend_backend, captured_kwargs=factory_kwargs
            ),
        ):
            worker.init_attention_backend()

        self.assertIs(worker.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_extend_attn_backend, draft_extend_backend)
        self.assertIs(worker.draft_runner.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_runner.attn_backend, draft_extend_backend)
        self.assertTrue(factory_kwargs["seed_dsa_topk_from_draft_extend"])

    def _make_adaptive_worker(self, runner_attn_backend):
        """An EAGLEWorkerV2 with a draft worker whose state-machine fields are
        filled with sentinels, sufficient to drive _override_worker_state /
        apply_runtime_state without touching the GPU."""
        draft_runner = SimpleNamespace(
            draft_attn_backend=object(),
            attn_backend=runner_attn_backend,
        )
        draft_worker = SimpleNamespace(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            cuda_graph_runner=object(),
            cuda_graph_runner_for_draft_extend=object(),
            draft_runner=draft_runner,
            # _override_worker_state / apply_runtime_state call this hook; the
            # topk=1 buffers are exercised by the fast-path tests above.
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = draft_worker
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=object(), decode_cuda_graph_runner=object()
            )
        )
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.server_args = _fake_server_args(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            cuda_graph_bs_decode=None,
            disable_cuda_graph=False,
        )
        return worker, draft_worker

    def test_override_worker_state_restores_runner_attn_backend(self):
        # build_adaptive_runtime_state runs init_attention_backend inside this
        # context for each candidate step; the runner backend it assigns must
        # not leak into the live worker.
        initial_backend = object()
        candidate_backend = object()
        worker, dw = self._make_adaptive_worker(initial_backend)

        with worker._override_worker_state(3, 4):
            dw.draft_runner.attn_backend = candidate_backend
            self.assertIs(dw.draft_runner.attn_backend, candidate_backend)

        self.assertIs(dw.draft_runner.attn_backend, initial_backend)

    def test_apply_runtime_state_updates_runner_attn_backend(self):
        # Switching to another step config must repoint the runner backend at
        # that config's draft-extend backend (read by the draft-extend forward).
        new_extend_backend = object()
        worker, dw = self._make_adaptive_worker(object())

        state = SpecRuntimeState(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            draft_attn_backend=object(),
            cuda_graph_runner=object(),
            target_attn_backend=object(),
            target_graph_runner=object(),
            draft_extend_attn_backend=new_extend_backend,
            cuda_graph_runner_for_draft_extend=object(),
        )
        worker.apply_runtime_state(state)

        self.assertIs(dw.draft_runner.attn_backend, new_extend_backend)

    def test_spec_v2_attn_backends_include_draft_extend_fallback(self):
        target_backend = object()
        decode_backend = object()
        fallback_backend = object()

        worker = object.__new__(EAGLEWorkerV2)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=target_backend)
        )
        worker._draft_worker = SimpleNamespace(
            draft_attn_backend=decode_backend,
            draft_extend_attn_backend=None,
            draft_runner=SimpleNamespace(attn_backend=fallback_backend),
        )

        self.assertEqual(
            worker.spec_v2_attn_backends,
            (target_backend, decode_backend, fallback_backend),
        )


if __name__ == "__main__":
    unittest.main()
