"""Equivalence tests for the EagleDraftWorker topk=1 chain fast path.

For topk=1 the draft tree degenerates to a chain, so `draft_forward` skips the
cat/topk/sort/gather of the slow path and returns pre-allocated constants. These
tests check that the pre-allocated `parent_list` / `top_scores_index` match the
slow path (`organize_draft_results`) for num_steps in {1, 2, 3, 4}.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
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


class TestDraftExtendTopk1PostprocessRouting(CustomTestCase):
    def _make_draft_extend_case(self, *, use_graph: bool, seed_dsa: bool):
        num_requests = 2
        num_tokens_per_req = 6
        num_tokens = num_requests * num_tokens_per_req
        logits = torch.randn((num_tokens, 31), device=DEVICE)
        hidden_states = torch.randn((num_tokens, 7), device=DEVICE)
        logits_output = SimpleNamespace(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(num_tokens, dtype=torch.long, device=DEVICE),
            spec_info=SimpleNamespace(dsa_seed_topk_capture=None),
        )

        graph_dsa = (
            torch.arange(40 * 5, dtype=torch.int32, device=DEVICE).view(40, 5)
            if seed_dsa
            else None
        )
        graph_runner = (
            SimpleNamespace(
                can_run_graph=MagicMock(return_value=True),
                execute=MagicMock(return_value=logits_output),
                buffers=SimpleNamespace(dsa_seed_topk_capture=graph_dsa),
            )
            if use_graph
            else None
        )
        draft_runner = SimpleNamespace(
            canary_manager=None,
            forward=MagicMock(
                return_value=SimpleNamespace(logits_output=logits_output)
            ),
        )
        worker = object.__new__(EagleDraftWorker)
        worker.device = DEVICE
        worker.speculative_num_draft_tokens = num_tokens_per_req
        worker.plan_stream_ctx = contextlib.nullcontext()
        worker.plan_stream = None
        worker.cuda_graph_runner_for_draft_extend = graph_runner
        worker.draft_runner = draft_runner
        worker.seed_dsa_topk_from_draft_extend = seed_dsa
        worker.dsa_extend_topk_buf = None
        worker.dsa_index_topk = 5
        worker.topk = 1
        worker.server_args = _fake_server_args(speculative_use_rejection_sampling=False)

        next_draft_input = SimpleNamespace()
        batch = SimpleNamespace(
            seq_lens=[1, 1],
            sampling_info=SimpleNamespace(temperatures=None),
        )
        batch_result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden_states),
            accept_lens=torch.tensor([1, 3], dtype=torch.long, device=DEVICE),
            next_token_ids=torch.tensor([4, 5], dtype=torch.int32, device=DEVICE),
            next_draft_input=next_draft_input,
        )
        return worker, batch, batch_result, forward_batch, logits_output, graph_dsa

    def test_cuda_topk1_routes_graph_and_eager_outputs_through_fused_kernel(self):
        for use_graph in (False, True):
            for seed_dsa in (False, True):
                with self.subTest(use_graph=use_graph, seed_dsa=seed_dsa):
                    (
                        worker,
                        batch,
                        batch_result,
                        forward_batch,
                        logits_output,
                        graph_dsa,
                    ) = self._make_draft_extend_case(
                        use_graph=use_graph, seed_dsa=seed_dsa
                    )
                    expected_topk_p = torch.ones((2, 1), device=DEVICE)
                    expected_topk_index = torch.tensor(
                        [[7], [11]], dtype=torch.long, device=DEVICE
                    )
                    expected_hidden = torch.randn((2, 7), device=DEVICE)
                    expected_dsa = (
                        torch.randint(0, 10, (2, 5), dtype=torch.int32, device=DEVICE)
                        if seed_dsa
                        else None
                    )

                    with patch(
                        "sglang.srt.speculative.eagle_worker_v2._is_cuda", True
                    ), patch(
                        "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
                        return_value=forward_batch,
                    ), patch(
                        "sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan"
                    ), patch(
                        "sglang.srt.speculative.eagle_worker_v2.maybe_detect_inf"
                    ), patch(
                        "sglang.srt.speculative.eagle_worker_v2.draft_extend_topk1_postprocess",
                        return_value=(
                            expected_topk_p,
                            expected_topk_index,
                            expected_hidden,
                            expected_dsa,
                        ),
                    ) as fused:
                        worker._draft_extend_for_decode(batch, batch_result)

                    expected_select_index = torch.tensor(
                        [0, 8], dtype=torch.long, device=DEVICE
                    )
                    self.assertEqual(fused.call_count, 1)
                    args = fused.call_args.args
                    self.assertIs(args[0], logits_output.next_token_logits)
                    torch.testing.assert_close(args[1], expected_select_index)
                    self.assertIs(args[2], logits_output.hidden_states)
                    if seed_dsa and use_graph:
                        self.assertIs(args[3], graph_dsa)
                    elif seed_dsa:
                        self.assertIs(
                            args[3], forward_batch.spec_info.dsa_seed_topk_capture
                        )
                    else:
                        self.assertIsNone(args[3])
                    self.assertIs(batch_result.next_draft_input.topk_p, expected_topk_p)
                    self.assertIs(
                        batch_result.next_draft_input.topk_index,
                        expected_topk_index,
                    )
                    self.assertIs(
                        batch_result.next_draft_input.hidden_states, expected_hidden
                    )
                    if seed_dsa:
                        self.assertIs(
                            batch_result.next_draft_input.dsa_topk_indices,
                            expected_dsa,
                        )

    def test_non_cuda_topk1_keeps_torch_fallback(self):
        worker, batch, batch_result, forward_batch, logits_output, _ = (
            self._make_draft_extend_case(use_graph=False, seed_dsa=False)
        )
        expected_select_index = torch.tensor([0, 8], dtype=torch.long, device=DEVICE)
        expected_logits = logits_output.next_token_logits[expected_select_index].clone()
        expected_hidden = logits_output.hidden_states[expected_select_index].clone()

        with patch("sglang.srt.speculative.eagle_worker_v2._is_cuda", False), patch(
            "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
            return_value=forward_batch,
        ), patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan"), patch(
            "sglang.srt.speculative.eagle_worker_v2.maybe_detect_inf"
        ), patch(
            "sglang.srt.speculative.eagle_worker_v2.draft_extend_topk1_postprocess"
        ) as fused:
            worker._draft_extend_for_decode(batch, batch_result)

        fused.assert_not_called()
        torch.testing.assert_close(
            batch_result.next_draft_input.topk_index,
            torch.argmax(expected_logits, dim=-1, keepdim=True),
        )
        torch.testing.assert_close(
            batch_result.next_draft_input.hidden_states,
            expected_hidden,
        )

    def test_cuda_non_topk1_modes_keep_materialized_fallback(self):
        for topk, use_rejection_sampling in ((1, True), (2, False)):
            with self.subTest(topk=topk, use_rejection_sampling=use_rejection_sampling):
                worker, batch, batch_result, forward_batch, logits_output, _ = (
                    self._make_draft_extend_case(use_graph=False, seed_dsa=False)
                )
                worker.topk = topk
                worker.server_args.speculative_use_rejection_sampling = (
                    use_rejection_sampling
                )
                expected_topk_p = torch.ones((2, topk), device=DEVICE)
                expected_topk_index = torch.zeros(
                    (2, topk), dtype=torch.long, device=DEVICE
                )
                expected_draft_probs = torch.randn((2, 31), device=DEVICE)

                with patch(
                    "sglang.srt.speculative.eagle_worker_v2._is_cuda", True
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
                    return_value=forward_batch,
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan"
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.maybe_detect_inf"
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.draft_extend_topk1_postprocess"
                ) as fused, patch(
                    "sglang.srt.speculative.eagle_worker_v2.sample_draft_proposal",
                    return_value=(
                        expected_draft_probs,
                        expected_topk_p,
                        expected_topk_index,
                    ),
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.renorm_draft_probs",
                    return_value=torch.empty((2, 31), device=DEVICE),
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.fast_topk",
                    return_value=(expected_topk_p, expected_topk_index),
                ):
                    worker._draft_extend_for_decode(batch, batch_result)

                fused.assert_not_called()
                self.assertEqual(logits_output.next_token_logits.shape, (2, 31))
                self.assertEqual(logits_output.hidden_states.shape, (2, 7))
                self.assertIs(
                    batch_result.next_draft_input.topk_index, expected_topk_index
                )
                if use_rejection_sampling:
                    self.assertIs(
                        batch_result.next_draft_input.draft_probs,
                        expected_draft_probs,
                    )


class TestEagleWorkerV2BackendFallback(CustomTestCase):
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
                    get_verify_buffers_to_fill_after_draft=lambda: (None, None),
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
