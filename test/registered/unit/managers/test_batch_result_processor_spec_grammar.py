"""Unit tests for Spec V2 grammar truncation in _resolve_spec_v2_tokens.

The grammar-constrained spec path stops accepting at the grammar-terminating
token, so the over-drafted suffix is never committed to KV nor emitted.
"""

import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.decode import DecodeRequest, DecodeTransferQueue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import (
    REQUEST_REASONING_END_TOKEN_IDS_KEY,
    SamplingParams,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeGrammar:
    """Grammar stub that terminates after `terminate_after` accepted tokens."""

    def __init__(self, terminate_after: int):
        self.accepted = []
        self.finished = False
        self._terminate_after = terminate_after

    def accept_token(self, token_id: int):
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self._terminate_after


class _FakeSpecAlgorithm:
    def is_none(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False


class _FakeForwardMode:
    def is_decode(self) -> bool:
        return True

    def is_extend(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = any(req.grammar is not None for req in reqs)
        self.forward_mode = _FakeForwardMode()
        self.spec_algorithm = _FakeSpecAlgorithm()


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        beam_coordinator=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _make_req(terminate_after: int) -> Req:
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid="r0",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
    )
    req.grammar = _FakeGrammar(terminate_after=terminate_after)
    req.kv.kv_committed_len = 0
    return req


def _make_result(num_draft_tokens, accept_lens, flat_tokens):
    return GenerationBatchResult(
        next_token_ids=torch.tensor(flat_tokens, dtype=torch.long),
        accept_lens=torch.tensor(accept_lens, dtype=torch.long),
        speculative_num_draft_tokens=num_draft_tokens,
    )


def _commit_disagg_handoff(
    req: Req,
    processor: SchedulerBatchResultProcessor,
    token_id: int,
    *,
    replayed_boundary: bool = False,
) -> None:
    queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
    queue.scheduler = SimpleNamespace(
        batch_result_processor=processor, kv_checksum_computer=None
    )
    queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    queue.metadata_buffers = SimpleNamespace(
        get_buf=lambda _: (
            torch.tensor([token_id], dtype=torch.long),
            torch.zeros(7, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, dtype=torch.long),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.tensor([1], dtype=torch.long),
        )
    )
    req.bootstrap_host = "127.0.0.1"
    req.bootstrap_room = 1
    if replayed_boundary:
        req.pd_rebootstrap_forced_output_id = token_id
    decode_req = DecodeRequest(
        req=req,
        kv_receiver=SimpleNamespace(clear=lambda: None),
        metadata_buffer_index=0,
        is_rebootstrap=replayed_boundary,
    )

    queue._commit_transfer_to_req(decode_req)


class TestSpecV2GrammarTruncation(CustomTestCase):
    def test_resolve_truncates_after_grammar_completion(self):
        req = _make_req(terminate_after=2)
        proc = _make_processor()
        # stride=4, accept_len=3 -> proposed [101, 102, 103]; grammar finishes at 102.
        result = _make_result(4, [3], [101, 102, 103, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[101, 102]])
        # No pre-claim: commit the full retained run (no -1 refund).
        self.assertEqual(req.kv.kv_committed_len, 2)

    def test_resolve_keeps_all_when_grammar_not_terminated(self):
        req = _make_req(terminate_after=99)
        proc = _make_processor()
        result = _make_result(4, [3], [201, 202, 203, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[201, 202, 203]])
        self.assertEqual(req.kv.kv_committed_len, 3)


class TestReasoningTokenAccounting(CustomTestCase):
    def test_multi_token_end_can_span_decode_steps(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        processor = _make_processor()
        processor.model_config.think_end_ids = [7, 8]

        processor._maybe_update_reasoning_tokens(req, [10, 7])
        processor._maybe_update_reasoning_tokens(req, [8, 11])

        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)

    def test_request_selected_end_ignores_other_closer(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {
            REQUEST_REASONING_END_TOKEN_IDS_KEY: [17, 18]
        }
        processor = _make_processor()
        processor.model_config.think_end_ids = [7, 8]
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7, 8],
            [17, 18],
        ]

        # The global/default closer must not end a medium request.
        processor._maybe_update_reasoning_tokens(req, [10, 7])
        processor._maybe_update_reasoning_tokens(req, [8, 11])
        self.assertFalse(req._is_reasoning_over)

        processor._maybe_update_reasoning_tokens(req, [10, 17])
        processor._maybe_update_reasoning_tokens(req, [18, 11])

        self.assertEqual(req.reasoning_tokens, 7)
        self.assertTrue(req._is_reasoning_over)

    def test_disagg_handoff_can_start_multi_token_selected_end(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {
            REQUEST_REASONING_END_TOKEN_IDS_KEY: [17, 18]
        }
        processor = _make_processor()
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7, 8],
            [17, 18],
        ]

        _commit_disagg_handoff(req, processor, 17)
        self.assertEqual(req.reasoning_tokens, 1)
        self.assertFalse(req._is_reasoning_over)

        processor._maybe_update_reasoning_tokens(req, 18)

        self.assertEqual(req.reasoning_tokens, 2)
        self.assertTrue(req._is_reasoning_over)

    def test_disagg_rebootstrap_does_not_recount_boundary(self):
        req = _make_req(terminate_after=99)
        req.require_reasoning = True
        req.sampling_params.custom_params = {REQUEST_REASONING_END_TOKEN_IDS_KEY: [17]}
        processor = _make_processor()
        processor.model_config.request_selectable_think_end_id_sequences = [
            [7],
            [17],
        ]

        _commit_disagg_handoff(req, processor, 17, replayed_boundary=True)

        self.assertEqual(req.reasoning_tokens, 0)
        self.assertFalse(req._is_reasoning_over)


class TestFinishedMambaSpecCheckpoint(CustomTestCase):
    def test_truncated_live_state_is_not_cached_but_kv_is_released(self):
        """Finishing inside a verified run must not cache its later live state."""
        cases = (
            # kind, grammar retained, max output, stop token, non-draft count, cache
            ("grammar", 1, 256, None, 1, False),
            ("offloaded", 1, 256, None, 1, False),
            ("grammar_abort", 0, 256, None, 1, False),
            ("eos", None, 256, 4, 1, False),
            ("length", None, 2, None, 1, False),
            ("stop", None, 256, 4, 1, False),
            ("aligned", None, 3, None, 1, True),
            ("grammar_aligned", 3, 256, None, 1, True),
            ("two_non_draft", 2, 256, None, 2, False),
        )
        for kind, retained, limit, stop, non_draft, expected in cases:
            with (
                self.subTest(kind=kind),
                get_context().override_server_args(
                    speculative_algorithm="EAGLE",
                    mamba_radix_cache_strategy="no_buffer",
                    disable_overlap_schedule=True,
                    disaggregation_decode_enable_offload_kvcache=kind == "offloaded",
                ),
            ):
                processor = _make_processor()
                req = _make_req(retained or 3)
                req.origin_input_ids = array("q", req.origin_input_ids)
                if retained is None:
                    req.grammar = None
                elif retained == 0:
                    req.grammar.accept_token = MagicMock(
                        side_effect=ValueError("invalid")
                    )
                req.sampling_params.max_new_tokens = limit
                if kind == "eos":
                    req.eos_token_ids = {stop}
                else:
                    req.sampling_params.stop_token_ids = {stop} if stop else set()
                req.output_ids.append(3)  # Prefill's bonus has not entered state yet.
                req.kv.kv_committed_len = 3
                req.kv.kv_allocated_len = 6
                req.kv.req_pool_idx = 0
                result = _make_result(3, [3], [4, 5, 6])
                result.num_non_draft_tokens_per_req = non_draft
                batch = _FakeBatch([req])
                batch.mamba_track_mask_cpu = None
                tokens = processor._resolve_spec_v2_tokens(result, batch)[0]
                req.output_ids.extend(tokens)
                req.update_finish_state(len(tokens))
                self.assertTrue(req.finished())

                # Model allocation is the boundary stub; token settlement and
                # release_kv_cache both run their production implementations.
                cached_lengths = []
                freed_ranges = []
                tree = MagicMock()
                tree.supports_mamba.return_value = True
                tree.claim_kv_row.return_value = False
                tree.token_to_kv_pool_allocator.page_size = 1
                tree.checkpoint.side_effect = lambda req, *, up_to: (
                    cached_lengths.append(up_to)
                )
                tree.free_kv_row.side_effect = lambda kv, ranges: freed_ranges.extend(
                    ranges
                )
                processor = replace(
                    processor,
                    tree_cache=tree,
                    decode_offload_manager=SimpleNamespace(
                        offload_kv_cache=lambda req: True
                    ),
                )
                processor._handle_finish_state_updated_req(req, batch, result, 0, None)
                if kind == "offloaded":
                    self.assertFalse(req.kv.is_kv_released)
                    release_kv_cache(req, tree, checkpoint=True)

                self.assertEqual(cached_lengths, [6] if expected else [])
                self.assertEqual(req.skip_radix_cache_insert, not expected)
                self.assertTrue(req.kv.is_kv_released)
                self.assertEqual(sum(hi - lo for lo, hi in freed_ranges), 6)

    def test_dedicated_checkpoints_are_not_rejected_for_live_state_overshoot(self):
        processor = _make_processor()
        processor = replace(
            processor, tree_cache=SimpleNamespace(supports_mamba=lambda: True)
        )
        req = _make_req(1)
        result = _make_result(3, [3], [4, 5, 6])
        result.num_correct_drafts_per_req_cpu = [2]
        result.grammar_retained_tokens = [[4]]
        for slots in (1, 2):
            with self.subTest(slots=slots):
                req.kv.mamba_ping_pong_track_buffer = torch.arange(slots)
                self.assertTrue(processor._mamba_can_cache_finished_req(req, result, 0))


if __name__ == "__main__":
    unittest.main()
