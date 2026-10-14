from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import (
    prefill_owner_shell,
    result_processor_shell,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Req:
    def __init__(self, *, allocated: bool = True):
        self.rid = "aborted-prefill"
        self.cache_request_handle = CacheRequestHandle(self.rid, 0)
        self.kv = ReqKvInfo(
            req_pool_idx=1 if allocated else None,
            kv_allocated_len=1 if allocated else 0,
            mamba_pool_idx=object() if allocated else None,
        )
        self.disagg = DisaggReqState(
            metadata_buffer_index=7 if allocated else -1, pending_bootstrap=allocated
        )
        self.sender = Mock()
        self.to_finish = FINISH_ABORT() if allocated else None
        self.finished_reason = None if allocated else FINISH_ABORT()
        self.discard_output_reason = None
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        self.output_ids = []
        self.origin_input_ids = list(range(100))
        self.extend_end = None
        self.time_stats = SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_last_chunked_prefill_finish_time=Mock(),
            set_completion_time=Mock(),
        )

    def finished(self):
        return self.finished_reason is not None

    def update_finish_state(self):
        self.finished_reason = self.to_finish
        self.to_finish = None


def _owner():
    return prefill_owner_shell(
        batch_result_processor=result_processor_shell(),
        spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
        tree_cache=Mock(),
        pending_chunk_rids={"aborted-prefill"},
        req_to_metadata_buffer_idx_allocator=Mock(),
        send_kv_chunk=Mock(),
        output_streamer=Mock(),
    )


def _batch(req, *, middle_chunk=False):
    return SimpleNamespace(
        reqs=[req],
        chunked_req=req if middle_chunk else None,
        spec_info=None,
        prefill_stats=None,
        dp_cooperation_info=None,
        extend_lens=None,
        extend_logprob_start_lens=None,
    )


def _result():
    return GenerationBatchResult(next_token_ids=torch.tensor([11]))


def _free_req(req, _tree_cache, *, checkpoint):
    assert checkpoint is False
    req.kv.req_pool_idx = None
    req.kv.mark_kv_released()
    req.kv.mamba_pool_idx = None


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
@patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
def test_aborted_final_result_releases_hybrid_cache(
    checkpoint_kv_cache, release_kv_cache
):
    owner = _owner()
    req = _Req()
    owner.kv_senders[req.rid] = req.sender
    owner.prefilling = [req]

    owner.commit_extend(_batch(req), _result(), [], None)

    assert owner.prefilling == []
    release_kv_cache.assert_called_once_with(req, owner.tree_cache, checkpoint=False)
    checkpoint_kv_cache.assert_not_called()
    req.sender.abort.assert_called_once_with()
    owner.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    owner.tree_cache.finish.assert_called_once_with(
        req.cache_request_handle, CacheRequestOutcome.ABORT
    )
    owner.output_streamer.stream_output.assert_called_once_with([req], False)
    owner.send_kv_chunk.assert_not_called()
    assert req.output_ids == []
    assert req.finished()
    assert req.disagg.metadata_buffer_index == -1
    assert req.rid not in owner.pending_chunk_rids
    assert req.rid not in owner.kv_senders


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_aborted_middle_result_releases_after_last_chunk(release_kv_cache):
    owner = _owner()
    req = _Req()
    owner.kv_senders[req.rid] = req.sender
    req.extend_end = 50

    owner.commit_extend(_batch(req, middle_chunk=True), _result(), [], None)

    release_kv_cache.assert_called_once_with(req, owner.tree_cache, checkpoint=False)
    owner.output_streamer.stream_output.assert_called_once_with([req], False)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_aborted_middle_result_waits_for_inflight_chunk(release_kv_cache):
    owner = _owner()
    req = _Req()
    owner.kv_senders[req.rid] = req.sender
    req.extend_end = len(req.origin_input_ids)
    # Its final chunk launched while the middle chunk was still in flight.
    owner.prefilling = [req]

    owner.commit_extend(_batch(req, middle_chunk=True), _result(), [], None)

    release_kv_cache.assert_not_called()
    owner.output_streamer.stream_output.assert_not_called()
    assert owner.prefilling == [req]

    owner.commit_extend(_batch(req), _result(), [], None)

    assert owner.prefilling == []

    release_kv_cache.assert_called_once_with(req, owner.tree_cache, checkpoint=False)
    owner.output_streamer.stream_output.assert_called_once_with([req], False)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache")
def test_delayed_result_ignores_already_retired_request(release_kv_cache):
    owner = _owner()
    req = _Req(allocated=False)
    owner.kv_senders[req.rid] = req.sender

    owner.commit_extend(_batch(req), _result(), [], None)

    release_kv_cache.assert_not_called()
    req.sender.abort.assert_not_called()
    owner.output_streamer.stream_output.assert_not_called()


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_sender_abort_failure_does_not_skip_local_cleanup(release_kv_cache):
    owner = _owner()
    req = _Req()
    owner.kv_senders[req.rid] = req.sender
    req.sender.abort.side_effect = RuntimeError("transport is down")

    owner.commit_extend(_batch(req), _result(), [], None)

    release_kv_cache.assert_called_once_with(req, owner.tree_cache, checkpoint=False)
    owner.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    owner.output_streamer.stream_output.assert_called_once_with([req], False)
    assert req.finished()


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
@patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
def test_grammar_rejection_retires_prefill_before_transfer(
    checkpoint_kv_cache, release_kv_cache
):
    owner = _owner()
    req = _Req()
    owner.kv_senders[req.rid] = req.sender
    req.to_finish = None
    req.grammar = Mock()
    req.grammar.accept_token.side_effect = ValueError("invalid token")

    owner.commit_extend(_batch(req), _result(), [], None)

    req.grammar.accept_token.assert_called_once_with(11)
    assert req.grammar.finished
    assert req.finished()
    release_kv_cache.assert_called_once_with(req, owner.tree_cache, checkpoint=False)
    checkpoint_kv_cache.assert_not_called()
    owner.send_kv_chunk.assert_not_called()
    owner.output_streamer.stream_output.assert_called_once_with([req], False)
    assert owner.transfer == []


def test_aborted_result_releases_mamba_allocated_before_kv():
    owner = _owner()
    owner.enable_hicache_storage = False
    owner.tree_cache.supports_mamba.return_value = True
    owner.tree_cache.req_to_token_pool.mamba_allocator.free = Mock()
    req = _Req(allocated=False)
    owner.kv_senders[req.rid] = req.sender
    req.kv.mamba_pool_idx = torch.tensor([3])
    req.to_finish = FINISH_ABORT()
    req.finished_reason = None

    owner.commit_extend(_batch(req), _result(), [], None)

    owner.tree_cache.req_to_token_pool.mamba_allocator.free.assert_called_once()
    assert req.kv.mamba_pool_idx is None
    owner.output_streamer.stream_output.assert_called_once_with([req], False)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_failed_kv_load_is_aborted_instead_of_sent_to_decode(release_kv_cache):
    owner = _owner()
    req = _Req()
    owner.kv_senders[req.rid] = req.sender
    req.to_finish = None
    req.discard_output_reason = FINISH_ABORT("External KV cache load failed", 503)
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([11]),
        logits_output=LogitsProcessorOutput(next_token_logits=None),
    )

    owner.commit_extend(_batch(req), result, [], None)

    assert req.finished_reason.status_code == 503
    assert req.output_ids == []
    release_kv_cache.assert_called_once_with(req, owner.tree_cache, checkpoint=False)
    req.sender.abort.assert_called_once_with()
    owner.send_kv_chunk.assert_not_called()


@pytest.mark.parametrize("transport_error", [False, True])
@pytest.mark.parametrize(
    "status,http_status,err_type",
    [
        (SamplingMaskStatus.OVERFLOW, 400, "BadRequestError"),
        (SamplingMaskStatus.INVALID, 500, "InternalServerError"),
    ],
)
@patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
def test_sampling_mask_abort_preserves_error_and_releases_once(
    release_kv_cache, status, http_status, err_type, transport_error
):
    """A failed sender notification must not leak ownership or lose the API error."""
    owner = _owner()
    owner.batch_result_processor.get_sampling_mask_finish_reason = lambda **kwargs: (
        SchedulerBatchResultProcessor.get_sampling_mask_finish_reason(None, **kwargs)
    )
    req = _Req()
    owner.kv_senders[req.rid] = req.sender
    req.to_finish = None
    req.return_sampling_mask = True
    req.time_stats.trace_ctx = Mock()
    if transport_error:
        req.sender.abort.side_effect = RuntimeError("transport is down")
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([11]),
        logits_output=LogitsProcessorOutput(
            next_token_logits=None, next_token_sampling_mask_status=[status]
        ),
    )

    with get_context().override_server_args(sampling_mask_max_tokens=64):
        owner.commit_extend(_batch(req), result, [], None)
        owner.commit_extend(_batch(req), result, [], None)

    assert req.finished_reason.status_code == http_status
    assert req.finished_reason.err_type == err_type
    assert req.output_ids == []
    assert not req.kv.holds_kv and not req.kv.holds_mamba
    assert req.disagg.metadata_buffer_index == -1
    assert not req.disagg.pending_bootstrap
    assert req.rid not in owner.pending_chunk_rids
    release_kv_cache.assert_called_once_with(req, owner.tree_cache, checkpoint=False)
    req.sender.abort.assert_called_once_with()
    owner.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(7)
    owner.tree_cache.finish.assert_called_once_with(
        req.cache_request_handle, CacheRequestOutcome.ABORT
    )
    owner.output_streamer.stream_output.assert_called_once_with([req], False)
    owner.send_kv_chunk.assert_not_called()


class TestPrefillCompleteResult(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="prefill_complete"
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
    def test_ready_observes_accepted_token_and_speculative_sidecars(self, cache):
        owner = _owner()
        owner.spec_algorithm = SimpleNamespace(is_eagle=lambda: True)
        req = _Req()
        owner.kv_senders[req.rid] = req.sender
        req.to_finish = None
        req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
        draft = SimpleNamespace(
            hidden_states=torch.tensor([[1.0, 2.0]]),
            topk_p=torch.tensor([[0.75, 0.25]]),
            topk_index=torch.tensor([[3, 4]]),
            dsa_topk_indices=torch.tensor([[5, 6]]),
        )
        # The launched step keeps no draft input; the result carries it.
        batch = _batch(req)
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([11]), next_draft_input=draft
        )
        observed = []

        def ready():
            observed.append(
                (
                    list(req.output_ids),
                    req.hidden_states_tensor.tolist(),
                    req.output_dsa_topk_indices.tolist(),
                    req in owner.transfer,
                )
            )

        req.sender.mark_prefill_complete.side_effect = ready
        owner.commit_extend(batch, result, [], None)
        self.assertEqual(observed, [([11], [1.0, 2.0], [5, 6], True)])
        self.assertTrue(req.disagg.pending_bootstrap)
        owner.send_kv_chunk.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.release_kv_cache", side_effect=_free_req)
    def test_aborted_final_result_never_publishes_ready(self, release):
        owner = _owner()
        req = _Req()
        owner.kv_senders[req.rid] = req.sender
        owner.commit_extend(_batch(req), _result(), [], None)
        self.assertTrue(req.finished())
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(owner.transfer, [])
        req.sender.mark_prefill_complete.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.should_force_retry", return_value=True)
    def test_discarded_attempt_never_publishes_ready(self, force_retry):
        owner = _owner()
        req = _Req()
        owner.kv_senders[req.rid] = req.sender
        req.to_finish = None
        retried = []
        owner.optimistic_release_and_requeue = lambda r, waiting: (
            retried.append(r) or waiting
        )
        owner.commit_extend(_batch(req), _result(), [], None)
        self.assertEqual(retried, [req])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(owner.transfer, [])
        req.sender.mark_prefill_complete.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
