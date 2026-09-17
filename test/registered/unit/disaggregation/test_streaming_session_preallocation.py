"""Session ownership across decode admission and transfer publication."""

from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.session.session_controller import Session
from sglang.srt.session.streaming_session import StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def runtime_context():
    reset_context()
    publish(ServerArgs(model_path="dummy"), role="tokenizer")
    yield
    reset_context()


def _request(session, rid, tokens):
    wire_req = TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", tokens),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(id=session.session_id),
        bootstrap_host="127.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=42,
    )
    return session.create_req(
        wire_req, tokenizer=None, vocab_size=256, disagg_mode=DisaggregationMode.DECODE
    )


def _fixture(*, rows=1, tokens=64):
    pool = ReqToTokenPool(
        size=rows, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    allocator = TokenToKVPoolAllocator(
        size=tokens,
        dtype=torch.float32,
        device="cpu",
        kvcache=object(),
        need_sort=False,
    )
    cache = StreamingSession(
        ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
    )
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pp_size = 1
    queue.req_to_token_pool = pool
    queue.token_to_kv_pool_allocator = allocator
    queue.tree_cache = cache
    queue.num_reserved_decode_tokens = 0
    queue._num_published_destinations = 0
    queue._uses_swa_tail_prealloc = lambda: False
    queue.queue = []
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue._resolve_pending_reqs = lambda: None
    queue._update_handshake_waiters = lambda *args: None
    queue._hicache_pending_restore_tokens = lambda: 0
    queue._allocatable_token_budgets = lambda **kwargs: allocator.available_size()
    queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
        available_size=lambda: 1, alloc=lambda: 0
    )
    queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
    queue.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
    queue.scheduler = SimpleNamespace(
        enable_hisparse=False,
        enable_lora=False,
        enable_priority_scheduling=False,
        enable_decode_hicache=False,
        tp_worker=SimpleNamespace(is_hybrid_swa=False),
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        output_streamer=Mock(),
    )
    session = Session(4096, session_id="session-a", streaming=True)
    first = _request(session, "first", range(8))
    queue._pre_alloc(first)
    row = first.kv.req_pool_idx
    _finish(cache, first)
    return queue, session, row


def _finish(cache, req):
    req.output_ids.append(17)
    req.finished_reason = FINISH_LENGTH(1)
    req.finished_len = 1
    req._refresh_fill_ids()
    cache.cache_finished_req(req)


def _enqueue(queue, req):
    receiver = Mock()
    pending = DecodeRequest(req=req, kv_receiver=receiver, waiting_for_input=True)
    queue.queue = [pending]
    return pending


@pytest.mark.parametrize("rows", [1, 2])
def test_continuations_reuse_the_row_and_release_all_kv_on_close(rows):
    queue, session, row = _fixture(rows=rows)
    cache = queue.tree_cache
    for turn in range(3):
        req = _request(session, f"append-{turn}", [23, 24])
        # Exercise the real session input construction used by the scheduler.
        assert req.bootstrap_room == 42
        assert req.bootstrap_host == "127.0.0.1"
        assert req.bootstrap_port == 8998
        previous_len = cache.slots[session.session_id].kv.kv_committed_len
        pending = _enqueue(queue, req)
        admitted, failed = queue.pop_preallocated()
        assert admitted == [pending]
        assert failed == []
        assert req.kv is cache.slots[session.session_id].kv
        assert req.kv.req_pool_idx == row
        assert req.kv.cache_protected_len == 0
        assert queue.req_to_token_pool.available_size() == rows - 1
        assert queue.token_to_kv_pool_allocator.available_size() == (
            queue.token_to_kv_pool_allocator.size - len(req.origin_input_ids)
        )
        assert pending.kv_receiver.send_metadata.call_args.kwargs == {
            "decode_prefix_len": previous_len
        }
        _finish(cache, req)
    cache.release_session(session.session_id)
    assert queue.req_to_token_pool.available_size() == rows
    assert queue.token_to_kv_pool_allocator.available_size() == 64


def test_fresh_request_without_a_row_does_not_block_a_session_continuation():
    queue, session, row = _fixture()
    fresh_session = Session(4096, session_id="fresh-session", streaming=True)
    fresh = _request(fresh_session, "fresh", [21, 22])
    first_pending = _enqueue(queue, fresh)
    continuation = _request(session, "continuation", [23, 24])
    second_pending = _enqueue(queue, continuation)
    queue.queue = [first_pending, second_pending]
    queue.req_to_metadata_buffer_idx_allocator.alloc = Mock(return_value=0)

    assert queue.req_to_token_pool.available_size() == 0
    assert queue.pop_preallocated() == ([second_pending], [])
    assert queue.queue == [first_pending]
    assert continuation.kv.req_pool_idx == row
    assert not fresh.kv.holds_kv
    first_pending.kv_receiver.send_metadata.assert_not_called()
    second_pending.kv_receiver.send_metadata.assert_called_once()
    queue.req_to_metadata_buffer_idx_allocator.alloc.assert_called_once()

    # Removing a non-head entry must preserve the fresh request for later
    # admission, once the existing session releases its row.
    _finish(queue.tree_cache, continuation)
    queue.tree_cache.release_session(session.session_id)
    assert queue.pop_preallocated() == ([first_pending], [])
    assert queue.queue == []
    _finish(queue.tree_cache, fresh)
    queue.tree_cache.release_session(fresh_session.session_id)
    assert queue.req_to_token_pool.available_size() == 1
    assert queue.token_to_kv_pool_allocator.available_size() == 64


def test_admission_retry_keeps_existing_session_ownership():
    queue, session, row = _fixture()
    req = _request(session, "blocked", [23, 24])
    pending = _enqueue(queue, req)
    original = queue.req_to_token_pool.req_to_token[row, :8].clone()
    queue._allocatable_token_budgets = lambda **kwargs: 0
    assert queue.pop_preallocated() == ([], [])
    assert req.kv is queue.tree_cache.slots[session.session_id].kv
    assert req.kv.req_pool_idx == row
    torch.testing.assert_close(queue.req_to_token_pool.req_to_token[row, :8], original)
    pending.kv_receiver.send_metadata.assert_not_called()
    queue._allocatable_token_budgets = lambda **kwargs: 64
    assert queue.pop_preallocated() == ([pending], [])
    pending.kv_receiver.send_metadata.assert_called_once()


@pytest.mark.parametrize("prefix_len,fill_len", [(8, 22), (16, 33), (17, 35)])
def test_paged_transfer_includes_the_shared_partial_page(prefix_len, fill_len):
    queue, session, row = _fixture(rows=2)
    req = _request(session, "paged", [23, 24])
    req.origin_input_ids = array("q", range(fill_len))
    pending = _enqueue(queue, req)
    # Transfer planning is CPU-only; stand in for the paged allocation kernel.
    queue.token_to_kv_pool_allocator.page_size = 16
    indices = torch.arange(16, 16 + fill_len, dtype=torch.int64)
    queue._prepare_streaming_session_preallocation = lambda req: indices[:prefix_len]

    def allocate(req, *args):
        req.kv.req_pool_idx = row
        queue.req_to_token_pool.write((row, slice(0, fill_len)), indices)
        return indices[prefix_len:]

    queue._pre_alloc = allocate
    assert queue.pop_preallocated() == ([pending], [])
    sent = pending.kv_receiver.send_metadata.call_args
    aligned = prefix_len // 16 * 16
    assert sent.kwargs["decode_prefix_len"] == aligned
    assert sent.args[0].tolist() == list(
        range(1 + aligned // 16, 1 + (fill_len + 15) // 16)
    )


def test_prebuilt_batch_does_not_rematch_a_published_session_row():
    queue, session, _ = _fixture()
    req = _request(session, "append", [23, 24])
    req.init_next_round_input(queue.tree_cache)
    prefix = req.prefix_indices
    queue._pre_alloc(req, prefix, len(prefix), len(prefix))
    # A completed transfer contributes the first sampled token before prebuilt.
    req.output_ids.append(17)
    scheduler = SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        waiting_queue=[req],
        enable_priority_scheduling=False,
        req_to_token_pool=queue.req_to_token_pool,
        max_running_requests=1,
        tree_cache=queue.tree_cache,
        token_to_kv_pool_allocator=queue.token_to_kv_pool_allocator,
        model_config=object(),
        enable_overlap=False,
        spec_algorithm=object(),
        future_map=None,
    )
    with (
        patch.object(
            queue.tree_cache, "match_prefix", wraps=queue.tree_cache.match_prefix
        ) as match,
        patch(
            "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
            return_value=Mock(),
        ),
    ):
        SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch(
            scheduler, SimpleNamespace(batch_size=lambda: 0)
        )
    match.assert_not_called()


@pytest.mark.parametrize("allocated", [False, True])
def test_bootstrap_abort_preserves_last_successful_history(allocated):
    queue, session, _ = _fixture()
    last_success = session.req_nodes["first"].req
    failed = _request(session, "failed", [23, 24])
    if allocated:
        failed.init_next_round_input(queue.tree_cache)
        prefix = failed.prefix_indices
        queue._pre_alloc(failed, prefix, len(prefix), len(prefix))
    failed.disagg_kv_sender = Mock()
    failed.pending_bootstrap = True
    scheduler = SimpleNamespace(
        clear_pending_chunk_send=Mock(),
        ps=SimpleNamespace(tp_rank=0),
        tree_cache=queue.tree_cache,
        req_to_metadata_buffer_idx_allocator=Mock(),
        output_streamer=Mock(),
        metrics_reporter=SimpleNamespace(enable_metrics=False),
        enable_hicache_storage=False,
    )
    SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, failed)
    assert isinstance(failed.finished_reason, FINISH_ABORT)
    assert not session._inflight
    assert session.req_nodes["first"].req is last_success
    assert not failed.pending_bootstrap
    if allocated:
        assert not queue.tree_cache.has_slot(session.session_id)
        assert queue.req_to_token_pool.available_size() == 1
        assert queue.token_to_kv_pool_allocator.available_size() == 64
    else:
        assert queue.tree_cache.has_slot(session.session_id)
    retry = _request(session, "retry", [25])
    assert list(retry.origin_input_ids) == list(range(8)) + [17, 25]
    assert retry.to_finish is None
    pending = _enqueue(queue, retry)
    assert queue.pop_preallocated() == ([pending], [])
    _finish(queue.tree_cache, retry)
    queue.tree_cache.release_session(session.session_id)
    assert queue.req_to_token_pool.available_size() == 1
    assert queue.token_to_kv_pool_allocator.available_size() == 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
