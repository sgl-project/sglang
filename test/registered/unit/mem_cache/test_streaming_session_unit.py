from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    StreamingSessionAbortPolicy,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeAllocator:
    def __init__(self):
        self.freed = []

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.clone())


class _FakeMultimodalInputs:
    def __init__(self, items):
        self.mm_items = items

    def release_features(self):
        for item in self.mm_items:
            item.feature = None


class _FakeInnerCache:
    def __init__(self, req_to_token_pool, allocator, page_size, match_results=None):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = allocator
        self.page_size = page_size
        self.match_results = list(match_results or [])
        self.dec_lock_ref_calls = []
        self.dec_lock_ref_params = []

    def cache_finished_req(self, *args, **kwargs):
        raise AssertionError("Streaming requests should not delegate to inner cache")

    def match_prefix(self, *args, **kwargs):
        if not self.match_results:
            raise AssertionError("Unexpected match_prefix call")
        return self.match_results.pop(0)

    def dec_lock_ref(self, node, *args, **kwargs):
        self.dec_lock_ref_calls.append(node)
        self.dec_lock_ref_params.append(args[0] if args else kwargs.get("params"))

    def supports_mamba(self):
        return False

    def sanity_check(self):
        return None


class _FakeReq:
    def __init__(
        self, session_id: str, req_pool_idx: int, committed: int, allocated: int
    ):
        self.session = SimpleNamespace(
            session_id=session_id,
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda req=None: True,
            req_nodes={},
            _inflight=False,
        )
        self.session._inflight_req = self
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(
            kv_allocated_len=allocated,
            swa_evicted_seqlen=0,
        )
        self.origin_input_ids = list(range(committed))
        self.output_ids = []
        self.extra_key = None
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.skip_lock_node_ids = {}
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.to_finish = None
        self.finished_reason = None
        self.finished_len = None
        self.streaming_abort_policy = StreamingSessionAbortPolicy.RELEASE_SESSION
        self.drop_trailing_stop_token = False
        self.multimodal_inputs = None

    def finished(self):
        return self.finished_reason is not None


def test_decode_result_finds_composed_streaming_session_control():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    inner = _FakeInnerCache(
        SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
        _FakeAllocator(),
        page_size=16,
    )
    streaming_session = StreamingSession(inner)
    control = Mock()
    streaming_session.attach_session_lifecycle(control)
    processor = SimpleNamespace(
        tree_cache=SimpleNamespace(session=streaming_session),
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)

    SchedulerBatchResultProcessor._maybe_record_streaming_session_decode(processor, req)

    control.on_decode_token.assert_called_once_with(req)


def test_prefill_result_finds_composed_streaming_session_control():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    inner = _FakeInnerCache(
        SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
        _FakeAllocator(),
        page_size=16,
    )
    streaming_session = StreamingSession(inner)
    control = Mock()
    streaming_session.attach_session_lifecycle(control)
    processor = SimpleNamespace(
        tree_cache=SimpleNamespace(session=streaming_session),
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)

    SchedulerBatchResultProcessor._maybe_record_streaming_session_prefill(
        processor, req, start=0, end=1
    )

    control.on_prefill_forward_complete.assert_called_once_with(req, 0, 1)


def test_first_streaming_request_caps_prefix_match_at_lifecycle_boundary():
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    control = Mock()
    control.next_prefill_chunk_end.return_value = 4
    tree_cache.attach_session_lifecycle(control)
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=0)
    req.full_untruncated_fill_ids = list(range(8))
    key = SimpleNamespace(token_ids=list(range(8)), limit=7)

    assert tree_cache.try_match_prefix(SimpleNamespace(req=req, key=key)) is None

    assert key.limit == 3
    control.next_prefill_chunk_end.assert_called_once_with(req, 0, 8)


def test_default_streaming_lifecycle_is_noop() -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=0)
    req.full_untruncated_fill_ids = list(range(8))
    key = SimpleNamespace(token_ids=list(range(8)), limit=7)

    assert not tree_cache.has_attached_lifecycle
    assert tree_cache.try_match_prefix(SimpleNamespace(req=req, key=key)) is None
    assert tree_cache.next_prefill_chunk_end(req, 0, 8) == 8
    assert key.limit == 7
    assert tree_cache.session_held_mamba_slots() == 0


def test_non_streaming_request_does_not_apply_lifecycle_prefix_cap():
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    control = Mock()
    tree_cache.attach_session_lifecycle(control)
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=0)
    req.session = None
    key = SimpleNamespace(token_ids=list(range(8)), limit=7)

    assert tree_cache.try_match_prefix(SimpleNamespace(req=req, key=key)) is None

    assert key.limit == 7
    control.next_prefill_chunk_end.assert_not_called()


@pytest.mark.parametrize("session", [None, SimpleNamespace(streaming=False)])
def test_non_streaming_prefill_does_not_inspect_tree_cache(session):
    processor = SimpleNamespace()
    req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)
    req.session = session

    SchedulerBatchResultProcessor._maybe_record_streaming_session_prefill(
        processor, req, start=0, end=1
    )


def test_middle_prefill_uses_completion_hook_and_final_prefill_uses_decode_hook():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    streaming_session = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=16,
        )
    )
    control = Mock()
    streaming_session.attach_session_lifecycle(control)

    req = _FakeReq("session-a", req_pool_idx=0, committed=1, allocated=1)
    req.inflight_middle_chunks = 0
    req.is_retracted = False
    req.time_stats = Mock()
    req.require_reasoning = False
    req.return_sampling_mask = False
    req.return_hidden_states = False
    req.grammar = None
    req.customized_info = None
    finish_state_updated = False

    def update_finish_state():
        nonlocal finish_state_updated
        finish_state_updated = True

    req.update_finish_state = Mock(side_effect=update_finish_state)
    sampled_token = 99

    def assert_middle_callback_state(callback_req, start, end):
        assert callback_req.output_ids == []
        assert (start, end) == (0, 1)

    def assert_final_callback_state(callback_req):
        assert callback_req.output_ids == [sampled_token]
        assert finish_state_updated

    control.on_prefill_forward_complete.side_effect = assert_middle_callback_state
    control.on_decode_token.side_effect = assert_final_callback_state

    processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=Mock(),
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_hisparse=False),
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=SimpleNamespace(
            session=streaming_session, cache_unfinished_req=Mock()
        ),
        hisparse_coordinator=None,
        req_to_token_pool=Mock(),
        decode_offload_manager=None,
        metrics_collector=Mock(),
        metrics_reporter=Mock(),
        draft_worker=Mock(),
        model_worker=Mock(),
        logprob_result_processor=Mock(),
        output_streamer=Mock(),
        abort_request=Mock(),
    )
    batch = SimpleNamespace(
        reqs=[req],
        decoding_reqs=[],
        return_logprob=False,
        prefill_stats=Mock(),
        dp_cooperation_info=None,
        prefix_lens=[0],
        extend_lens=[1],
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        spec_info=None,
    )
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([sampled_token]),
        extend_input_len_per_req=[1],
        extend_logprob_start_len_per_req=[0],
    )

    with patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.get_memory",
        return_value=SimpleNamespace(enable_hisparse=False),
    ):
        req.inflight_middle_chunks = 1
        processor.process_batch_result_prefill(batch, result)
        assert req.output_ids == []
        control.on_prefill_forward_complete.assert_called_once_with(req, 0, 1)

        req.inflight_middle_chunks = 0
        batch.prefix_lens = [1]
        processor.process_batch_result_prefill(batch, result)

    control.on_prefill_forward_complete.assert_called_once_with(req, 0, 1)
    control.on_decode_token.assert_called_once_with(req)


def test_mixed_prefill_batch_dispatches_each_row_to_its_lifecycle():
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    streaming_session = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=16,
        )
    )
    control = Mock()
    streaming_session.attach_session_lifecycle(control)

    prefill_req = _FakeReq("session-prefill", 0, committed=1, allocated=1)
    decode_req = _FakeReq("session-decode", 1, committed=1, allocated=1)
    decode_req.output_ids = [77]
    for req in (prefill_req, decode_req):
        req.inflight_middle_chunks = 0
        req.is_retracted = False
        req.time_stats = Mock()
        req.require_reasoning = False
        req.return_sampling_mask = False
        req.return_hidden_states = False
        req.grammar = None
        req.customized_info = None
        req.update_finish_state = Mock()

    processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=Mock(),
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_hisparse=False),
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=SimpleNamespace(
            session=streaming_session, cache_unfinished_req=Mock()
        ),
        hisparse_coordinator=None,
        req_to_token_pool=Mock(),
        decode_offload_manager=None,
        metrics_collector=Mock(),
        metrics_reporter=Mock(),
        draft_worker=Mock(),
        model_worker=Mock(),
        logprob_result_processor=Mock(),
        output_streamer=Mock(),
        abort_request=Mock(),
    )
    batch = SimpleNamespace(
        reqs=[prefill_req, decode_req],
        decoding_reqs=[decode_req],
        return_logprob=False,
        prefill_stats=Mock(),
        dp_cooperation_info=None,
        prefix_lens=[0, 1],
        extend_lens=[1, 1],
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        spec_info=None,
    )
    result = GenerationBatchResult(
        next_token_ids=torch.tensor([91, 92]),
        extend_input_len_per_req=[1, 1],
        extend_logprob_start_len_per_req=[0, 0],
    )

    with patch(
        "sglang.srt.managers.scheduler_components.batch_result_processor.get_memory",
        return_value=SimpleNamespace(enable_hisparse=False),
    ):
        processor.process_batch_result_prefill(batch, result)

    control.on_prefill_forward_complete.assert_not_called()
    assert [args.args[0] for args in control.on_decode_token.call_args_list] == [
        prefill_req,
        decode_req,
    ]
    assert decode_req.output_ids == [77, 92]


def test_preabort_detaches_session_and_preserves_slot():
    """Pre-aborted req (to_finish set before match_prefix) is detached from
    the session: session=None, abort_req(req) called. Slot stays intact."""
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(
        req_to_token_pool,
        allocator,
        page_size=16,
        match_results=[
            MatchResult(
                device_indices=torch.tensor([], dtype=torch.int64),
                last_device_node=None,
                last_host_node=None,
                best_match_node=None,
            )
        ],
    )
    tree_cache = StreamingSession(inner)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=48,
        kv=SimpleNamespace(kv_allocated_len=48, swa_evicted_seqlen=0),
        cache_protected_len=16,
    )

    req = _FakeReq("session-a", req_pool_idx=1, committed=1, allocated=1)
    req.to_finish = FINISH_ABORT("too long")

    result = tree_cache.match_prefix(
        SimpleNamespace(
            req=req,
            key=SimpleNamespace(token_ids=list(range(64))),
        )
    )

    # Req detached from session.
    assert req.session is None
    # Slot untouched.
    slot = tree_cache.slots["session-a"]
    assert slot.req_pool_idx == 0
    assert slot.kv_committed_len == 48
    assert slot.kv.kv_allocated_len == 48
    assert len(result.device_indices) == 0


def test_detach_first_queued_request_releases_pending_session_control() -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    inner = _FakeInnerCache(
        SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
        _FakeAllocator(),
        page_size=1,
    )
    tree_cache = StreamingSession(inner)
    control = Mock()
    tree_cache.attach_session_lifecycle(control)
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=0)
    req.req_pool_idx = None
    session = req.session
    session.abort_req = Mock()

    assert tree_cache.detach_queued_request(req)

    assert req.session is None
    session.abort_req.assert_called_once_with(req)
    control.on_session_released.assert_called_once_with("session-a")


def test_unfinished_retraction_checkpoints_without_finishing_session() -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=8, allocated=8)
    session = req.session
    session.checkpoint_retracted_req = Mock()
    session.finish_req = Mock()

    tree_cache.cache_finished_req(req, is_insert=False)

    session.checkpoint_retracted_req.assert_called_once_with(req)
    session.finish_req.assert_not_called()
    assert tree_cache.slots["session-a"].req_pool_idx == 0
    assert req.req_pool_idx is None


def test_unfinished_streaming_cache_requires_retraction_mode() -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=8, allocated=8)

    with pytest.raises(RuntimeError, match="only be cached for retraction"):
        tree_cache.cache_finished_req(req)

    assert tree_cache.slots == {}
    assert req.req_pool_idx == 0


def test_first_mid_abort_nukes_ephemeral_slot():
    """First-request mid-processing abort: no slot exists yet, ephemeral
    slot is created from req state and nuked via release_session."""
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # No slot exists yet (first request).
    req = _FakeReq("session-a", req_pool_idx=0, committed=0, allocated=20)
    session = req.session
    session.abort_req = Mock()
    item = SimpleNamespace(feature=object())
    req.multimodal_inputs = _FakeMultimodalInputs([item])
    req.finished_reason = FINISH_ABORT("input too long")

    tree_cache.cache_finished_req(req)

    # Slot must NOT be created.
    assert "session-a" not in tree_cache.slots
    # Transient pool slot freed.
    assert req.req_pool_idx is None
    assert req_to_token_pool.free_slots == [0]
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(20))
    assert req.kv is None
    assert req.session is None
    assert req.multimodal_inputs is None
    assert item.feature is None
    session.abort_req.assert_called_once_with(req)


def test_nth_mid_abort_nukes_session_slot():
    """Nth-request mid-processing abort: slot exists, restore_to_req ran.
    ALL KV is wiped (release_session). Slot is deleted. Token IDs stay
    in req_nodes for next turn's re-prefill."""
    page_size = 1
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    mamba_allocator = _FakeAllocator()
    req_to_token_pool = SimpleNamespace(
        req_to_token=req_to_token,
        free_slots=[],
        mamba_allocator=mamba_allocator,
    )
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size)
    tree_cache = StreamingSession(inner)

    # Session already has a slot from a previous turn.
    mamba_pool_idx = torch.tensor(3)
    ping_pong = torch.tensor([4, 5])
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=None,
        cache_protected_len=0,
        mamba_pool_idx=mamba_pool_idx,
        mamba_ping_pong_track_buffer=ping_pong,
    )

    # Mid-processing abort: req has the SESSION slot's pool_idx (restore_to_req ran).
    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    session = req.session
    session.abort_req = Mock()
    old_item = SimpleNamespace(feature=object())
    new_item = SimpleNamespace(feature=object())
    previous_req = SimpleNamespace(multimodal_inputs=_FakeMultimodalInputs([old_item]))
    session.req_nodes["previous"] = SimpleNamespace(req=previous_req)
    req.multimodal_inputs = _FakeMultimodalInputs([old_item, new_item])
    req.mamba_pool_idx = mamba_pool_idx
    req.mamba_ping_pong_track_buffer = ping_pong
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 32
    req.mamba_branching_seqlen = 16
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    # Slot wiped — deleted from slots dict.
    assert "session-a" not in tree_cache.slots
    # All KV freed: [0, 65) from release_session (slot extended to req's allocated).
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(65))
    # Pool slot returned.
    assert req_to_token_pool.free_slots == [0]
    assert req.req_pool_idx is None
    assert req.kv is None
    assert req.session is None
    assert req.multimodal_inputs is None
    assert old_item.feature is not None
    assert new_item.feature is None
    assert previous_req.multimodal_inputs.mm_items == [old_item]
    assert req.mamba_pool_idx is None
    assert req.mamba_ping_pong_track_buffer is None
    assert req.mamba_next_track_idx is None
    assert req.mamba_last_track_seqlen is None
    assert req.mamba_branching_seqlen is None
    session.abort_req.assert_called_once_with(req)


def test_mid_abort_preserves_retracted_request_boundary() -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    mamba_allocator = _FakeAllocator()
    req_to_token_pool = SimpleNamespace(
        req_to_token=req_to_token,
        free_slots=[],
        mamba_allocator=mamba_allocator,
    )
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, _FakeAllocator(), page_size=1)
    )
    mamba_pool_idx = torch.tensor(3)
    ping_pong = torch.tensor([4, 5])
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        mamba_pool_idx=mamba_pool_idx,
        mamba_ping_pong_track_buffer=ping_pong,
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=60, allocated=65)
    session = req.session
    session.abort_req = Mock()
    session.req_nodes["current"] = SimpleNamespace(req=req)
    item = SimpleNamespace(feature=object())
    multimodal_inputs = _FakeMultimodalInputs([item])
    multimodal_inputs.release_features = Mock()
    req.multimodal_inputs = multimodal_inputs
    req.mamba_pool_idx = mamba_pool_idx
    req.mamba_ping_pong_track_buffer = ping_pong
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 32
    req.mamba_branching_seqlen = 16
    req.finished_reason = FINISH_ABORT("client disconnected")

    tree_cache.cache_finished_req(req)

    assert req.session is session
    assert req.multimodal_inputs is multimodal_inputs
    assert item.feature is not None
    multimodal_inputs.release_features.assert_not_called()
    assert req.mamba_pool_idx is None
    assert req.mamba_ping_pong_track_buffer is None
    assert req.mamba_next_track_idx is None
    assert req.mamba_last_track_seqlen is None
    assert req.mamba_branching_seqlen is None
    session.abort_req.assert_called_once_with(req)


def test_release_session_threads_mamba_skip_ids():
    """release_session must forward the slot's skip_lock_node_ids to
    dec_lock_ref. The first req's last_node may be full-only-locked (mamba
    skipped at inc), so without the skip set the release would drop a mamba
    lock the session never took -- another request's, on a shared node."""
    from sglang.srt.mem_cache.unified_cache.components import ComponentType

    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    inner = _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    tree_cache = StreamingSession(inner)

    lock_node = SimpleNamespace(id=42)
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=50,
        kv=SimpleNamespace(kv_allocated_len=50, swa_evicted_seqlen=0),
        last_node=lock_node,
        cache_protected_len=0,
        skip_lock_node_ids={ComponentType.MAMBA: {42}},
    )

    tree_cache.release_session("session-a")

    assert inner.dec_lock_ref_calls == [lock_node]
    params = inner.dec_lock_ref_params[0]
    assert params is not None
    assert params.skip_lock_node_ids.get(ComponentType.MAMBA) == {42}


# Shrink tests removed: streaming sessions are append-only after the
# rollback fix in session_controller (rollback_aborted_req).  The shrink
# code path in cache_finished_req no longer exists.


def test_truncate_kv_rejects_the_radix_protected_prefix() -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, _FakeAllocator(), page_size=1)
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=20,
        kv=SimpleNamespace(kv_allocated_len=20, swa_evicted_seqlen=0),
        cache_protected_len=8,
    )

    with pytest.raises(ValueError, match="below the protected session prefix"):
        tree_cache.truncate_kv("session-a", 7)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 20
    assert slot.kv.kv_allocated_len == 20


def test_truncate_kv_frees_only_complete_pages_after_target() -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            allocator,
            page_size=4,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=12,
        kv=SimpleNamespace(kv_allocated_len=12, swa_evicted_seqlen=0),
        cache_protected_len=4,
    )

    slot = tree_cache.truncate_kv("session-a", 6)

    assert slot.kv_committed_len == 6
    assert slot.kv.kv_allocated_len == 6
    assert allocator.freed[0].tolist() == [8, 9, 10, 11]


@pytest.mark.parametrize("target", [-1, 13])
def test_truncate_kv_rejects_target_outside_committed_range(target: int) -> None:
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=12,
        kv=SimpleNamespace(kv_allocated_len=12, swa_evicted_seqlen=0),
    )

    with pytest.raises(ValueError, match="outside"):
        tree_cache.truncate_kv("session-a", target)


def test_detach_refuses_request_that_advanced_beyond_retained_slot() -> None:
    req_to_token = torch.arange(256, dtype=torch.int32).reshape(2, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    tree_cache.slots["session-a"] = SessionSlot(
        req_pool_idx=0,
        kv_committed_len=10,
        kv=SimpleNamespace(kv_allocated_len=10, swa_evicted_seqlen=0),
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=11, allocated=11)

    assert not tree_cache.detach_queued_request(req)
    assert req.session is not None


def test_trim_overshoot_postcondition():
    """`_trim_overshoot` postcondition: every per-req KV field is capped at
    target = origin+finished_len, output_ids is truncated, and the tail
    KV slots are freed. Covers both non-SWA fields (kv_committed_len,
    kv_allocated_len, output_ids) and SWA bookkeeping (swa_evicted_seqlen)
    in one shot — same invariant `_free_tail` enforces on the match_prefix
    path.
    """
    page_size = 1
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size)
    )

    # Overshoot scenario: origin=26, finished_len=12 -> target=38.
    # committed=40 (overshoot 2), allocated=44, swa_evicted=42 (> target),
    # output_ids extended to 14 by the overshoot round.
    req = _FakeReq("session-a", req_pool_idx=0, committed=40, allocated=44)
    req.origin_input_ids = list(range(26))
    req.output_ids = list(range(14))
    req.kv.swa_evicted_seqlen = 42

    tree_cache._trim_overshoot(req, finished_len=12)

    target = 38
    assert req.kv_committed_len == target
    assert req.kv.kv_allocated_len == target
    assert req.kv.swa_evicted_seqlen == target
    assert len(req.output_ids) == 12
    # Tail [38, 44) freed by _free_kv_aligned.
    assert len(allocator.freed) == 1
    assert allocator.freed[0].tolist() == list(range(38, 44))


def test_cancel_keeps_forwarded_partial_output_and_session_slot():
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=7, allocated=8)
    req.origin_input_ids = list(range(5))
    req.output_ids = [5, 6, 7]
    req.finished_reason = FINISH_ABORT()
    req.streaming_abort_policy = StreamingSessionAbortPolicy.COMMIT_FORWARDED

    tree_cache.cache_finished_req(req)

    slot = tree_cache.slots["session-a"]
    assert slot.kv_committed_len == 7
    assert slot.kv.kv_allocated_len == 7
    assert req.output_ids == [5, 6]
    assert allocator.freed[0].tolist() == [7]


def test_matched_stop_token_is_not_committed_when_requested():
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token, free_slots=[])
    allocator = _FakeAllocator()
    tree_cache = StreamingSession(
        _FakeInnerCache(req_to_token_pool, allocator, page_size=1)
    )

    req = _FakeReq("session-a", req_pool_idx=0, committed=7, allocated=8)
    req.origin_input_ids = list(range(5))
    req.output_ids = [5, 6, 200008]
    req.finished_reason = FINISH_MATCHED_TOKEN(200008)
    req.finished_len = 3
    req.drop_trailing_stop_token = True

    tree_cache.cache_finished_req(req)

    assert tree_cache.slots["session-a"].kv_committed_len == 7
    assert req.output_ids == [5, 6]
    assert allocator.freed[0].tolist() == [7]


def test_nonmatched_finish_keeps_trailing_output_token():
    req_to_token = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    tree_cache = StreamingSession(
        _FakeInnerCache(
            SimpleNamespace(req_to_token=req_to_token, free_slots=[]),
            _FakeAllocator(),
            page_size=1,
        )
    )
    req = _FakeReq("session-a", req_pool_idx=0, committed=8, allocated=8)
    req.origin_input_ids = list(range(5))
    req.output_ids = [5, 6, 7]
    req.finished_reason = FINISH_LENGTH(3)
    req.finished_len = 3
    req.drop_trailing_stop_token = True

    tree_cache.cache_finished_req(req)

    assert tree_cache.slots["session-a"].kv_committed_len == 8
    assert req.output_ids == [5, 6, 7]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
