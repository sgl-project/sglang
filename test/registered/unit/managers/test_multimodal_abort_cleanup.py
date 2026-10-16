import asyncio
import concurrent.futures
import copy
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers import mm_schedule
from sglang.srt.managers.io_struct import (
    AbortReq,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.runtime_context import get_context, get_parallel, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.srt.utils import cuda_vmm_transport_utils as vmm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture
def vmm_pool():
    # exercise the real slice allocator and ACK protocol with host-backed storage
    pool = object.__new__(vmm.CudaVmmMemoryPool)
    pool.memory_pool = torch.zeros(1024, dtype=torch.uint8)
    pool.consumer_count = 2
    pool.device_index = 0
    pool._recycle_stream = None
    pool._lock = threading.Lock()
    pool.available_chunks = [vmm._CudaVmmMemoryChunk(0, 1024)]
    pool.occupied_chunks = []
    with (
        get_context().override_server_args(tp_size=2),
        patch.object(torch.cuda, "device", return_value=nullcontext()),
        patch.object(torch.cuda, "stream", return_value=nullcontext()),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(
            vmm.CudaVmmTensorTransportProxy,
            "_pool",
            return_value=SimpleNamespace(memory=pool.memory_pool),
        ),
    ):
        yield pool


def _publish(pool):
    chunk = pool._reserve_chunk(1024)
    assert chunk is not None, "a previous request stranded the pool slice"
    pool.memory_pool.zero_()
    pool.occupied_chunks.append(chunk)
    proxy = vmm.CudaVmmTensorTransportProxy(
        fabric_handle=b"test",
        posix_socket_path=None,
        allocation_size=1024,
        data_offset=256,
        data_nbytes=768,
        control_offset=0,
        consumer_count=2,
        shape=(768,),
        dtype=torch.uint8,
    )
    return [
        MultimodalDataItem(modality=Modality.IMAGE, feature=copy.deepcopy(proxy)),
        MultimodalDataItem(modality=Modality.IMAGE, feature=copy.deepcopy(proxy)),
    ]


def _rank(rank):
    return get_parallel().override(
        tp_rank=rank, attn_tp_rank=rank, attn_tp_size=2, attn_cp_rank=0, attn_cp_size=1
    )


def _queued_scheduler(req):
    scheduler = object.__new__(Scheduler)
    scheduler.waiting_queue = [req]
    scheduler.chunked_req = scheduler.mm_receiver = scheduler.dllm_config = None
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.tree_cache = MagicMock()
    scheduler.beam_coordinator = MagicMock()
    scheduler.ipc_channels = MagicMock()
    scheduler.grammar_manager = MagicMock()
    scheduler.collect_inflight_reqs = lambda: set()
    return scheduler


def test_queued_abort_recycles_unconsumed_vmm_slices(vmm_pool):
    """Client cancellations before prefill must not exhaust the bounded pool."""
    for _ in range(3):
        for rank, item in enumerate(_publish(vmm_pool)):
            req = Req("queued-image", "", [1], SamplingParams())
            req.multimodal_inputs = MultimodalInputs(mm_items=[item])
            scheduler = _queued_scheduler(req)
            with _rank(rank):
                scheduler.abort_request(AbortReq(rid=req.rid))
            assert scheduler.waiting_queue == []
        vmm_pool._recycle_chunks()
        assert not vmm_pool.occupied_chunks


@pytest.mark.parametrize("path", ["full", "per_image", "batch"])
def test_embedding_hit_cannot_recycle_another_ranks_live_proxy(vmm_pool, path):
    """A rank's cache hit must not retire another rank's proxy before cleanup."""
    items = _publish(vmm_pool)
    cache = MultiModalStaticCache(1 << 20)
    cached = torch.ones(1, 2)
    cache.set(cache.combine_hashes([123]), EmbeddingResult(embedding=cached))
    cache.set(123, EmbeddingResult(embedding=cached))

    def encode(_):
        pytest.fail("cache hit unexpectedly ran the encoder")

    for rank, item in enumerate(items):
        item.hash = 123
        item.offsets = [(0, 0)]
        with _rank(rank), patch.object(mm_schedule, "embedding_cache", cache):
            if path == "full":
                embedding, _ = mm_schedule._get_chunked_embedding_full(
                    encode,
                    [item],
                    [(0, 0)],
                    0,
                    1,
                    torch.tensor([1]),
                    torch.device("cpu"),
                )
            elif path == "per_image":
                [embedding] = mm_schedule._get_chunked_embedding_by_item(
                    encode,
                    [item],
                    0,
                    1,
                    torch.device("cpu"),
                )
            else:
                request = mm_schedule.PerImageRequestInfo(0, [item], [(0, 0)], 0, 1)
                embeddings = mm_schedule._batch_encode_per_image_misses(
                    encode,
                    [request],
                    torch.device("cpu"),
                )
                embedding = embeddings[(123, 1)]
        assert torch.equal(embedding, cached)
        assert not item.feature._consumer_acknowledged, (
            "cache eviction may need this input again"
        )
    for rank, item in enumerate(items):
        with _rank(rank):
            item.release_transport_proxies()
    vmm_pool._recycle_chunks()
    assert not vmm_pool.occupied_chunks

    _publish(vmm_pool)
    for rank, item in enumerate(items):
        with _rank(rank):
            MultimodalInputs(mm_items=[item]).release_features()
    vmm_pool._recycle_chunks()
    assert vmm_pool.occupied_chunks, "late cleanup acknowledged a recycled slice"


def test_aborted_session_turn_releases_new_media_not_history(vmm_pool):
    """An aborted append owns new media, but only borrows historical features."""
    for rank, item in enumerate(_publish(vmm_pool)):
        session = Session(100, streaming=True)
        session._inflight = True
        history = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(3))
        req = Req("session-image", "", [1], SamplingParams(), session=session)
        req.multimodal_inputs = MultimodalInputs(mm_items=[history])
        req.extend_image_inputs(MultimodalInputs(mm_items=[item]))
        with _rank(rank):
            _queued_scheduler(req).abort_request(AbortReq(rid=req.rid))
        assert torch.equal(history.feature, torch.ones(3))
        assert not session._inflight
    vmm_pool._recycle_chunks()
    assert not vmm_pool.occupied_chunks


def test_rejected_session_releases_media_before_request_admission(vmm_pool):
    """An invalid session is rejected before the request attaches its media."""
    for rank, item in enumerate(_publish(vmm_pool)):
        recv = TokenizedGenerateReqInput(
            rid="invalid-session",
            input_text="",
            input_ids=[1],
            input_embeds=None,
            mm_inputs=MultimodalProcessorOutput(mm_items=[item]),
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            session_params=SessionParams(id="missing"),
        )
        scheduler = _queued_scheduler(None)
        scheduler.session_controller = {}
        scheduler.model_config = SimpleNamespace(vocab_size=10)
        scheduler.tokenizer = None
        scheduler.init_req_max_new_tokens = lambda req: None
        scheduler._add_request_to_queue = lambda req: None
        with _rank(rank):
            scheduler.handle_generate_request(recv)
    vmm_pool._recycle_chunks()
    assert not vmm_pool.occupied_chunks


def test_other_rank_materialization_failure_releases_local_proxies(vmm_pool):
    """A healthy rank still owns lazy slices when another rank rejects the input."""
    for rank, item in enumerate(_publish(vmm_pool)):
        recv = TokenizedGenerateReqInput(
            rid="peer-failure",
            input_text="",
            input_ids=[1],
            input_embeds=None,
            mm_inputs=MultimodalInputs(mm_items=[item]),
            token_type_ids=None,
            sampling_params=SamplingParams(),
            return_logprob=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
        )
        scheduler = _queued_scheduler(None)
        scheduler._gather_vmm_materialization_errors = lambda error: [
            None,
            "copy failed",
        ]
        with _rank(rank):
            errors = scheduler._materialize_cuda_vmm_inputs(recv)
        assert "copy failed" in errors[0]
        assert recv.mm_inputs is None
    vmm_pool._recycle_chunks()
    assert not vmm_pool.occupied_chunks


def test_repeated_cancellation_during_publication_does_not_leak(vmm_pool):
    """Cancellation must not orphan slices still being published on another thread."""
    transport = object.__new__(vmm.CudaVmmFeatureTransport)
    transport.pool = vmm_pool
    started, release = threading.Event(), threading.Event()
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=torch.ones(2))

    def publish(_):
        started.set()
        assert release.wait(timeout=5)
        return _publish(vmm_pool)[0].feature

    async def run():
        task = asyncio.create_task(
            transport.prepare_for_dispatch_async(
                [MultimodalProcessorOutput(mm_items=[item])]
            )
        )
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        transport._publisher_executor = executor
        with patch.object(vmm_pool, "wrap_tensor", side_effect=publish):
            try:
                asyncio.run(run())
            finally:
                release.set()
                executor.shutdown(wait=True)
    assert not vmm_pool.occupied_chunks


def test_recycler_does_not_hold_allocator_lock_or_free_reused_slice(vmm_pool):
    """A slow ACK poll cannot block cancellation or retire a republished offset."""
    old = _publish(vmm_pool)
    vmm_pool._stop_recycler = threading.Event()
    vmm_pool._recycle_interval = 0.001
    vmm_pool._pool_error = None
    vmm_pool.memory_pool[:8].view(torch.int32).fill_(1)
    stack = torch.stack

    def poll(tensors):
        vmm_pool._stop_recycler.set()
        assert vmm_pool._lock.acquire(blocking=False), (
            "CUDA poll held the allocator lock"
        )
        vmm_pool._lock.release()
        snapshot = stack(tensors)
        vmm_pool.cancel_proxy(old[0].feature)
        _publish(vmm_pool)
        return snapshot

    with patch.object(torch, "stack", side_effect=poll):
        vmm_pool._recycle_loop()
    assert vmm_pool._pool_error is None
    assert vmm_pool.occupied_chunks, "old poll result retired a new allocation"


def _deferred_proxy():
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    proxy.total_consumer_count = 1
    proxy.acknowledge_consumption = MagicMock()
    return proxy


def test_release_features_acknowledges_deferred_transport():
    proxy = _deferred_proxy()
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
    mm_inputs = MultimodalInputs(mm_items=[item])

    mm_inputs.release_features()

    proxy.acknowledge_consumption.assert_called_once_with(1)
    assert item.feature is None


def test_release_features_keeps_cleanup_error_request_local():
    proxy = _deferred_proxy()
    proxy.acknowledge_consumption.side_effect = RuntimeError("ack failed")
    item = MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)
    mm_inputs = MultimodalInputs(mm_items=[item])

    mm_inputs.release_features()

    assert item.feature is None


def test_request_abort_releases_multimodal_features():
    mm_inputs = MagicMock()
    req = object.__new__(Req)
    req.rid = "rejected-vlm-request"
    req.session = None
    req.multimodal_inputs = mm_inputs
    req.grammar = object()
    req.return_logprob = True
    req.logprob_start_len = 0

    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(tp_rank=1),
    ):
        req.set_finish_with_abort("invalid multimodal request")

    mm_inputs.release_features.assert_called_once_with()
    assert req.multimodal_inputs is None


def test_session_abort_preserves_shared_multimodal_features():
    mm_inputs = MagicMock()
    req = object.__new__(Req)
    req.rid = "rejected-session-turn"
    req.session = object()
    req._session_mm_items = []
    req.multimodal_inputs = mm_inputs
    req.grammar = object()
    req.return_logprob = True
    req.logprob_start_len = 0

    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(tp_rank=1),
    ):
        req.set_finish_with_abort("invalid session turn")

    mm_inputs.release_features.assert_not_called()
    assert req.multimodal_inputs is None


@pytest.fixture
def kept_rows():
    """One kept entry of mm_schedule, as left by the chunk before."""
    mm_schedule.release_prefill_rows()
    mm_schedule._prefill_rows[("item", ((0, 9),))] = object()
    yield mm_schedule._prefill_rows
    mm_schedule.release_prefill_rows()


def _admission_scheduler(chunked_req):
    """A scheduler whose admission records how many kept entries it found."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = chunked_req
    scheduler.prefill_delayer = None
    scheduler.entries_at_admission = []

    def admit(prefill_delayer_single_pass, running_batch):
        scheduler.entries_at_admission.append(len(mm_schedule._prefill_rows))
        return None, running_batch

    scheduler._get_new_batch_prefill_raw = admit
    return scheduler


def test_kept_rows_stay_while_a_request_is_mid_prefill_and_go_after(kept_rows):
    scheduler = _admission_scheduler(chunked_req=object())
    scheduler.get_new_batch_prefill(running_batch=None)

    # An abort or the last chunk leaves no chunked request for the next step.
    scheduler.chunked_req = None
    scheduler.get_new_batch_prefill(running_batch=None)

    assert scheduler.entries_at_admission == [1, 0]


def test_last_chunk_still_reads_its_kept_rows(kept_rows):
    scheduler = _admission_scheduler(chunked_req=object())

    def last_chunk(prefill_delayer_single_pass, running_batch):
        # add_chunked_req returns None for a request's last chunk, whose
        # forward in this step still reads the kept rows.
        scheduler.chunked_req = None
        return None, running_batch

    scheduler._get_new_batch_prefill_raw = last_chunk
    scheduler.get_new_batch_prefill(running_batch=None)

    assert len(kept_rows) == 1


def test_bootstrap_failure_releases_kept_rows_before_admission(kept_rows):
    scheduler = _admission_scheduler(chunked_req=object())
    scheduler.scheduler_stage_metrics = None
    scheduler.process_pending_chunked_abort = Mock()
    scheduler._process_hicache_events = Mock()
    scheduler.resolve_waiting_queue_bootstrap = Mock()

    def bootstrap_failed(last_batch, running_batch):
        # process_prefill_chunk drops a chunked request whose bootstrap failed.
        scheduler.chunked_req = None

    scheduler.process_prefill_chunk = Mock(side_effect=bootstrap_failed)
    scheduler.dp_attn_adapter = Mock()
    scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None

    scheduler.get_next_disagg_prefill_batch_to_run(
        running_batch=SimpleNamespace(batch_is_full=False), last_batch=None
    )

    assert scheduler.entries_at_admission == [0]


def _pause_scheduler(disaggregation_mode):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine_paused = False
    scheduler.enable_overlap = False
    scheduler.enable_overlap_mlx = False
    scheduler.last_batch = None
    scheduler.cur_batch_for_debug = None
    scheduler.running_batch = MagicMock()
    scheduler.running_batch.reqs = []
    scheduler.disaggregation_mode = disaggregation_mode
    scheduler.decode_offload_manager = None
    scheduler.hisparse_coordinator = None
    scheduler.tree_cache = MagicMock()
    scheduler.req_to_token_pool = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.waiting_queue = []
    scheduler._add_request_to_queue = MagicMock()
    scheduler.metrics_reporter = MagicMock()
    scheduler.metrics_reporter.current_scheduler_metrics_enabled = False
    scheduler.kv_events_publisher = MagicMock()
    scheduler.chunked_req = MagicMock()
    scheduler.chunked_req.finished.return_value = False
    return scheduler


@pytest.fixture
def published_server_args():
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path="dummy"), role="test")
    yield
    reset_context()


@pytest.mark.parametrize(
    "mode, disaggregation_mode, kept",
    [
        ("retract", DisaggregationMode.NULL, 0),
        # The live chunked request stays and resumes after the pause.
        ("retract", DisaggregationMode.PREFILL, 1),
        ("in_place", DisaggregationMode.NULL, 1),
    ],
)
def test_pause_drops_kept_rows_only_with_the_chunked_request(
    published_server_args, kept_rows, mode, disaggregation_mode, kept
):
    scheduler = _pause_scheduler(disaggregation_mode)

    with patch("sglang.srt.managers.scheduler.retract_all"):
        scheduler.pause_generation(PauseGenerationReqInput(mode=mode))

    assert len(kept_rows) == kept


def test_flush_cache_drops_kept_rows(kept_rows):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine_paused = False
    scheduler.is_fully_idle = Mock(return_value=True)
    scheduler.tree_cache = MagicMock()
    scheduler.req_to_token_pool = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.grammar_manager = MagicMock()
    scheduler.metrics_reporter = MagicMock()
    scheduler.draft_worker = None

    assert scheduler.flush_cache(empty_cache=False)
    assert not kept_rows


def test_weights_only_memory_release_drops_kept_rows(kept_rows):
    with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
        manager = SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=None,
            memory_saver_adapter=Mock(),
            # This path never reaches flush_cache; a stub that keeps the rows
            # shows the release comes from the manager itself.
            flush_cache=Mock(return_value=True),
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(_engine_paused=False),
        )

    with patch("torch.get_device_module"):
        manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
        )

    manager.flush_cache.assert_not_called()
    assert not kept_rows


@pytest.mark.parametrize("prefix_len, in_progress", [(4, True), (0, False)])
def test_dllm_context_is_in_progress_once_part_is_prefilled(prefix_len, in_progress):
    # dLLM prefills its context over several steps without chunked_req.
    manager = DllmManager(dllm_config=None)
    manager.waiting_queue.append(
        SimpleNamespace(dllm_phase=DllmReqPhase.STAGING_PREFILL, prefix_len=prefix_len)
    )

    assert manager.any_context_in_progress() is in_progress


def test_dllm_abort_drops_kept_rows(published_server_args, kept_rows):
    # Another dLLM context can stay part-way for many steps, so the table is not
    # cleared by the scheduling check; the aborted context's rows have no reader.
    req = Req("dllm-video", "", [1], SamplingParams())
    scheduler = _queued_scheduler(req)
    scheduler.waiting_queue = []
    scheduler.dllm_config = object()
    scheduler.dllm_manager = DllmManager(dllm_config=None)
    scheduler.dllm_manager.staging_queue.append(req)

    scheduler.abort_request(AbortReq(rid=req.rid))

    assert not scheduler.dllm_manager.staging_queue
    assert not kept_rows


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
