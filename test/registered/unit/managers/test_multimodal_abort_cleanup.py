import asyncio
import concurrent.futures
import copy
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    AbortReq,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_schedule import _acknowledge_deferred_cuda_ipc_cache_hits
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.runtime_context import get_context, get_parallel
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


def test_embedding_hit_cannot_recycle_another_ranks_live_proxy(vmm_pool):
    """A rank's cache hit must not retire another rank's proxy before cleanup."""
    items = _publish(vmm_pool)
    with _rank(0):
        _acknowledge_deferred_cuda_ipc_cache_hits([items[0]])
    vmm_pool._recycle_chunks()
    assert vmm_pool.occupied_chunks

    with _rank(1):
        _acknowledge_deferred_cuda_ipc_cache_hits([items[1]])
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
    vmm_pool.memory_pool[:8].view(torch.int32).fill_(1)
    stack = torch.stack

    def poll(tensors):
        assert vmm_pool._lock.acquire(blocking=False), (
            "CUDA poll held the allocator lock"
        )
        vmm_pool._lock.release()
        snapshot = stack(tensors)
        vmm_pool.cancel_proxy(old[0].feature)
        _publish(vmm_pool)
        return snapshot

    with patch.object(torch, "stack", side_effect=poll):
        vmm_pool._recycle_chunks()
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
