import copy
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.mm_schedule import _acknowledge_deferred_cuda_ipc_cache_hits
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.multimodal.transport.cuda_ipc import CudaIpcTensorTransportProxy
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
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
