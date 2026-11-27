import asyncio
import errno
import os
import pickle
import resource
import subprocess
import sys
import threading
from array import array
from contextlib import contextmanager, nullcontext
from multiprocessing import resource_tracker, shared_memory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch
import zmq
import zmq.asyncio

from sglang.srt.managers import io_struct, mm_utils
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import ShmPointerMMData
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.managers.req_types import (
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.transport import memory_pool
from sglang.srt.multimodal.transport.cuda_ipc import (
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _request(feature, rid="request"):
    return TokenizedEmbeddingReqInput(
        rid=rid,
        input_text="",
        input_ids=array("q", [1]),
        mm_inputs=MultimodalProcessorOutput(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=feature)]
        ),
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        time_stats=APIServerReqTimeStats(),
    )


def _manager():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.tokenizer_ipc_name = None
    manager.mm_processor = None
    manager.cuda_vmm_feature_transport = SimpleNamespace(
        prepare_for_dispatch_async=AsyncMock(return_value=[]),
        cancel_for_dispatch=Mock(),
    )
    manager.positional_embeds_transport = SimpleNamespace(
        prepare_for_dispatch=Mock(return_value=[]), cancel_for_dispatch=Mock()
    )
    manager.rid_to_state = {
        "request": SimpleNamespace(dispatched=False, abort_sent=False)
    }
    manager.encoder_dispatch_ready = {}
    return manager


@pytest.mark.parametrize("pickle_ipc", [False, True])
def test_async_dispatch_preserves_enqueue_future(pickle_ipc):
    async def drive():
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        receiver = context.socket(zmq.PULL)
        try:
            receiver.bind("inproc://dispatch-contract")
            sender.connect("inproc://dispatch-contract")
            manager = _manager()
            manager.send_to_scheduler = sender
            request = _request(None)
            with patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc):
                request.wrap_pickle_fields()
                future = manager._dispatch_to_scheduler(request)
                assert asyncio.isfuture(future)
                await asyncio.wait_for(future, 3)
                received = await asyncio.wait_for(
                    io_struct.async_sock_recv(receiver), 3
                )
            assert received.rid == request.rid
        finally:
            sender.close(linger=0)
            receiver.close(linger=0)
            context.term()

    asyncio.run(drive())


@pytest.fixture
def _cuda_ipc_pool():
    # Model CUDA control words on CPU; use the actual lease allocator and proxy API.
    storage = memory_pool.StreamOrderedMmFeaturePool.__new__(
        memory_pool.StreamOrderedMmFeaturePool
    )
    storage.byte_tensor = torch.zeros(1280, dtype=torch.uint8)
    storage.base_address = storage.byte_tensor.data_ptr()
    storage.device_id = 0
    storage.consumer_count = 2
    storage.control_words_per_slot = 3
    storage.transport_name = "CUDA IPC"
    storage.data_start = 256
    storage._control_words = storage.byte_tensor[:48].view(torch.int32).view(4, 3)
    storage._available_ranges = [(256, 1280)]
    storage._available_slots = list(reversed(range(4)))
    storage._slot_generations = [0] * 4
    storage._occupied = {}
    storage._lock = threading.Lock()
    pool = MmItemMemoryPool.__new__(MmItemMemoryPool)
    pool._pool = storage
    pool._pool_ipc_handle = ("test-pool",)
    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda:"):
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    def write(_device, address, value, _transport):
        storage._control_words.view(-1)[(address - storage.base_address) // 4] = value

    def wait(_device, address, value, _transport):
        assert (
            storage._control_words.view(-1)[(address - storage.base_address) // 4]
            == value
        )

    with (
        patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
        patch.object(memory_pool, "stream_wait_value32", side_effect=wait),
        patch.object(memory_pool, "stream_write_value32", side_effect=write),
        patch.object(torch, "tensor", side_effect=cpu_tensor),
    ):
        yield pool


def _publish_cuda_ipc_proxy(pool, source):
    storage = pool._pool
    with storage._lock:
        lease = storage._allocate_locked(source.numel() * source.element_size())
    assert lease is not None
    destination = storage.byte_tensor[lease.start : lease.start + lease.nbytes]
    destination.copy_(source.view(torch.uint8))
    storage._control_words[lease.slot, 0] = lease.generation
    return CudaIpcTensorTransportProxy(
        data=destination,
        info_data=source,
        pool_ipc_handle=pool._pool_ipc_handle,
        pool_byte_offset=lease.start,
        ready_byte_offset=lease.ready_byte_offset,
        ack_byte_offset=lease.ack_byte_offset,
        generation=lease.generation,
        total_consumer_count=2,
        use_pool_handle_cache=True,
    )


@pytest.mark.parametrize("pickle_ipc", [False, True])
@pytest.mark.parametrize("reused", [False, True])
def test_cuda_ipc_cleanup_preserves_reused_pool_slots(
    _cuda_ipc_pool, pickle_ipc, reused
):
    pool = _cuda_ipc_pool
    source = torch.arange(16, dtype=torch.float32)
    old = _publish_cuda_ipc_proxy(pool, source)
    old._acknowledge_on_stream(pool._pool.base_address, 0, 2)
    pool._pool._recycle_ready_leases_locked()
    if reused:
        fresh = _publish_cuda_ipc_proxy(pool, source + 1)
        assert fresh.ready_byte_offset == old.ready_byte_offset
        assert fresh.generation != old.generation
    before = pool._pool.byte_tensor.clone()
    manager = _manager()
    manager.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=pool)
    request = _request(old)
    with patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc):
        request.wrap_pickle_fields()
        manager._cancel_cuda_ipc_features([request])
        manager._cancel_cuda_ipc_features([request])
    assert pool.active_lease_count == int(reused)
    assert torch.equal(pool._pool.byte_tensor, before)


@pytest.mark.parametrize("pickle_ipc", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_cuda_ipc_cleanup_deduplicates_shared_leases(
    _cuda_ipc_pool, pickle_ipc, batched
):
    pool = _cuda_ipc_pool
    source = torch.arange(16, dtype=torch.float32)
    shared = _publish_cuda_ipc_proxy(pool, source)
    requests = [_request(shared)]
    if batched:
        requests.append(_request(shared, rid="second"))
    else:
        requests[0].mm_inputs.mm_items[0].precomputed_embeddings = shared
    requests[-1].mm_inputs.mm_items.append(
        MultimodalDataItem(
            modality=Modality.IMAGE, feature=_publish_cuda_ipc_proxy(pool, source + 1)
        )
    )
    manager = _manager()
    manager.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=pool)
    cancel = pool.cancel_proxy
    cancelled = []

    def cancel_and_recycle(proxy):
        cancelled.append((proxy.ready_byte_offset, proxy.generation))
        cancel(proxy)
        pool._pool._recycle_ready_leases_locked()

    with (
        patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
        patch.object(pool, "cancel_proxy", side_effect=cancel_and_recycle),
    ):
        for request in requests:
            request.wrap_pickle_fields()
        manager._cancel_cuda_ipc_features(requests)
    assert len(cancelled) == len(set(cancelled)) == 2
    assert pool.active_lease_count == 0


@pytest.mark.parametrize("invalid", ["pool", "offset"])
def test_cuda_ipc_cleanup_rejects_invalid_proxies(_cuda_ipc_pool, invalid):
    pool = _cuda_ipc_pool
    proxy = _publish_cuda_ipc_proxy(pool, torch.arange(16, dtype=torch.float32))
    if invalid == "pool":
        proxy.proxy_state["ipc_extra"]["pool_handle"] = ("other-pool",)
    else:
        proxy.ready_byte_offset += 1
    manager = _manager()
    manager.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=pool)
    with pytest.raises(RuntimeError, match="does not belong|Invalid"):
        manager._cancel_cuda_ipc_features([_request(proxy)])
    assert pool.active_lease_count == 1
    assert not pool._pool._control_words[:, 1:].any()


@pytest.mark.parametrize("pickle_ipc", [False, True])
@pytest.mark.parametrize("http_request", [False, True])
def test_parallel_cuda_ipc_cancel_after_warmup(
    _cuda_ipc_pool, pickle_ipc, http_request
):
    async def run():
        pool = _cuda_ipc_pool
        manager = _manager()
        manager.canary_info_hook = None
        manager.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=pool)
        manager.rid_to_state = {}
        source = torch.arange(16, dtype=torch.float32)
        proxy = _publish_cuda_ipc_proxy(pool, source)
        obj = GenerateReqInput(
            text=["prompt"],
            sampling_params={"n": 3, "max_new_tokens": 1},
        )
        obj.normalize_batch_and_arguments()

        def init_state(req):
            state = ReqState(
                out_list=[],
                finished=False,
                event=asyncio.Event(),
                obj=req,
                time_stats=APIServerReqTimeStats(),
            )
            manager.rid_to_state[req.rid] = state
            return state

        manager._init_req_state = init_state
        states = {rid: init_state(obj[index]) for index, rid in enumerate(obj.rid)}

        async def tokenize(req):
            return TokenizedGenerateReqInput(
                rid=req.rid,
                input_text=req.text,
                input_ids=array("q", [1]),
                input_embeds=None,
                mm_inputs=_request(proxy).mm_inputs,
                token_type_ids=None,
                sampling_params=SamplingParams(max_new_tokens=1),
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
                time_stats=states[req.rid].time_stats,
            )

        manager._tokenize_one_request = tokenize
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        receiver = context.socket(zmq.PULL)
        stalled = context.socket(zmq.PUSH)
        sender.bind("inproc://parallel-ipc-cancel")
        receiver.connect("inproc://parallel-ipc-cancel")
        dispatch = manager._dispatch_to_scheduler
        futures = []
        caller = asyncio.current_task()
        unrelated_bytes = None

        def dispatch_and_cancel(packet):
            warmup = packet.sampling_params.max_new_tokens == 0
            manager.send_to_scheduler = sender if warmup else stalled
            future = dispatch(packet)
            futures.append(future)
            if not warmup:
                assert not future.done()
                asyncio.get_running_loop().call_soon(caller.cancel)
            return future

        manager._dispatch_to_scheduler = dispatch_and_cancel

        async def wait_warmup(req, request, state):
            nonlocal unrelated_bytes
            received = await io_struct.async_sock_recv(receiver)
            received.unwrap_pickle_fields()
            received_proxy = received.mm_inputs.mm_items[0].feature
            received_proxy._wait_until_ready(pool._pool.base_address, 0)
            received_proxy._acknowledge_on_stream(pool._pool.base_address, 0, 2)
            pool._pool._recycle_ready_leases_locked()
            assert pool.active_lease_count == 0
            fresh = _publish_cuda_ipc_proxy(pool, source + 1)
            assert fresh.ready_byte_offset == proxy.ready_byte_offset
            assert fresh.generation != proxy.generation
            unrelated_bytes = pool._pool.byte_tensor.clone()
            state.finished = True
            del manager.rid_to_state[req.rid]
            yield {"meta_info": {"id": req.rid}}

        manager._wait_one_response = wait_warmup
        request = Mock() if http_request else None
        generator = manager._handle_batch_request(obj, request, states)
        try:
            with (
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                pytest.raises(asyncio.CancelledError),
            ):
                await anext(generator)
            assert len(futures) == 2
            assert futures[0].done() and not futures[0].cancelled()
            assert futures[1].cancelled()
            assert pool.active_lease_count == 1
            assert torch.equal(pool._pool.byte_tensor, unrelated_bytes)
            if http_request:
                request.is_disconnected.assert_not_called()
        finally:
            await generator.aclose()
            sender.close(linger=0)
            receiver.close(linger=0)
            stalled.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run())


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("pickle_ipc", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "cancel_pending",
        "timeout",
        "dispatch_error",
        "prepare_error",
        "aborted",
        "queued_cancel",
        "timing_error",
    ],
)
def test_cuda_ipc_dispatch_lease_ownership(
    _cuda_ipc_pool, batched, pickle_ipc, failure
):
    async def run():
        pool = _cuda_ipc_pool
        manager = _manager()
        manager.mm_processor = SimpleNamespace(cudaipc_mmfeature_pool=pool)
        requests = []
        source = torch.arange(16, dtype=torch.float32)
        for index in range(2 if batched else 1):
            request = _request(_publish_cuda_ipc_proxy(pool, source), rid=str(index))
            request.mm_inputs.mm_items[
                0
            ].precomputed_embeddings = _publish_cuda_ipc_proxy(pool, source + 1)
            requests.append(request)
            manager.rid_to_state[request.rid] = SimpleNamespace(
                dispatched=False, abort_sent=failure == "aborted"
            )
        leases = list(pool._pool._occupied.values())
        queued = failure in ("queued_cancel", "timing_error")
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        receiver = context.socket(zmq.PULL)
        sender.bind("inproc://cudaipc-cleanup")
        if queued:
            receiver.connect("inproc://cudaipc-cleanup")
        if failure == "timeout":
            sender.setsockopt(zmq.SNDTIMEO, 10)
        manager.send_to_scheduler = sender
        original_dispatch = manager._dispatch_to_scheduler
        caller = asyncio.current_task()

        def dispatch(packet):
            if failure == "dispatch_error":
                raise RuntimeError("dispatch failed")
            future = original_dispatch(packet)
            assert future.done() == queued
            if failure == "queued_cancel":
                completion = asyncio.get_running_loop().create_future()

                def finish_and_cancel():
                    completion.set_result(None)
                    caller.cancel()

                asyncio.get_running_loop().call_soon(finish_and_cancel)
                return completion
            if failure == "cancel_pending":
                asyncio.get_running_loop().call_soon(caller.cancel)
            return future

        manager._dispatch_to_scheduler = dispatch
        if failure == "prepare_error":
            manager.cuda_vmm_feature_transport.prepare_for_dispatch_async.side_effect = RuntimeError(
                "preparation failed"
            )
        timing_patch = (
            patch.object(
                APIServerReqTimeStats,
                "set_api_server_dispatch_finish_time",
                side_effect=RuntimeError("timing failed"),
            )
            if failure == "timing_error"
            else nullcontext()
        )
        error = (
            asyncio.CancelledError
            if "cancel" in failure
            else zmq.Again
            if failure == "timeout"
            else RuntimeError
        )
        try:
            with patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc), timing_patch:
                expected = (
                    nullcontext() if failure == "aborted" else pytest.raises(error)
                )
                with expected:
                    if batched:
                        await manager._send_batch_request(requests)
                    else:
                        await manager._send_one_request(requests[0])
                assert all(
                    manager.rid_to_state[request.rid].dispatched == queued
                    for request in requests
                )
                if queued:
                    assert pool.active_lease_count == len(leases)
                    packet = await asyncio.wait_for(
                        io_struct.async_sock_recv(receiver), 2
                    )
                    received = packet.batch if batched else [packet]
                    for request in received:
                        request.unwrap_pickle_fields()
                        for offset, proxy in enumerate(
                            (
                                request.mm_inputs.mm_items[0].feature,
                                request.mm_inputs.mm_items[0].precomputed_embeddings,
                            )
                        ):
                            assert isinstance(proxy, CudaIpcTensorTransportProxy)
                            extra = proxy.proxy_state["ipc_extra"]
                            start = extra["pool_byte_offset"]
                            actual = pool._pool.byte_tensor[
                                start : start + extra["nbytes"]
                            ].view(torch.float32)
                            assert torch.equal(actual, source + offset)
                    assert all(
                        not pool._pool._control_words[lease.slot, 1:].any()
                        for lease in leases
                    )
                    for lease in leases:
                        pool._pool._control_words[lease.slot, 1:] = lease.generation
                pool._pool._recycle_ready_leases_locked()
                assert pool.active_lease_count == 0
                assert pool._pool._available_ranges == [(256, 1280)]
        finally:
            sender.close(linger=0)
            receiver.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run())


@pytest.fixture(autouse=True)
def _shm_transport():
    with (
        patch.object(mm_utils, "_get_is_default_transport", return_value=False),
        patch.object(
            mm_utils,
            "get_serving",
            return_value=SimpleNamespace(skip_tokenizer_init=False),
        ),
    ):
        yield


@contextmanager
def _exhaust_descriptors():
    original = resource.getrlimit(resource.RLIMIT_NOFILE)
    opened = []
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(original[0], 256), original[1]))
        while True:
            try:
                opened.append(os.open(os.devnull, os.O_RDONLY))
            except OSError as error:
                assert error.errno == errno.EMFILE
                break
        yield
    finally:
        for fd in opened:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, original)


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX shared memory")
def test_descriptor_exhaustion_cleanup_preserves_opened_receiver():
    source = torch.arange(256, dtype=torch.float32)
    pointer = ShmPointerMMData(source, precomputed_hash=123)
    receiver = pickle.loads(pickle.dumps(pointer))
    try:
        with _exhaust_descriptors():
            pointer.close_and_unlink()
        assert not os.path.exists(f"/dev/shm/{pointer.shm_name}")
        assert receiver.precomputed_hash == 123
        actual = receiver.materialize()
        assert torch.equal(actual, source)
        actual[0] = -1
        assert source[0] == 0
        pointer.close_and_unlink()  # Repeated rank cleanup is harmless.
    finally:
        pointer.close_and_unlink()
        receiver.close_and_unlink()


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX shared memory")
def test_failed_receiver_open_is_unlinked_at_descriptor_limit():
    pointer = ShmPointerMMData(torch.ones(256))
    wire = pickle.dumps(pointer)
    try:
        with _exhaust_descriptors():
            receiver = pickle.loads(wire)
            with pytest.raises(RuntimeError, match="Too many open files"):
                receiver.materialize()
        assert not os.path.exists(f"/dev/shm/{pointer.shm_name}")
    finally:
        pointer.close_and_unlink()


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX shared memory")
@pytest.mark.parametrize("warm_tracker,free_count", [(False, 2), (True, 1)])
def test_creation_failure_reclaims_unregistered_segment(warm_tracker, free_count):
    name = mm_utils.make_shm_name("mm")
    child_code = """
import errno
import os
import resource
import sys
from multiprocessing import resource_tracker
from unittest.mock import patch

import torch
from sglang.srt.managers import mm_utils

name, free_count, warm_tracker = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
assert resource_tracker._resource_tracker._fd is None
if warm_tracker:
    resource_tracker.ensure_running()
source = torch.ones(16)
original = resource.getrlimit(resource.RLIMIT_NOFILE)
opened = []
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(original[0], 128), original[1]))
    while True:
        try:
            opened.append(os.open(os.devnull, os.O_RDONLY))
        except OSError as error:
            assert error.errno == errno.EMFILE
            break
    for _ in range(free_count):
        os.close(opened.pop())
    with patch.object(mm_utils, "make_shm_name", return_value=name):
        try:
            pointer = mm_utils.ShmPointerMMData(source)
        except OSError as error:
            assert error.errno == errno.EMFILE
        else:
            pointer.close_and_unlink()
    assert not os.path.exists("/dev/shm/" + name)
finally:
    resource.setrlimit(resource.RLIMIT_NOFILE, original)
    for fd in opened:
        os.close(fd)
"""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                child_code,
                name,
                str(free_count),
                str(int(warm_tracker)),
            ],
            capture_output=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert b"KeyError:" not in result.stderr
        assert not os.path.exists(f"/dev/shm/{name}")
    finally:
        try:
            shared_memory._posixshmem.shm_unlink(f"/{name}")
        except FileNotFoundError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX shared memory")
@pytest.mark.parametrize("checkpoint", ["allocation", "copy"])
def test_producer_death_reclaims_segment_after_tracker_family_exit(
    checkpoint, tmp_path
):
    name = mm_utils.make_shm_name("mm")
    child = tmp_path / "producer_death.py"
    child.write_text("""
import multiprocessing
import os
import signal
import sys
from unittest.mock import patch


def produce(name, checkpoint, ready):
    import torch
    from sglang.srt.managers import mm_utils

    torch.set_num_threads(1)
    source = torch.ones(16384)
    reserve = os.posix_fallocate
    copy = torch.Tensor.copy_

    def wait_for_death():
        ready.send(os.getpid())
        multiprocessing.parent_process().join(timeout=30)
        raise AssertionError("producer was not killed")

    def allocation(fd, offset, length):
        reserve(fd, offset, length)
        wait_for_death()

    def partial_copy(dst, src, *args, **kwargs):
        half = dst.numel() // 2
        copy(dst[:half], src[:half])
        wait_for_death()

    target = os if checkpoint == "allocation" else torch.Tensor
    attribute = "posix_fallocate" if checkpoint == "allocation" else "copy_"
    replacement = allocation if checkpoint == "allocation" else partial_copy
    with (
        patch.object(mm_utils, "make_shm_name", return_value=name),
        patch.object(target, attribute, replacement),
    ):
        mm_utils.ShmPointerMMData(source)


if __name__ == "__main__":
    name, checkpoint = sys.argv[1:]
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    producer = context.Process(target=produce, args=(name, checkpoint, sender))
    producer.start()
    sender.close()
    try:
        assert receiver.poll(30), "producer did not reach checkpoint"
        assert receiver.recv() == producer.pid
        assert os.path.exists("/dev/shm/" + name)
        os.kill(producer.pid, signal.SIGKILL)
        producer.join(timeout=10)
        assert producer.exitcode == -signal.SIGKILL
        # The surviving parent still holds the shared tracker's writer.
        assert os.path.exists("/dev/shm/" + name)
    finally:
        if producer.is_alive():
            producer.kill()
            producer.join(timeout=10)
        receiver.close()
        producer.close()
""")
    try:
        # Captured stderr stays open in the tracker until its EOF cleanup ends.
        result = subprocess.run(
            [sys.executable, str(child), name, checkpoint],
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert b"KeyError:" not in result.stderr
        assert not os.path.exists(f"/dev/shm/{name}")
    finally:
        try:
            shared_memory._posixshmem.shm_unlink(f"/{name}")
        except FileNotFoundError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX shared memory")
@pytest.mark.parametrize("failure", ["allocation", "copy"])
def test_constructor_failure_balances_resource_tracker(failure):
    name = mm_utils.make_shm_name("mm")
    child_code = """
import errno
import os
import sys
from multiprocessing import resource_tracker
from unittest.mock import patch

import torch
from sglang.srt.managers import mm_utils

name, failure = sys.argv[1:]
resource_tracker.ensure_running()
source = torch.ones(16384)
copy = torch.Tensor.copy_

def allocation(*args):
    raise OSError(errno.ENOSPC, "allocation failed")

def partial_copy(dst, src, *args, **kwargs):
    half = dst.numel() // 2
    copy(dst[:half], src[:half])
    raise RuntimeError("copy failed")

target = os if failure == "allocation" else torch.Tensor
attribute = "posix_fallocate" if failure == "allocation" else "copy_"
replacement = allocation if failure == "allocation" else partial_copy
with (
    patch.object(mm_utils, "make_shm_name", return_value=name),
    patch.object(target, attribute, replacement),
):
    try:
        mm_utils.ShmPointerMMData(source)
    except (OSError, RuntimeError) as error:
        assert failure + " failed" in str(error)
    else:
        raise AssertionError("constructor unexpectedly succeeded")
assert not os.path.exists("/dev/shm/" + name)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", child_code, name, failure],
            capture_output=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert b"KeyError:" not in result.stderr
        assert b"leaked shared_memory" not in result.stderr
        assert not os.path.exists(f"/dev/shm/{name}")
    finally:
        try:
            shared_memory._posixshmem.shm_unlink(f"/{name}")
        except FileNotFoundError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX shared memory")
def test_failed_open_cleanup_preserves_independent_resource_tracker():
    pointer = ShmPointerMMData(torch.ones(16))
    child_code = """
import errno
import os
import pickle
import resource
import sys
from multiprocessing import shared_memory
from sglang.srt.managers.mm_utils import ShmPointerMMData

wire = sys.stdin.buffer.read()
# Start a tracker independent of the sender's, without registering its segment.
dummy = shared_memory.SharedMemory(create=True, size=1)
original = resource.getrlimit(resource.RLIMIT_NOFILE)
opened = []
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(original[0], 256), original[1]))
    while True:
        try:
            opened.append(os.open(os.devnull, os.O_RDONLY))
        except OSError as error:
            assert error.errno == errno.EMFILE
            break
    receiver = pickle.loads(wire)
    try:
        receiver.materialize()
    except RuntimeError as error:
        assert "Too many open files" in str(error)
    else:
        raise AssertionError("receiver unexpectedly opened the segment")
finally:
    for fd in opened:
        os.close(fd)
    resource.setrlimit(resource.RLIMIT_NOFILE, original)
    dummy.close()
    dummy.unlink()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", child_code],
            input=pickle.dumps(pointer),
            capture_output=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert not os.path.exists(f"/dev/shm/{pointer.shm_name}")
        assert b"KeyError:" not in result.stderr
    finally:
        # The independent receiver unlinks; retire the sender's registration too.
        if not os.path.exists(f"/dev/shm/{pointer.shm_name}"):
            resource_tracker.unregister(f"/{pointer.shm_name}", "shared_memory")
        pointer.close_and_unlink()


@pytest.mark.parametrize("failure", ["partial", "serialize", "sync", "async"])
def test_send_failure_releases_new_allocations_only(failure):
    async def run():
        manager = _manager()
        borrowed = ShmPointerMMData(torch.tensor([7.0]))
        request = _request([torch.ones(16), torch.ones(16), borrowed])
        created = []

        def allocate(*args, **kwargs):
            if failure == "partial" and created:
                raise RuntimeError("preparation failed")
            pointer = ShmPointerMMData(*args, **kwargs)
            created.append(pointer)
            return pointer

        manager._dispatch_to_scheduler = Mock(side_effect=RuntimeError("send failed"))
        if failure == "async":
            future = asyncio.get_running_loop().create_future()
            future.set_exception(RuntimeError("send failed"))
            manager._dispatch_to_scheduler = Mock(return_value=future)
        if failure == "serialize":
            request.time_stats.unpicklable = lambda: None
        try:
            with (
                patch.object(mm_utils, "ShmPointerMMData", side_effect=allocate),
                patch.object(io_struct, "_USE_PICKLE_IPC", False),
            ):
                with pytest.raises(Exception):
                    await manager._send_one_request(request)
            assert created
            assert all(not os.path.exists(f"/dev/shm/{p.shm_name}") for p in created)
            assert os.path.exists(f"/dev/shm/{borrowed.shm_name}")
            manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once()
            manager.positional_embeds_transport.cancel_for_dispatch.assert_called_once()
        finally:
            for pointer in [*created, borrowed]:
                pointer.close_and_unlink()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["partial", "send"])
def test_send_failure_releases_model_specific_allocations(failure):
    async def run():
        manager = _manager()
        borrowed = ShmPointerMMData(torch.tensor([7.0]))
        request = _request(None)
        request.mm_inputs.shm_model_specific_data = True
        item = request.mm_inputs.mm_items[0]
        item.model_specific_data = {
            "first": torch.ones(mm_utils._SHM_MODEL_DATA_MIN_BYTES // 4),
            "second": torch.ones(mm_utils._SHM_MODEL_DATA_MIN_BYTES // 4),
            "borrowed": borrowed,
        }
        created = []

        def allocate(*args, **kwargs):
            if failure == "partial" and created:
                raise RuntimeError("preparation failed")
            pointer = ShmPointerMMData(*args, **kwargs)
            created.append(pointer)
            return pointer

        manager._dispatch_to_scheduler = Mock(side_effect=RuntimeError("send failed"))
        try:
            with (
                patch.object(mm_utils, "ShmPointerMMData", side_effect=allocate),
                patch.object(
                    mm_utils.envs.SGLANG_ENABLE_UOCR_SHM_WRAP_MODEL_DATA,
                    "get",
                    return_value=True,
                ),
                patch.object(io_struct, "_USE_PICKLE_IPC", False),
            ):
                with pytest.raises(
                    RuntimeError, match="preparation failed|send failed"
                ):
                    await manager._send_one_request(request)
            assert created
            assert all(not os.path.exists(f"/dev/shm/{p.shm_name}") for p in created)
            assert os.path.exists(f"/dev/shm/{borrowed.shm_name}")
        finally:
            for pointer in [*created, borrowed]:
                pointer.close_and_unlink()

    asyncio.run(run())


@pytest.mark.parametrize(
    "batched,model_data", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("pickle_ipc", [False, True])
def test_real_send_cancellation_transfers_ownership(
    batched, model_data, queued, pickle_ipc
):
    async def run():
        manager = _manager()
        size = mm_utils._SHM_MODEL_DATA_MIN_BYTES // 4 if model_data else 16
        source = torch.arange(size, dtype=torch.float32)
        request = _request(None if model_data else source)
        if model_data:
            request.mm_inputs.shm_model_specific_data = True
            request.mm_inputs.mm_items[0].model_specific_data = {"tensor": source}
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        receiver = context.socket(zmq.PULL)
        sender.bind("inproc://cleanup")
        if queued:
            receiver.connect("inproc://cleanup")
        manager.send_to_scheduler = sender
        original = manager._dispatch_to_scheduler
        caller = asyncio.current_task()

        def dispatch(packet):
            future = original(packet)
            if queued:
                assert future.done() and future.exception() is None
                # Enqueue finishes while the caller is suspended in await.
                completion = asyncio.get_running_loop().create_future()

                def finish_and_cancel():
                    completion.set_result(None)
                    caller.cancel()

                asyncio.get_running_loop().call_soon(finish_and_cancel)
                return completion
            else:
                asyncio.get_running_loop().call_soon(caller.cancel)
            return future

        manager._dispatch_to_scheduler = dispatch
        try:
            with (
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                patch.object(
                    mm_utils.envs.SGLANG_ENABLE_UOCR_SHM_WRAP_MODEL_DATA,
                    "get",
                    return_value=True,
                ),
            ):
                with pytest.raises(asyncio.CancelledError):
                    if batched:
                        await manager._send_batch_request([request])
                    else:
                        await manager._send_one_request(request)
                assert manager.rid_to_state[request.rid].dispatched == queued
                if queued:
                    packet = await asyncio.wait_for(
                        io_struct.async_sock_recv(receiver), 5
                    )
                    received = packet.batch[0] if batched else packet
                    mm_utils.unwrap_shm_features(received)
                    item = received.mm_inputs.mm_items[0]
                    actual = (
                        item.model_specific_data["tensor"]
                        if model_data
                        else item.feature
                    )
                    assert torch.equal(actual, source)
                    actual[0] = -1
                    assert source[0] == 0
                    manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()
                else:
                    manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once()
                    if not batched:
                        item = request.mm_inputs.mm_items[0]
                        pointer = (
                            item.model_specific_data["tensor"]
                            if model_data
                            else item.feature
                        )
                        assert not os.path.exists(f"/dev/shm/{pointer.shm_name}")
        finally:
            mm_utils.discard_shm_features(request)
            sender.close(linger=0)
            receiver.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run())


@pytest.mark.parametrize("feature_kind", ["tensor", "list", "tuple"])
@pytest.mark.parametrize("pickle_ipc", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_router_failed_forward_releases_received_features(
    feature_kind, pickle_ipc, batched
):
    async def run():
        tensors = [torch.ones(16384), torch.zeros(16384)]
        if feature_kind == "tensor":
            feature = tensors[0]
        elif feature_kind == "tuple":
            feature = tuple(tensors)
        else:
            feature = tensors
        requests = [_request(feature)]
        requests[0].mm_inputs.mm_items.append(
            MultimodalDataItem(modality=Modality.IMAGE, precomputed_embeddings=feature)
        )
        if batched:
            child = _request(None, rid="second")
            child.mm_inputs.mm_items[0].precomputed_embeddings = feature
            requests.append(child)
        handles = []
        context = zmq.asyncio.Context()
        worker = context.socket(zmq.PUSH)
        receiver = context.socket(zmq.PULL)
        scheduler_sender = context.socket(zmq.PUSH)
        worker.bind("inproc://router-cleanup")
        receiver.connect("inproc://router-cleanup")
        scheduler_sender.setsockopt(zmq.SNDTIMEO, 10)
        router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        router.receive_from_worker = receiver
        router.send_to_scheduler = scheduler_sender
        try:
            pointer = ShmPointerMMData(torch.ones(16))
            handles.append(pointer)
            requests[0].mm_inputs.mm_items[0].model_specific_data["tensor"] = pointer
            with patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc):
                for request in requests:
                    mm_utils.wrap_shm_features(request, pending_handles=handles)
                    request.wrap_pickle_fields()
                packet = (
                    io_struct.BatchTokenizedEmbeddingReqInput(batch=requests)
                    if batched
                    else requests[0]
                )
                await asyncio.wait_for(io_struct.async_sock_send(worker, packet), 5)
                with pytest.raises(zmq.Again):
                    await asyncio.wait_for(router.router_worker_obj(), 5)
            assert handles
            assert all(not os.path.exists(f"/dev/shm/{p.shm_name}") for p in handles)
        finally:
            for pointer in handles:
                pointer.close_and_unlink()
            worker.close(linger=0)
            receiver.close(linger=0)
            scheduler_sender.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run())


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("pickle_ipc", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("failure", ["mark_dispatched", "timing_finish", "lost_future"])
def test_bookkeeping_failure_preserves_queued_request(
    batched, pickle_ipc, pending, failure
):
    async def run():
        manager = _manager()
        source = torch.arange(16, dtype=torch.float32)
        request = _request(source)
        context = zmq.asyncio.Context()
        sender = context.socket(zmq.PUSH)
        receiver = context.socket(zmq.PULL)
        sender.bind("inproc://bookkeeping")
        if not pending:
            receiver.connect("inproc://bookkeeping")
        manager.send_to_scheduler = sender
        original_dispatch = manager._dispatch_to_scheduler
        enqueue_futures = []

        def dispatch(packet):
            future = original_dispatch(packet)
            enqueue_futures.append(future)
            assert future.done() == (not pending)
            if pending:
                asyncio.get_running_loop().call_soon(
                    receiver.connect, "inproc://bookkeeping"
                )
            return None if failure == "lost_future" else future

        manager._dispatch_to_scheduler = dispatch
        if batched:
            mm_utils.wrap_shm_features(request)
        if failure == "mark_dispatched":
            failure_patch = patch.object(
                manager,
                "_mark_state_dispatched",
                side_effect=RuntimeError("bookkeeping failed"),
            )
        elif failure == "timing_finish":
            failure_patch = patch.object(
                APIServerReqTimeStats,
                "set_api_server_dispatch_finish_time",
                side_effect=RuntimeError("bookkeeping failed"),
            )
        else:
            failure_patch = nullcontext()
        error_type, error_message = (
            (TypeError, "must return its enqueue Future")
            if failure == "lost_future"
            else (RuntimeError, "bookkeeping failed")
        )
        try:
            with (
                patch.object(io_struct, "_USE_PICKLE_IPC", pickle_ipc),
                failure_patch,
            ):
                with pytest.raises(error_type, match=error_message):
                    if batched:
                        await manager._send_batch_request([request])
                    else:
                        await manager._send_one_request(request)
                if failure == "lost_future":
                    await asyncio.wait_for(enqueue_futures[0], 5)
                packet = await asyncio.wait_for(io_struct.async_sock_recv(receiver), 5)
                received = packet.batch[0] if batched else packet
                mm_utils.unwrap_shm_features(received)
                assert torch.equal(received.mm_inputs.mm_items[0].feature, source)
                manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_not_called()
                manager.positional_embeds_transport.cancel_for_dispatch.assert_not_called()
        finally:
            mm_utils.discard_shm_features(request)
            sender.close(linger=0)
            receiver.close(linger=0)
            context.destroy(linger=0)

    asyncio.run(run())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
