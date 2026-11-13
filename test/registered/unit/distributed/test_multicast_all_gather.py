"""Direct multicast allgather: layout, barrier reuse, and foreground progress."""

import ctypes
import json
import math
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators.gated_transfer import (
    Link,
    TransferChain,
    paused_background_transfers,
)
from sglang.srt.distributed.device_communicators.multicast_all_gather import (
    MulticastAllGather,
    MulticastUnavailableError,
    _multicast_address,
    has_multicast_support,
)
from sglang.srt.environ import envs
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=90, stage="base-c", runner_config="4-gpu-b200")


class TestMulticastAllGatherLayout(CustomTestCase):
    def test_multicast_view_offset_and_mapping_validation(self):
        storage = torch.empty(100, dtype=torch.uint8)
        output = storage[7:67]
        handle = SimpleNamespace(
            rank=1,
            world_size=3,
            multicast_ptr=5000,
            buffer_ptrs=[1000, 2000, 3000],
            offset=16,
            buffer_size=116,
        )
        # The handle's offset has already been applied to its multicast pointer.
        self.assertEqual(_multicast_address(output, handle, 1, 3), 5007)
        self.assertEqual(_multicast_address(storage[7:7], handle, 1, 3), 5007)
        for change in (
            {"rank": 0},
            {"world_size": 2},
            {"multicast_ptr": 0},
            {"buffer_ptrs": [1000, 0, 3000]},
            {"buffer_ptrs": [1000, 2000]},
            {"buffer_size": 82},
        ):
            with self.subTest(change=change):
                bad = SimpleNamespace(**(vars(handle) | change))
                with self.assertRaises((ValueError, RuntimeError)):
                    _multicast_address(output, bad, 1, 3)

    def test_direct_rank_major_chunks_with_tail_and_input_alias(self):
        world, count, chunk_elements, multicast_base = 3, 19, 8, 1 << 40
        for dtype in (torch.uint8, torch.float32):
            for in_place in (False, True):
                with self.subTest(dtype=dtype, in_place=in_place):
                    values = torch.arange(world * count).to(dtype).view(world, count)
                    storage = [
                        torch.full((world * count + 10,), 99, dtype=dtype)
                        for _ in range(world)
                    ]
                    outputs = [tensor[3:-7] for tensor in storage]
                    for rank in range(world):
                        outputs[rank][rank * count : (rank + 1) * count].copy_(
                            values[rank]
                        )
                    for rank in range(world):
                        events, submitted = [], []
                        producer = object()

                        def run(chunks, *, setup, finish, after):
                            self.assertIs(after, producer)
                            for operation in setup:
                                operation(None)
                            for copy in chunks:
                                submitted.append(copy)
                                offset = copy.dst - multicast_base
                                for output in outputs:
                                    ctypes.memmove(
                                        output.data_ptr() + offset,
                                        copy.src,
                                        copy.nbytes,
                                    )
                            for operation in finish:
                                operation(None)
                            return 17

                        gather = MulticastAllGather.__new__(MulticastAllGather)
                        gather.output = outputs[rank].view(world * count, 1)
                        gather.rank, gather.world_size = rank, world
                        gather.chunk_bytes = chunk_elements * values.element_size()
                        gather._multicast_ptr = multicast_base
                        gather.chain = SimpleNamespace(run=run)
                        gather._entry = lambda stream: events.append("entry")
                        gather._exit = lambda stream: events.append("exit")
                        source = (
                            outputs[rank][rank * count : (rank + 1) * count]
                            if in_place
                            else values[rank]
                        )
                        ticket = gather.run(source.view(count, 1), after=producer)
                        self.assertEqual(ticket, 17)
                        self.assertEqual(events, ["entry", "exit"])
                        self.assertEqual(
                            len(submitted), math.ceil(count / chunk_elements)
                        )
                        self.assertEqual(
                            sum(copy.nbytes for copy in submitted), source.nbytes
                        )
                        self.assertTrue(
                            all(copy.nbytes <= gather.chunk_bytes for copy in submitted)
                        )
                    for output, backing in zip(outputs, storage):
                        torch.testing.assert_close(output.view(world, count), values)
                        self.assertTrue(bool((backing[:3] == 99).all()))
                        self.assertTrue(bool((backing[-7:] == 99).all()))

    def test_output_views_land_at_their_offset_in_the_mapping(self):
        world, count, multicast_base = 3, 5, 1 << 40
        dtype = torch.float32
        values = torch.arange(world * count, dtype=dtype).view(world, count)
        # Every rank binds the same symmetric range; gathers use views of it.
        bound = [torch.full((40,), -1.0, dtype=dtype) for _ in range(world)]

        def gather_for(rank):
            def run(chunks, *, setup, finish, after):
                for copy in chunks:
                    offset = copy.dst - multicast_base
                    for output in bound:
                        ctypes.memmove(
                            output.data_ptr() + offset, copy.src, copy.nbytes
                        )
                return 0

            gather = MulticastAllGather.__new__(MulticastAllGather)
            gather.output = bound[rank]
            gather.rank, gather.world_size, gather.chunk_bytes = rank, world, 8
            gather._multicast_ptr = multicast_base
            gather.chain = SimpleNamespace(run=run)
            gather._entry = gather._exit = lambda stream: None
            return gather

        for start in (0, 3, 40 - world * count):
            with self.subTest(start=start):
                for output in bound:
                    output.fill_(-1)
                for rank in range(world):
                    view = bound[rank][start : start + world * count]
                    gather_for(rank).run(values[rank], output=view, after=None)
                for output in bound:
                    view = output[start : start + world * count]
                    torch.testing.assert_close(view.view(world, count), values)
                    outside = torch.cat(
                        [output[:start], output[start + world * count :]]
                    )
                    self.assertTrue(bool((outside == -1).all()))

        gather = gather_for(1)
        backing = torch.empty(60, dtype=dtype)
        gather.output = backing[10:50]
        for output in (
            torch.empty(world * count, dtype=dtype),
            backing[40 : 40 + world * count],
            backing[10:40].view(torch.int32)[: world * count],
            backing[10:40:2],
        ):
            with self.subTest(offset=output.storage_offset(), dtype=output.dtype):
                with self.assertRaises(ValueError):
                    gather.run(values[1], output=output, after=None)

    def test_rejects_invalid_inputs_before_submission(self):
        gather = MulticastAllGather.__new__(MulticastAllGather)
        gather.output = torch.empty(57)
        gather.rank, gather.world_size, gather.chunk_bytes = 1, 3, 32
        gather._multicast_ptr = 1 << 40
        submitted = []
        gather.chain = SimpleNamespace(
            run=lambda *args, **kwargs: submitted.append(args)
        )
        gather._entry = gather._exit = lambda stream: None
        for source in (
            torch.empty(18),
            torch.empty(19, dtype=torch.float64),
            torch.empty(38)[::2],
            gather.output[:19],
            gather.output[20:39],
        ):
            with self.subTest(
                shape=source.shape, dtype=source.dtype, offset=source.storage_offset()
            ):
                with self.assertRaises(ValueError):
                    gather.run(source, after=object())
        self.assertFalse(submitted)

    def test_empty_input_still_submits_both_barriers(self):
        events = []

        def run(chunks, *, setup, finish, after):
            self.assertFalse(chunks)
            for operation in (*setup, *finish):
                operation(None)
            return 23

        gather = MulticastAllGather.__new__(MulticastAllGather)
        gather.output = torch.empty(3)[:0]
        gather.rank, gather.world_size, gather.chunk_bytes = 1, 3, 32
        gather._multicast_ptr = 1 << 40
        gather.chain = SimpleNamespace(run=run)
        gather._entry = lambda stream: events.append("entry")
        gather._exit = lambda stream: events.append("exit")
        self.assertEqual(gather.run(torch.empty(0), after=object()), 23)
        self.assertEqual(events, ["entry", "exit"])


_WORLD = 4 if torch.cuda.device_count() >= 4 else 2


def _wait_until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, (
            "multicast transfer or foreground collective hung"
        )
        time.sleep(0.005)


def _run_multicast(rank, port, backend):
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    from cuda.bindings import driver as cu

    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.utils.cuda_vmm_utils import check_drv

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD,
        device_id=device,
    )
    group = dist.group.WORLD
    cpu_group = dist.new_group(backend="gloo")
    if backend is not None:
        symm_mem.set_backend(backend)
    foreground = PyNcclCommunicator(cpu_group, device)
    chain = TransferChain(Link.NVLINK)
    count, chunk_bytes = (1 << 18) + 17, 128 << 10
    backing = symm_mem.empty(_WORLD * count + 10, dtype=torch.float32, device=device)
    output = backing[3:-7]
    gather = MulticastAllGather(group, output, chain, chunk_bytes=chunk_bytes)
    backing2 = symm_mem.empty(_WORLD * count + 10, dtype=torch.float32, device=device)
    output2 = backing2[3:-7]
    gather2 = MulticastAllGather(group, output2, chain, chunk_bytes=chunk_bytes)
    producer, producer2, consumer, foreground_stream = (
        torch.cuda.Stream() for _ in range(4)
    )
    finished, foreground_done = torch.cuda.Event(), torch.cuda.Event()
    finished.record()
    foreground_done.record()
    local = torch.arange(count, dtype=torch.float32, device=device) + rank * count
    expected = torch.arange(_WORLD * count, dtype=torch.float32, device=device)
    source, source2 = torch.empty_like(local), torch.empty_like(local)
    x = torch.full((4096,), rank + 1.0, device=device)
    with torch.cuda.stream(foreground_stream), foreground.change_state(enable=True):
        foreground.all_reduce(x)
    torch.cuda.synchronize()

    def prepare(generation, source=source, backing=backing):
        backing.fill_(-1)
        source.copy_(local)
        source.add_(generation * _WORLD * count)

    def check(generation, backing=backing):
        torch.testing.assert_close(
            backing[3:-7], expected + generation * _WORLD * count, rtol=0, atol=0
        )
        assert bool((backing[:3] == -1).all()) and bool((backing[-7:] == -1).all())

    def wait(ticket):
        chain.wait(ticket, stream=consumer)
        finished.record(consumer)
        _wait_until(finished.query)

    # Warm up every operation before a gate can park. Then reuse the same
    # barrier lanes with fresh input, including an exact own-rank input alias.
    for generation in range(3):
        prepare(generation)
        own = output[rank * count : (rank + 1) * count]
        own.copy_(source)
        ticket = gather.run(own if generation else source)
        wait(ticket)
        check(generation)

    # Independent producers and outputs, serialized on one transfer chain.
    with torch.cuda.stream(producer):
        prepare(3)
    with torch.cuda.stream(producer2):
        prepare(4, source2, backing2)
    first = gather.run(source, after=producer)
    second = gather2.run(source2, after=producer2)
    wait(second)
    assert chain.query(first)
    check(3)
    check(4, backing2)

    # A paused sender holds completion on every rank. The independent,
    # prewarmed foreground communicator must still complete before release.
    prepare(5)
    x.fill_(rank + 1)
    torch.cuda.synchronize()
    if rank == 0:
        chain._gate._word.value = 7
    dist.barrier(group=cpu_group)
    ticket = gather.run(source)
    try:
        time.sleep(0.1)
        assert not chain.query(ticket), "a rank completed without the paused sender"
        with torch.cuda.stream(foreground_stream), foreground.change_state(enable=True):
            foreground.all_reduce(x)
            foreground_done.record()
        _wait_until(foreground_done.query)
        assert not chain.query(ticket), "background transfer ignored its pause"
        dist.barrier(group=cpu_group)
    finally:
        if rank == 0:
            chain._gate.force_open()
    wait(ticket)
    check(5)
    assert bool((x == _WORLD * (_WORLD + 1) / 2).all())

    # Exercise the public bracket with deterministic pending background work.
    blocker = torch.zeros(1, dtype=torch.int32).pin_memory()

    def hold(stream):
        check_drv(
            cu.cuStreamWaitValue32(stream.cuda_stream, blocker.data_ptr(), 1, 1),
            "test blocker",
        )

    prepare(6)
    x.fill_(rank + 1)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    chain.run([hold])
    ticket = gather.run(source)
    pauses, forced = chain._gate._seq, chain._gate.forced_opens
    try:
        with torch.cuda.stream(foreground_stream), foreground.change_state(enable=True):
            with paused_background_transfers(Link.NVLINK):
                foreground.all_reduce(x)
            foreground_done.record()
        _wait_until(foreground_done.query)
        assert chain._gate._seq == pauses + 1
        assert not chain.query(ticket)
    finally:
        blocker[0] = 1
    wait(ticket)
    check(6)
    assert bool((x == _WORLD * (_WORLD + 1) / 2).all())
    assert chain._gate.forced_opens == forced

    # Trace a complete warmed gather: no SM kernel or local layout-copy pass.
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
        prepare(7)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            ticket = gather.run(source)
            wait(ticket)
        with tempfile.TemporaryDirectory(
            prefix="multicast-gather-profile-"
        ) as directory:
            path = Path(directory) / "trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        assert not any(event.get("cat") == "kernel" for event in events), (
            "gather used SM kernels"
        )
        copies = [event for event in events if event.get("cat") == "gpu_memcpy"]
        assert len(copies) == math.ceil(source.nbytes / chunk_bytes) + 2
        assert sum(event["args"]["bytes"] for event in copies) == source.nbytes + 8
        check(7)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    # Keep the process group alive while symmetric-memory handles remain.


def _run_matches_nccl(rank, port, backend):
    """Bitwise agreement with NCCL's allgather across dtypes, sizes, chunk
    sizes, in-place inputs, and output views at varying offsets."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD,
        device_id=device,
    )
    group = dist.group.WORLD
    if backend is not None:
        symm_mem.set_backend(backend)
    counts = (1, 7, 4099, (1 << 19) + 3)
    max_bytes, slack = counts[-1] * 4, (64 << 10) + 1
    # One bound mapping, as in the KV-shard pool: each gather lands in a view
    # at its own offset, and the bytes around that view must stay untouched.
    # Its size need not divide by the group size.
    bound = symm_mem.empty(_WORLD * max_bytes + slack, dtype=torch.uint8, device=device)
    chain = TransferChain(Link.NVLINK)
    # Chunks with a partial tail (the smallest queues more ops than the chain
    # keeps on the GPU at once), and inputs that fit in one chunk.
    gathers = [
        MulticastAllGather(group, bound, chain, chunk_bytes=chunk_bytes)
        for chunk_bytes in (4100, 64 << 10, 8 << 20)
    ]
    generator = torch.Generator(device=device)
    case = 0
    for dtype in (torch.uint8, torch.bfloat16, torch.float32):
        for count in counts:
            for gather in gathers:
                for in_place in (False, True):
                    case += 1
                    nbytes = count * dtype.itemsize
                    # Random bytes, so float inputs include NaN/Inf patterns.
                    generator.manual_seed(case * _WORLD + rank)
                    local = torch.randint(
                        0,
                        256,
                        (nbytes,),
                        dtype=torch.uint8,
                        device=device,
                        generator=generator,
                    ).view(dtype)
                    expected = torch.empty(_WORLD * count, dtype=dtype, device=device)
                    dist.all_gather_into_tensor(expected, local, group=group)

                    start = 16 * (case * 257 % 4096)
                    end = start + _WORLD * nbytes
                    bound.fill_(0xA5)
                    output = bound[start:end].view(dtype)
                    source = local
                    if in_place:
                        source = output[rank * count : (rank + 1) * count]
                        source.copy_(local)
                    chain.wait(gather.run(source, output=output))
                    label = (
                        f"rank {rank}, {dtype}, {count} per rank, chunk "
                        f"{gather.chunk_bytes} B, in_place={in_place}"
                    )
                    assert torch.equal(
                        output.view(torch.uint8), expected.view(torch.uint8)
                    ), f"{label}: differs from NCCL"
                    assert bool((bound[:start] == 0xA5).all()), label
                    assert bool((bound[end:] == 0xA5).all()), label
    torch.cuda.synchronize()
    dist.barrier()
    # Keep the process group alive while symmetric-memory handles remain.


class _WithoutMulticast:
    """A rendezvous handle whose multicast mapping is missing."""

    multicast_ptr = 0

    def __init__(self, handle):
        self._handle = handle

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _run_without_control_mapping(rank, port, backend):
    """Rank 1's control buffer gets no multicast mapping: every rank raises,
    and none waits on rank 1 in the construction barrier."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD,
        device_id=device,
    )
    if backend is not None:
        symm_mem.set_backend(backend)
    output = symm_mem.empty(_WORLD * 1024, dtype=torch.uint8, device=device)
    rendezvous, handles = symm_mem.rendezvous, []

    def rendezvous_without_control_multicast(tensor, group):
        handles.append(rendezvous(tensor, group))
        # The output's rendezvous comes first, the control buffer's second.
        if rank == 1 and len(handles) == 2:
            return _WithoutMulticast(handles[-1])
        return handles[-1]

    with unittest.mock.patch.object(
        symm_mem, "rendezvous", rendezvous_without_control_multicast
    ):
        try:
            MulticastAllGather(dist.group.WORLD, output, TransferChain(Link.NVLINK))
        except MulticastUnavailableError as error:
            assert str(error).startswith("rank 1: "), error
        else:
            raise AssertionError("built a gather without rank 1's control mapping")
    assert len(handles) == 2
    dist.barrier()


@unittest.skipIf(
    torch.cuda.device_count() < 2
    or torch.version.cuda is None
    or not has_multicast_support(torch.device("cuda", 0)),
    "needs two CUDA GPUs with multicast support",
)
class TestMulticastAllGatherMultiGpu(CustomTestCase):
    def _spawn(self, backend, port, target=_run_multicast):
        with (
            envs.SGLANG_GATED_TRANSFER_WATCHDOG_MS.override(0),
            temp_set_env(
                NCCL_CUMEM_ENABLE="1",
                NCCL_NVLS_ENABLE="1",
                CUDA_DEVICE_MAX_CONNECTIONS="32",
            ),
        ):
            mp.spawn(target, args=(port, backend), nprocs=_WORLD, join=True)

    def test_nccl_backend(self):
        self._spawn("NCCL", 29836)

    def test_default_backend(self):
        self._spawn(None, 29837)

    def test_matches_nccl_all_gather(self):
        self._spawn(None, 29838, target=_run_matches_nccl)

    def test_matches_nccl_all_gather_with_nccl_backend(self):
        self._spawn("NCCL", 29839, target=_run_matches_nccl)

    def test_rank_without_mapping_raises_on_every_rank(self):
        self._spawn(None, 29840, target=_run_without_control_mapping)


if __name__ == "__main__":
    unittest.main()
