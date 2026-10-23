"""One-sided zero-SM transfers over an NCCL symmetric window
(``symmetric_transfer``), gated by ``gated_transfer``.

- CPU: the order and size of the push chunks (unicast peer rotation, one
  copy per gated chunk, the in-place self skip; one copy per span through a
  multicast VA).
- 2-4 GPUs: allgather and broadcast through the multicast VA (skipped without
  NVLS) and by unicast push (NVLS off, as the engine sets it): data, an
  unconsumed transfer consumed by the next one, and a paused rank holding
  every rank's read.

Pause words are flipped from the host (they live in pinned memory): GPU work
queued while a gate is parked can share its hardware queue.
"""

import time
import unittest
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from sglang.srt.distributed.device_communicators.gated_transfer import (
    Link,
    TransferChain,
    get_transfer_gate,
)
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-c", runner_config="4-gpu-b200")

_MiB = 1 << 20


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


class TestSymmetricPushChunks(CustomTestCase):
    def _transfer(self, *, rank, targets, chunk_bytes):
        from sglang.srt.distributed.device_communicators.symmetric_transfer import (
            SymmetricAllGather,
        )

        transfer = SymmetricAllGather.__new__(SymmetricAllGather)
        transfer.barrier = SimpleNamespace(
            rank=rank,
            world_size=4,
            local_base=targets[rank] if len(targets) > 1 else 7000,
            targets=targets,
            unicast=len(targets) > 1,
        )
        transfer.chunk_bytes = chunk_bytes
        return transfer

    def test_unicast_rotates_peers_one_copy_per_chunk(self):
        """Every rank pushes to a different peer at each step (no incast),
        and each gated chunk is one chunk_bytes copy."""
        bases = (1000, 2000, 3000, 4000)
        steps = []
        for rank in range(4):
            transfer = self._transfer(rank=rank, targets=bases, chunk_bytes=8)
            chunks = transfer._push_chunks(bases[rank] + 16, 16, 16)  # in place
            self.assertTrue(all(len(c.copies) == 1 for c in chunks))
            dsts = [c.copies[0][0] for c in chunks]
            # Two spans x three peers, this rank skipped.
            peers = [(rank + i) % 4 for i in (1, 2, 3)]
            self.assertEqual(
                dsts, [bases[p] + 16 + off for off in (0, 8) for p in peers]
            )
            steps.append([d // 1000 - 1 for d in dsts])
        for step in zip(*steps):
            self.assertEqual(sorted(step), [0, 1, 2, 3])

    def test_unicast_out_of_place_includes_this_rank_last(self):
        bases = (1000, 2000, 3000, 4000)
        transfer = self._transfer(rank=2, targets=bases, chunk_bytes=None)
        chunks = transfer._push_chunks(9000, 0, 4)
        self.assertEqual([c.copies[0][0] for c in chunks], [4000, 1000, 2000, 3000])

    def test_multicast_one_copy_per_span(self):
        transfer = self._transfer(rank=1, targets=(5000,), chunk_bytes=8)
        chunks = transfer._push_chunks(100, 16, 20)
        self.assertEqual(
            [c.copies for c in chunks],
            [((5016, 100, 8),), ((5024, 108, 8),), ((5032, 116, 4),)],
        )


# =============================================================================
# 2-4 GPUs: allgather and broadcast through the window.
# =============================================================================

_WORLD = 4 if torch.cuda.device_count() >= 4 else 2


def _init(rank, port):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=_WORLD
    )
    from sglang.srt.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )

    return PyNcclCommunicator(
        group=torch.distributed.group.WORLD, device=torch.device("cuda", rank)
    )


def _run_symmetric(rank, port, mode, skipped):
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        allocate_symmetric_window,
    )
    from sglang.srt.distributed.device_communicators.symmetric_transfer import (
        SymmetricAllGather,
        SymmetricBroadcast,
        WindowBarrier,
        window_targets,
    )

    comm = _init(rank, port)
    block = 3 * _MiB + 4096  # a ragged last chunk
    control_offset = _WORLD * block
    buffer = allocate_symmetric_window(comm, control_offset + 4096)
    targets = window_targets(comm, buffer, mode)
    ok = torch.tensor([int(targets is not None)], dtype=torch.int32)
    torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
    if not ok.item():
        if rank == 0:
            skipped.put(f"the NCCL window has no {mode} targets")
        return
    window = buffer.tensor
    barrier = WindowBarrier(
        rank=rank,
        world_size=_WORLD,
        local_base=window.data_ptr(),
        targets=targets,
        control_offset=control_offset,
    )
    barrier.init_control(window)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    gate = get_transfer_gate(Link.NVLINK)
    chain = TransferChain(rank, gate)
    after = torch.cuda.Stream()
    consumer = torch.cuda.Stream()

    def consume(transfer, done):
        """Wait on the consumer stream (the exit barrier runs there) and
        return an event marking it."""
        transfer.wait(consumer, done)
        consumed = torch.cuda.Event()
        consumed.record(consumer)
        return consumed

    def wait(event):
        assert _wait_until(event.query), f"[rank {rank}] {mode} transfer hung"

    # Broadcast rank 1's bytes into [0, block) of every rank, out of place.
    src = torch.full((block,), 7 + rank, dtype=torch.uint8, device="cuda")
    bcast = SymmetricBroadcast(barrier, chunk_bytes=_MiB, chain=chain)
    torch.cuda.synchronize()
    done = torch.cuda.Event()
    bcast.broadcast(1, src.data_ptr(), 0, block, after, done=done)
    wait(consume(bcast, done))
    assert bool((window[:block] == 8).all()), f"[rank {rank}] {mode} broadcast"

    gather = SymmetricAllGather(barrier, chunk_bytes=_MiB, chain=chain)
    own = window[rank * block : (rank + 1) * block]

    def run_gather():
        window[:control_offset].zero_()
        own.fill_(rank + 1)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        done = torch.cuda.Event()
        # In place: the own block is pushed from the slot it already sits in.
        gather.all_gather([(own.data_ptr(), rank * block, block)], after, done=done)
        return consume(gather, done)

    def check(step):
        for r in range(_WORLD):
            got = window[r * block : (r + 1) * block]
            bad = (got != r + 1).nonzero()
            assert not bad.numel(), (
                f"[rank {rank}] {mode} {step}: slot {r} has {bad.numel()} wrong "
                f"bytes from {int(bad[0])} (values {got[bad[:4, 0]].tolist()})"
            )

    # One gather left unconsumed: the next one consumes it on its `after`
    # stream first. Nothing may write the window while one is in flight (the
    # in-place push reads this rank's slot), so the second reuses the data.
    window[:control_offset].zero_()
    own.fill_(rank + 1)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    blocks = [(own.data_ptr(), rank * block, block)]
    gather.all_gather(blocks, after)
    done = torch.cuda.Event()
    gather.all_gather(blocks, after, done=done)
    wait(consume(gather, done))
    check("after an unconsumed gather")

    # Rank 0 paused (from the host) before the chunks are enqueued: its copies
    # hold, and so does every rank's read.
    if rank == 0:
        gate._word.value = 3
    consumed = run_gather()
    time.sleep(0.2)
    stalled = not consumed.query()
    torch.distributed.barrier()  # host only: no GPU work behind a parked gate
    if rank == 0:
        gate.force_open()
    wait(consumed)
    assert stalled, f"[rank {rank}] {mode} gather ran through rank 0's pause"
    check("after a pause")
    torch.distributed.barrier()


@unittest.skipIf(torch.cuda.device_count() < 2, "needs 2 CUDA devices")
class TestSymmetricTransferMultiGpu(CustomTestCase):
    def _spawn(self, mode, port):
        from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
            NCCLLibrary,
        )

        # The library PyNcclCommunicator loads, which may differ from torch's.
        if NCCLLibrary().ncclGetRawVersion() < 22902:
            self.skipTest("symmetric window pointers need NCCL >= 2.29.2")
        skipped = mp.get_context("spawn").SimpleQueue()
        # The window needs cuMem; the multicast VA needs NVLS too. Unicast
        # runs with NVLS off, as the engine sets it for that transport.
        nvls = "1" if mode == "multicast" else "0"
        with temp_set_env(NCCL_CUMEM_ENABLE="1", NCCL_NVLS_ENABLE=nvls):
            mp.spawn(
                _run_symmetric, args=(port, mode, skipped), nprocs=_WORLD, join=True
            )
        if not skipped.empty():
            self.skipTest(skipped.get())

    def test_multicast_push(self):
        self._spawn("multicast", 29832)

    def test_unicast_push(self):
        self._spawn("unicast", 29833)


if __name__ == "__main__":
    unittest.main()
