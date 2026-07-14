"""
Numeric correctness tests for the direct-oneCCL XPU communicator
(PyXcclCommunicator). Each test spawns ``world_size`` ranks, binds each to its
own Intel XPU, drives a collective through oneCCL directly, and compares the
result against the ``torch.distributed`` reference (or a closed-form expected
value).

Requires >= 2 Intel XPUs; skips otherwise. pyxccl is opt-in, so the workers set
SGLANG_ENABLE_PYXCCL=1; oneCCL needs an OFI transport, so CCL_ATL_TRANSPORT=ofi
is set too.

Follows the multi-process pattern in test/manual/cpu/test_comm.py: module-level
worker/collective functions (picklable for the "spawn" start method) plus a Pipe
that reports per-rank success back to the parent.

Usage:
python3 -m unittest test_pyxccl_collectives.TestPyXcclCollectives.test_all_reduce
"""

import multiprocessing
import os
import traceback
import unittest
from multiprocessing import Process

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase, find_available_port

register_xpu_ci(est_time=120, suite="stage-a-test-2-gpu-xpu")

WORLD_SIZE = 2
# oneCCL supports these element types via onecclDataTypeEnum.from_torch.
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _make_comm(rank: int, world_size: int):
    """Init a gloo world + a PyXcclCommunicator bound to xpu:{rank}."""
    from sglang.srt.distributed.device_communicators.pyxccl import PyXcclCommunicator

    torch.xpu.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # pyxccl broadcasts its oneCCL unique id over the CPU (gloo) group, mirroring
    # how GroupCoordinator attaches pynccl/pyxccl to cpu_group.
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    comm = PyXcclCommunicator(group=cpu_group, device=torch.device(f"xpu:{rank}"))
    assert not comm.disabled, "PyXcclCommunicator unexpectedly disabled"
    return comm, cpu_group


def all_reduce_fn(rank: int, world_size: int):
    comm, _ = _make_comm(rank, world_size)
    dev = f"xpu:{rank}"
    for dtype in DTYPES:
        # Deterministic per-rank input so the reference is exact.
        base = torch.arange(2 * 10, dtype=dtype, device=dev).reshape(2, 10)
        tensor = base + rank

        comm.all_reduce(tensor)  # oneCCL, in-place
        torch.xpu.synchronize()

        # Sum over ranks of (base + r) == world_size*base + sum(range(world_size)).
        expected = world_size * base + sum(range(world_size))
        torch.testing.assert_close(tensor, expected)
    comm.destroy()
    dist.destroy_process_group()


def broadcast_fn(rank: int, world_size: int):
    comm, _ = _make_comm(rank, world_size)
    dev = f"xpu:{rank}"
    for dtype in DTYPES:
        if rank == 0:
            tensor = torch.arange(12, dtype=dtype, device=dev)
        else:
            tensor = torch.full((12,), -1.0, dtype=dtype, device=dev)
        comm.broadcast(tensor, src=0)
        torch.xpu.synchronize()
        expected = torch.arange(12, dtype=dtype, device=dev)
        torch.testing.assert_close(tensor, expected)
    comm.destroy()
    dist.destroy_process_group()


def all_gather_fn(rank: int, world_size: int):
    comm, _ = _make_comm(rank, world_size)
    dev = f"xpu:{rank}"
    for dtype in DTYPES:
        chunk = 4
        inp = torch.full((chunk,), float(rank + 1), dtype=dtype, device=dev)
        out = torch.empty((chunk * world_size,), dtype=dtype, device=dev)
        comm.all_gather(out, inp)
        torch.xpu.synchronize()
        expected = torch.cat(
            [
                torch.full((chunk,), float(r + 1), dtype=dtype, device=dev)
                for r in range(world_size)
            ]
        )
        torch.testing.assert_close(out, expected)
    comm.destroy()
    dist.destroy_process_group()


def reduce_scatter_fn(rank: int, world_size: int):
    comm, _ = _make_comm(rank, world_size)
    dev = f"xpu:{rank}"
    for dtype in DTYPES:
        per_rank = 4
        # Full input on every rank: value (r+1) contributed by rank r.
        full = torch.full(
            (per_rank * world_size,), float(rank + 1), dtype=dtype, device=dev
        )
        out = torch.empty((per_rank,), dtype=dtype, device=dev)
        comm.reduce_scatter(out, full)
        torch.xpu.synchronize()
        # Each output element is the sum over ranks of (r+1).
        expected = torch.full(
            (per_rank,),
            float(sum(r + 1 for r in range(world_size))),
            dtype=dtype,
            device=dev,
        )
        torch.testing.assert_close(out, expected)
    comm.destroy()
    dist.destroy_process_group()


def sglang_tp_collectives_fn(rank: int, world_size: int):
    """End-to-end: drive SGLang's public TP-collective API through the real
    initialize_model_parallel() -> GroupCoordinator -> pyxccl_comm path on XPU.

    This exercises the wiring added to GroupCoordinator (the pyxccl_comm branches
    in _all_reduce_in_place / _all_gather_into_tensor / broadcast), not just the
    raw communicator.
    """
    from sglang.srt.distributed.communication_op import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from sglang.srt.distributed.parallel_state import (
        get_default_distributed_backend,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.xpu.set_device(rank)
    backend = get_default_distributed_backend("xpu")  # -> "xccl"
    init_distributed_environment(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    tp = get_tp_group()
    # The whole point of this port: on XPU tp>1 the TP group drives pyxccl.
    assert (
        tp.pyxccl_comm is not None and not tp.pyxccl_comm.disabled
    ), "TP group is not using the pyxccl (direct oneCCL) path"

    dev = f"xpu:{rank}"

    # all_reduce via the public API -> get_tp_group().all_reduce -> inplace_all_reduce
    # custom op -> _all_reduce_in_place -> pyxccl_comm.all_reduce.
    x = torch.full((2, 8), float(rank + 1), device=dev, dtype=torch.float32)
    out = tensor_model_parallel_all_reduce(x)
    torch.xpu.synchronize()
    expected_sum = float(world_size * (world_size + 1) // 2)
    torch.testing.assert_close(out, torch.full_like(out, expected_sum))

    # all_gather via the public API -> get_tp_group().all_gather ->
    # _all_gather_into_tensor -> pyxccl_comm.all_gather. dim=0 concatenation.
    g = torch.full((3, 4), float(rank + 1), device=dev, dtype=torch.float32)
    gathered = tensor_model_parallel_all_gather(g, dim=0)
    torch.xpu.synchronize()
    expected_gather = torch.cat(
        [
            torch.full((3, 4), float(r + 1), device=dev, dtype=torch.float32)
            for r in range(world_size)
        ],
        dim=0,
    )
    torch.testing.assert_close(gathered, expected_gather)

    dist.barrier()
    tp.destroy()
    dist.destroy_process_group()


def _run(rank, world_size, master_port, output_writer, fn):
    try:
        os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
        # These tests exercise the direct-oneCCL path, which is opt-in.
        os.environ["SGLANG_ENABLE_PYXCCL"] = "1"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        fn(rank, world_size)
        ok = True
    except Exception as e:  # surface the traceback to the parent
        print(f"subprocess[rank={rank}] error: {e}", flush=True)
        traceback.print_exc()
        ok = False
    output_writer.send(ok)
    output_writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


class TestPyXcclCollectives(CustomTestCase):
    def _spawn_and_check(self, fn, world_size=WORLD_SIZE):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            self.skipTest("XPU not available")
        if torch.xpu.device_count() < world_size:
            self.skipTest(f"need >= {world_size} XPUs")

        mp.set_start_method("spawn", force=True)
        master_port = find_available_port(23456)
        output_reader, output_writer = multiprocessing.Pipe(duplex=False)

        processes = []
        for rank in range(world_size):
            p = Process(
                target=_run,
                kwargs=dict(
                    rank=rank,
                    world_size=world_size,
                    master_port=master_port,
                    output_writer=output_writer,
                    fn=fn,
                ),
            )
            p.start()
            processes.append(p)

        for _ in range(world_size):
            self.assertTrue(
                output_reader.recv(), "Subprocess failed. Check logs above."
            )
        for p in processes:
            p.join()

    def test_all_reduce(self):
        self._spawn_and_check(all_reduce_fn)

    def test_broadcast(self):
        self._spawn_and_check(broadcast_fn)

    def test_all_gather(self):
        self._spawn_and_check(all_gather_fn)

    def test_reduce_scatter(self):
        self._spawn_and_check(reduce_scatter_fn)

    def test_sglang_tp_collectives(self):
        # End-to-end: SGLang's public TP-collective API through the real
        # initialize_model_parallel -> GroupCoordinator -> pyxccl path.
        self._spawn_and_check(sglang_tp_collectives_fn)


if __name__ == "__main__":
    unittest.main()
