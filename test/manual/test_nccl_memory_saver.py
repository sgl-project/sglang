"""Two-node GPU regression for registered NCCL graphs across TMS pause/resume.

Run one process per node with --rank 0/1 and the same --master-addr/--master-port.
Use TMS's preload launcher and NCCL_GRAPH_REGISTER=1. NCCL_DEBUG=TRACE with
NCCL_DEBUG_SUBSYS=INIT,NET,REG provides registration and transport evidence.
--legacy-caller reproduces the unprotected call path using the same workload.
"""

import argparse
import ctypes
import datetime
import json
import os

import torch
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver as tms

from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.utils import StatelessProcessGroup


def mapping(tensor):
    driver = ctypes.CDLL("libcuda.so.1")
    query = driver.cuPointerGetAttribute
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    query.restype = ctypes.c_int
    identity = ctypes.c_uint64()
    assert query(ctypes.byref(identity), 7, tensor.data_ptr()) == 0
    return {"ptr": hex(tensor.data_ptr()), "buffer_id": identity.value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--nccl-library")
    parser.add_argument("--legacy-caller", action="store_true")
    parser.add_argument("--capture-allocation", action="store_true")
    args = parser.parse_args()
    expect_resident = (
        not args.legacy_caller and os.environ.get("NCCL_GRAPH_REGISTER") != "0"
    )
    torch.cuda.set_device(0)
    store = dist.TCPStore(
        args.master_addr,
        args.master_port,
        2,
        args.rank == 0,
        timeout=datetime.timedelta(seconds=90),
    )

    def barrier(label):
        store.set(f"barrier/{label}/{args.rank}", b"1")
        store.wait([f"barrier/{label}/{rank}" for rank in range(2)])

    def emit(event, **fields):
        print(json.dumps(dict(event=event, rank=args.rank, **fields)), flush=True)

    kwargs = {} if args.legacy_caller else {"enable_memory_saver": True}
    comm = PyNcclCommunicator(
        StatelessProcessGroup(args.rank, 2, store),
        0,
        library_path=args.nccl_library,
        **kwargs,
    )
    assert comm.available
    count = 1024**2
    ordinary_bytes = 64 * 1024**2
    stream = torch.cuda.Stream()
    seed = torch.full((count,), args.rank + 1, device="cuda", dtype=torch.float32)
    warm_out = torch.empty(2 * count, device="cuda")
    with comm.change_state(enable=True), torch.cuda.stream(stream):
        for _ in range(3):
            comm.all_gather(warm_out, seed)
    stream.synchronize()

    def allocate():
        # Small send/recv tensors share a 20 MiB allocator segment. The ordinary
        # 64 MiB allocation is separate but belongs to the same tag/graph pool.
        x = seed.clone()
        out = torch.empty(2 * count, device="cuda")
        ordinary = torch.full((ordinary_bytes,), 29, device="cuda", dtype=torch.uint8)
        return x, out, ordinary

    if not args.capture_allocation:
        with tms.region(tag="payload", enable_cpu_backup=True):
            x, out, ordinary = allocate()
    torch.cuda.synchronize()
    barrier("capture")
    graph = torch.cuda.CUDAGraph()
    ctx = (
        tms.cuda_graph(graph, tag="payload", enable_cpu_backup=True, stream=stream)
        if args.capture_allocation
        else torch.cuda.graph(graph, stream=stream)
    )
    with comm.change_state(enable=True), ctx:
        if args.capture_allocation:
            x, out, ordinary = allocate()
        comm.all_gather(out, x)
    torch.cuda.synchronize()
    initial_x, initial_out = mapping(x), mapping(out)
    emit(
        "captured",
        x=initial_x,
        out=initial_out,
        ordinary=mapping(ordinary),
        legacy_caller=args.legacy_caller,
        capture_allocation=args.capture_allocation,
        nccl_version=comm.nccl_version,
    )

    passed = True
    for cycle in range(4):
        if cycle:
            torch.cuda.synchronize()
            barrier(f"pause-{cycle}")
            before = mapping(ordinary)
            free_before = torch.cuda.mem_get_info()[0]
            tag = "payload" if cycle % 2 else None
            tms.pause(tag)
            freed = torch.cuda.mem_get_info()[0] - free_before
            assert freed >= ordinary_bytes
            tms.resume(tag)
            after = mapping(ordinary)
            assert after["ptr"] == before["ptr"]
            assert after["buffer_id"] != before["buffer_id"]
            if expect_resident:
                assert mapping(x) == initial_x
                assert mapping(out) == initial_out
            else:
                assert mapping(x)["buffer_id"] != initial_x["buffer_id"]
                assert mapping(out)["buffer_id"] != initial_out["buffer_id"]
            emit(
                "resumed",
                cycle=cycle,
                freed_bytes=freed,
                x=mapping(x),
                out=mapping(out),
                ordinary_before=before,
                ordinary_after=after,
            )
        seed.fill_(cycle * 100 + args.rank + 1)
        x.copy_(seed)
        out.fill_(-999)
        torch.cuda.synchronize()
        barrier(f"replay-{cycle}")
        graph.replay()
        torch.cuda.synchronize()
        actual = out.cpu().view(2, count)
        expected = torch.arange(1, 3, dtype=torch.float32) + cycle * 100
        mismatches = (actual != expected[:, None]).sum().item()
        ordinary_ok = torch.all(ordinary == 29).item()
        passed = passed and mismatches == 0 and ordinary_ok
        emit(
            "replay_result",
            cycle=cycle,
            mismatches=mismatches,
            ordinary_ok=ordinary_ok,
            first_values=actual[:, :4].tolist(),
        )
    barrier("finished")
    graph.reset()
    comm.nccl.ncclCommDestroy(comm.comm)
    emit("complete", passed=passed)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
