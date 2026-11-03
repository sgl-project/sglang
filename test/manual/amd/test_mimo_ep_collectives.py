"""Four-GPU smoke for AITER collectives used by MiMo MORI EP.

Run with matching editable SGLang/AITER installs:
    python test/manual/amd/test_mimo_ep_collectives.py

Uses a 64 MiB IPC input pool per GPU and checks BF16 eager/capture/replay,
all reduce-scatter dimensions, and empty collectives. It does not load a model.
"""

import datetime
import os
import socket

os.environ["SGLANG_DP_USE_REDUCE_SCATTER"] = "1"
os.environ["SGLANG_AITER_CAPTURE_COPY_IN"] = "1"
os.environ.pop("ROCM_QUICK_REDUCE_QUANTIZATION", None)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method="tcp://127.0.0.1:" + str(rendezvous),
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=90),
    )
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from sglang.srt.distributed.parallel_state import GroupCoordinator

    ca = CustomAllreduce(dist.group.WORLD, rank, max_size=64 * 1024 * 1024)
    assert not ca.disabled
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.ca_comm = ca
    group.world_size = 4
    source = torch.full((32, 16), rank + 1, device="cuda", dtype=torch.bfloat16)
    reduced = torch.empty((8, 16), device="cuda", dtype=source.dtype)
    gathered = torch.empty_like(source)
    local = source[:8].clone()
    # Eager, then graph replay with changed values: verifies copy-in IPC reads.
    assert group._maybe_aiter_reduce_scatter(reduced, source)
    torch.testing.assert_close(reduced, torch.full_like(reduced, 10))
    group._all_gather_into_tensor(gathered, local)
    expected = (
        torch.arange(1, 5, device="cuda", dtype=source.dtype)
        .repeat_interleave(8)[:, None]
        .expand_as(gathered)
    )
    torch.testing.assert_close(gathered, expected)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture():
        with torch.cuda.stream(stream):
            group._maybe_aiter_reduce_scatter(reduced, source)
            group._all_gather_into_tensor(gathered, local)
        stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            group._maybe_aiter_reduce_scatter(reduced, source)
            group._all_gather_into_tensor(gathered, local)
    for factor in (1, 2, 3):
        source.fill_((rank + 1) * factor)
        local.fill_((rank + 1) * factor)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(reduced, torch.full_like(reduced, 10 * factor))
        torch.testing.assert_close(gathered, expected * factor)
    # Exercise the newer first/last/middle dimension encoding as well.
    for shape, dim in [((32, 16), 0), ((8, 16), 1), ((2, 8, 16), 1)]:
        inp = torch.full(shape, rank + 1, device="cuda", dtype=source.dtype)
        out_shape = list(shape)
        out_shape[dim] //= 4
        out = torch.empty(out_shape, device="cuda", dtype=source.dtype)
        ca.reduce_scatter(inp, out, dim=dim)
        torch.testing.assert_close(out, torch.full_like(out, 10))
        empty_shape = list(shape)
        empty_shape[0] = 0
        zero = torch.empty(empty_shape, device="cuda", dtype=source.dtype)
        ca.reduce_scatter(zero, zero, dim=dim)
    # Call AITER directly to exercise the C++ zero-size entry guards.
    empty = source[:0]
    ca.all_reduce(empty)
    ca.reduce_scatter(empty, empty)
    ca.all_gather_unreg(empty, out=empty, dim=0)
    ca.fused_ar_rms(
        empty, empty, w=torch.ones(16, device="cuda", dtype=source.dtype), eps=1e-6
    )
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(
            "PASS: 4-GPU BF16 RS/AG eager + 3 graph replays with changing inputs; first/last/middle RS; empty AR/RS/AG/RMSNorm",
            flush=True,
        )
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=4, join=True)
