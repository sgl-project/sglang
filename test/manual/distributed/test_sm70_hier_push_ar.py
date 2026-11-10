"""sm70_hier_push_ar must equal the quad-then-pair custom-AR chain it replaces,
bitwise, eager and under CUDA-graph replay, and leave every receive slot empty.

Needs the 8xV100 hybrid mesh. Run: python test/manual/distributed/test_sm70_hier_push_ar.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
SIZES = [8, 16, 24, 512, 4096, 4104, 8192, 16384, 40960, 65536, 131072]
ITERS = 40


def _special_fp16(n: int, gen: torch.Generator) -> torch.Tensor:
    """Random fp16 with zeros of both signs, subnormals, infs, NaN payloads, the
    all-ones empty marker and values whose sums overflow."""
    x = torch.randn(n, generator=gen, device="cuda", dtype=torch.float32)
    scale = torch.tensor([1e-6, 1e-2, 1.0, 1e2, 3e4], device="cuda")
    pick = torch.randint(0, 5, (n,), generator=gen, device="cuda")
    x = (x * scale[pick]).half()
    bits = x.view(torch.int16)
    special = torch.tensor(
        [0x0000, -0x8000, 0x0001, -0x7FFF, 0x7C00, -0x0400, 0x7E00, 0x7C01, -1, -2],
        dtype=torch.int16,
        device="cuda",
    )
    mask = torch.rand(n, generator=gen, device="cuda") < 0.02
    idx = torch.randint(0, len(special), (n,), generator=gen, device="cuda")
    bits[mask] = special[idx][mask]
    # Whole 32-bit words equal to the empty marker.
    pairs = bits.view(-1, 2)
    word_mask = torch.rand(pairs.shape[0], generator=gen, device="cuda") < 0.01
    pairs[word_mask] = -1
    return x


def _reference(x: torch.Tensor, quads, pairs, rank: int) -> torch.Tensor:
    """fp32 sums over the group in rank order, one fp16 rounding per step."""
    gathered = [torch.empty_like(x) for _ in range(WORLD)]
    dist.all_gather(gathered, x)
    quad_sum = {}
    for q in quads:
        acc = gathered[q[0]].float()
        for r in q[1:]:
            acc = acc + gathered[r].float()
        for r in q:
            quad_sum[r] = acc.half()
    pair = next(p for p in pairs if rank in p)
    return (quad_sum[pair[0]].float() + quad_sum[pair[1]].float()).half()


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _all_empty(hier) -> bool:
    from sglang.kernels.ops.communication import sm70_hier_push_ar as push
    from sglang.srt.distributed.device_communicators.dsv41_hier_ar import (
        _PUSH_SLOT_BYTES,
        _DeviceBuffer,
    )

    p = hier._push
    ok = True
    for ptrs, rank, nbytes in (
        (p.quad_ptrs, p.quad_rank, push.quad_workspace_bytes(_PUSH_SLOT_BYTES)),
        (p.pair_ptrs, p.pair_rank, push.pair_workspace_bytes(_PUSH_SLOT_BYTES)),
    ):
        buf = torch.as_tensor(_DeviceBuffer(ptrs[rank], nbytes), device="cuda")
        ok &= bool((buf == push.EMPTY_BYTE).all())
    return ok


def _check(rank: int, name: str, ok: bool, failures: list) -> None:
    flag = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flag)
    if flag.item() and rank == 0:
        failures.append(name)
        print(f"FAIL {name}", flush=True)


def _worker(rank: int, port: int, result) -> None:
    from sglang.srt.environ import envs

    from sglang.srt.distributed import init_distributed_environment

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=WORLD,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=rank,
    )
    from sglang.srt.distributed.device_communicators.dsv41_hier_ar import (
        Dsv41HierAllReduce,
        partition_quads_and_pairs,
    )

    with envs.SGLANG_DSV41_HIER_AR_CA.override(
        True
    ), envs.SGLANG_DSV41_HIER_AR_PUSH.override(True):
        hier = Dsv41HierAllReduce(list(range(WORLD)), rank, torch.device("cuda", rank))
    assert hier._push is not None and hier._ca_chain is not None
    quads, pairs = partition_quads_and_pairs(list(range(WORLD)))
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    failures = []

    # Eager: in place, out of place, and against the chain and the reference.
    for n in SIZES:
        ok = True
        for _ in range(ITERS):
            x = _special_fp16(n, gen)
            inplace = x.clone()
            hier._push.all_reduce(inplace, inplace)
            out = torch.empty_like(x)
            hier._push.all_reduce(x, out)
            chain = x.clone()
            assert hier._reduce_ca_chain(chain)
            ref = _reference(x, quads, pairs, rank)
            ok &= _same(inplace, chain) and _same(out, chain) and _same(chain, ref)
        _check(rank, f"eager n={n}", ok, failures)

    # Back to back without host syncs, sizes shared across ranks.
    size_gen = torch.Generator().manual_seed(99)
    xs, outs = [], []
    for _ in range(500):
        n = SIZES[torch.randint(0, len(SIZES), (1,), generator=size_gen).item()]
        x = _special_fp16(n, gen)
        xs.append(x)
        outs.append(torch.empty_like(x))
        hier._push.all_reduce(x, outs[-1])
    ok = True
    for x, out in zip(xs, outs):
        chain = x.clone()
        hier._reduce_ca_chain(chain)
        ok &= _same(out, chain)
    _check(rank, "back-to-back 500", ok, failures)

    # CUDA graph: several reductions with work in between, replayed on new data.
    graph_sizes = [4096, 8192, 65536, 4096]
    static_in = [torch.zeros(n, device="cuda", dtype=torch.float16) for n in graph_sizes]
    static_out = [torch.empty_like(t) for t in static_in]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for i, (a, b) in enumerate(zip(static_in, static_out)):
                t = a.clone() if i % 2 else a
                hier._push.all_reduce(t, t)
                b.copy_(t)
    torch.cuda.current_stream().wait_stream(stream)
    ok = True
    for _ in range(200):
        data = [_special_fp16(n, gen) for n in graph_sizes]
        for a, d in zip(static_in, data):
            a.copy_(d)
        graph.replay()
        for i, (d, b) in enumerate(zip(data, static_out)):
            chain = d.clone()
            hier._reduce_ca_chain(chain)
            ok &= _same(b, chain)
            if i % 2 == 0:
                # Graph reduced static_in[i] in place.
                ok &= _same(static_in[i], chain)
    _check(rank, "graph replay 200", ok, failures)

    torch.cuda.synchronize()
    dist.barrier()
    _check(rank, "receive slots empty", _all_empty(hier), failures)

    # Latency: 100 reductions of 8 KiB (one decode token) per graph.
    x = torch.randn(4096, device="cuda", dtype=torch.float16)
    timings = {}
    for name, fn in (
        ("push", lambda: hier._push.all_reduce(x, x)),
        ("chain", lambda: hier._reduce_ca_chain(x)),
    ):
        g = torch.cuda.CUDAGraph()
        with hier.graph_capture_contexts():
            with torch.cuda.graph(g):
                for _ in range(100):
                    fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - t0) / 2000 * 1e6
        dist.barrier()
    if rank == 0:
        print(
            f"8 KiB AR in graph: push {timings['push']:.2f} us, "
            f"chain {timings['chain']:.2f} us",
            flush=True,
        )
        result.value = 0 if not failures else 1
    dist.barrier()


def main() -> int:
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")
    ctx = mp.get_context("spawn")
    result = ctx.Value("i", 2)
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=_worker, args=(r, port, result)) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode for p in procs) or result.value:
        print("FAILED")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
