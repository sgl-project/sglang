"""
Microbenchmark: greedy AG+argmax shortcut vs. baseline (full-vocab AG + full-vocab argmax).

Measures BOTH regimes:
  * eager  — standalone kernel launches (upper-bound on launch overhead)
  * graph  — captured HIP/CUDA graph + replay (matches production decode regime,
             where every kernel launch in the decode step is absorbed into the
             model graph)

Why graph mode matters:
  The shortcut issues 4–5 small kernels (max, stack, tiny-AG, transpose, argmax).
  In eager, each launch costs ~20 µs on ROCm, giving the shortcut a ~90 µs floor
  that hides its asymptotic win. Under graph replay those launches are free, so
  the comparison reflects what production actually sees.

Baseline:
    full = tensor_model_parallel_all_gather(local_logits, dim=-1)   # [M, V]
    ids  = torch.argmax(full, dim=-1)                                # [M]

Shortcut:
    _fused_greedy_argmax_across_tp(local_logits) -> [M]              # local argmax
                                                                       + small AG(val,idx)
                                                                       + global pick

Launch (MI355X × 8, Qwen3.5 FP8 LM-head shape included):
    torchrun --nproc_per_node=8 --master_port=29510 \
        benchmark/kernels/all_gather/benchmark_ag_argmax_shortcut.py
"""

import os
import sys

import torch
import torch.distributed as dist

# Dense M sweep to locate the eager→graph crossover precisely.
M_SWEEP = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
# V_local for Qwen3.5 FP8 LM-head on TP=8 (vocab 248320 / 8).
V_LOCAL = 31040
DTYPE = torch.bfloat16


def time_eager(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device=device)
    return start.elapsed_time(end) / iters  # ms/call


def time_graph(fn, iters, warmup, device):
    """Capture `fn` into a HIP graph under SGLang's graph-capture context,
    which routes aiter AG through the graph-safe `all_gather_reg` path and
    suppresses the RCCL watchdog collision with HIP stream capture.
    """
    from sglang.srt.distributed import graph_capture

    with graph_capture() as gc:
        stream = gc.stream
        # Warm up on the capture stream so NCCL/aiter metadata is initialized.
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.synchronize(device=device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            fn()

    # Exit graph_capture before replaying so the watchdog can run normally.
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize(device=device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize(device=device)
    return start.elapsed_time(end) / iters  # ms/call


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size
    )

    import sglang.srt.distributed.parallel_state as ps

    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    ps.initialize_model_parallel(tensor_model_parallel_size=world_size)

    from sglang.srt.distributed import tensor_model_parallel_all_gather
    from sglang.srt.layers.logits_processor import _fused_greedy_argmax_across_tp

    rows = []

    for M in M_SWEEP:
        torch.manual_seed(1234 + rank * 37 + M)
        # Pre-allocated input — in production this is the LM-head matmul output.
        local = torch.randn(M, V_LOCAL, dtype=DTYPE, device=device)

        # --- Correctness (eager reference) ---
        ref_full = tensor_model_parallel_all_gather(local, dim=-1)
        ref_ids = torch.argmax(ref_full, dim=-1).to(torch.int64)
        got_ids = _fused_greedy_argmax_across_tp(local).to(torch.int64)
        assert torch.equal(ref_ids, got_ids), f"eager mismatch at M={M}"

        # --- Closures under test ---
        # NOTE: these must not keep Python-side references that the graph capture
        # can't reuse. Allocations inside the closure are captured into the
        # graph's private memory pool on first capture.
        def baseline():
            full = tensor_model_parallel_all_gather(local, dim=-1)
            _ = torch.argmax(full, dim=-1)

        def shortcut():
            _ = _fused_greedy_argmax_across_tp(local)

        # --- Eager timing ---
        base_eager = time_eager(baseline, iters=200, warmup=20, device=device)
        short_eager = time_eager(shortcut, iters=200, warmup=20, device=device)

        # --- Graph timing (the regime that matches production) ---
        try:
            base_graph = time_graph(baseline, iters=200, warmup=20, device=device)
        except Exception as exc:
            base_graph = float("nan")
            if rank == 0:
                print(f"[M={M}] baseline graph capture failed: {exc}", flush=True)
        try:
            short_graph = time_graph(shortcut, iters=200, warmup=20, device=device)
        except Exception as exc:
            short_graph = float("nan")
            if rank == 0:
                print(f"[M={M}] shortcut graph capture failed: {exc}", flush=True)

        # Rank-avg to smooth per-rank jitter.
        buf = torch.tensor(
            [base_eager, short_eager, base_graph, short_graph], device=device
        )
        # all_reduce(AVG) doesn't handle NaN well; mask them first.
        nan_mask = torch.isnan(buf)
        buf = torch.where(nan_mask, torch.zeros_like(buf), buf)
        dist.all_reduce(buf, op=dist.ReduceOp.AVG)
        be, se, bg, sg = buf.tolist()

        rows.append((M, be, se, bg, sg))

    if rank == 0:
        hdr = (
            f"{'M':>4}  "
            f"{'base_eager (ms)':>16}  {'short_eager (ms)':>17}  {'spdup_eager':>11}  "
            f"{'base_graph (ms)':>16}  {'short_graph (ms)':>17}  {'spdup_graph':>11}"
        )
        print(
            f"\nShape: V_local={V_LOCAL}  V_total={V_LOCAL * world_size}  "
            f"dtype={DTYPE}  TP={world_size}\n"
        )
        print(hdr)
        print("-" * len(hdr))
        cutoff_eager = None
        cutoff_graph = None
        for M, be, se, bg, sg in rows:
            sp_e = be / se if se > 0 else float("inf")
            sp_g = bg / sg if sg > 0 else float("inf")
            if cutoff_eager is None and sp_e >= 1.0:
                cutoff_eager = M
            if cutoff_graph is None and sp_g >= 1.0:
                cutoff_graph = M
            print(
                f"{M:>4}  "
                f"{be:>16.4f}  {se:>17.4f}  {sp_e:>10.2f}x  "
                f"{bg:>16.4f}  {sg:>17.4f}  {sp_g:>10.2f}x"
            )
        print()
        print(f"Crossover (eager, baseline/shortcut ≥ 1.0×): M ≈ {cutoff_eager}")
        print(f"Crossover (graph, baseline/shortcut ≥ 1.0×): M ≈ {cutoff_graph}")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    main()
