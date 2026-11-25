"""Multi-rank check of the DeepSeek-V4.1 engram peer-shard lookup.

Each rank owns a real-size row shard of one engram table (random fp8 rows and
e8m0 scales), looks up random ids for its own tokens through the peer mappings,
and compares bitwise with the DP-sharded path it replaces (all-gather ids, zero-
filled owned rows, reduce-scatter). Launch one process per GPU, e.g. under srun:

    MASTER_ADDR=<node0> MASTER_PORT=29511 python test_engram_peer_shards.py
"""

import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.kernels.ops.quantization.fp4_fake_quant import FP8_BLOCK_SIZE
from sglang.srt.layers.engram import EngramLayout, _PeerShards, shard_range

DIM = 256
CONFIG = SimpleNamespace(
    engram_layer_ids=[1, 14],
    engram_num_embeddings=[384006168, 384016682],
    engram_max_ngram_size=4,
    engram_n_heads=8,
    engram_vocab_size=16000000,
    engram_head_dim=DIM,
)


def fill_shard(weight, scale, seed):
    g = torch.Generator(device=weight.device).manual_seed(seed)
    chunk = 1 << 22
    w8 = weight.view(torch.uint8)
    for i in range(0, w8.shape[0], chunk):
        part = torch.randint(
            0,
            256,
            w8[i : i + chunk].shape,
            generator=g,
            device=w8.device,
            dtype=torch.uint8,
        )
        # 0x7f / 0xff are e4m3 NaN; keep every value finite.
        part = torch.where((part & 0x7F) == 0x7F, part - 1, part)
        w8[i : i + chunk] = part
    s8 = scale.view(torch.uint8)
    s8.copy_(
        torch.randint(
            100, 150, s8.shape, generator=g, device=s8.device, dtype=torch.uint8
        )
    )


def random_ids(bounds, tokens, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    lo = torch.tensor(bounds[:-1], device=device, dtype=torch.int64)
    hi = torch.tensor(bounds[1:], device=device, dtype=torch.int64)
    u = torch.rand(tokens, len(lo), generator=g, device=device, dtype=torch.float64)
    ids = lo + (u * (hi - lo).double()).long()
    return torch.minimum(ids, hi - 1)


def reference(ids, weight, scale, row_start, rows, world):
    """The current _dp_sharded_lookup mechanics with equal per-rank token counts."""
    t = ids.shape[0]
    gathered = torch.empty(world * t, ids.shape[1], dtype=ids.dtype, device=ids.device)
    dist.all_gather_into_tensor(gathered, ids.contiguous())
    values = torch.empty(
        world * t * ids.shape[1], DIM, dtype=torch.bfloat16, device=ids.device
    )
    engram_gather(
        weight.data_ptr(),
        scale.data_ptr(),
        gathered.reshape(-1),
        values,
        DIM,
        FP8_BLOCK_SIZE,
        row_lo=row_start,
        row_hi=row_start + rows,
    )
    values = values.view(world * t, -1)
    local = torch.empty(t, values.shape[1], dtype=values.dtype, device=values.device)
    dist.reduce_scatter_tensor(local, values)
    return local.view(t, ids.shape[1], DIM)


def bench(fn, calls=20, replays=20):
    """GPU time per call (us) of ``fn`` captured ``calls`` times in one CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            for _ in range(calls):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (calls * replays) * 1000.0


def main():
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, device_id=torch.device("cuda", local_rank)
    )
    cpu_group = dist.new_group(backend="gloo")
    group = SimpleNamespace(cpu_group=cpu_group, rank_in_group=rank, world_size=world)
    layout = EngramLayout.from_config(CONFIG)
    n = layout.num_embeddings[0]
    col_bounds = layout.column_bounds(0)
    token_list = [
        int(x) for x in os.environ.get("TOKENS", "1 64 256 1024 16384").split()
    ]
    split = os.environ.get("SPLIT", "auto")  # auto (column if it divides) | rows
    shard_bounds = col_bounds if split == "auto" else (0, n)
    row_start, row_end = shard_range(n, shard_bounds, rank, world)
    rows = row_end - row_start
    t0 = time.perf_counter()
    peer = _PeerShards(rows, DIM, group)
    weight, scale = peer.views()
    fill_shard(weight, scale, seed=1000 + rank)
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0
    t0 = time.perf_counter()
    peer.connect(row_start, "test")
    t_connect = time.perf_counter() - t0
    if rank == 0:
        print(
            f"world={world} split={split} rows/rank~{rows} shard={peer.size / 2**30:.2f} GiB "
            f"alloc+fill={t_alloc:.1f}s connect={t_connect:.1f}s",
            flush=True,
        )
    ok = True
    for tokens in token_list:
        ids = random_ids(col_bounds, tokens, seed=7 * tokens + rank, device="cuda")
        # The model passes a strided [T, 24] slice of [T, n_layers, 24].
        both = torch.stack([ids, ids], dim=1)
        got = peer.lookup(both[:, 0])
        ref = reference(ids, weight, scale, row_start, rows, world)
        g16, r16 = got.view(torch.int16), ref.view(torch.int16)
        bit_diff = int((g16 != r16).sum())
        val_diff = int((got != ref).sum())
        zero_sign = int(((g16 != r16) & (got == 0) & (ref == 0)).sum())
        # The same lookup inside a CUDA graph.
        static_ids = both[:, 0].clone()
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            peer.lookup(static_ids)
            with torch.cuda.graph(graph, stream=s):
                graph_out = peer.lookup(static_ids)
        torch.cuda.current_stream().wait_stream(s)
        graph.replay()
        torch.cuda.synchronize()
        graph_diff = int((graph_out.view(torch.int16) != g16).sum())
        t_peer = bench(lambda: peer.lookup(both[:, 0]))
        t_ref = bench(lambda: reference(ids, weight, scale, row_start, rows, world))
        res = torch.tensor([bit_diff, val_diff, zero_sign, graph_diff], device="cuda")
        dist.all_reduce(res)
        times = torch.tensor([t_peer, t_ref], device="cuda")
        tmax = times.clone()
        dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"tokens/rank={tokens}: bit_mismatch={int(res[0])} value_mismatch={int(res[1])} "
                f"zero_sign_only={int(res[2])} graph_mismatch={int(res[3])} "
                f"peer={t_peer:.1f}us (max {tmax[0]:.1f}) current={t_ref:.1f}us (max {tmax[1]:.1f})",
                flush=True,
            )
        ok &= int(res[1]) == 0 and int(res[3]) == 0
    dist.barrier()
    if rank == 0:
        print("PASS" if ok else "FAIL", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
