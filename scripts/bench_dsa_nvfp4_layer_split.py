"""Measure NVFP4 pool writes, indexer transfer, and prefix materialization."""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp


@torch.inference_mode()
def worker(rank, world, output, repeats):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29830 + world)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.runtime_context import get_parallel

    torch.manual_seed(42)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{29830 + world}",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world, attention_context_model_parallel_size=world
    )
    device = f"cuda:{rank}"
    records = []
    layers = [SimpleNamespace(layer_id=i) for i in range(78)]
    loc = torch.arange(4096, device=device)
    nope = torch.randn(4096, 1, 512, device=device).bfloat16()
    rope = torch.randn(4096, 1, 64, device=device).bfloat16()
    index = torch.ones(4096, 128, device=device).to(torch.float8_e4m3fn)
    scale = torch.ones(4096, 1, device=device)

    for capacity in (32768, 131072):
        for mode in ("baseline", "layer_split_sync", "layer_split_prefetch"):
            kwargs = dict(
                size=capacity,
                page_size=64,
                dtype=torch.float4_e2m1fn_x2,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=78,
                device=device,
                index_head_dim=128,
                enable_memory_saver=False,
                kv_cache_dim=416,
            )
            if mode == "baseline":
                pool = DSATokenToKVPool(**kwargs)
            else:
                pool = LayerSplitDSATokenToKVPool(
                    **kwargs, layer_shard_rank=rank, layer_shard_size=world
                )
            for layer in layers:
                pool.set_mla_kv_global_scale(layer.layer_id, 0.5 + layer.layer_id / 128)
            sizes = [None] * world
            torch.distributed.all_gather_object(
                sizes,
                int(pool.get_kv_size_bytes()),
                group=get_parallel().attn_cp_group.cpu_group,
            )

            def step(current_pool):
                for layer in layers:
                    if mode == "layer_split_prefetch":
                        current_pool.prefetch_kv_buffer(layer.layer_id)
                    current_pool.set_mla_kv_buffer(layer, loc, nope, rope)
                    current_pool.index_key_cache.store_quantized(
                        layer.layer_id, loc, index, scale
                    )
                    current_pool.index_key_cache.get_buffer(layer.layer_id)
                    current_pool.get_mla_kv_buffer(layer, loc)

            # These are kernel warmups. No model or dataset is loaded.
            step(pool)
            step(pool)
            samples = []
            for _ in range(3):
                torch.cuda.synchronize()
                torch.distributed.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(repeats):
                    step(pool)
                end.record()
                end.synchronize()
                times = torch.tensor(
                    [
                        start.elapsed_time(end) / repeats,
                        1000 * (time.perf_counter() - wall_start) / repeats,
                    ],
                    device=device,
                )
                torch.distributed.all_reduce(times, op=torch.distributed.ReduceOp.MAX)
                samples.append(times.tolist())
            records.append(
                {
                    "mode": mode,
                    "cp_size": world,
                    "capacity_tokens": capacity,
                    "layers": 78,
                    "rows_written_and_materialized": 4096,
                    "pool_bytes_per_rank": sizes,
                    "cuda_ms_per_78_layers": statistics.median(s[0] for s in samples),
                    "wall_ms_per_78_layers": statistics.median(s[1] for s in samples),
                    "samples_cuda_wall_ms": samples,
                }
            )
            del step, pool
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.barrier()
    if rank == 0:
        Path(output).write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(records, indent=2), flush=True)
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if torch.cuda.device_count() < 4:
        parser.error("Four SM100-family GPUs are required")
    if any(torch.cuda.get_device_capability(i)[0] != 10 for i in range(4)):
        parser.error("NVFP4 DSA requires SM100-family GPUs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for world in (2, 4):
        output = args.output_dir / f"cp{world}.json"
        if output.exists():
            parser.error(f"Existing results are preserved: {output}")
        mp.spawn(
            worker, args=(world, str(output), args.repeats), nprocs=world, join=True
        )


if __name__ == "__main__":
    main()
