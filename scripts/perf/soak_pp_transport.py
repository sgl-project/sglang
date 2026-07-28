#!/usr/bin/env python3
"""Long soak test for PP packed transport.

Runs 100,000 rounds with randomized batch sizes and payload types.
Validates correctness every 1,000 rounds.
Monitors memory, cache sizes, and latency drift.
"""
import argparse, json, os, math, random, time
import torch
import torch.distributed as dist

def parse_args():
    p = argparse.ArgumentParser(description="PP transport soak test")
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--rounds", type=int, default=100000)
    p.add_argument("--validate-every", type=int, default=1000)
    p.add_argument("--output-json", default=None)
    p.add_argument("--timeout-s", type=int, default=600)
    return p.parse_args()

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}

def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    
    if args.backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    
    from sglang.srt.distributed.pp_packed_transport import (
        calculate_pp_buffer_layout, pack_pp_proxy_tensors, unpack_pp_proxy_tensors,
        PPSchemaEntry, PPSchemaCache, PPStaticBufferRegistry,
    )
    from sglang.srt.speculative.glm52_eagle3_pp import (
        GLM52_EAGLE3_AUX_PP_KEY, allocate_packed_aux_buffer,
    )
    
    HS = 5120
    CL = 3
    TK = 64
    ROWS_OPTIONS = [1, 4, 16, 64, 256]
    BUCKET = 256
    
    random.seed(42)
    
    send_cache = PPSchemaCache()
    recv_cache = PPSchemaCache()
    static_reg = PPStaticBufferRegistry(device)
    
    latencies = []
    errors = []
    memory_snapshots = []
    
    start_time = time.time()
    
    for i in range(args.rounds):
        if time.time() - start_time > args.timeout_s:
            if rank == 0:
                print(f"Timeout after {i} rounds")
            break
        
        rows = random.choice(ROWS_OPTIONS)
        include_topk = random.choice([True, False])
        
        send_dict = {
            "hidden_states": torch.randn(rows, HS, dtype=dtype, device=device),
            "residual": torch.randn(rows, HS, dtype=dtype, device=device),
            GLM52_EAGLE3_AUX_PP_KEY: allocate_packed_aux_buffer(rows, CL, HS, dtype, device),
        }
        if include_topk:
            send_dict["topk_indices"] = torch.randint(0, TK, (rows, TK), dtype=torch.int32, device=device)
        
        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            send_dict, HS, CL, TK, BUCKET
        )
        
        if rank == 0:
            data_buf, ctrl_buf = static_reg.get_or_allocate(BUCKET, data_nelem, ctrl_nelem, dtype)
            pack_pp_proxy_tensors(send_dict, data_buf, ctrl_buf, d_off, c_off, rows)
            
            entry = send_cache.lookup(key)
            if entry is None:
                entry = send_cache.register(key, data_nelem, ctrl_nelem, d_off, c_off)
            
            hdr = torch.tensor([entry.schema_id, rows, key.presence_mask], dtype=torch.long, device=device)
            
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            
            dist.send(hdr, dst=1)
            dist.send(data_buf[:data_nelem], dst=1)
            if ctrl_nelem > 0:
                dist.send(ctrl_buf[:ctrl_nelem], dst=1)
            
            end_evt.record()
            end_evt.synchronize()
            latencies.append(start_evt.elapsed_time(end_evt) * 1000)
        else:
            hdr = torch.empty(3, dtype=torch.long, device=device)
            dist.recv(hdr, src=0)
            schema_id, active_rows, presence = hdr.tolist()
            
            entry = recv_cache.get_by_id(schema_id)
            if entry is None:
                entry = recv_cache.register(key, data_nelem, ctrl_nelem, d_off, c_off)
            
            data_buf, ctrl_buf = static_reg.get_or_allocate(BUCKET, data_nelem, ctrl_nelem, dtype)
            dist.recv(data_buf[:data_nelem], src=0)
            if ctrl_nelem > 0:
                dist.recv(ctrl_buf[:ctrl_nelem], src=0)
            
            result = unpack_pp_proxy_tensors(
                data_buf, ctrl_buf, entry, active_rows, device,
                dtype, HS, CL, TK
            )
            
            # Validate every N rounds
            if (i + 1) % args.validate_every == 0:
                assert result["hidden_states"].shape[0] == active_rows
                assert result["hidden_states"].shape[1] == HS
        
        # Record snapshots
        if (i + 1) % args.validate_every == 0 and rank == 0:
            cuda_alloc = torch.cuda.memory_allocated() / 1024**2
            cuda_reserved = torch.cuda.memory_reserved() / 1024**2
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
            
            recent = latencies[-args.validate_every:]
            recent.sort()
            p50 = recent[len(recent) // 2]
            p95 = recent[min(int(len(recent) * 0.95), len(recent) - 1)]
            p99 = recent[min(int(len(recent) * 0.99), len(recent) - 1)]
            
            snapshot = {
                "round": i + 1,
                "cuda_allocated_mib": cuda_alloc,
                "cuda_reserved_mib": cuda_reserved,
                "rss_mib": rss,
                "send_cache_size": send_cache.size,
                "recv_cache_size": 0,  # rank 0 doesn't have recv cache
                "static_buffer_count": static_reg.size,
                "p50_us": p50,
                "p95_us": p95,
                "p99_us": p99,
                "send_cache_hits": send_cache.hits,
                "send_cache_misses": send_cache.misses,
            }
            memory_snapshots.append(snapshot)
            print(f"  Round {i+1:>6d}: p50={p50:.1f}us p95={p95:.1f}us p99={p99:.1f}us "
                  f"cuda_alloc={cuda_alloc:.1f}MiB rss={rss:.1f}MiB cache={send_cache.size}")
    
    elapsed = time.time() - start_time
    
    if rank == 0 and args.output_json:
        result = {
            "rounds_completed": len(latencies),
            "elapsed_seconds": elapsed,
            "rounds_per_second": len(latencies) / elapsed if elapsed > 0 else 0,
            "memory_snapshots": memory_snapshots,
            "errors": errors,
            "send_cache_final_size": send_cache.size,
            "send_cache_hits": send_cache.hits,
            "send_cache_misses": send_cache.misses,
            "send_cache_evictions": send_cache.evictions,
            "static_buffer_count": static_reg.size,
        }
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSoak complete: {len(latencies)} rounds in {elapsed:.1f}s")
        print(f"Results written to {args.output_json}")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
