# Simple microbenchmark to compare TP all-reduce overlap on/off

import argparse
import os
import time

import torch
import torch.distributed as dist


def setup_dist():
    backend = os.environ.get("SGLANG_DIST_BACKEND", "nccl")
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def teardown_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def run_once(tensor, iters: int) -> float:
    # simple compute kernel to overlap with comm
    def compute_step(x):
        y = torch.nn.functional.silu(x)
        y = y @ x.transpose(0, 1)
        return y

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        # launch compute on default stream
        tmp = compute_step(tensor)
        # in-place all-reduce on tensor
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    setup_dist()
    try:
        world_size = dist.get_world_size()
        if world_size == 1:
            print("world_size=1; this benchmark needs TP>1")
            return
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        x = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)

        # Baseline: overlap disabled
        os.environ["SGLANG_ENABLE_TP_ALLREDUCE_OVERLAP"] = "false"
        torch.cuda.synchronize()
        baseline = run_once(x, args.iters)

        # Overlap enabled
        os.environ["SGLANG_ENABLE_TP_ALLREDUCE_OVERLAP"] = "true"
        torch.cuda.synchronize()
        overlap = run_once(x, args.iters)

        if dist.get_rank() == 0:
            print({
                "tokens": args.tokens,
                "hidden": args.hidden,
                "dtype": args.dtype,
                "iters": args.iters,
                "per_iter_ms_baseline": baseline * 1000.0,
                "per_iter_ms_overlap": overlap * 1000.0,
                "speedup_percent": (baseline - overlap) / baseline * 100.0,
            })
    finally:
        teardown_dist()


if __name__ == "__main__":
    main()


