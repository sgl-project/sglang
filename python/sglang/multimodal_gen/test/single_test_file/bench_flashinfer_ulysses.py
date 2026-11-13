# SPDX-License-Identifier: Apache-2.0
"""Correctness and latency of Wan/H3's actual USP exchange entry points.

Run on an otherwise idle node, e.g.:
    torchrun --standalone --nproc-per-node=8 \
      python/sglang/multimodal_gen/test/single_test_file/bench_flashinfer_ulysses.py \
      --seq-lens 4096 16384 --output results.json

Uses synthetic BF16 QKV, without weights. Measures the complete QKV input and
output exchange, including packing, unpacking and allocations, then the same
pair surrounding dense SDPA. These are attention-block timings, not end-to-end
video generation. Times exclude collective initialization and JIT compilation.
NCCL baseline disables both IPC transports and head-group pipelining. Every
rank must see the same options. FlashInfer is selected explicitly, with errors
propagated; the report records the actual communicator backend.
"""

import argparse
import json
import os
import statistics
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    flashinfer_ulysses as fi,
)
from sglang.multimodal_gen.runtime.layers import usp


def measure(fn, warmup, iterations, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        dist.barrier()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end) / iterations, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backend", default="nvlink", choices=["nvlink", "nccl", "pcie", "auto"]
    )
    parser.add_argument("--output", default="flashinfer-ulysses.json")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    group = SimpleNamespace(ulysses_group=dist.group.WORLD, ulysses_shm_handle=None)
    os.environ.update(
        SGLANG_DIFFUSION_IPC_A2A="0",
        SGLANG_DIFFUSION_IPC_A2A_MULTI="0",
        SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS="0",
        SGLANG_DIFFUSION_FLASHINFER_ULYSSES_BACKEND=args.backend,
    )
    results = []
    try:
        with (
            patch.object(usp, "get_sp_group", return_value=group),
            patch.object(usp, "get_ulysses_parallel_world_size", return_value=world),
        ):
            for model, heads in (("wan", 40), ("minimax_h3", 56)):
                if heads % world:
                    continue
                for seq in args.seq_lens:
                    assert seq % world == 0
                    torch.manual_seed(1234 + rank)
                    # Wan's separate 4D projections / H3's strided 3D views.
                    packed = torch.randn(
                        1,
                        seq // world,
                        heads,
                        3,
                        128,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    qkv = packed.unbind(-2)
                    if model == "minimax_h3":
                        qkv = tuple(t.squeeze(0) for t in qkv)

                    def exchange():
                        if model == "wan":
                            return usp._usp_input_all_to_all_qkv(*qkv)
                        return usp._usp_input_all_to_all_packed_qkv(*qkv)

                    def output(x):
                        if model == "minimax_h3":
                            return usp._usp_output_all_to_all(
                                x.unsqueeze(0), head_dim=2
                            ).squeeze(0)
                        return usp._usp_output_all_to_all(x, head_dim=2)

                    def pair():
                        return output(exchange()[0])

                    def attention():
                        q, k, v = exchange()
                        if q.ndim == 3:
                            q, k, v = (t.unsqueeze(0) for t in (q, k, v))
                        out = (
                            F.scaled_dot_product_attention(
                                *(t.transpose(1, 2) for t in (q, k, v))
                            )
                            .transpose(1, 2)
                            .contiguous()
                        )
                        return output(out.squeeze(0) if model == "minimax_h3" else out)

                    os.environ["SGLANG_DIFFUSION_FLASHINFER_ULYSSES"] = "0"
                    expected = tuple(t.clone() for t in exchange())
                    ref_attention = attention().clone()
                    baseline = {
                        "exchange": measure(pair, args.warmup, args.iterations, device),
                        "attention": measure(
                            attention, args.warmup, args.iterations, device
                        ),
                    }
                    os.environ["SGLANG_DIFFUSION_FLASHINFER_ULYSSES"] = "1"
                    actual = exchange()
                    for ref, got in zip(expected, actual):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)
                    torch.testing.assert_close(pair(), qkv[0], rtol=0, atol=0)
                    torch.testing.assert_close(
                        attention(), ref_attention, rtol=0, atol=0
                    )
                    assert fi._CONTEXTS, "FlashInfer path was bypassed"
                    candidate = {
                        "exchange": measure(pair, args.warmup, args.iterations, device),
                        "attention": measure(
                            attention, args.warmup, args.iterations, device
                        ),
                    }
                    row = {
                        "model": model,
                        "global_seq": seq,
                        "heads": heads,
                        "head_dim": 128,
                        "world_size": world,
                        "dtype": "bfloat16",
                        "bitwise_equal": True,
                        "actual_backends": sorted(
                            {c.comm.backend for c in fi._CONTEXTS.values()}
                        ),
                        "nccl": baseline,
                        "flashinfer": candidate,
                        "speedup": {
                            key: baseline[key]["median_ms"]
                            / candidate[key]["median_ms"]
                            for key in baseline
                        },
                    }
                    results.append(row)
                    if rank == 0:
                        print(json.dumps(row), flush=True)
                    fi.close_all()
                    usp.drop_a2a_staging_buffers()
            if rank == 0:
                with open(args.output, "w") as f:
                    json.dump(
                        {
                            "torch": torch.__version__,
                            "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(),
                            "results": results,
                        },
                        f,
                        indent=2,
                    )
    finally:
        fi.close_all()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
