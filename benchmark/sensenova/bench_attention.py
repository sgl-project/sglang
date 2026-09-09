# SPDX-License-Identifier: Apache-2.0
"""Compare NEO-Unify attention backends on the same CUDA tensors."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.neo_unify import (
    build_image_token_end,
    neo_unify_attention,
    resolve_neo_backend,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prefill", "denoise"), default="prefill")
    parser.add_argument("--backends", nargs="+", default=["legacy", "sdpa", "triton"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-length", type=int, default=1024)
    parser.add_argument("--prefix-length", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("neo_attention_results.json")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA device is required")
    if (
        args.iterations <= 0
        or args.warmup < 0
        or args.query_length <= 0
        or args.prefix_length < 0
        or args.batch_size <= 0
    ):
        parser.error("Invalid lengths, batch size or iteration count")
    torch.manual_seed(42)
    device, dtype = (
        torch.device("cuda", torch.cuda.current_device()),
        getattr(torch, args.dtype),
    )
    q = torch.randn(
        args.batch_size,
        args.query_length,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        args.batch_size,
        args.query_length + args.prefix_length,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    causal = args.mode == "prefill"
    ids = torch.arange(args.query_length, device=device)
    # Two image spans separated by text, including a boundary across a tile.
    start, end = args.query_length // 8, args.query_length // 2
    ids[start:end] = start
    start, end = args.query_length * 5 // 8, args.query_length * 7 // 8
    ids[start:end] = start
    ends = build_image_token_end(ids, args.prefix_length) if causal else None
    allow = None
    legacy_mask = None
    flash_attn_func = None
    if args.mode == "denoise" and "legacy" in args.backends:
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            pass

    def run(backend):
        if backend not in ("legacy", "eager", "sdpa"):
            return neo_unify_attention(
                q, k, v, image_token_end=ends, causal=causal, backend=backend
            )
        if backend == "legacy" and not causal and flash_attn_func is not None:
            return flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2).repeat_interleave(args.heads // args.kv_heads, 1)
        vh = v.transpose(1, 2).repeat_interleave(args.heads // args.kv_heads, 1)
        if backend == "sdpa" or (backend == "legacy" and not causal):
            return F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=allow
            ).transpose(1, 2)
        scores = (qh @ kh.transpose(-1, -2)) * args.head_dim**-0.5
        if backend == "legacy" and legacy_mask is not None:
            scores = scores + legacy_mask
        elif allow is not None:
            scores = scores.masked_fill(~allow, float("-inf"))
        return (scores.softmax(-1, dtype=torch.float32).to(dtype) @ vh).transpose(1, 2)

    results = []
    for backend in args.backends:
        # Like the model, prepare masks once outside the layer/denoising loop.
        allow = None
        if causal and backend in ("legacy", "eager", "sdpa"):
            kp = torch.arange(k.shape[1], device=device)
            qp = torch.arange(q.shape[1], device=device) + args.prefix_length
            allow = (kp <= qp[:, None]) | (kp < ends[:, None])
            del kp, qp
            legacy_mask = None
            if backend == "legacy":
                legacy_mask = torch.where(
                    allow[None, None],
                    torch.tensor(0.0, device=device),
                    torch.tensor(float("-inf"), device=device),
                )
        # Forced backends must succeed: a missing FA3 build is not a fallback timing.
        if backend == "legacy":
            if causal:
                actual_backend = "legacy/eager"
            else:
                actual_backend = (
                    "legacy/flash" if flash_attn_func is not None else "legacy/sdpa"
                )
        elif backend in ("eager", "sdpa"):
            actual_backend = backend
        else:
            actual_backend = resolve_neo_backend(
                q, image_aware=ends is not None, backend=backend
            )
        for _ in range(args.warmup):
            run(backend)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        samples = []
        for _ in range(args.iterations):
            start, stop = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            run(backend)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
        samples.sort()
        result = dict(
            backend=actual_backend,
            median_ms=samples[len(samples) // 2],
            min_ms=samples[0],
            max_ms=samples[-1],
            peak_memory_bytes=torch.cuda.max_memory_allocated(),
            extra_peak_bytes=torch.cuda.max_memory_allocated() - baseline,
        )
        print(json.dumps(result))
        results.append(result)
    report = dict(
        arguments={**vars(args), "output": str(args.output)},
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
