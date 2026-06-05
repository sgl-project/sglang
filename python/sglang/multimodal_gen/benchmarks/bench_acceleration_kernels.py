# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.acceleration_policy import (
    KERNEL_COMPILE_POLICY_ENV,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import (
    LocalAttention,
    _LOCAL_ATTENTION_AUTOTUNE_CACHE,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.layers.elementwise import MulAdd
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _emit(name: str, policy: str, ms: float, args: argparse.Namespace) -> None:
    print(
        json.dumps(
            {
                "name": name,
                "policy": policy,
                "ms": ms,
                "batch": args.batch,
                "seq": args.seq,
                "kv_seq": args.kv_seq or args.seq,
                "hidden": args.hidden,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "dtype": args.dtype,
                "attention_autotune_mode": getattr(
                    args, "attention_autotune_mode", None
                ),
            },
            sort_keys=True,
        )
    )


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def bench_mul_add(args: argparse.Namespace) -> None:
    dtype = _dtype(args.dtype)
    a = torch.randn(args.batch, args.seq, args.hidden, device="cuda", dtype=dtype)
    b = torch.randn(args.batch, 1, args.hidden, device="cuda", dtype=dtype)
    c = torch.randn(args.batch, args.seq, args.hidden, device="cuda", dtype=dtype)

    for policy in args.kernel_policies:
        os.environ[KERNEL_COMPILE_POLICY_ENV] = policy
        op = MulAdd().cuda()
        ms = _time_cuda(lambda: op(a, b, c, 1), args.warmup, args.iters)
        _emit("mul_add", policy, ms, args)


def bench_norm_scale_shift(args: argparse.Namespace) -> None:
    dtype = _dtype(args.dtype)
    x = torch.randn(args.batch, args.seq, args.hidden, device="cuda", dtype=dtype)
    shift = torch.randn(args.batch, 1, args.hidden, device="cuda", dtype=dtype)
    scale = torch.randn(args.batch, 1, args.hidden, device="cuda", dtype=dtype)

    for policy in args.kernel_policies:
        os.environ[KERNEL_COMPILE_POLICY_ENV] = policy
        op = LayerNormScaleShift(args.hidden, dtype=dtype).cuda()
        ms = _time_cuda(lambda: op(x, shift, scale), args.warmup, args.iters)
        _emit("layer_norm_scale_shift", policy, ms, args)


def bench_sdpa(args: argparse.Namespace) -> None:
    dtype = _dtype(args.dtype)
    kv_seq = args.kv_seq or args.seq
    q = torch.randn(
        args.batch,
        args.heads,
        args.seq,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        args.batch,
        args.heads,
        kv_seq,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k)

    policies = {
        "torch_default": None,
        "cudnn_allowed": [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        "cudnn_disabled": [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
    }
    for policy, backends in policies.items():
        if backends is None:
            fn = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            fn = lambda backends=backends: _sdpa_with_backends(q, k, v, backends)
        ms = _time_cuda(fn, args.warmup, args.iters)
        _emit("sdpa", policy, ms, args)


def bench_attention_backends(args: argparse.Namespace) -> None:
    dtype = _dtype(args.dtype)
    kv_seq = args.kv_seq or args.seq
    q = torch.randn(
        args.batch,
        args.seq,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        args.batch,
        kv_seq,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    scale = args.head_dim**-0.5
    flash = FlashAttentionImpl(
        num_heads=args.heads,
        head_size=args.head_dim,
        causal=False,
        softmax_scale=scale,
        num_kv_heads=args.heads,
    )
    metadata = FlashAttentionMetadata(max_seqlen_q=args.seq, max_seqlen_k=kv_seq)

    policies = {
        "fa": lambda: flash.forward(q, k, v, attn_metadata=metadata),
        "sdpa_cudnn": lambda: _sdpa_with_backends(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
        ).transpose(1, 2),
        "sdpa_no_cudnn": lambda: _sdpa_with_backends(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
        ).transpose(1, 2),
    }
    for policy, fn in policies.items():
        ms = _time_cuda(fn, args.warmup, args.iters)
        _emit("attention_backend", policy, ms, args)


def bench_local_attention_autotune(args: argparse.Namespace) -> None:
    dtype = _dtype(args.dtype)
    kv_seq = args.kv_seq or args.seq
    q = torch.randn(
        args.batch,
        args.seq,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        args.batch,
        kv_seq,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k)
    metadata = FlashAttentionMetadata(max_seqlen_q=args.seq, max_seqlen_k=kv_seq)
    attention_kwargs = {}
    if args.attention_autotune_mode == "explicit":
        attention_kwargs = {
            "attention_autotune": True,
            "attention_autotune_warmup": args.autotune_warmup,
            "attention_autotune_iters": args.autotune_iters,
            "attention_autotune_min_speedup": args.autotune_min_speedup,
        }
    elif args.attention_autotune_mode == "disabled":
        attention_kwargs = {"attention_autotune": False}
    with global_force_attn_backend_context_manager(AttentionBackendEnum.FA):
        layer = LocalAttention(
            num_heads=args.heads,
            head_size=args.head_dim,
            num_kv_heads=args.heads,
            **attention_kwargs,
        ).cuda()
    _LOCAL_ATTENTION_AUTOTUNE_CACHE.clear()
    with torch.inference_mode(), set_forward_context(0, metadata):
        layer(q, k, v)
        selected = next(
            iter(_LOCAL_ATTENTION_AUTOTUNE_CACHE.values()),
            f"{args.attention_autotune_mode}_uncached",
        )
        ms = _time_cuda(lambda: layer(q, k, v), args.warmup, args.iters)
    _emit("local_attention_autotune", selected, ms, args)


def _sdpa_with_backends(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backends: list[SDPBackend],
) -> torch.Tensor:
    with sdpa_kernel(backends):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        choices=[
            "all",
            "mul_add",
            "norm",
            "sdpa",
            "attention_backends",
            "local_attention_autotune",
        ],
        default="all",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--kv-seq", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--autotune-warmup", type=int, default=3)
    parser.add_argument("--autotune-iters", type=int, default=10)
    parser.add_argument("--autotune-min-speedup", type=float, default=1.02)
    parser.add_argument(
        "--attention-autotune-mode",
        choices=["default", "explicit", "disabled"],
        default="default",
    )
    parser.add_argument(
        "--kernel-policies",
        nargs="+",
        default=["force_fused", "force_torch_compile"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("bench_acceleration_kernels.py requires CUDA")

    if args.op in {"all", "mul_add"}:
        bench_mul_add(args)
    if args.op in {"all", "norm"}:
        bench_norm_scale_shift(args)
    if args.op in {"all", "sdpa"}:
        bench_sdpa(args)
    if args.op in {"all", "attention_backends"}:
        bench_attention_backends(args)
    if args.op in {"all", "local_attention_autotune"}:
        bench_local_attention_autotune(args)


if __name__ == "__main__":
    main()
