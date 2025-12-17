#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import torch
from sgl_kernel.rotary_embedding import rotary_embedding_cos_sin


def compute_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def profile_rotary_embedding():
    # 配置参数：选取 Benchmark 中出现过的最大维度组合
    # Batch Size: 32
    # Seq Len: 8192 (Benchmark 列表中的最大值)
    # Num Heads: 32 (常规配置，保证计算量)
    # Num KV Heads: 32 (与 Heads 相同，最大化 Key 的负载)
    # Head Size: 256 (Benchmark Configs 中的最大值)
    batch_size = 1
    seq_len = 8192
    num_heads = 32
    num_kv_heads = 8
    head_size = 80
    rotary_dim = head_size
    dtype = torch.bfloat16
    device = "cuda"

    print(
        f"Profiling Config: Batch={batch_size}, SeqLen={seq_len}, "
        f"Heads={num_heads}, KV_Heads={num_kv_heads}, HeadSize={head_size}, Dtype={dtype}"
    )

    # 1. 准备 Cos/Sin Cache
    try:
        cos_cache, sin_cache = compute_cos_sin_cache(seq_len, rotary_dim, dtype=dtype)
        cos_cache = cos_cache.to(device)
        sin_cache = sin_cache.to(device)
    except Exception as e:
        print(f"Error creating cache: {e}")
        return

    # 2. 准备输入数据
    # 计算总 token 数
    num_tokens = batch_size * seq_len
    print(f"Total tokens: {num_tokens}")

    try:
        # 生成 Query 和 Key
        query = torch.randn(
            num_tokens, num_heads * head_size, dtype=dtype, device=device
        )
        key = torch.randn(
            num_tokens, num_kv_heads * head_size, dtype=dtype, device=device
        )

        # 生成对应的 Cos/Sin (Expand per token)
        positions = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(
            batch_size
        )

        cos = cos_cache[positions]
        sin = sin_cache[positions]
    except torch.cuda.OutOfMemoryError:
        print("OOM: The configuration is too large for the current GPU memory.")
        return

    # 3. Warmup
    print("Warming up...")
    for _ in range(5):
        rotary_embedding_cos_sin(cos, sin, query, key, head_size, True)
    torch.cuda.synchronize()

    # 4. Profile Run (只运行一次，便于 NCU 捕获)
    print("Running for profiling...")

    # 开启 Profiler (如果使用 ncu --launch-count 1 ... python script.py，这行其实是可选的，
    # 但显式开启有助于 ncu --profile-from-start off 模式)
    torch.cuda.profiler.start()

    rotary_embedding_cos_sin(cos, sin, query, key, head_size, True)

    torch.cuda.profiler.stop()

    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    profile_rotary_embedding()
