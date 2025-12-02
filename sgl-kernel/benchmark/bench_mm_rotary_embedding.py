#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import torch
import triton

from sgl_kernel.rotary_embedding import rotary_embedding as sgl_rotary_embedding

def compute_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute separate cos and sin caches.

    Returns:
        cos, sin: shape (max_seq_len, rotary_dim / 2)
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def benchmark_mm_rotary_embedding() -> None:
    """Benchmark sgl_kernel.rotary_embedding vs vLLM & flash_attn (if available)."""
    try:
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as vLLMRotaryEmbedding

        HAS_VLLM = True
    except ImportError:
        vLLMRotaryEmbedding = None
        HAS_VLLM = False
        print("vLLM not available")

    try:
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

        HAS_FLASH_ATTN = True
    except ImportError:
        FlashRotaryEmbedding = None
        HAS_FLASH_ATTN = False
        print("flash_attn not available")

    device = "cuda"
    dtype = torch.bfloat16
    max_seq_len = 65536

    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # Test configurations: (batch_size, num_heads, num_kv_heads, head_size)
    configs = [
        # Decoding scenarios (batch_size=1)
        (1, 32, 8, 128),
        (1, 64, 8, 128),
        (1, 32, 8, 64),
        (1, 32, 8, 256),
        (32, 32, 8, 64),
        # # Edge cases
        (1, 32, 1, 128),   # MQA
        (32, 8, 8, 128),   # MHA
        (1, 32, 8, 80),    # Non-standard head_size
    ]

    for batch_size, num_heads, num_kv_heads, head_size in configs:
        try:
            torch.cuda.synchronize()
        except torch.AcceleratorError:
            torch.cuda.empty_cache()
            pass
        
        try:
            rotary_dim = head_size
            cos_cache, sin_cache = compute_cos_sin_cache(max_seq_len, rotary_dim, dtype=dtype)
            cos_cache = cos_cache.to(device)
            sin_cache = sin_cache.to(device)

            print(f"\nConfig: batch_size={batch_size}, heads={num_heads}/{num_kv_heads}, head_size={head_size}, dtype={dtype}")
            print("-" * 100)
        except Exception as e:
            print(f"\nSkipping config (batch_size={batch_size}, heads={num_heads}/{num_kv_heads}, head_size={head_size}): {e}")
            continue

        header = f"{'seq_len':>8}"
        header += f" | {'ours (ms)':>10}"
        if HAS_VLLM:
            header += f" | {'vLLM (ms)':>10}"
        if HAS_FLASH_ATTN:
            header += f" | {'flash_attn (ms)':>14}"
        header += f" | {'speedup':>9}"
        print(header)
        print("-" * 100)

        results = []

        for seq_len in seq_lens:
            try:
                num_tokens = batch_size * seq_len
                query = torch.randn(num_tokens, num_heads * head_size, dtype=dtype, device=device)  
                key = torch.randn(num_tokens, num_kv_heads * head_size, dtype=dtype, device=device)
                positions = torch.arange(seq_len, device=device, dtype=torch.int64).repeat(batch_size)
                cos = cos_cache[positions]
                sin = sin_cache[positions]

                row_str = f"{seq_len:8d}"
                ours_time = None
                vllm_time = None
                fa_time = None
            except (RuntimeError, torch.AcceleratorError) as e:
                print(f"{seq_len:8d} | SKIP (CUDA error from previous iteration)")
                torch.cuda.synchronize()
                continue

            # Ours: sgl_kernel.rotary_embedding (triton.testing.do_bench style)
            def fn_ours() -> None:
                q = query.clone()
                k = key.clone()
                sgl_rotary_embedding(cos, sin, q, k, head_size, True)

            ms, _, _ = triton.testing.do_bench(fn_ours, quantiles=[0.5, 0.2, 0.8])
            ours_time = 1000 * ms
            row_str += f" | {ours_time:10.4f}"

            # vLLM
            if HAS_VLLM:
                vllm_rope = vLLMRotaryEmbedding(
                    head_size=head_size,
                    rotary_dim=rotary_dim,
                    max_position_embeddings=max_seq_len,
                    base=10000,
                    is_neox_style=True,
                    dtype=dtype,
                ).cuda()
                
                def fn_vllm() -> None:
                    q = query.clone()
                    k = key.clone()
                    vllm_rope.forward_cuda(positions, q, k)

                ms, _, _ = triton.testing.do_bench(fn_vllm, quantiles=[0.5, 0.2, 0.8])
                vllm_time = 1000 * ms
                row_str += f" | {vllm_time:10.4f}"
            else:
                row_str += f" | {'N/A':>10}"

            # FlashAttention RotaryEmbedding (NeoX-style, QKV layout)
            if HAS_FLASH_ATTN:
                try:
                    flash_rotary = FlashRotaryEmbedding(rotary_dim, device=device)
                    qkv = torch.randn(batch_size, seq_len, 3, num_heads, head_size, dtype=dtype, device=device)

                    def fn_flash_attn() -> None:
                        qkv_fa = qkv.clone()
                        flash_rotary(qkv_fa, seqlen_offset=0)

                    ms, _, _ = triton.testing.do_bench(fn_flash_attn, quantiles=[0.5, 0.2, 0.8])
                    fa_time = 1000 * ms
                    row_str += f" | {fa_time:14.4f}"
                except Exception:
                    row_str += f" | {'ERROR':>14}"
                    fa_time = None
            else:
                row_str += f" | {'N/A':>14}"

            if HAS_VLLM and ours_time is not None and vllm_time is not None:
                speedup = vllm_time / ours_time
                row_str += f" | {speedup:9.2f}x"

            print(row_str)
            results.append({"seq_len": seq_len, "ours": ours_time, "vllm": vllm_time, "flash_attn": fa_time})

        print("-" * 100)
        print("\nAverage time:")
        ours_vals = [r["ours"] for r in results if r["ours"] is not None]
        if ours_vals:
            print(f"  SGLang: {np.mean(ours_vals):.4f} ms")
        if HAS_VLLM:
            vllm_vals = [r["vllm"] for r in results if r["vllm"] is not None]
            if vllm_vals:
                print(f"  vLLM: {np.mean(vllm_vals):.4f} ms")
        if HAS_FLASH_ATTN:
            fa_vals = [r["flash_attn"] for r in results if r["flash_attn"] is not None]
            if fa_vals:
                print(f"  flash_attn: {np.mean(fa_vals):.4f} ms")
   

if __name__ == "__main__":
    benchmark_mm_rotary_embedding()