#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import torch
import triton

from sgl_kernel.rotary_embedding import rotary_embedding_cos_sin as sgl_rotary_cos_sin
from sgl_kernel.testing.rotary_embedding import RotaryEmbedding as NativeRotaryEmbedding

def compute_cos_sin_cache(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def benchmark_mm_rotary_embedding() -> None:
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
        (1, 32, 8, 128),
        (1, 64, 8, 128),
        (1, 32, 8, 64),
        (1, 32, 8, 256),
        (32, 32, 8, 64),
        (1, 32, 1, 128), 
        (32, 8, 8, 128), 
        (1, 32, 8, 80),  
    ]

    for batch_size, num_heads, num_kv_heads, head_size in configs:
        try:
            torch.cuda.synchronize()
        except torch.AcceleratorError:
            torch.cuda.empty_cache()
            pass

        try:
            rotary_dim = head_size
            cos_cache, sin_cache = compute_cos_sin_cache(
                max_seq_len, rotary_dim, dtype=dtype
            )
            cos_cache = cos_cache.to(device)
            sin_cache = sin_cache.to(device)

            cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1).to(
                device=device, dtype=dtype
            )

            native_rope = NativeRotaryEmbedding(
                head_size=head_size,
                rotary_dim=rotary_dim,
                max_position_embeddings=max_seq_len,
                base=10000,
                is_neox_style=True,
                dtype=dtype,
            ).to(device)

            print(f"\nConfig: batch_size={batch_size}, heads={num_heads}/{num_kv_heads}, head_size={head_size}, dtype={dtype}")
            print("-" * 100)
        except Exception as e:
            print(f"\nSkipping config (batch_size={batch_size}, heads={num_heads}/{num_kv_heads}, head_size={head_size}): {e}")
            continue

        header = f"{'seq_len':>8}"
        header += f" | {'native (ms)':>12}"
        header += f" | {'sgl_cos_sin (ms)':>16}"
        header += f" | {'sgl_pos (ms)':>12}"
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
                native_time = None
                sgl_cos_sin_time = None
                sgl_pos_time = None
                vllm_time = None
                fa_time = None
            except (RuntimeError, torch.AcceleratorError) as e:
                print(f"{seq_len:8d} | SKIP (CUDA error from previous iteration)")
                torch.cuda.synchronize()
                continue

            def fn_native() -> None:
                q = query.clone()
                k = key.clone()
                native_rope.forward_native(positions, q, k)

            ms, _, _ = triton.testing.do_bench(fn_native, quantiles=[0.5, 0.2, 0.8])
            native_time = 1000 * ms
            row_str += f" | {native_time:12.4f}"

            def fn_sgl_cos_sin() -> None:
                q = query.clone()
                k = key.clone()
                sgl_rotary_cos_sin(cos, sin, q, k, head_size, True)

            ms, _, _ = triton.testing.do_bench(fn_sgl_cos_sin, quantiles=[0.5, 0.2, 0.8])
            sgl_cos_sin_time = 1000 * ms
            row_str += f" | {sgl_cos_sin_time:16.4f}"

            def fn_sgl_pos() -> None:
                q = query.clone()
                k = key.clone()
                torch.ops.sgl_kernel.rotary_embedding(
                    positions,
                    q,
                    k,
                    head_size,
                    cos_sin_cache,
                    True,
                )

            ms, _, _ = triton.testing.do_bench(fn_sgl_pos, quantiles=[0.5, 0.2, 0.8])
            sgl_pos_time = 1000 * ms
            row_str += f" | {sgl_pos_time:12.4f}"

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

            if HAS_FLASH_ATTN:
                try:
                    flash_rotary = FlashRotaryEmbedding(rotary_dim, device=device)
                    qkv = torch.randn(
                        batch_size,
                        seq_len,
                        3,
                        num_heads,
                        head_size,
                        dtype=dtype,
                        device=device,
                    )

                    def fn_flash_attn() -> None:
                        qkv_fa = qkv.clone()
                        flash_rotary(qkv_fa, seqlen_offset=0)

                    ms, _, _ = triton.testing.do_bench(
                        fn_flash_attn, quantiles=[0.5, 0.2, 0.8]
                    )
                    fa_time = 1000 * ms
                    row_str += f" | {fa_time:14.4f}"
                except Exception:
                    row_str += f" | {'ERROR':>14}"
                    fa_time = None

            if sgl_cos_sin_time is not None:
                if HAS_VLLM and vllm_time is not None:
                    speedup = vllm_time / sgl_cos_sin_time
                    row_str += f" | {speedup:9.2f}x"
                elif (not HAS_VLLM) and native_time is not None:
                    speedup = native_time / sgl_cos_sin_time
                    row_str += f" | {speedup:9.2f}x"
                else:
                    row_str += f" | {'N/A':>9}"

            print(row_str)
            results.append(
                {
                    "seq_len": seq_len,
                    "native": native_time,
                    "sgl_cos_sin": sgl_cos_sin_time,
                    "sgl_pos": sgl_pos_time,
                    "vllm": vllm_time,
                    "flash_attn": fa_time,
                }
            )

        print("-" * 100)
        print("\nAverage time:")
        native_vals = [r["native"] for r in results if r["native"] is not None]
        if native_vals:
            print(f"  Native: {np.mean(native_vals):.4f} ms")

        sgl_cos_sin_vals = [r["sgl_cos_sin"] for r in results if r["sgl_cos_sin"] is not None]
        if sgl_cos_sin_vals:
            print(f"  SGLang (cos/sin API): {np.mean(sgl_cos_sin_vals):.4f} ms")

        sgl_pos_vals = [r["sgl_pos"] for r in results if r["sgl_pos"] is not None]
        if sgl_pos_vals:
            print(f"  SGLang (positions API): {np.mean(sgl_pos_vals):.4f} ms")
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