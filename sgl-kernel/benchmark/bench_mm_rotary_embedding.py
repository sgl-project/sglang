#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import triton
from sgl_kernel.rotary_embedding import rotary_embedding_cos_sin as sgl_rotary_cos_sin
from sgl_kernel.testing.rotary_embedding import RotaryEmbedding as NativeRotaryEmbedding


def compute_cos_sin_cache_half(
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


def _expand_neox_full_repeat(half: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        (half.shape[0], half.shape[1] * 2), dtype=half.dtype, device=half.device
    )
    out[:, 0::2] = half
    out[:, 1::2] = half
    return out


def _expand_llama_full_cat(half: torch.Tensor) -> torch.Tensor:
    return torch.cat([half, half], dim=-1)


def _make_misaligned_contiguous_like(
    t: torch.Tensor, offset_elems: int = 1
) -> torch.Tensor:
    # Create a contiguous tensor with same shape/dtype/device but misaligned base pointer.
    numel = t.numel()
    buf = torch.empty((numel + offset_elems,), device=t.device, dtype=t.dtype)
    out = buf[offset_elems : offset_elems + numel].view_as(t)
    out.copy_(t)
    assert out.is_contiguous()
    return out


def _make_tensor(
    shape: Tuple[int, ...], *, dtype: torch.dtype, device: str, misalign: bool
) -> torch.Tensor:
    numel = int(np.prod(shape))
    if not misalign:
        return torch.randn(*shape, dtype=dtype, device=device)
    buf = torch.randn((numel + 1,), dtype=dtype, device=device)
    return buf[1 : 1 + numel].view(shape)


def _torch_naive_rope_inplace(
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    interleaved: bool,
) -> None:
    """Naive RoPE on GPU using PyTorch tensor ops (in-place on q/k)."""

    def _as_3d(x: torch.Tensor, h: int) -> torch.Tensor:
        if x.dim() == 3:
            return x
        return x.view(x.shape[0], h, head_size)

    if interleaved and cos.shape[1] == head_size:
        half = head_size // 2
        cos = cos.view(cos.shape[0], half, 2).select(2, 0).contiguous()
        sin = sin.view(sin.shape[0], half, 2).select(2, 1).contiguous()

    if interleaved:
        embed_dim = int(cos.shape[1])
        rot_dim = embed_dim * 2
    else:
        embed_dim = int(cos.shape[1]) // 2
        rot_dim = embed_dim * 2

    cos_b = cos[:, None, :embed_dim]
    sin_b = sin[:, None, :embed_dim]

    def _apply(x: torch.Tensor, h: int) -> None:
        x3 = _as_3d(x, h)
        xr = x3[..., :rot_dim]
        if interleaved:
            xr2 = xr.view(xr.shape[0], xr.shape[1], embed_dim, 2)
            x0 = xr2[..., 0]
            x1 = xr2[..., 1]
            out0 = x0 * cos_b - x1 * sin_b
            out1 = x1 * cos_b + x0 * sin_b
            xr2[..., 0].copy_(out0)
            xr2[..., 1].copy_(out1)
        else:
            x0 = xr[..., :embed_dim]
            x1 = xr[..., embed_dim:rot_dim]
            cos_x = cos[:, None, :embed_dim]
            sin_x = sin[:, None, :embed_dim]
            cos_y = cos[:, None, embed_dim:rot_dim]
            sin_y = sin[:, None, embed_dim:rot_dim]
            out0 = x0 * cos_x - x1 * sin_x
            out1 = x1 * cos_y + x0 * sin_y
            xr[..., :embed_dim].copy_(out0)
            xr[..., embed_dim:rot_dim].copy_(out1)

    _apply(q, num_heads)
    if k is not None:
        _apply(k, num_kv_heads)


def _predict_vec_flags(
    *,
    dtype: torch.dtype,
    interleaved: bool,
    embed_dim_for_rotation: int,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_token_stride_elems: int,
    key_token_stride_elems: int,
    head_stride_query_elems: int,
    head_stride_key_elems: int,
) -> Tuple[bool, bool]:
    """Predict flags for the *optimized* kernel dispatch:
    - use_vec_compute: vectorized compute loop (pairs_per_step) is allowed
    - qk_aligned16: Q/K supports 16B vec load/store (otherwise use unaligned Q/K path)
    """
    kVecBytes = 16
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    kElePerVec = kVecBytes // elem_bytes
    pairs_per_step = (kElePerVec // 2) if interleaved else kElePerVec

    # A) vectorized compute viability (cos/sin alignment + embed_dim multiple)
    use_vec_compute = True
    if embed_dim_for_rotation % pairs_per_step != 0:
        use_vec_compute = False

    # cos/sin alignment & row stride for vector loads (float2/float4 reinterprets)
    if (cos.data_ptr() % kVecBytes) != 0:
        use_vec_compute = False
    if (sin.data_ptr() % kVecBytes) != 0:
        use_vec_compute = False
    if ((cos.shape[1] * elem_bytes) % kVecBytes) != 0:
        use_vec_compute = False
    if ((sin.shape[1] * elem_bytes) % kVecBytes) != 0:
        use_vec_compute = False

    # B) Q/K 16B alignment (only affects aligned_qk template)
    qk_aligned16 = True
    if (query.data_ptr() % kVecBytes) != 0:
        qk_aligned16 = False
    if (query_token_stride_elems * elem_bytes) % kVecBytes != 0:
        qk_aligned16 = False
    if (head_stride_query_elems * elem_bytes) % kVecBytes != 0:
        qk_aligned16 = False

    if key is not None:
        if (key.data_ptr() % kVecBytes) != 0:
            qk_aligned16 = False
        if (key_token_stride_elems * elem_bytes) % kVecBytes != 0:
            qk_aligned16 = False
        if (head_stride_key_elems * elem_bytes) % kVecBytes != 0:
            qk_aligned16 = False

    return use_vec_compute, qk_aligned16


def _predict_use_grid_2d(
    *,
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim_for_rotation: int,
    use_vec_compute: bool,
    dtype: torch.dtype,
    interleaved: bool,
) -> Tuple[bool, int]:
    kVecBytes = 16
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    kElePerVec = kVecBytes // elem_bytes
    pairs_per_step = (kElePerVec // 2) if interleaved else kElePerVec
    launch_pairs_per_thread = pairs_per_step if use_vec_compute else 1

    max_pairs = max(
        num_heads * embed_dim_for_rotation, num_kv_heads * embed_dim_for_rotation
    )
    total_threads_needed = (
        max_pairs + launch_pairs_per_thread - 1
    ) // launch_pairs_per_thread

    def _round_up32(x: int) -> int:
        return ((x + 31) // 32) * 32

    threads_per_block_2d = min(
        512, max(128, _round_up32(min(total_threads_needed, 512)))
    )
    blocks_per_token_2d = (
        total_threads_needed + threads_per_block_2d - 1
    ) // threads_per_block_2d
    use_grid_2d = (num_tokens <= 4) and (blocks_per_token_2d > 1)
    return use_grid_2d, blocks_per_token_2d


@dataclass(frozen=True)
class BenchCase:
    name: str
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    rotary_dim: int
    interleaved: bool
    dtype: torch.dtype
    layout: str  # "2d" or "3d"
    key_mode: str  # "with_k" or "no_k"
    cache_format: str  # "half" | "neox_full_repeat" | "llama_full_cat"
    misalign: str  # "none" | "q" | "cos"
    seq_lens: List[int]
    bench_native: bool = True


def benchmark_mm_rotary_embedding(args: argparse.Namespace) -> None:
    try:
        from vllm.model_executor.layers.rotary_embedding import (
            RotaryEmbedding as vLLMRotaryEmbedding,
        )

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
    max_seq_len = 65536

    print(
        "Legend:\n"
        "  - vec: vec16 => predicted 16B-aligned vector path; vec16u => vectorized compute but unaligned Q/K load/store; scalar => scalar/fallback\n"
        "  - grid: 1D => one block per token; 2D => split one token across multiple blocks (only when num_tokens<=4)\n"
        "  - bpt2d: estimated blocks-per-token if 2D were used (useful to judge 'small grid but many blocks per token')\n"
    )

    cases: List[BenchCase] = [
        BenchCase(
            name="base_bf16_int2d_k_halfcache",
            batch_size=1,
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            rotary_dim=128,
            interleaved=True,
            dtype=torch.bfloat16,
            layout="2d",
            key_mode="with_k",
            cache_format="half",
            misalign="none",
            seq_lens=[1, 2, 4, 32, 256, 3015, 8192],
            bench_native=True,
        ),
        BenchCase(
            name="base_bf16_nonint2d_k_llamacache",
            batch_size=1,
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            rotary_dim=128,
            interleaved=False,
            dtype=torch.bfloat16,
            layout="2d",
            key_mode="with_k",
            cache_format="llama_full_cat",
            misalign="none",
            seq_lens=[1, 4, 256, 8192],
            bench_native=False,
        ),
        BenchCase(
            name="base_bf16_int2d_k_halfcache_misalignQ",
            batch_size=1,
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            rotary_dim=128,
            interleaved=True,
            dtype=torch.bfloat16,
            layout="2d",
            key_mode="with_k",
            cache_format="half",
            misalign="q",
            seq_lens=[1, 4, 256, 8192],
            bench_native=False,
        ),
        BenchCase(
            name="base_bf16_int2d_noK_halfcache",
            batch_size=1,
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            rotary_dim=128,
            interleaved=True,
            dtype=torch.bfloat16,
            layout="2d",
            key_mode="no_k",
            cache_format="half",
            misalign="none",
            seq_lens=[1, 4, 256, 8192],
            bench_native=False,
        ),
        BenchCase(
            name="base_bf16_int3d_k_halfcache",
            batch_size=1,
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            rotary_dim=128,
            interleaved=True,
            dtype=torch.bfloat16,
            layout="3d",
            key_mode="with_k",
            cache_format="half",
            misalign="none",
            seq_lens=[1, 4, 256, 8192],
            bench_native=False,
        ),
        BenchCase(
            name="base_fp16_int2d_k_halfcache",
            batch_size=1,
            num_heads=32,
            num_kv_heads=8,
            head_size=128,
            rotary_dim=128,
            interleaved=True,
            dtype=torch.float16,
            layout="2d",
            key_mode="with_k",
            cache_format="half",
            misalign="none",
            seq_lens=[1, 4, 256, 8192],
            bench_native=False,
        ),
        BenchCase(
            name="qwen_image_bf16_int2d_k_halfcache",
            batch_size=1,
            num_heads=24,
            num_kv_heads=24,
            head_size=128,
            rotary_dim=128,
            interleaved=True,
            dtype=torch.bfloat16,
            layout="2d",
            key_mode="with_k",
            cache_format="half",
            misalign="none",
            seq_lens=[1, 4, 3015],
            bench_native=False,
        ),
        BenchCase(
            name="grid2d_stress_bf16_int2d_k_neoxfullrepeat",
            batch_size=1,
            num_heads=128,
            num_kv_heads=128,
            head_size=256,
            rotary_dim=256,
            interleaved=True,
            dtype=torch.bfloat16,
            layout="2d",
            key_mode="with_k",
            cache_format="neox_full_repeat",
            misalign="none",
            seq_lens=[1, 2, 4],
            bench_native=False,
        ),
        BenchCase(
            name="batch32_bf16_int2d_k_halfcache",
            batch_size=32,
            num_heads=32,
            num_kv_heads=8,
            head_size=64,
            rotary_dim=64,
            interleaved=True,
            dtype=torch.bfloat16,
            layout="2d",
            key_mode="with_k",
            cache_format="half",
            misalign="none",
            seq_lens=[1, 4, 128, 1024],
            bench_native=False,
        ),
    ]

    if args.case is not None:
        allow = set(args.case)
        cases = [c for c in cases if c.name in allow]
        missing = allow.difference({c.name for c in cases})
        if missing:
            print(f"WARNING: unknown --case entries ignored: {sorted(missing)}")

    for case in cases:
        try:
            torch.cuda.synchronize()
        except torch.AcceleratorError:
            torch.cuda.empty_cache()
            pass

        try:
            cos_half, sin_half = compute_cos_sin_cache_half(
                max_seq_len, case.rotary_dim, dtype=torch.float32
            )
            cos_half = cos_half.to(device=device, dtype=case.dtype)
            sin_half = sin_half.to(device=device, dtype=case.dtype)

            if case.cache_format == "half":
                cos_cache = cos_half
                sin_cache = sin_half
            elif case.cache_format == "neox_full_repeat":
                cos_cache = _expand_neox_full_repeat(cos_half)
                sin_cache = _expand_neox_full_repeat(sin_half)
            elif case.cache_format == "llama_full_cat":
                cos_cache = _expand_llama_full_cat(cos_half)
                sin_cache = _expand_llama_full_cat(sin_half)
            else:
                raise ValueError(f"Unknown cache_format={case.cache_format}")

            cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)

            native_rope = NativeRotaryEmbedding(
                head_size=case.head_size,
                rotary_dim=case.rotary_dim,
                max_position_embeddings=max_seq_len,
                base=10000,
                is_neox_style=case.interleaved,
                dtype=case.dtype,
            ).to(device)

            print(
                f"\nCase: {case.name} | "
                f"bs={case.batch_size} heads={case.num_heads}/{case.num_kv_heads} "
                f"hs={case.head_size} rd={case.rotary_dim} "
                f"dtype={case.dtype} interleaved={case.interleaved} layout={case.layout} "
                f"key={case.key_mode} cache={case.cache_format} misalign={case.misalign}"
            )
            print("-" * 100)
        except Exception as e:
            print(f"\nSkipping case {case.name}: {e}")
            continue

        header = f"{'seq_len':>8}"
        if case.bench_native:
            header += f" | {'native (ms)':>12}"
        header += f" | {'naive (ms)':>11}"
        header += f" | {'sgl_cos_sin (ms)':>16}"
        header += f" | {'sgl_pos (ms)':>12}"
        header += f" | {'vec':>6} | {'grid':>4} | {'bpt2d':>5}"
        if HAS_VLLM:
            header += f" | {'vLLM (ms)':>10}"
        if HAS_FLASH_ATTN:
            header += f" | {'flash_attn (ms)':>14}"
        header += f" | {'pos/cos':>9}"
        print(header)
        print("-" * 100)

        results = []

        for seq_len in case.seq_lens:
            try:
                num_tokens = case.batch_size * seq_len
                if case.layout == "2d":
                    query_shape = (num_tokens, case.num_heads * case.head_size)
                    key_shape = (num_tokens, case.num_kv_heads * case.head_size)
                elif case.layout == "3d":
                    query_shape = (num_tokens, case.num_heads, case.head_size)
                    key_shape = (num_tokens, case.num_kv_heads, case.head_size)
                else:
                    raise ValueError(f"Unknown layout={case.layout}")

                query = _make_tensor(
                    query_shape,
                    dtype=case.dtype,
                    device=device,
                    misalign=(case.misalign == "q"),
                )
                key = _make_tensor(
                    key_shape, dtype=case.dtype, device=device, misalign=False
                )
                if case.key_mode == "no_k":
                    key = None

                positions = torch.arange(
                    seq_len, device=device, dtype=torch.int64
                ).repeat(case.batch_size)
                cos = cos_cache[positions]
                sin = sin_cache[positions]

                if case.misalign == "cos":
                    cos = _make_misaligned_contiguous_like(cos)
                    sin = _make_misaligned_contiguous_like(sin)

                row_str = f"{seq_len:8d}"
                native_time = None
                naive_time = None
                sgl_cos_sin_time = None
                sgl_pos_time = None
                vllm_time = None
                fa_time = None
            except (RuntimeError, torch.AcceleratorError):
                print(f"{seq_len:8d} | SKIP (CUDA error from previous iteration)")
                torch.cuda.synchronize()
                continue

            def fn_native() -> None:
                q = query.clone()
                if key is None:
                    raise RuntimeError("native requires key")
                k = key.clone()
                native_rope.forward_native(positions, q, k)

            if case.bench_native and key is not None:
                ms, _, _ = triton.testing.do_bench(fn_native, quantiles=[0.5, 0.2, 0.8])
                native_time = 1000 * ms
                row_str += f" | {native_time:12.4f}"
            elif case.bench_native:
                row_str += f" | {'N/A':>12}"

            @torch.no_grad()
            def fn_naive() -> None:
                qn = query.clone()
                kn = None if key is None else key.clone()
                _torch_naive_rope_inplace(
                    cos=cos,
                    sin=sin,
                    q=qn,
                    k=kn,
                    num_heads=case.num_heads,
                    num_kv_heads=case.num_kv_heads,
                    head_size=case.head_size,
                    interleaved=case.interleaved,
                )

            ms, _, _ = triton.testing.do_bench(fn_naive, quantiles=[0.5, 0.2, 0.8])
            naive_time = 1000 * ms
            row_str += f" | {naive_time:11.4f}"

            def fn_sgl_cos_sin() -> None:
                q = query.clone()
                k = None if key is None else key.clone()
                sgl_rotary_cos_sin(cos, sin, q, k, case.head_size, case.interleaved)

            ms, _, _ = triton.testing.do_bench(
                fn_sgl_cos_sin, quantiles=[0.5, 0.2, 0.8]
            )
            sgl_cos_sin_time = 1000 * ms
            row_str += f" | {sgl_cos_sin_time:16.4f}"

            def fn_sgl_pos() -> None:
                q = query.clone()
                k = None if key is None else key.clone()
                torch.ops.sgl_kernel.rotary_embedding(
                    positions,
                    q,
                    k,
                    case.head_size,
                    cos_sin_cache,
                    case.interleaved,
                )

            try:
                ms, _, _ = triton.testing.do_bench(
                    fn_sgl_pos, quantiles=[0.5, 0.2, 0.8]
                )
                sgl_pos_time = 1000 * ms
                row_str += f" | {sgl_pos_time:12.4f}"
            except Exception:
                row_str += f" | {'ERROR':>12}"
                sgl_pos_time = None

            # Prediction flags (updated for optimized kernel)
            rot_dim_from_cache = int(cos.shape[1])
            embed_dim_for_rotation = (
                rot_dim_from_cache if case.interleaved else (rot_dim_from_cache // 2)
            )

            if query.dim() == 2:
                query_hidden = int(query.size(1))
                query_token_stride = query_hidden
                head_stride_query = case.head_size
            else:
                query_hidden = int(query.size(1) * query.size(2))
                query_token_stride = query_hidden
                head_stride_query = int(query.stride(1))

            if key is None:
                key_token_stride = 0
                head_stride_key = case.head_size
            else:
                if key.dim() == 2:
                    key_hidden = int(key.size(1))
                    key_token_stride = key_hidden
                    head_stride_key = case.head_size
                else:
                    key_hidden = int(key.size(1) * key.size(2))
                    key_token_stride = key_hidden
                    head_stride_key = int(key.stride(1))

            use_vec_compute, qk_aligned16 = _predict_vec_flags(
                dtype=case.dtype,
                interleaved=case.interleaved,
                embed_dim_for_rotation=embed_dim_for_rotation,
                query=query,
                key=key,
                cos=cos,
                sin=sin,
                query_token_stride_elems=query_token_stride,
                key_token_stride_elems=key_token_stride,
                head_stride_query_elems=head_stride_query,
                head_stride_key_elems=head_stride_key,
            )

            use_grid2d, bpt2d = _predict_use_grid_2d(
                num_tokens=num_tokens,
                num_heads=case.num_heads,
                num_kv_heads=(0 if key is None else case.num_kv_heads),
                embed_dim_for_rotation=embed_dim_for_rotation,
                use_vec_compute=use_vec_compute,
                dtype=case.dtype,
                interleaved=case.interleaved,
            )

            if use_vec_compute:
                vec_str = "vec16" if qk_aligned16 else "vec16u"
            else:
                vec_str = "scalar"

            grid_str = "2D" if use_grid2d else "1D"
            row_str += f" | {vec_str:>6} | {grid_str:>4} | {str(bpt2d):>5}"

            if HAS_VLLM:
                vllm_rope = vLLMRotaryEmbedding(
                    head_size=case.head_size,
                    rotary_dim=case.rotary_dim,
                    max_position_embeddings=max_seq_len,
                    base=10000,
                    is_neox_style=case.interleaved,
                    dtype=case.dtype,
                ).cuda()

                def fn_vllm() -> None:
                    q = query.clone()
                    if key is None:
                        raise RuntimeError("vLLM requires key")
                    k = key.clone()
                    vllm_rope.forward_cuda(positions, q, k)

                try:
                    ms, _, _ = triton.testing.do_bench(
                        fn_vllm, quantiles=[0.5, 0.2, 0.8]
                    )
                    vllm_time = 1000 * ms
                    row_str += f" | {vllm_time:10.4f}"
                except Exception:
                    row_str += f" | {'ERROR':>10}"
                    vllm_time = None

            if HAS_FLASH_ATTN:
                try:
                    from flash_attn.layers.rotary import (  # noqa
                        RotaryEmbedding as FlashRotaryEmbedding,
                    )

                    flash_rotary = FlashRotaryEmbedding(case.rotary_dim, device=device)
                    qkv = torch.randn(
                        case.batch_size,
                        seq_len,
                        3,
                        case.num_heads,
                        case.head_size,
                        dtype=case.dtype,
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

            if (
                (sgl_pos_time is not None)
                and (sgl_cos_sin_time is not None)
                and (sgl_cos_sin_time > 0)
            ):
                speedup = sgl_pos_time / sgl_cos_sin_time
                row_str += f" | {speedup:9.2f}x"
            else:
                row_str += f" | {'N/A':>9}"

            print(row_str)
            results.append(
                {
                    "seq_len": seq_len,
                    "native": native_time,
                    "naive": naive_time,
                    "sgl_cos_sin": sgl_cos_sin_time,
                    "sgl_pos": sgl_pos_time,
                    "vllm": vllm_time,
                    "flash_attn": fa_time,
                }
            )

        native_vals = [r["native"] for r in results if r["native"] is not None]
        naive_vals = [r["naive"] for r in results if r["naive"] is not None]
        sgl_cos_sin_vals = [
            r["sgl_cos_sin"] for r in results if r["sgl_cos_sin"] is not None
        ]
        sgl_pos_vals = [r["sgl_pos"] for r in results if r["sgl_pos"] is not None]
        vllm_vals = [r["vllm"] for r in results if r["vllm"] is not None]
        fa_vals = [r["flash_attn"] for r in results if r["flash_attn"] is not None]

        avg_row = f"{'AVG':>8}"
        if case.bench_native:
            avg_row += (
                f" | {np.mean(native_vals):12.4f}" if native_vals else f" | {'N/A':>12}"
            )
        avg_row += (
            f" | {np.mean(naive_vals):11.4f}" if naive_vals else f" | {'N/A':>11}"
        )
        avg_row += (
            f" | {np.mean(sgl_cos_sin_vals):16.4f}"
            if sgl_cos_sin_vals
            else f" | {'N/A':>16}"
        )
        avg_row += (
            f" | {np.mean(sgl_pos_vals):12.4f}" if sgl_pos_vals else f" | {'N/A':>12}"
        )
        avg_row += f" | {'-':>6} | {'-':>4} | {'-':>5}"

        if HAS_VLLM:
            avg_row += (
                f" | {np.mean(vllm_vals):10.4f}" if vllm_vals else f" | {'N/A':>10}"
            )
        if HAS_FLASH_ATTN:
            avg_row += f" | {np.mean(fa_vals):14.4f}" if fa_vals else f" | {'N/A':>14}"
        if sgl_pos_vals and sgl_cos_sin_vals and (np.mean(sgl_cos_sin_vals) > 0):
            avg_speedup = float(np.mean(sgl_pos_vals) / np.mean(sgl_cos_sin_vals))
            avg_row += f" | {avg_speedup:9.2f}x"
        else:
            avg_row += f" | {'N/A':>9}"

        print("-" * 100)
        print(avg_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark rotary embedding ")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only the named case. If omitted, runs all cases.",
    )
    benchmark_mm_rotary_embedding(parser.parse_args())
