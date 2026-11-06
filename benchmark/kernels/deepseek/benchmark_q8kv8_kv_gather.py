#!/usr/bin/env python3
"""Microbenchmark: Q8KV8 sparse-prefill KV gather overhaul.

Compares the legacy gather (``gather_dequant_requant_fp8_paged_legacy``:
fresh ``torch.zeros`` destination + one program per (token, 128-elem
slice)) against the new gather (``gather_dequant_requant_fp8_paged_dsa``:
no pre-zeroing needed, fused pad-row zero-fill, TOKENS_PER_PROG tokens
per program with 16B-vectorized access), in three flavors:

    legacy      torch.zeros alloc + legacy kernel          (baseline)
    new_alloc   torch.empty alloc + vectorized kernel      (vectorized copy only)
    new_cached  persistent grow-only buffer + vec kernel   (changes 1 + 2,
                                                            = production path)

For each scenario it checks BIT-EXACT equality of the fp8 output bytes
(``torch.equal`` on ``uint8`` views; the requant scale is the identity
scalar 1.0 on this path, so the buffer is the entire output), including
the `topk` zero landing-pad rows, then reports us/call and effective
TB/s.

Shapes model GLM / DeepSeek-V3.2 DSA prefill on one rank: d = 576 fp8
out (512 nope + 64 rope), 656 B/token paged cache rows (512 nope fp8 +
16 B f32 group scales + 128 B bf16 rope), page_size 64, topk 2048.
NOTE: gather traffic scales with kv_len (= number of gathered KV rows
= len(page_table_1_flattened)), NOT with s_q; s_q below only labels the
chunk that a scenario represents.  The GLM-5.2 il=64k profile point
(112.0 us/call, ~0.7 TB/s effective) corresponds to the
(s_q=4096, kv_len=65536) row.

Usage (single GPU, < 2 min):
    python benchmark/kernels/deepseek/benchmark_q8kv8_kv_gather.py [--device cuda:0]
        [--iters 200] [--warmup 20]

Also runs correctness-only edge cases: ragged tail (kv_len % 4 != 0),
extra_rows=0, num_tokens=0, and a cached-buffer SHRINK reuse (big call
then small call) that proves stale bytes from the earlier, larger call
cannot leak into the smaller call's pad rows.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MOD_PATH = _REPO_ROOT / "python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py"

# Import the module straight from its file so the benchmark stays
# standalone (no sglang package import side effects; needs only
# torch + triton).
_spec = importlib.util.spec_from_file_location("dequant_k_cache", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

gather_new = _mod.gather_dequant_requant_fp8_paged_dsa
_infer_dsa_dims = _mod._infer_dsa_dims


def gather_dequant_requant_fp8_paged_legacy(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
    extra_rows: int = 0,
) -> torch.Tensor:
    """Legacy (pre-vectorization) gather + dequant + requant.

    Kept as the bit-exactness / performance reference for
    ``gather_dequant_requant_fp8_paged_dsa`` (see
    ``benchmark/kernels/deepseek/benchmark_q8kv8_kv_gather.py``).  Allocates and zero-fills the
    full destination each call, then launches one program per
    (token, 128-elem slice).
    """
    dim_quant = quant_k_cache.shape[-1]
    dim_nope, dim_rope = _infer_dsa_dims(dim_quant)
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    num_tiles = dim_nope // group_size
    out_dim = dim_nope + dim_rope
    assert num_tiles * group_size == dim_nope

    total_rows = num_tokens + extra_rows
    # Allocate a fresh zero-filled buffer.  The extra landing-pad rows at
    # the tail must read as zeros (the kernel may over-read past
    # num_tokens for masked indices).
    output = torch.zeros(
        (total_rows, 1, out_dim),
        dtype=torch.float8_e4m3fn,
        device=quant_k_cache.device,
    )

    num_blocks_per_token = num_tiles + triton.cdiv(dim_rope, group_size)

    input_nope_q = quant_k_cache[:, :dim_nope]
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    _gather_dequant_requant_fp8_paged_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output


@triton.jit
def _gather_dequant_requant_fp8_paged_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    """Fused gather + dequant(per-group) + requant(per-tensor) -> fp8."""
    token_id = tl.program_id(0)
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int32)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # nope: read fp8, mul group scale -> f32, cast to fp8_e4m3fn
        effective_block_id = raw_block_id
        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE

        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = (
            input_nope_s_ptr
            + token_id_paged * input_nope_s_stride_0
            + effective_block_id
        )

        y_q = tl.load(ptr_q, mask=mask, other=0.0).to(tl.float32)
        y_s = tl.load(ptr_s)

        # dequant -> f32 -> requant to fp8
        y = (y_q * y_s).to(tl.float8e4nv)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # rope: read bf16, cast to fp8
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.float8e4nv)
        tl.store(dst_ptr, data, mask=mask)


gather_legacy = gather_dequant_requant_fp8_paged_legacy

PAGE_SIZE = 64
DIM_QUANT = 656  # 512 nope fp8 + 16 scale bytes + 128 rope bytes
OUT_DIM = 576  # 512 nope + 64 rope, fp8
TOPK = 2048

# (s_q label, kv_len = gathered rows).  kv_len drives the bytes moved.
SCENARIOS = [
    (512, 8192),
    (2048, 32768),
    (4096, 65536),
]


def build_paged_kv_pool(pool_tokens: int, device: str) -> torch.Tensor:
    """Synthetic paged fp8 KV cache: [pool_tokens, 1, 656] fp8_e4m3fn."""
    g = torch.Generator(device=device).manual_seed(0)
    nope = (torch.randn(pool_tokens, 512, generator=g, device=device) * 2.0).to(
        torch.float8_e4m3fn
    )
    # Positive, realistically small per-group dequant scales.
    scales = (torch.rand(pool_tokens, 4, generator=g, device=device) * 0.05 + 1e-3).to(
        torch.float32
    )
    rope = torch.randn(pool_tokens, 64, generator=g, device=device).to(torch.bfloat16)

    raw = torch.empty(pool_tokens, DIM_QUANT, dtype=torch.uint8, device=device)
    raw[:, :512] = nope.view(torch.uint8)
    raw[:, 512:528] = scales.view(torch.uint8)
    raw[:, 528:] = rope.view(torch.uint8)
    return raw.view(torch.float8_e4m3fn).view(pool_tokens, 1, DIM_QUANT)


def build_page_table_flattened(
    kv_lens, pool_tokens: int, device: str, seed: int = 1
) -> torch.Tensor:
    """Realistic page_table_1_flattened: per request, random distinct
    64-token pages, tokens contiguous within a page (production paged
    layout), requests concatenated."""
    n_pool_pages = pool_tokens // PAGE_SIZE
    g = torch.Generator(device="cpu").manual_seed(seed)
    parts = []
    for kv_len in kv_lens:
        n_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        assert n_pages <= n_pool_pages, "pool too small for scenario"
        pages = torch.randperm(n_pool_pages, generator=g)[:n_pages]
        toks = (pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE)[None, :]).reshape(
            -1
        )[:kv_len]
        parts.append(toks)
    return torch.cat(parts).to(torch.int32).to(device)


class CachedGather:
    """Mimics the dsa_backend production path: persistent grow-only fp8
    destination buffer, gather zero-fills only the pad tail in-kernel."""

    def __init__(self):
        self.buf = None

    def __call__(self, pool, pt, extra_rows):
        total = pt.shape[0] + extra_rows
        if self.buf is None or self.buf.shape[0] < total:
            self.buf = torch.empty(
                (total, OUT_DIM), dtype=torch.float8_e4m3fn, device=pool.device
            )
        return gather_new(pool, pt, extra_rows=extra_rows, out=self.buf[:total])


def assert_bit_exact(ref: torch.Tensor, got: torch.Tensor, what: str):
    assert ref.shape == got.shape, f"{what}: shape {got.shape} != {ref.shape}"
    ok = torch.equal(
        ref.contiguous().view(torch.uint8), got.contiguous().view(torch.uint8)
    )
    assert ok, f"{what}: fp8 bytes NOT bit-exact"


def bench_us(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / iters  # ms -> us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this microbench needs 1 GPU.", file=sys.stderr)
        sys.exit(1)
    torch.cuda.set_device(args.device)
    dev = args.device

    pool_tokens = 131072  # 2x the largest kv_len; 131072*656 B ~ 86 MB
    pool = build_paged_kv_pool(pool_tokens, dev)

    print(f"device={dev} ({torch.cuda.get_device_name(dev)})")
    print(f"pool: {pool_tokens} tokens x {DIM_QUANT} B, topk={TOPK}")
    print()

    # ------------------------------------------------------------------
    # Correctness edge cases (not timed)
    # ------------------------------------------------------------------
    print("== correctness edge cases ==")
    cached_edge = CachedGather()
    for kv_len, extra in [(1234, TOPK), (8192, 0), (0, TOPK), (63, 17)]:
        pt = build_page_table_flattened([kv_len], pool_tokens, dev, seed=7)
        ref = gather_legacy(pool, pt, extra_rows=extra)
        got_alloc = gather_new(pool, pt, extra_rows=extra)
        got_cached = cached_edge(pool, pt, extra_rows=extra)
        assert_bit_exact(ref, got_alloc, f"kv_len={kv_len},extra={extra} new_alloc")
        assert_bit_exact(ref, got_cached, f"kv_len={kv_len},extra={extra} new_cached")
        # Pad rows must be exactly zero bytes.
        if extra > 0:
            pad = got_cached[kv_len:].view(torch.uint8)
            assert int(pad.max()) == 0 if pad.numel() else True
        print(f"  kv_len={kv_len:6d} extra_rows={extra:5d}: bit-exact OK")

    # Cached-buffer SHRINK reuse: big call dirties the buffer, then a
    # smaller call must still produce zero pad rows (stale-data test for
    # the grow-only buffer + tail-only zeroing invariant).
    cached_shrink = CachedGather()
    pt_big = build_page_table_flattened([65536], pool_tokens, dev, seed=11)
    cached_shrink(pool, pt_big, TOPK)
    pt_small = build_page_table_flattened([4096], pool_tokens, dev, seed=13)
    ref_small = gather_legacy(pool, pt_small, extra_rows=TOPK)
    got_small = cached_shrink(pool, pt_small, TOPK)
    assert_bit_exact(ref_small, got_small, "shrink-reuse (65536 -> 4096)")
    print("  shrink-reuse 65536 -> 4096 rows: pad rows clean, bit-exact OK")
    print()

    # ------------------------------------------------------------------
    # Timed scenarios
    # ------------------------------------------------------------------
    print("== timing ==")
    print(
        "metrics: us/call = mean wall time per gather call incl. any alloc/"
        "zero-fill (LOWER = faster); eff TB/s = payload (656+4 B/token read"
        " + 576 B/row written incl. pad) / time (HIGHER = faster);"
        " speedup = legacy_us / variant_us (>1 = faster than legacy)."
    )
    header = (
        f"{'s_q':>5} {'kv_len':>7} {'variant':>10} {'us/call':>9} "
        f"{'eff TB/s':>9} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for s_q, kv_len in SCENARIOS:
        pt = build_page_table_flattened([kv_len], pool_tokens, dev, seed=s_q)
        total_rows = kv_len + TOPK
        payload_bytes = kv_len * (DIM_QUANT + 4) + total_rows * OUT_DIM

        cached = CachedGather()
        variants = [
            ("legacy", lambda: gather_legacy(pool, pt, extra_rows=TOPK)),
            ("new_alloc", lambda: gather_new(pool, pt, extra_rows=TOPK)),
            ("new_cached", lambda: cached(pool, pt, TOPK)),
        ]

        # Bit-exactness at the benchmarked shape before timing.
        ref = gather_legacy(pool, pt, extra_rows=TOPK)
        for name, fn in variants[1:]:
            assert_bit_exact(ref, fn(), f"s_q={s_q} {name}")
        del ref

        legacy_us = None
        for name, fn in variants:
            us = bench_us(fn, args.warmup, args.iters)
            tbps = payload_bytes / (us * 1e-6) / 1e12
            if name == "legacy":
                legacy_us = us
                speedup = "1.00x"
            else:
                speedup = f"{legacy_us / us:.2f}x"
            print(
                f"{s_q:>5} {kv_len:>7} {name:>10} {us:>9.1f} {tbps:>9.3f} {speedup:>8}"
            )
        print()

    print(
        "note: legacy additionally writes a full-buffer zero fill "
        f"({total_rows * OUT_DIM / 1e6:.1f} MB at the largest shape) that is "
        "NOT counted in its payload bytes; its true HW bandwidth is higher "
        "than the eff TB/s shown, which is exactly why us/call is the "
        "decision metric."
    )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
