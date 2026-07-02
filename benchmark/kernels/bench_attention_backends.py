"""Benchmark HPC attention backend against Triton, FlashInfer, and FA3.

Similar to vLLM PR #46020's attention benchmark, this script measures raw kernel
execution time for various batch sizes and sequence lengths.

Usage:
    python3 benchmark/kernels/bench_attention_backends.py
    python3 benchmark/kernels/bench_attention_backends.py --backends hpc triton flashinfer
    python3 benchmark/kernels/bench_attention_backends.py --repeats 100
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# ── Config ──────────────────────────────────────────────────────────────────

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8  # GQA ratio = 4
PAGE_SIZE = 64
DTYPE = torch.bfloat16
DEVICE = "cuda"

SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)

# Max KV cache size (tokens).  Enough for 64 × 8k = 512k tokens.
MAX_KV_TOKENS = 512 * 1024
NUM_PAGES = MAX_KV_TOKENS // PAGE_SIZE

WARMUP_ITERS = 10
DEFAULT_REPEATS = 50


# ── Batch spec parsing ──────────────────────────────────────────────────────

@dataclass
class BatchSpec:
    """Parsed batch specification.

    For decode/extend: ``num_reqs`` requests, each with ``new_tokens`` new Q
    tokens and ``cached_kv`` cached KV tokens.
    For prefill: ``num_reqs=1`` request with ``new_tokens`` tokens and
    ``cached_kv=0``.
    For mixed: ``prefill_specs`` + ``decode_specs`` lists.
    """
    name: str
    batch_type: str  # "prefill" | "decode" | "extend" | "spec-decode" | "mixed"
    batch_size: int
    # Simple specs
    num_reqs: int = 1
    new_tokens: int = 0
    cached_kv: int = 0
    # Mixed specs
    prefill_reqs: List[Tuple[int, int]] = field(default_factory=list)  # (new_tokens, cached_kv)
    decode_reqs: List[Tuple[int, int]] = field(default_factory=list)  # (new_tokens, cached_kv)

    @property
    def total_q_tokens(self) -> int:
        if self.batch_type == "mixed":
            return sum(n for n, _ in self.prefill_reqs) + sum(n for n, _ in self.decode_reqs)
        return self.num_reqs * self.new_tokens


def parse_batch_specs() -> List[BatchSpec]:
    """Define batch specs matching the vLLM PR #46020 table."""
    specs = []

    # Prefill (1 request, N tokens, no cached KV)
    for n, label in [(512, "q512"), (2048, "q2k"), (4096, "q4k"), (8192, "q8k")]:
        specs.append(BatchSpec(
            name=label, batch_type="prefill", batch_size=1,
            num_reqs=1, new_tokens=n, cached_kv=0,
        ))

    # Extend (1 request, 1k new tokens, 2k cached)
    specs.append(BatchSpec(
        name="q1ks2k", batch_type="extend", batch_size=1,
        num_reqs=1, new_tokens=1024, cached_kv=2048,
    ))

    # Extend (2 requests, 1k new tokens each, 4k cached each)
    specs.append(BatchSpec(
        name="2q1ks4k", batch_type="extend", batch_size=2,
        num_reqs=2, new_tokens=1024, cached_kv=4096,
    ))

    # Decode (N requests, 1 new token each, K cached)
    for n, k, label in [
        (8, 1024, "8q1s1k"),
        (16, 2048, "16q1s2k"),
        (32, 1024, "32q1s1k"),
        (64, 4096, "64q1s4k"),
    ]:
        specs.append(BatchSpec(
            name=label, batch_type="decode", batch_size=n,
            num_reqs=n, new_tokens=1, cached_kv=k,
        ))

    # Spec-decode (N requests, M new tokens each, K cached)
    for n, m, k, label in [
        (8, 8, 4096, "8q8s4k"),
        (16, 2, 1024, "16q2s1k"),
        (16, 4, 1024, "16q4s1k"),
        (16, 8, 1024, "16q8s1k"),
        (32, 4, 2048, "32q4s2k"),
    ]:
        specs.append(BatchSpec(
            name=label, batch_type="spec-decode", batch_size=n,
            num_reqs=n, new_tokens=m, cached_kv=k,
        ))

    # Mixed (prefill + decode)
    specs.append(BatchSpec(
        name="2q2k_8q1s1k", batch_type="mixed", batch_size=10,
        prefill_reqs=[(2048, 0)],
        decode_reqs=[(1, 1024)] * 8,
    ))
    specs.append(BatchSpec(
        name="4q1k_16q1s2k", batch_type="mixed", batch_size=20,
        prefill_reqs=[(1024, 0)],
        decode_reqs=[(1, 2048)] * 16,
    ))
    specs.append(BatchSpec(
        name="2q4k_32q1s1k", batch_type="mixed", batch_size=34,
        prefill_reqs=[(4096, 0)],
        decode_reqs=[(1, 1024)] * 32,
    ))

    return specs


# ── KV cache & input preparation ────────────────────────────────────────────

@dataclass
class AttentionInputs:
    """Prepared inputs for a batch spec, shared across backends."""
    # Q: [total_q_tokens, num_q_heads, head_dim]
    q: torch.Tensor
    # K/V new: [total_new_kv_tokens, num_kv_heads, head_dim]
    k_new: torch.Tensor
    v_new: torch.Tensor
    # Paged KV cache: [num_pages, page_size, num_kv_heads, head_dim]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    # Block table (page indices): [bs, max_blocks] int32 — used by HPC and FA3
    block_table: torch.Tensor
    # Sequence lengths (total KV tokens including new): [bs] int32
    seq_lens: torch.Tensor
    # For extend: cu_seqlens_q [bs+1] int32, max_seqlens_q int
    cu_seqlens_q: Optional[torch.Tensor] = None
    max_seqlens_q: int = 0
    # For triton: kv_indptr [bs+1], kv_indices [total_tokens] — TOKEN indices
    triton_kv_indptr: Optional[torch.Tensor] = None
    triton_kv_indices: Optional[torch.Tensor] = None
    # For flashinfer: kv_indptr [bs+1], kv_indices [total_pages] — PAGE indices
    fi_kv_indptr: Optional[torch.Tensor] = None
    fi_kv_indices: Optional[torch.Tensor] = None
    fi_kv_last_page_len: Optional[torch.Tensor] = None
    # Batch info
    batch_size: int = 0
    total_q_tokens: int = 0
    is_decode: bool = False


def prepare_inputs(spec: BatchSpec) -> AttentionInputs:
    """Prepare dummy inputs for a batch spec."""
    if spec.batch_type == "mixed":
        return _prepare_mixed_inputs(spec)
    return _prepare_simple_inputs(spec)


def _prepare_simple_inputs(spec: BatchSpec) -> AttentionInputs:
    """Prepare inputs for non-mixed batch specs."""
    bs = spec.num_reqs
    new_tokens = spec.new_tokens
    cached_kv = spec.cached_kv
    total_q = bs * new_tokens
    total_kv = bs * cached_kv + total_q  # cached + new

    # Allocate paged KV cache
    num_pages_needed = (total_kv + PAGE_SIZE - 1) // PAGE_SIZE
    k_cache = torch.randn(
        max(num_pages_needed, NUM_PAGES), PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    v_cache = torch.randn_like(k_cache)

    # Q
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # New K/V
    k_new = torch.randn(total_q, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_new = torch.randn_like(k_new)

    # Block table: [bs, max_blocks] — page indices for HPC and FA3
    # Pages are allocated contiguously per request
    max_seq_len = cached_kv + new_tokens
    max_blocks = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    block_table = torch.zeros(bs, max_blocks, dtype=torch.int32, device=DEVICE)
    page_offset = 0
    for i in range(bs):
        for j in range(max_blocks):
            block_table[i, j] = page_offset + j
        page_offset += max_blocks

    # Sequence lengths (total KV tokens including new)
    seq_lens = torch.full((bs,), max_seq_len, dtype=torch.int32, device=DEVICE)

    # For triton: kv_indptr and kv_indices — TOKEN indices
    # Token indices must be consistent with page assignments:
    #   token_index = page_index * PAGE_SIZE + offset_within_page
    triton_kv_indices_list = []
    triton_kv_indptr = [0]
    for i in range(bs):
        base_page = int(block_table[i, 0].item())
        for j in range(max_seq_len):
            # token index = base_page * PAGE_SIZE + j
            triton_kv_indices_list.append(base_page * PAGE_SIZE + j)
        triton_kv_indptr.append(len(triton_kv_indices_list))
    triton_kv_indices = torch.tensor(triton_kv_indices_list, dtype=torch.int32, device=DEVICE)
    triton_kv_indptr = torch.tensor(triton_kv_indptr, dtype=torch.int32, device=DEVICE)

    # For flashinfer: kv_indptr and kv_indices — PAGE indices
    fi_kv_indices_list = []
    fi_kv_indptr = [0]
    for i in range(bs):
        for j in range(max_blocks):
            fi_kv_indices_list.append(int(block_table[i, j].item()))
        fi_kv_indptr.append(len(fi_kv_indices_list))
    fi_kv_indices = torch.tensor(fi_kv_indices_list, dtype=torch.int32, device=DEVICE)
    fi_kv_indptr = torch.tensor(fi_kv_indptr, dtype=torch.int32, device=DEVICE)

    # For flashinfer: kv_last_page_len — number of valid tokens in last page
    last_page_len = max_seq_len % PAGE_SIZE
    if last_page_len == 0:
        last_page_len = PAGE_SIZE
    fi_kv_last_page_len = torch.full((bs,), last_page_len, dtype=torch.int32, device=DEVICE)

    # For extend: cu_seqlens_q
    cu_seqlens_q = None
    max_q = new_tokens
    if spec.batch_type != "decode":
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
        for i in range(bs):
            cu_seqlens_q[i + 1] = (i + 1) * new_tokens
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        max_q = new_tokens
    else:
        max_q = 1

    return AttentionInputs(
        q=q, k_new=k_new, v_new=v_new,
        k_cache=k_cache, v_cache=v_cache,
        block_table=block_table, seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q, max_seqlens_q=max_q,
        triton_kv_indptr=triton_kv_indptr, triton_kv_indices=triton_kv_indices,
        fi_kv_indptr=fi_kv_indptr, fi_kv_indices=fi_kv_indices,
        fi_kv_last_page_len=fi_kv_last_page_len,
        batch_size=bs, total_q_tokens=total_q,
        is_decode=(spec.batch_type == "decode"),
    )


def _prepare_mixed_inputs(spec: BatchSpec) -> AttentionInputs:
    """Prepare inputs for mixed batch specs (prefill + decode requests)."""
    all_reqs = []  # (new_tokens, cached_kv)
    all_reqs.extend(spec.prefill_reqs)
    all_reqs.extend(spec.decode_reqs)

    bs = len(all_reqs)
    total_q = sum(n for n, _ in all_reqs)

    # Max sequence length across all requests
    max_seq_len = max(n + c for n, c in all_reqs)
    max_blocks = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE

    # Allocate KV cache
    total_kv = sum(n + c for n, c in all_reqs)
    num_pages_needed = (total_kv + PAGE_SIZE - 1) // PAGE_SIZE
    k_cache = torch.randn(
        max(num_pages_needed, NUM_PAGES), PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )
    v_cache = torch.randn_like(k_cache)

    # Q
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k_new = torch.randn(total_q, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_new = torch.randn_like(k_new)

    # Block table — page indices
    block_table = torch.zeros(bs, max_blocks, dtype=torch.int32, device=DEVICE)
    page_offset = 0
    seq_lens_list = []
    blocks_per_req = []
    for i, (nt, ck) in enumerate(all_reqs):
        sl = nt + ck
        seq_lens_list.append(sl)
        req_blocks = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        blocks_per_req.append(req_blocks)
        for j in range(max_blocks):
            block_table[i, j] = page_offset + j if j < req_blocks else 0
        page_offset += req_blocks

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)

    # Triton: kv_indptr / kv_indices — TOKEN indices
    # Token indices must be consistent with page assignments
    triton_kv_indices_list = []
    triton_kv_indptr = [0]
    for i, (nt, ck) in enumerate(all_reqs):
        sl = nt + ck
        base_page = int(block_table[i, 0].item())
        for j in range(sl):
            triton_kv_indices_list.append(base_page * PAGE_SIZE + j)
        triton_kv_indptr.append(len(triton_kv_indices_list))

    triton_kv_indices = torch.tensor(triton_kv_indices_list, dtype=torch.int32, device=DEVICE)
    triton_kv_indptr = torch.tensor(triton_kv_indptr, dtype=torch.int32, device=DEVICE)

    # FlashInfer: kv_indptr / kv_indices — PAGE indices
    fi_kv_indices_list = []
    fi_kv_indptr = [0]
    for i in range(bs):
        rb = blocks_per_req[i]
        for j in range(rb):
            fi_kv_indices_list.append(int(block_table[i, j].item()))
        fi_kv_indptr.append(len(fi_kv_indices_list))
    fi_kv_indices = torch.tensor(fi_kv_indices_list, dtype=torch.int32, device=DEVICE)
    fi_kv_indptr = torch.tensor(fi_kv_indptr, dtype=torch.int32, device=DEVICE)

    # FlashInfer: kv_last_page_len
    fi_last_page_len_list = []
    for sl in seq_lens_list:
        lpl = sl % PAGE_SIZE
        if lpl == 0:
            lpl = PAGE_SIZE
        fi_last_page_len_list.append(lpl)
    fi_kv_last_page_len = torch.tensor(fi_last_page_len_list, dtype=torch.int32, device=DEVICE)

    # cu_seqlens_q
    cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    offset = 0
    for i, (nt, _) in enumerate(all_reqs):
        offset += nt
        cu_seqlens_q[i + 1] = offset

    max_q = max(n for n, _ in all_reqs)

    return AttentionInputs(
        q=q, k_new=k_new, v_new=v_new,
        k_cache=k_cache, v_cache=v_cache,
        block_table=block_table, seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q, max_seqlens_q=max_q,
        triton_kv_indptr=triton_kv_indptr, triton_kv_indices=triton_kv_indices,
        fi_kv_indptr=fi_kv_indptr, fi_kv_indices=fi_kv_indices,
        fi_kv_last_page_len=fi_kv_last_page_len,
        batch_size=bs, total_q_tokens=total_q,
        is_decode=False,
    )


# ── Timing utilities ────────────────────────────────────────────────────────

def benchmark_kernel(fn, warmup=WARMUP_ITERS, repeats=DEFAULT_REPEATS):
    """Benchmark a kernel function using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Return median in seconds
    times.sort()
    return times[len(times) // 2] / 1000.0


# ── Backend implementations ─────────────────────────────────────────────────

def bench_hpc(inputs: AttentionInputs, repeats: int) -> float:
    """Benchmark HPC attention backend."""
    import hpc

    o = torch.empty(
        inputs.total_q_tokens, NUM_Q_HEADS, HEAD_DIM,
        dtype=DTYPE, device=DEVICE,
    )

    if inputs.is_decode:
        # Decode path
        def fn():
            hpc.attention_decode_bf16(
                q=inputs.q.view(inputs.batch_size, NUM_Q_HEADS, HEAD_DIM),
                kcache=inputs.k_cache,
                vcache=inputs.v_cache,
                block_ids=inputs.block_table,
                num_seq_kvcache=inputs.seq_lens,
                mtp=0,
                new_kv_included=True,
                splitk=True,
                output=o,
            )
    else:
        # Prefill/extend path
        def fn():
            hpc.attention_with_kvcache_prefill_bf16(
                q=inputs.q.view(-1, NUM_Q_HEADS, HEAD_DIM),
                kcache=inputs.k_cache,
                vcache=inputs.v_cache,
                cu_seqlens_q=inputs.cu_seqlens_q,
                block_ids=inputs.block_table,
                seqlens_kvcache=inputs.seq_lens,
                max_seqlens_q=inputs.max_seqlens_q,
                output=o,
            )

    return benchmark_kernel(fn, repeats=repeats)


def bench_triton(inputs: AttentionInputs, repeats: int) -> float:
    """Benchmark Triton attention backend."""
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd,
    )
    from sglang.srt.layers.attention.triton_ops.extend_attention import (
        extend_attention_fwd,
    )

    # KV buffer in 3D NHD format for triton
    k_buffer = inputs.k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)
    v_buffer = inputs.v_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)

    if inputs.is_decode:
        # Decode path
        # Compute max_kv_splits based on max sequence length (split_tile=256)
        max_seq_len = int(inputs.seq_lens.max().item())
        split_tile_size = 256
        max_kv_splits = max(1, (max_seq_len + split_tile_size - 1) // split_tile_size)
        # Cap at 128 to limit buffer size
        max_kv_splits = min(max_kv_splits, 128)
        num_kv_splits = torch.full(
            (inputs.batch_size,), max_kv_splits, dtype=torch.int32, device=DEVICE,
        )

        # attn_logits: [bs, num_heads, max_kv_splits, v_head_dim]
        # attn_lse: [bs, num_heads, max_kv_splits]
        attn_logits = torch.empty(
            inputs.batch_size, NUM_Q_HEADS, max_kv_splits, HEAD_DIM,
            dtype=torch.float32, device=DEVICE,
        )
        attn_lse = torch.empty(
            inputs.batch_size, NUM_Q_HEADS, max_kv_splits,
            dtype=torch.float32, device=DEVICE,
        )
        o = torch.empty(
            inputs.batch_size, NUM_Q_HEADS, HEAD_DIM,
            dtype=DTYPE, device=DEVICE,
        )

        q_3d = inputs.q.view(inputs.batch_size, NUM_Q_HEADS, HEAD_DIM)

        def fn():
            decode_attention_fwd(
                q=q_3d,
                k_buffer=k_buffer,
                v_buffer=v_buffer,
                o=o,
                kv_indptr=inputs.triton_kv_indptr,
                kv_indices=inputs.triton_kv_indices,
                attn_logits=attn_logits,
                attn_lse=attn_lse,
                num_kv_splits=num_kv_splits,
                max_kv_splits=max_kv_splits,
                sm_scale=SM_SCALE,
                k_scale=1.0,
                v_scale=1.0,
                page_size=PAGE_SIZE,
            )
    else:
        # Extend/prefill path
        o = torch.empty(
            inputs.total_q_tokens, NUM_Q_HEADS, HEAD_DIM,
            dtype=DTYPE, device=DEVICE,
        )
        max_extend_len = inputs.max_seqlens_q

        def fn():
            extend_attention_fwd(
                q_extend=inputs.q.view(-1, NUM_Q_HEADS, HEAD_DIM),
                k_extend=inputs.k_new,
                v_extend=inputs.v_new,
                o_extend=o,
                k_buffer=k_buffer,
                v_buffer=v_buffer,
                qo_indptr=inputs.cu_seqlens_q,
                kv_indptr=inputs.triton_kv_indptr,
                kv_indices=inputs.triton_kv_indices,
                custom_mask=None,
                is_causal=True,
                mask_indptr=None,
                max_len_extend=max_extend_len,
                k_scale=1.0,
                v_scale=1.0,
                sm_scale=SM_SCALE,
                page_size=PAGE_SIZE,
            )

    return benchmark_kernel(fn, repeats=repeats)


def bench_flashinfer(inputs: AttentionInputs, repeats: int) -> float:
    """Benchmark FlashInfer attention backend."""
    import flashinfer

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=DEVICE)

    # Paged KV cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    # FlashInfer expects interleaved K/V stacked along dim 1
    paged_kv_cache = torch.stack(
        [inputs.k_cache, inputs.v_cache], dim=1
    ).contiguous()  # [num_pages, 2, page_size, num_kv_heads, head_dim]

    # Use begin_forward (newer API) or plan (older API).
    # Decode wrapper: accepts data_type=, q_data_type=
    # Prefill wrapper: accepts kv_data_type=, q_data_type= (NOT data_type=)
    def _call_plan(wrapper, *args, **kwargs):
        for method_name in ("begin_forward", "plan"):
            method = getattr(wrapper, method_name, None)
            if method is None:
                continue
            try:
                method(*args, **kwargs)
                return
            except TypeError:
                continue
        raise RuntimeError("FlashInfer wrapper plan/begin_forward failed")

    if inputs.is_decode:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        _call_plan(
            wrapper,
            inputs.fi_kv_indptr,
            inputs.fi_kv_indices,
            inputs.fi_kv_last_page_len,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            data_type=DTYPE,
            q_data_type=DTYPE,
        )

        def fn():
            return wrapper.forward(
                inputs.q.view(inputs.batch_size, NUM_Q_HEADS, HEAD_DIM),
                paged_kv_cache,
                sm_scale=SM_SCALE,
            )
    else:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        # Prefill wrapper uses kv_data_type instead of data_type
        _call_plan(
            wrapper,
            inputs.cu_seqlens_q,
            inputs.fi_kv_indptr,
            inputs.fi_kv_indices,
            inputs.fi_kv_last_page_len,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )

        def fn():
            return wrapper.forward(
                inputs.q.view(-1, NUM_Q_HEADS, HEAD_DIM),
                paged_kv_cache,
                causal=True,
                sm_scale=SM_SCALE,
            )

    return benchmark_kernel(fn, repeats=repeats)


def bench_fa3(inputs: AttentionInputs, repeats: int) -> float:
    """Benchmark FlashAttention v3 backend."""
    try:
        from sgl_kernel.flash_attn import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
        )
    except ImportError:
        try:
            from flash_attn import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
            )
        except ImportError:
            print("  [SKIP] flash_attn not available")
            return float("inf")

    # FA3 uses [num_pages, page_size, num_kv_heads, head_dim] for k_cache/v_cache
    k_cache = inputs.k_cache.contiguous()
    v_cache = inputs.v_cache.contiguous()

    if inputs.is_decode:
        # Decode path: flash_attn_with_kvcache
        # FA3 expects q: [batch_size, seqlen_q, num_heads, head_size] (4D)
        q_4d = inputs.q.view(inputs.batch_size, 1, NUM_Q_HEADS, HEAD_DIM)
        # FA3 expects page_table as [bs, max_pages] int32
        # and cache_seqlens as [bs] int32
        cache_seqlens = inputs.seq_lens

        def fn():
            flash_attn_with_kvcache(
                q=q_4d,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=inputs.block_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=None,
                max_seqlen_q=1,
                softmax_scale=SM_SCALE,
                causal=True,
            )
    else:
        # Prefill/extend path
        o = torch.empty(
            inputs.total_q_tokens, NUM_Q_HEADS, HEAD_DIM,
            dtype=DTYPE, device=DEVICE,
        )
        q_3d = inputs.q.view(-1, NUM_Q_HEADS, HEAD_DIM)

        # For pure prefill (no cached KV), use flash_attn_varlen_func
        # For extend, use flash_attn_with_kvcache with cu_seqlens_q
        if inputs.cu_seqlens_q is not None and inputs.seq_lens.max().item() > inputs.max_seqlens_q:
            # Extend path
            def fn():
                flash_attn_with_kvcache(
                    q=q_3d,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    page_table=inputs.block_table,
                    cache_seqlens=inputs.seq_lens,
                    cu_seqlens_q=inputs.cu_seqlens_q,
                    max_seqlen_q=inputs.max_seqlens_q,
                    softmax_scale=SM_SCALE,
                    causal=True,
                )
        else:
            # Pure prefill path
            cu_seqlens_k = inputs.cu_seqlens_q.clone()

            def fn():
                flash_attn_varlen_func(
                    q=q_3d,
                    k=inputs.k_new,
                    v=inputs.v_new,
                    cu_seqlens_q=inputs.cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=inputs.max_seqlens_q,
                    max_seqlen_k=inputs.max_seqlens_q,
                    softmax_scale=SM_SCALE,
                    causal=True,
                )

    return benchmark_kernel(fn, repeats=repeats)


def bench_trtllm_mha(inputs: AttentionInputs, repeats: int) -> float:
    """Benchmark TensorRT-LLM MHA backend."""
    try:
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
    except ImportError:
        print("  [SKIP] trtllm_mha not available")
        return float("inf")

    print("  [SKIP] trtllm_mha requires full server context (not standalone benchmarkable)")
    return float("inf")


# ── Backend registry ────────────────────────────────────────────────────────

BACKENDS = {
    "hpc": bench_hpc,
    "triton": bench_triton,
    "flashinfer": bench_flashinfer,
    "fa3": bench_fa3,
    "trtllm_mha": bench_trtllm_mha,
}

BACKEND_DISPLAY = {
    "hpc": "HPC_ATTN",
    "triton": "TRITON_ATTN",
    "flashinfer": "FLASHINFER",
    "fa3": "FLASH_ATTN",
    "trtllm_mha": "TRTLLM_MHA",
}


# ── Table generation ────────────────────────────────────────────────────────

def format_table(results: Dict[str, Dict[str, float]], specs: List[BatchSpec],
                 backend_names: List[str]) -> str:
    """Generate a comparison table like the vLLM PR."""
    display_names = [BACKEND_DISPLAY.get(b, b) for b in backend_names]

    # Build all rows first, then compute column widths dynamically
    # Each backend gets two columns: "Time (s)" and "vs Best"
    col_labels = []
    for name in display_names:
        col_labels.append(f"{name} Time (s)")
        col_labels.append("vs Best")

    # Collect row data
    row_data = []
    for spec in specs:
        times = []
        for b in backend_names:
            t = results.get(spec.name, {}).get(b, float("inf"))
            times.append(t)
        best = min(times)

        row = [spec.name, spec.batch_type, str(spec.batch_size)]
        for t in times:
            if t == float("inf"):
                row.append("N/A")
                row.append("N/A")
            else:
                pct = (t / best) * 100 if best > 0 else 0
                row.append(f"{t:.6f}")
                row.append(f"{pct:.1f}%")
        row_data.append(row)

    # Column headers
    headers = ["Batch Spec", "Type", "Batch Size"] + col_labels

    # Compute column widths from headers and data
    num_cols = len(headers)
    col_widths = [0] * num_cols
    for i in range(num_cols):
        col_widths[i] = len(headers[i])
        for row in row_data:
            if i < len(row):
                col_widths[i] = max(col_widths[i], len(row[i]))

    # Pad width = max content width
    def fmt_cell(text, width, align="left"):
        if align == "right":
            return f" {text:>{width}} "
        if align == "center":
            total = width - len(text)
            left = total // 2
            right = total - left
            return f" {' ' * left}{text}{' ' * right} "
        return f" {text:<{width}} "

    # Build table
    lines = []

    # Header row
    header_cells = []
    for i, h in enumerate(headers):
        header_cells.append(fmt_cell(h, col_widths[i], "center"))
    lines.append("|" + "|".join(header_cells) + "|")

    # Separator row
    sep_cells = []
    for i in range(num_cols):
        sep_cells.append("-" * (col_widths[i] + 2))
    lines.append("|" + "|".join(sep_cells) + "|")

    # Data rows
    for row in row_data:
        cells = []
        for i in range(num_cols):
            val = row[i] if i < len(row) else ""
            align = "center" if i >= 3 else "left"
            cells.append(fmt_cell(val, col_widths[i], align))
        lines.append("|" + "|".join(cells) + "|")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention backends (HPC vs Triton vs FlashInfer vs FA3)"
    )
    parser.add_argument(
        "--backends", nargs="+", default=["hpc", "triton", "flashinfer", "fa3"],
        choices=list(BACKENDS.keys()),
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="Number of timing iterations per benchmark",
    )
    parser.add_argument(
        "--specs", nargs="+", default=None,
        help="Specific batch specs to run (default: all)",
    )
    args = parser.parse_args()

    print(f"Attention Backend Benchmark")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  Config: head_dim={HEAD_DIM}, num_q_heads={NUM_Q_HEADS}, "
          f"num_kv_heads={NUM_KV_HEADS}, page_size={PAGE_SIZE}, dtype={DTYPE}")
    print(f"  Backends: {', '.join(args.backends)}")
    print(f"  Warmup: {WARMUP_ITERS} iters, Measure: {args.repeats} iters")
    print()

    specs = parse_batch_specs()
    if args.specs:
        specs = [s for s in specs if s.name in args.specs]

    results: Dict[str, Dict[str, float]] = {}

    for spec in specs:
        print(f"Preparing {spec.name} ({spec.batch_type}, bs={spec.batch_size})...", end=" ", flush=True)
        inputs = prepare_inputs(spec)
        print(f"q_tokens={inputs.total_q_tokens}")

        results[spec.name] = {}
        for backend in args.backends:
            display = BACKEND_DISPLAY.get(backend, backend)
            print(f"  Benchmarking {display}...", end=" ", flush=True)
            try:
                t = BACKENDS[backend](inputs, args.repeats)
                results[spec.name][backend] = t
                if t == float("inf"):
                    print("SKIP")
                else:
                    print(f"{t:.6f}s")
            except Exception as e:
                results[spec.name][backend] = float("inf")
                print(f"ERROR: {e}")

        # Free memory
        del inputs
        torch.cuda.empty_cache()

    print()
    print("=" * 120)
    print("Results:")
    print("=" * 120)
    print()
    table = format_table(results, specs, args.backends)
    print(table)
    print()


if __name__ == "__main__":
    main()
