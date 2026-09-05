#!/usr/bin/env python3
"""Microbenchmark the two DeepSeek-V4-Flash prefill attention paths.

This benchmark intentionally excludes model, CP/NCCL, PD transfer and scheduler
time. It models one rank of a CP-v2 interleave prefill, then compares the
production-shaped calls of:

* ``DeepseekV4AttnBackend._forward_prefill_sparse`` (including FP8-cache
  dequantization, BF16 workspace construction and sparse attention), and
* ``flash_mla_with_kvcache`` over the same packed FP8 KV cache.

The default workload matches the serving benchmark: a 61,440-token cached
prefix plus a 4,096-token extend, CP2, and batch sizes 8/16/24/32. C0/C4/C128
are reported separately and, when all three are requested, combined using the
DeepSeek-V4-Flash 3/20/20-layer weights.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

# Prefer the checkout containing this script; the standalone copy under
# /upfs/abing/sglang/deepseek-v4 falls back to the explicitly configured repo.
_SCRIPT_REPO = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(os.environ.get("SGLANG_REPO_ROOT", _SCRIPT_REPO))
if not (_REPO_ROOT / "python" / "sglang").is_dir():
    _REPO_ROOT = Path("/upfs/abing/sglang/sglang")
sys.path.insert(0, str(_REPO_ROOT / "python"))

import torch  # noqa: E402

from sglang.kernels.ops.attention.dsv4.index_buf_accessor import SetKAndS  # noqa: E402
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (  # noqa: E402
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.deepseek_v4_backend import (  # noqa: E402
    DeepseekV4AttnBackend,
)
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (  # noqa: E402
    SparsePrefillWorkspace,
)
from sglang.srt.layers.cp.base import init_cp_strategy  # noqa: E402
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel  # noqa: E402

DSV4_HEAD_DIM = 512
DSV4_HEAD_DIM_V = 512
DSV4_SWA_WINDOW = 128
DSV4_PAGE_SIZE = 256
DSV4_C4_TOPK = 512
PACKED_BYTES_PER_TOKEN = 448 + 64 * 2 + 8
FLASHMLA_TMA_K_STRIDE = 576
DEFAULT_LAYER_WEIGHTS = {0: 3, 4: 20, 128: 20}


class _PoolLayout:
    def __init__(self, page_size: int):
        self.page_size = page_size


class _TokenToKVPool:
    """Minimum DeepSeekV4TokenToKVPool surface used by sparse prefill."""

    def __init__(
        self,
        *,
        swa_cache: torch.Tensor,
        extra_cache: torch.Tensor,
        full_to_swa: torch.Tensor,
        swa_page_size: int,
        extra_page_size: int,
    ) -> None:
        self._swa_cache = swa_cache
        self._extra_cache = extra_cache
        self.full_to_swa_index_mapping = full_to_swa
        self.swa_window_size = swa_page_size
        self._extra_page_size = extra_page_size

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self._swa_cache

    def get_extra_key_page_size(self, layer_id: int) -> int:
        del layer_id
        return self._extra_page_size

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self._extra_cache


@dataclass
class BenchmarkCase:
    prefix_len: int
    extend_len: int
    context_len: int
    batch_size: int
    cp_size: int
    cp_rank: int
    local_q_tokens: int
    compress_ratio: int
    backend: DeepseekV4AttnBackend
    forward_batch: ForwardBatch
    token_pool: _TokenToKVPool
    core_metadata: SimpleNamespace
    q: torch.Tensor
    attn_sink: torch.Tensor
    dense_k_cache: torch.Tensor
    dense_indices: torch.Tensor
    dense_topk_length: torch.Tensor
    dense_extra_k_cache: torch.Tensor | None
    dense_extra_indices: torch.Tensor | None
    dense_extra_topk_length: torch.Tensor | None


@dataclass
class Timing:
    mean_ms: float
    median_ms: float
    p10_ms: float
    p90_ms: float
    minimum_ms: float
    maximum_ms: float


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _make_packed_cache(
    *,
    total_slots: int,
    page_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    total_slots = max(total_slots, 1)
    num_pages = math.ceil(total_slots / page_size)
    padded_slots = num_pages * page_size
    page_bytes = page_size * PACKED_BYTES_PER_TOKEN
    padded_page_bytes = _align_up(page_bytes, FLASHMLA_TMA_K_STRIDE)
    raw_cache = torch.zeros(
        num_pages,
        padded_page_bytes,
        dtype=torch.uint8,
        device=device,
    )
    k_bf16 = (
        torch.randn(
            padded_slots,
            DSV4_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.05
    )
    packed = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16)
    locations = torch.arange(padded_slots, dtype=torch.int32, device=device)
    SetKAndS.torch(_PoolLayout(page_size), raw_cache, locations, packed)
    return raw_cache


def _make_local_query_layout(
    *,
    prefix_len: int,
    extend_len: int,
    batch_size: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global causal positions and request ids owned by one CP rank."""
    global_q = extend_len * batch_size
    flat_query = torch.arange(global_q, dtype=torch.int64, device=device)
    owned = flat_query.remainder(cp_size) == cp_rank
    owned_query = flat_query[owned]
    request_ids = torch.div(owned_query, extend_len, rounding_mode="floor")
    query_positions = prefix_len + owned_query.remainder(extend_len)
    return query_positions.to(torch.int32), request_ids.to(torch.int32)


def _make_forward_batch(
    *,
    prefix_len: int,
    extend_len: int,
    batch_size: int,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> tuple[ForwardBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    context_len = prefix_len + extend_len
    global_q = extend_len * batch_size
    seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
    extend_seq_lens = torch.full(
        (batch_size,), extend_len, dtype=torch.int32, device=device
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    req_to_token = torch.arange(
        batch_size * context_len, dtype=torch.int32, device=device
    ).view(batch_size, context_len)
    global_positions = torch.arange(
        prefix_len, context_len, dtype=torch.int32, device=device
    ).repeat(batch_size)
    local_positions, local_request_ids = _make_local_query_layout(
        prefix_len=prefix_len,
        extend_len=extend_len,
        batch_size=batch_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=device,
    )
    extend_start_loc = (
        torch.arange(batch_size, dtype=torch.int32, device=device) * extend_len
    )
    request_base = (
        torch.arange(batch_size, dtype=torch.int64, device=device).repeat_interleave(
            extend_len
        )
        * context_len
    )
    extend_offset = torch.arange(
        prefix_len, context_len, dtype=torch.int64, device=device
    ).repeat(batch_size)
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=torch.zeros(global_q, dtype=torch.int64, device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        out_cache_loc=request_base + extend_offset,
        seq_lens_sum=batch_size * context_len,
        positions=global_positions.to(torch.int64),
        extend_num_tokens=global_q,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=[extend_len] * batch_size,
        extend_start_loc=extend_start_loc,
        extend_prefix_lens=torch.zeros(
            batch_size, dtype=torch.int32, device=device
        ).fill_(prefix_len),
        extend_prefix_lens_cpu=[prefix_len] * batch_size,
    )
    strategy = InterleaveCPStrategy(cp_size=cp_size)
    batch.attn_cp_metadata = strategy.build_metadata(
        num_tokens=global_q,
        seqs_len=[context_len] * batch_size,
        extend_seqs_len=[extend_len] * batch_size,
    )
    return batch, req_to_token, local_positions, local_request_ids


def _make_swa_indices(
    *,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    context_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    request_base = request_ids * context_len
    offsets = (
        query_positions[:, None]
        - torch.arange(DSV4_SWA_WINDOW, dtype=torch.int32, device=device)[None, :]
    )
    valid = offsets >= 0
    physical = request_base[:, None] + offsets.clamp_min(0)
    indices = torch.where(valid, physical, torch.full_like(physical, -1))
    topk_length = (query_positions + 1).clamp(max=DSV4_SWA_WINDOW)
    return indices.unsqueeze(1).contiguous(), topk_length.contiguous()


def _make_c4_metadata(
    *,
    context_len: int,
    batch_size: int,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    c4_per_request = max(context_len // 4, 1)
    page_size = DSV4_PAGE_SIZE // 4
    pages_per_request = math.ceil(c4_per_request / page_size)

    page_ids = (
        torch.arange(pages_per_request, dtype=torch.int32, device=device)[None, :]
        + torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
        * pages_per_request
    )
    page_table = page_ids.index_select(0, request_ids.long()).contiguous()

    raw = torch.arange(DSV4_C4_TOPK, dtype=torch.int32, device=device)[None, :]
    raw = raw.expand(query_positions.shape[0], -1).clone()
    available = ((query_positions + 1) // 4).clamp(max=DSV4_C4_TOPK)
    valid = raw < available[:, None]
    raw.masked_fill_(~valid, -1)

    physical_base = request_ids[:, None] * pages_per_request * page_size
    physical = torch.where(valid, physical_base + raw.clamp_min(0), -1)
    topk_length = available.clamp(min=1)
    total_slots = batch_size * pages_per_request * page_size
    return (
        page_table,
        raw.contiguous(),
        physical.unsqueeze(1).contiguous(),
        topk_length.contiguous(),
        total_slots,
    )


def _make_c128_metadata(
    *,
    context_len: int,
    batch_size: int,
    query_positions: torch.Tensor,
    request_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    c128_per_request = max(context_len // 128, 1)
    width = _align_up(c128_per_request, 64)
    local = torch.arange(width, dtype=torch.int32, device=device)[None, :].expand(
        query_positions.shape[0], -1
    )
    available = ((query_positions + 1) // 128).clamp(min=1, max=c128_per_request)
    valid = local < available[:, None]
    physical = request_ids[:, None] * c128_per_request + local
    physical = torch.where(valid, physical, torch.full_like(physical, -1))
    total_slots = batch_size * c128_per_request
    return (
        physical.contiguous(),
        available.contiguous(),
        total_slots,
    )


def build_case(
    *,
    prefix_len: int,
    extend_len: int,
    batch_size: int,
    cp_size: int,
    cp_rank: int,
    compress_ratio: int,
    num_heads: int,
    device: torch.device,
    seed: int,
) -> BenchmarkCase:
    if compress_ratio not in (0, 4, 128):
        raise ValueError(f"Unsupported compress ratio: {compress_ratio}")
    context_len = prefix_len + extend_len
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + context_len * 17 + batch_size * 31 + compress_ratio)

    forward_batch, req_to_token, query_positions, request_ids = _make_forward_batch(
        prefix_len=prefix_len,
        extend_len=extend_len,
        batch_size=batch_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=device,
    )
    total_q = query_positions.shape[0]
    total_context_slots = context_len * batch_size
    swa_cache_raw = _make_packed_cache(
        total_slots=total_context_slots,
        page_size=DSV4_PAGE_SIZE,
        device=device,
        generator=generator,
    )
    dense_k_cache = swa_cache_raw[:, : DSV4_PAGE_SIZE * PACKED_BYTES_PER_TOKEN].view(
        swa_cache_raw.shape[0],
        DSV4_PAGE_SIZE,
        1,
        PACKED_BYTES_PER_TOKEN,
    )
    full_to_swa = torch.arange(total_context_slots, dtype=torch.int64, device=device)
    dense_indices, dense_topk_length = _make_swa_indices(
        query_positions=query_positions,
        request_ids=request_ids,
        context_len=context_len,
        device=device,
    )

    page_table = None
    c4_raw_indices = None
    c128_page_indices = None
    dense_extra_indices = None
    dense_extra_topk_length = None
    dense_extra_k_cache = None
    extra_page_size = DSV4_PAGE_SIZE
    extra_cache_raw = swa_cache_raw

    if compress_ratio == 4:
        (
            page_table,
            c4_raw_indices,
            dense_extra_indices,
            dense_extra_topk_length,
            extra_slots,
        ) = _make_c4_metadata(
            context_len=context_len,
            batch_size=batch_size,
            query_positions=query_positions,
            request_ids=request_ids,
            device=device,
        )
        extra_page_size = DSV4_PAGE_SIZE // 4
        extra_cache_raw = _make_packed_cache(
            total_slots=extra_slots,
            page_size=extra_page_size,
            device=device,
            generator=generator,
        )
    elif compress_ratio == 128:
        (
            c128_page_indices,
            dense_extra_topk_length,
            extra_slots,
        ) = _make_c128_metadata(
            context_len=context_len,
            batch_size=batch_size,
            query_positions=query_positions,
            request_ids=request_ids,
            device=device,
        )
        dense_extra_indices = c128_page_indices.unsqueeze(1)
        extra_page_size = DSV4_PAGE_SIZE // 128
        extra_cache_raw = _make_packed_cache(
            total_slots=extra_slots,
            page_size=extra_page_size,
            device=device,
            generator=generator,
        )

    if compress_ratio != 0:
        dense_extra_k_cache = extra_cache_raw[
            :, : extra_page_size * PACKED_BYTES_PER_TOKEN
        ].view(
            extra_cache_raw.shape[0],
            extra_page_size,
            1,
            PACKED_BYTES_PER_TOKEN,
        )

    token_pool = _TokenToKVPool(
        swa_cache=swa_cache_raw,
        extra_cache=extra_cache_raw,
        full_to_swa=full_to_swa,
        swa_page_size=DSV4_PAGE_SIZE,
        extra_page_size=extra_page_size,
    )
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(sparse_prefill_cache=None)
    backend.req_to_token = req_to_token
    backend.sparse_prefill_workspace = SparsePrefillWorkspace(device)
    backend.softmax_scale = DSV4_HEAD_DIM**-0.5
    backend.head_dim_v = DSV4_HEAD_DIM_V
    backend.dsv4_prefill_backend = "flashmla_sparse"

    q = (
        torch.randn(
            total_q,
            1,
            num_heads,
            DSV4_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.05
    ).contiguous()
    attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)
    core_metadata = SimpleNamespace(
        positions_casual=query_positions,
        page_table=page_table,
        c4_sparse_raw_indices=c4_raw_indices,
        c128_page_indices=c128_page_indices,
    )
    return BenchmarkCase(
        prefix_len=prefix_len,
        extend_len=extend_len,
        context_len=context_len,
        batch_size=batch_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        local_q_tokens=total_q,
        compress_ratio=compress_ratio,
        backend=backend,
        forward_batch=forward_batch,
        token_pool=token_pool,
        core_metadata=core_metadata,
        q=q,
        attn_sink=attn_sink,
        dense_k_cache=dense_k_cache,
        dense_indices=dense_indices,
        dense_topk_length=dense_topk_length,
        dense_extra_k_cache=dense_extra_k_cache,
        dense_extra_indices=dense_extra_indices,
        dense_extra_topk_length=dense_extra_topk_length,
    )


def _make_sparse_call(case: BenchmarkCase) -> Callable[[], torch.Tensor]:
    def call() -> torch.Tensor:
        return case.backend._forward_prefill_sparse(
            q=case.q,
            layer_id=0,
            compress_ratio=case.compress_ratio,
            forward_batch=case.forward_batch,
            token_to_kv_pool=case.token_pool,
            core_attn_metadata=case.core_metadata,
            attn_sink=case.attn_sink,
        )

    return call


def _make_dense_call(case: BenchmarkCase) -> Callable[[], torch.Tensor]:
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

    scheduler_metadata = get_mla_metadata()[0]

    def call() -> torch.Tensor:
        return flash_mla_with_kvcache(
            q=case.q,
            k_cache=case.dense_k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=DSV4_HEAD_DIM_V,
            tile_scheduler_metadata=scheduler_metadata,
            softmax_scale=DSV4_HEAD_DIM**-0.5,
            is_fp8_kvcache=True,
            indices=case.dense_indices,
            topk_length=case.dense_topk_length,
            attn_sink=case.attn_sink,
            extra_k_cache=case.dense_extra_k_cache,
            extra_indices_in_kvcache=case.dense_extra_indices,
            extra_topk_length=case.dense_extra_topk_length,
        )[0]

    return call


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


@torch.inference_mode()
def benchmark_cuda(
    fn: Callable[[], torch.Tensor], *, warmup: int, repeats: int
) -> tuple[Timing, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        output = fn()
        end.record()
    torch.cuda.synchronize()
    elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    assert output is not None
    return (
        Timing(
            mean_ms=statistics.mean(elapsed),
            median_ms=statistics.median(elapsed),
            p10_ms=_percentile(elapsed, 0.10),
            p90_ms=_percentile(elapsed, 0.90),
            minimum_ms=min(elapsed),
            maximum_ms=max(elapsed),
        ),
        output,
    )


def _format_row(row: dict[str, object]) -> str:
    return (
        f"prefix={row['prefix_len']:>6} extend={row['extend_len']:>5} "
        f"batch={row['batch_size']:>2} local_q={row['local_q_tokens']:>7} "
        f"C{row['compress_ratio']:<3}  "
        f"sparse={row['sparse_median_ms']:>8.3f} ms  "
        f"dense={row['dense_median_ms']:>8.3f} ms  "
        f"dense/sparse={row['dense_over_sparse']:>6.3f}x  "
        f"winner={row['winner']}"
    )


def _weighted_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_shape: dict[tuple[int, int, int, int, int], dict[int, dict[str, object]]] = {}
    for row in rows:
        key = (
            int(row["prefix_len"]),
            int(row["extend_len"]),
            int(row["batch_size"]),
            int(row["cp_size"]),
            int(row["cp_rank"]),
        )
        by_shape.setdefault(key, {})[int(row["compress_ratio"])] = row

    weighted = []
    for shape, ratio_rows in sorted(by_shape.items()):
        if not all(ratio in ratio_rows for ratio in DEFAULT_LAYER_WEIGHTS):
            continue
        prefix_len, extend_len, batch_size, cp_size, cp_rank = shape
        sparse_ms = sum(
            float(ratio_rows[ratio]["sparse_median_ms"]) * weight
            for ratio, weight in DEFAULT_LAYER_WEIGHTS.items()
        )
        dense_ms = sum(
            float(ratio_rows[ratio]["dense_median_ms"]) * weight
            for ratio, weight in DEFAULT_LAYER_WEIGHTS.items()
        )
        weighted.append(
            {
                "prefix_len": prefix_len,
                "extend_len": extend_len,
                "batch_size": batch_size,
                "cp_size": cp_size,
                "cp_rank": cp_rank,
                "sparse_ms": sparse_ms,
                "dense_ms": dense_ms,
                "dense_over_sparse": dense_ms / sparse_ms,
                "winner": "sparse" if sparse_ms < dense_ms else "dense",
            }
        )
    return weighted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-len", type=int, default=0)
    parser.add_argument(
        "--extend-len",
        "--extend-lens",
        dest="extend_lens",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="One or more extend lengths to benchmark (default: 4096).",
    )
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--compress-ratios", type=int, nargs="+", default=[0, 4, 128])
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare a small output sample after timing (not included in timing).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to a timestamped file in the current directory.",
    )
    return parser.parse_args()


def _main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    extend_lens = sorted(set(args.extend_lens))
    batch_sizes = sorted(set(args.batch_sizes))
    if args.prefix_len < 0 or any(extend_len <= 0 for extend_len in extend_lens):
        raise SystemExit(
            "--prefix-len must be non-negative and every --extend-len positive"
        )
    if args.cp_size <= 1 or not 0 <= args.cp_rank < args.cp_size:
        raise SystemExit("--cp-size must be >1 and --cp-rank must be in range")
    if args.num_heads <= 0 or any(batch <= 0 for batch in args.batch_sizes):
        raise SystemExit("--batch-sizes and --num-heads must be positive")
    if any(
        batch_size * extend_len % args.cp_size
        for extend_len in extend_lens
        for batch_size in batch_sizes
    ):
        raise SystemExit(
            "Every batch_size * extend_len pair must be divisible by cp_size so the "
            "single-rank benchmark has production-shaped equal CP padding"
        )
    if args.warmup < 1 or args.repeats < 1:
        raise SystemExit("--warmup and --repeats must be positive")
    invalid_ratios = set(args.compress_ratios) - {0, 4, 128}
    if invalid_ratios:
        raise SystemExit(f"Unsupported compress ratios: {sorted(invalid_ratios)}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    csv_path = args.csv or Path(
        f"dsv4-prefill-cpv2-functions-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    )

    print("DeepSeek-V4-Flash prefill function benchmark")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(
        f"prefix={args.prefix_len}, extends={extend_lens}, "
        f"contexts={[args.prefix_len + extend_len for extend_len in extend_lens]}, "
        f"cp={args.cp_size}, rank={args.cp_rank}"
    )
    print(
        f"batches={batch_sizes}, heads={args.num_heads}, "
        f"warmup={args.warmup}, repeats={args.repeats}"
    )
    print("Q/indices are CP-v2 interleave rank-local; KV capacity is global-context.")
    print("sparse timing includes dequant/workspace/index-combine; CP/NCCL is excluded")

    rows: list[dict[str, object]] = []
    case_shapes = [
        (extend_len, batch_size)
        for extend_len in extend_lens
        for batch_size in batch_sizes
    ]
    for shape_index, (extend_len, batch_size) in enumerate(case_shapes):
        for ratio_index, compress_ratio in enumerate(args.compress_ratios):
            try:
                case = build_case(
                    prefix_len=args.prefix_len,
                    extend_len=extend_len,
                    batch_size=batch_size,
                    cp_size=args.cp_size,
                    cp_rank=args.cp_rank,
                    compress_ratio=compress_ratio,
                    num_heads=args.num_heads,
                    device=device,
                    seed=args.seed,
                )
                sparse_call = _make_sparse_call(case)
                dense_call = _make_dense_call(case)

                # Alternate the first measured path by batch size to reduce
                # persistent clock/thermal ordering bias.
                if (shape_index + ratio_index) % 2 == 0:
                    sparse_timing, sparse_out = benchmark_cuda(
                        sparse_call, warmup=args.warmup, repeats=args.repeats
                    )
                    dense_timing, dense_out = benchmark_cuda(
                        dense_call, warmup=args.warmup, repeats=args.repeats
                    )
                else:
                    dense_timing, dense_out = benchmark_cuda(
                        dense_call, warmup=args.warmup, repeats=args.repeats
                    )
                    sparse_timing, sparse_out = benchmark_cuda(
                        sparse_call, warmup=args.warmup, repeats=args.repeats
                    )

                if args.check:
                    q_rows = torch.linspace(
                        0,
                        sparse_out.shape[0] - 1,
                        steps=min(16, sparse_out.shape[0]),
                        device=device,
                    ).long()
                    torch.testing.assert_close(
                        sparse_out.index_select(0, q_rows)[:, :2, :32].float(),
                        dense_out.squeeze(1)
                        .index_select(0, q_rows)[:, :2, :32]
                        .float(),
                        atol=0.2,
                        rtol=0.2,
                    )

                speedup = dense_timing.median_ms / sparse_timing.median_ms
                row: dict[str, object] = {
                    "prefix_len": args.prefix_len,
                    "extend_len": extend_len,
                    "context_len": args.prefix_len + extend_len,
                    "batch_size": batch_size,
                    "cp_size": args.cp_size,
                    "cp_rank": args.cp_rank,
                    "global_q_tokens": batch_size * extend_len,
                    "local_q_tokens": case.local_q_tokens,
                    "compress_ratio": compress_ratio,
                    "sparse_mean_ms": sparse_timing.mean_ms,
                    "sparse_median_ms": sparse_timing.median_ms,
                    "sparse_p10_ms": sparse_timing.p10_ms,
                    "sparse_p90_ms": sparse_timing.p90_ms,
                    "dense_mean_ms": dense_timing.mean_ms,
                    "dense_median_ms": dense_timing.median_ms,
                    "dense_p10_ms": dense_timing.p10_ms,
                    "dense_p90_ms": dense_timing.p90_ms,
                    "dense_over_sparse": speedup,
                    "winner": "sparse" if speedup > 1.0 else "dense",
                }
                rows.append(row)
                print(_format_row(row), flush=True)
                del case, sparse_out, dense_out, sparse_call, dense_call
            except torch.OutOfMemoryError as exc:
                print(
                    f"OOM: prefix={args.prefix_len}, extend={extend_len}, "
                    f"batch={batch_size}, "
                    f"C{compress_ratio}: {exc}",
                    flush=True,
                )
            finally:
                torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No benchmark case completed")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nWeighted 43-layer estimate (C0x3 + C4x20 + C128x20):")
    weighted = _weighted_rows(rows)
    for row in weighted:
        print(
            f"prefix={row['prefix_len']:>6} extend={row['extend_len']:>5} "
            f"batch={row['batch_size']:>2}  "
            f"sparse={row['sparse_ms']:>9.3f} ms  "
            f"dense={row['dense_ms']:>9.3f} ms  "
            f"dense/sparse={row['dense_over_sparse']:>6.3f}x  "
            f"winner={row['winner']}"
        )

    print("\nFirst measured sparse-winning batch size:")
    for extend_len in extend_lens:
        print(f"extend={extend_len}:")
        for ratio in args.compress_ratios:
            sparse_wins = [
                int(row["batch_size"])
                for row in rows
                if int(row["extend_len"]) == extend_len
                and int(row["compress_ratio"]) == ratio
                and row["winner"] == "sparse"
            ]
            print(f"  C{ratio}: {min(sparse_wins) if sparse_wins else 'not found'}")
        weighted_wins = [
            int(row["batch_size"])
            for row in weighted
            if int(row["extend_len"]) == extend_len and row["winner"] == "sparse"
        ]
        if any(int(row["extend_len"]) == extend_len for row in weighted):
            print(f"  weighted: {min(weighted_wins) if weighted_wins else 'not found'}")
    print(f"CSV: {csv_path.resolve()}")


def main() -> None:
    args = parse_args()
    # Model CP communication is intentionally excluded, but the attention
    # backend must see the same CP-v2/interleave topology and local coordinates
    # as production. No process group is touched by either measured function.
    with envs.SGLANG_ENABLE_CP_V2.override(True), get_parallel().override(
        world_size=args.cp_size,
        world_rank=args.cp_rank,
        tp_size=args.cp_size,
        tp_rank=args.cp_rank,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_cp_size=args.cp_size,
        attn_cp_rank=args.cp_rank,
    ):
        init_cp_strategy(
            enable_prefill_cp=True,
            cp_size=args.cp_size,
            cp_strategy="interleave",
        )
        if not hasattr(DeepseekV4AttnBackend, "_get_or_build_sparse_prefill_cache"):
            raise SystemExit(
                "This benchmark requires the DSV4 CP-v2 sparse-prefill patch"
            )
        _main(args)


if __name__ == "__main__":
    main()
