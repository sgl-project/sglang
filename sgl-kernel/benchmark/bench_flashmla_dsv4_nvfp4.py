"""Benchmark one DeepSeek-V4 NVFP4 decode layer against the old fallback.

The two providers consume the same quantized caches, physical indices,
top-k lengths, attention sink, and query tensor:

* ``fused`` calls ``dsv4_sparse_decode_fwd_nvfp4`` through its Python wrapper.
* ``head_fallback`` reproduces the fallback at this branch's ``HEAD``: it
  dequantizes the complete padded primary and extra index widths (including
  ``-1`` suffixes) into a persistent BF16 workspace, rebuilds/rebases the
  local indices, and calls ``flash_mla_sparse_fwd`` once.

CUDA-event timing covers each provider's single-layer decode data path; it is
not model/server end-to-end latency. Cache construction/quantization and
persistent workspace allocation are outside the timed region. Fused scheduler
metadata is initialized once before timing and then reused, so its construction
cost is untimed/amortized. The default ``cold`` mode evicts L2 before every
timed provider invocation and is the only primary acceptance mode. ``warm`` is
diagnostic only: multi-kernel providers can contain host-submission gaps, so it
must not be described as pure-kernel latency.

By default, each request models an independent 32K-token original context. Its
primary cache owns one disjoint 256-token physical page with a 128-token SWA
live range, while its extra cache owns
``floor(context / compression_ratio)`` disjoint physical tokens, matching the
production compression rule for non-divisible lengths. Indices are a seeded,
shuffled sample without replacement from each request's range. For C128 at
32K, only 256 of the 1024 index slots are active and the remaining suffix is
``-1``. ``--full-topk`` grows the effective context when necessary to make
every extra index slot active (131072 original tokens for C128).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from sgl_kernel import flashmla_ops
from sgl_kernel.flash_mla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache_dsv4_nvfp4,
)

from sglang.srt.layers.attention.dsv4.nvfp4_k_cache import (
    DSV4_NVFP4_BYTES_PER_TOKEN,
    dequantize_dsv4_nvfp4_k_cache_paged,
    quantize_dsv4_nvfp4_k_cache_into,
)

HEAD_DIM = 512
PRIMARY_PAGE_SIZE = 256
PRIMARY_TOPK = 128
DEFAULT_CONTEXT_TOKENS_PER_REQUEST = 32 * 1024
MIN_FLUSH_BYTES = 256 * 1024 * 1024
FALLBACK_SM90_L2_BYTES = 64 * 1024 * 1024
PRIMARY_GLOBAL_SCALE = 0.625
EXTRA_GLOBAL_SCALE = 1.75
INDEX_PATTERN = (
    "per_request_disjoint_seeded_randperm_without_replacement_"
    "active_prefix_minus_one_suffix"
)
FUSED_PROVIDER = "fused"
FALLBACK_PROVIDER = "head_fallback"
FUSED_PROVIDER_KIND = "fused_nvfp4_splitk_combine"
FALLBACK_PROVIDER_KIND = "head_full_padded_dequant_rebase_flashmla_sparse"
BENCHMARK_SCOPE = "single_layer_decode_data_path_not_model_e2e"
SCHEDULER_METADATA_TIMING = "initialized_before_timing_then_reused_amortized"


@dataclass(frozen=True)
class Variant:
    name: str
    h_q: int
    extra_page_size: int
    extra_topk: int
    extra_compress_ratio: int


VARIANTS = {
    "flash-c4": Variant(
        name="flash-c4",
        h_q=64,
        extra_page_size=64,
        extra_topk=512,
        extra_compress_ratio=4,
    ),
    "flash-c128": Variant(
        name="flash-c128",
        h_q=64,
        extra_page_size=2,
        extra_topk=1024,
        extra_compress_ratio=128,
    ),
    "pro-c4": Variant(
        name="pro-c4",
        h_q=128,
        extra_page_size=64,
        extra_topk=1024,
        extra_compress_ratio=4,
    ),
    "pro-c128": Variant(
        name="pro-c128",
        h_q=128,
        extra_page_size=2,
        extra_topk=1024,
        extra_compress_ratio=128,
    ),
}


@dataclass(frozen=True)
class CaseLayout:
    requested_context_tokens_per_request: int
    context_tokens_per_request: int
    primary_physical_tokens_per_request: int
    primary_active_length: int
    extra_physical_tokens_per_request: int
    extra_active_length: int


@dataclass
class Inputs:
    q: torch.Tensor
    primary_raw: torch.Tensor
    primary_view: torch.Tensor
    primary_scale: torch.Tensor
    primary_indices: torch.Tensor
    primary_lengths: torch.Tensor
    extra_raw: torch.Tensor
    extra_view: torch.Tensor
    extra_scale: torch.Tensor
    extra_indices: torch.Tensor
    extra_lengths: torch.Tensor
    attn_sink: torch.Tensor
    workspace: torch.Tensor


def _resolve_layout(
    variant: Variant,
    context_tokens_per_request: int,
    full_topk: bool,
) -> CaseLayout:
    effective_context = context_tokens_per_request
    if full_topk:
        effective_context = max(
            effective_context,
            variant.extra_topk * variant.extra_compress_ratio,
        )

    # Production uses a 256-token physical SWA page while the logical window
    # exposed to attention is 128 tokens. Keep every request in its own page.
    primary_physical_tokens = PRIMARY_PAGE_SIZE
    primary_active_length = min(PRIMARY_TOPK, effective_context)
    # The compressed pools only materialize complete compression groups.
    extra_physical_tokens = effective_context // variant.extra_compress_ratio
    extra_active_length = min(variant.extra_topk, extra_physical_tokens)
    return CaseLayout(
        requested_context_tokens_per_request=context_tokens_per_request,
        context_tokens_per_request=effective_context,
        primary_physical_tokens_per_request=primary_physical_tokens,
        primary_active_length=primary_active_length,
        extra_physical_tokens_per_request=extra_physical_tokens,
        extra_active_length=extra_active_length,
    )


def _make_disjoint_indices(
    *,
    batch_size: int,
    index_width: int,
    physical_tokens_per_request: int,
    active_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.full(
        (batch_size, 1, index_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for request_idx in range(batch_size):
        selected = torch.randperm(
            physical_tokens_per_request,
            dtype=torch.int32,
            device=device,
            generator=generator,
        )[:active_length]
        indices[request_idx, 0, :active_length] = (
            selected + request_idx * physical_tokens_per_request
        )
    return indices


def _make_cache(
    *,
    page_size: int,
    minimum_tokens: int,
    scale_value: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_pages = math.ceil(minimum_tokens / page_size)
    capacity = num_pages * page_size
    raw = torch.zeros(
        (num_pages, page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    source = (
        torch.randn(
            (capacity, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    scale = torch.tensor([scale_value], dtype=torch.float32, device=device)
    quantize_dsv4_nvfp4_k_cache_into(
        cache_k=source,
        kv_buffer=raw,
        loc=torch.arange(capacity, dtype=torch.int32, device=device),
        page_size=page_size,
        global_scale=scale,
    )
    view = raw.view(num_pages, page_size, 1, DSV4_NVFP4_BYTES_PER_TOKEN)
    return raw, view, scale


def _build_inputs(
    variant: Variant,
    batch_size: int,
    seed: int,
    layout: CaseLayout,
    device: torch.device,
) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    index_generator = torch.Generator(device=device).manual_seed(seed + 1)
    q = (
        torch.randn(
            (batch_size, 1, variant.h_q, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    primary_raw, primary_view, primary_scale = _make_cache(
        page_size=PRIMARY_PAGE_SIZE,
        minimum_tokens=(
            batch_size * layout.primary_physical_tokens_per_request
        ),
        scale_value=PRIMARY_GLOBAL_SCALE,
        generator=generator,
        device=device,
    )
    extra_raw, extra_view, extra_scale = _make_cache(
        page_size=variant.extra_page_size,
        minimum_tokens=batch_size * layout.extra_physical_tokens_per_request,
        scale_value=EXTRA_GLOBAL_SCALE,
        generator=generator,
        device=device,
    )

    primary_indices = _make_disjoint_indices(
        batch_size=batch_size,
        index_width=PRIMARY_TOPK,
        physical_tokens_per_request=layout.primary_physical_tokens_per_request,
        active_length=layout.primary_active_length,
        generator=index_generator,
        device=device,
    )
    extra_indices = _make_disjoint_indices(
        batch_size=batch_size,
        index_width=variant.extra_topk,
        physical_tokens_per_request=layout.extra_physical_tokens_per_request,
        active_length=layout.extra_active_length,
        generator=index_generator,
        device=device,
    )
    primary_lengths = torch.full(
        (batch_size,),
        layout.primary_active_length,
        dtype=torch.int32,
        device=device,
    )
    extra_lengths = torch.full(
        (batch_size,),
        layout.extra_active_length,
        dtype=torch.int32,
        device=device,
    )
    attn_sink = torch.linspace(
        -0.75, 0.25, variant.h_q, dtype=torch.float32, device=device
    )

    # Match HEAD's persistent sparse-prefill workspace: both padded index
    # widths are materialized, including rows addressed by an inactive -1
    # suffix. The provider reuses this allocation on every invocation.
    num_primary = batch_size * PRIMARY_TOPK
    num_extra = batch_size * variant.extra_topk
    workspace = torch.empty(
        (num_primary + num_extra, 1, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )

    return Inputs(
        q=q,
        primary_raw=primary_raw,
        primary_view=primary_view,
        primary_scale=primary_scale,
        primary_indices=primary_indices,
        primary_lengths=primary_lengths,
        extra_raw=extra_raw,
        extra_view=extra_view,
        extra_scale=extra_scale,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=attn_sink,
        workspace=workspace,
    )


def _make_providers(
    inputs: Inputs,
    variant: Variant,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    scheduler = FlashMLASchedMeta()
    sm_scale = HEAD_DIM**-0.5

    def fused() -> torch.Tensor:
        out, _ = flash_mla_with_kvcache_dsv4_nvfp4(
            q=inputs.q,
            k_cache=inputs.primary_view,
            kv_global_scale=inputs.primary_scale,
            indices=inputs.primary_indices,
            topk_length=inputs.primary_lengths,
            attn_sink=inputs.attn_sink,
            tile_scheduler_metadata=scheduler,
            head_dim_v=HEAD_DIM,
            softmax_scale=sm_scale,
            extra_k_cache=inputs.extra_view,
            extra_kv_global_scale=inputs.extra_scale,
            extra_indices_in_kvcache=inputs.extra_indices,
            extra_topk_length=inputs.extra_lengths,
        )
        return out.squeeze(1)

    def head_fallback() -> torch.Tensor:
        # Keep this data path in lockstep with HEAD's
        # DeepSeekV4AttentionBackend._forward_nvfp4_sparse. In particular,
        # dequantize both complete padded widths and rebuild every arange,
        # clamp, where, and local-index rebase on every invocation.
        q_flat = inputs.q.squeeze(1)
        batch_size = q_flat.shape[0]
        swa_indices_2d = inputs.primary_indices.squeeze(1)
        swa_width = swa_indices_2d.shape[-1]
        swa_flat = swa_indices_2d.reshape(-1)
        extra_indices_2d = inputs.extra_indices.squeeze(1)
        extra_width = extra_indices_2d.shape[-1]

        num_swa = swa_flat.numel()
        num_extra = batch_size * extra_width
        workspace = inputs.workspace
        swa_kv = workspace[:num_swa]
        dequantize_dsv4_nvfp4_k_cache_paged(
            inputs.primary_raw,
            swa_flat,
            page_size=PRIMARY_PAGE_SIZE,
            global_scale=inputs.primary_scale,
            out=swa_kv,
        )

        extra_kv = workspace[num_swa : num_swa + num_extra]
        dequantize_dsv4_nvfp4_k_cache_paged(
            inputs.extra_raw,
            extra_indices_2d.reshape(-1),
            page_size=variant.extra_page_size,
            global_scale=inputs.extra_scale,
            out=extra_kv,
        )

        kv = workspace
        total_width = swa_width + extra_width
        positions = torch.arange(
            total_width, dtype=torch.int32, device=inputs.q.device
        ).unsqueeze(0)
        swa_lens = (
            inputs.primary_lengths.reshape(-1)
            .to(torch.int32)
            .clamp(min=0, max=swa_width)
        )
        extra_lens = (
            inputs.extra_lengths.reshape(-1)
            .to(torch.int32)
            .clamp(min=0, max=extra_width)
        )
        total_lens = swa_lens + extra_lens
        batch_offsets = torch.arange(
            batch_size, dtype=torch.int32, device=inputs.q.device
        )[:, None]
        swa_local = batch_offsets * swa_width + positions
        extra_position = positions - swa_lens[:, None]
        extra_local = (
            batch_size * swa_width + batch_offsets * extra_width + extra_position
        )
        local_indices = torch.where(
            positions < swa_lens[:, None], swa_local, extra_local
        )
        local_indices = torch.where(
            positions < total_lens[:, None],
            local_indices,
            torch.zeros_like(local_indices),
        )
        out, _, _ = flash_mla_sparse_fwd(
            q=q_flat,
            kv=kv,
            indices=local_indices.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=HEAD_DIM,
            attn_sink=inputs.attn_sink,
            topk_length=total_lens,
        )
        return out

    return fused, head_fallback


def _device_l2_cache_bytes(device: torch.device) -> tuple[int, str]:
    properties = torch.cuda.get_device_properties(device)
    for attribute in ("L2_cache_size", "l2_cache_size"):
        value = getattr(properties, attribute, None)
        if isinstance(value, int) and value > 0:
            return value, f"torch_device_properties.{attribute}"
    return FALLBACK_SM90_L2_BYTES, "conservative_sm90_64mib_fallback"


def _make_flush_buffer(
    cache_mode: str,
    l2_cache_bytes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if cache_mode == "warm":
        return None
    flush_bytes = max(4 * l2_cache_bytes, MIN_FLUSH_BYTES)
    buffer = torch.empty(flush_bytes, dtype=torch.uint8, device=device)
    buffer.zero_()
    return buffer


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _event_samples_us(
    fn: Callable[[], torch.Tensor],
    iterations: int,
    flush_buffer: torch.Tensor | None,
) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    last_output = None
    for start, end in zip(starts, ends):
        if flush_buffer is not None:
            # The read-modify-write touches every byte. Because it and the
            # event use the current stream, eviction completes before the
            # provider starts, while the start timestamp excludes the flush.
            flush_buffer.add_(1)
        start.record()
        last_output = fn()
        end.record()
    torch.cuda.synchronize()
    del last_output
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _warmup(
    fused: Callable[[], torch.Tensor],
    head_fallback: Callable[[], torch.Tensor],
    iterations: int,
) -> None:
    for _ in range(iterations):
        fused()
    for _ in range(iterations):
        head_fallback()
    torch.cuda.synchronize()


def _benchmark_case(
    *,
    variant: Variant,
    batch_size: int,
    seed: int,
    context_tokens_per_request: int,
    full_topk: bool,
    cache_mode: str,
    warmup_iterations: int,
    iterations: int,
    rounds: int,
    atol: float,
    rtol: float,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    layout = _resolve_layout(variant, context_tokens_per_request, full_topk)
    inputs = _build_inputs(variant, batch_size, seed, layout, device)
    fused, head_fallback = _make_providers(inputs, variant)
    providers = {
        FUSED_PROVIDER: fused,
        FALLBACK_PROVIDER: head_fallback,
    }
    provider_kinds = {
        FUSED_PROVIDER: FUSED_PROVIDER_KIND,
        FALLBACK_PROVIDER: FALLBACK_PROVIDER_KIND,
    }

    # This first pair also initializes lazy Triton/FlashMLA state and the fused
    # scheduler metadata. Its construction is deliberately outside all timing
    # regions and is amortized by reusing the same metadata object thereafter.
    fused_out = fused()
    fallback_out = head_fallback()
    torch.cuda.synchronize()
    diff = (fused_out.float() - fallback_out.float()).abs()
    max_abs_diff = float(diff.max().item())
    max_rel_diff = float(
        (diff / fallback_out.float().abs().clamp_min(1e-6)).max().item()
    )
    torch.testing.assert_close(fused_out, fallback_out, atol=atol, rtol=rtol)

    _warmup(fused, head_fallback, warmup_iterations)
    l2_cache_bytes, l2_cache_size_source = _device_l2_cache_bytes(device)
    flush_buffer = _make_flush_buffer(cache_mode, l2_cache_bytes, device)
    flush_bytes = 0 if flush_buffer is None else flush_buffer.numel()
    round_samples: dict[str, list[list[float]]] = {
        provider: [] for provider in providers
    }
    for round_idx in range(rounds):
        order = (
            (FUSED_PROVIDER, FALLBACK_PROVIDER)
            if round_idx % 2 == 0
            else (
                FALLBACK_PROVIDER,
                FUSED_PROVIDER,
            )
        )
        for provider in order:
            round_samples[provider].append(
                _event_samples_us(providers[provider], iterations, flush_buffer)
            )

    round_medians = {
        provider: [statistics.median(values) for values in provider_rounds]
        for provider, provider_rounds in round_samples.items()
    }
    fused_median_us = statistics.median(round_medians[FUSED_PROVIDER])
    fallback_median_us = statistics.median(round_medians[FALLBACK_PROVIDER])
    fused_round_median_cv = _coefficient_of_variation(
        round_medians[FUSED_PROVIDER]
    )
    fallback_round_median_cv = _coefficient_of_variation(
        round_medians[FALLBACK_PROVIDER]
    )
    summary = {
        "variant": variant.name,
        "batch_size": batch_size,
        "h_q": variant.h_q,
        "benchmark_scope": BENCHMARK_SCOPE,
        "fused_provider": FUSED_PROVIDER,
        "fused_provider_kind": FUSED_PROVIDER_KIND,
        "fallback_provider": FALLBACK_PROVIDER,
        "fallback_provider_kind": FALLBACK_PROVIDER_KIND,
        "fused_scheduler_metadata_timing": SCHEDULER_METADATA_TIMING,
        "requested_context_tokens_per_request": (
            layout.requested_context_tokens_per_request
        ),
        "context_tokens_per_request": layout.context_tokens_per_request,
        "full_topk": full_topk,
        "cache_mode": cache_mode,
        "acceptance_role": (
            "primary" if cache_mode == "cold" else "diagnostic_only"
        ),
        "warm_timing_caveat": (
            "not_applicable"
            if cache_mode == "cold"
            else "may_include_host_submission_gaps_not_pure_kernel"
        ),
        "l2_cache_bytes": l2_cache_bytes,
        "l2_cache_size_source": l2_cache_size_source,
        "flush_bytes": flush_bytes,
        "primary_topk": PRIMARY_TOPK,
        "primary_page_size": PRIMARY_PAGE_SIZE,
        "primary_physical_tokens_per_request": (
            layout.primary_physical_tokens_per_request
        ),
        "primary_active_length": layout.primary_active_length,
        "extra_topk": variant.extra_topk,
        "extra_page_size": variant.extra_page_size,
        "extra_compress_ratio": variant.extra_compress_ratio,
        "extra_physical_tokens_per_request": (
            layout.extra_physical_tokens_per_request
        ),
        "extra_active_length": layout.extra_active_length,
        "index_pattern": INDEX_PATTERN,
        "index_seed": seed + 1,
        "primary_global_scale": PRIMARY_GLOBAL_SCALE,
        "extra_global_scale": EXTRA_GLOBAL_SCALE,
        "warmup_iterations": warmup_iterations,
        "iterations_per_round": iterations,
        "rounds": rounds,
        "samples_per_provider": iterations * rounds,
        "correct": True,
        "atol": atol,
        "rtol": rtol,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "fused_median_us": fused_median_us,
        "fallback_median_us": fallback_median_us,
        "fused_round_median_cv": fused_round_median_cv,
        "fallback_round_median_cv": fallback_round_median_cv,
        "fused_speedup": fallback_median_us / fused_median_us,
        "device": torch.cuda.get_device_name(device),
        "seed": seed,
    }
    round_rows = []
    for provider in (FUSED_PROVIDER, FALLBACK_PROVIDER):
        for round_idx, median_us in enumerate(round_medians[provider], start=1):
            round_rows.append(
                {
                    "variant": variant.name,
                    "batch_size": batch_size,
                    "h_q": variant.h_q,
                    "benchmark_scope": BENCHMARK_SCOPE,
                    "cache_mode": cache_mode,
                    "acceptance_role": (
                        "primary" if cache_mode == "cold" else "diagnostic_only"
                    ),
                    "provider": provider,
                    "provider_kind": provider_kinds[provider],
                    "round": round_idx,
                    "iterations": iterations,
                    "round_median_us": median_us,
                    "round_median_cv": (
                        fused_round_median_cv
                        if provider == FUSED_PROVIDER
                        else fallback_round_median_cv
                    ),
                    "seed": seed,
                }
            )
    return summary, round_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 12, 17])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--context-tokens-per-request",
        "--context-tokens",
        dest="context_tokens_per_request",
        type=int,
        default=DEFAULT_CONTEXT_TOKENS_PER_REQUEST,
        help=(
            "Original, uncompressed context tokens represented by each "
            "request (default: 32768)."
        ),
    )
    parser.add_argument(
        "--full-topk",
        action="store_true",
        help=(
            "Grow the effective context when needed so every extra top-k "
            "slot is active (C128 requires at least 131072 tokens)."
        ),
    )
    parser.add_argument(
        "--cache-mode",
        choices=("warm", "cold"),
        default="cold",
        help=(
            "Use cold L2 for every timed provider invocation (default and "
            "the only primary acceptance mode). Warm is diagnostic only and "
            "may include host-submission gaps between provider kernels."
        ),
    )
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("flashmla_dsv4_nvfp4.csv"),
    )
    parser.add_argument(
        "--output-rounds-csv",
        type=Path,
        help=(
            "Per-round median CSV (default: <output-csv stem>.rounds.csv)."
        ),
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        raise RuntimeError(f"SM90 is required, got compute capability {capability}")
    if args.warmup < 1 or args.iterations < 1 or args.rounds < 1:
        raise ValueError("warmup, iterations, and rounds must all be positive")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise ValueError("batch sizes must all be positive")
    if args.context_tokens_per_request < 1:
        raise ValueError("context tokens per request must be positive")

    flashmla_ops_path = Path(flashmla_ops.__file__).resolve()
    flashmla_ops_sha256 = _sha256(flashmla_ops_path)

    rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    case_idx = 0
    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        for batch_size in args.batch_sizes:
            row, case_round_rows = _benchmark_case(
                variant=variant,
                batch_size=batch_size,
                seed=args.seed + case_idx,
                context_tokens_per_request=args.context_tokens_per_request,
                full_topk=args.full_topk,
                cache_mode=args.cache_mode,
                warmup_iterations=args.warmup,
                iterations=args.iterations,
                rounds=args.rounds,
                atol=args.atol,
                rtol=args.rtol,
                device=device,
            )
            row["flashmla_ops_path"] = str(flashmla_ops_path)
            row["flashmla_ops_sha256"] = flashmla_ops_sha256
            for round_row in case_round_rows:
                round_row["flashmla_ops_path"] = str(flashmla_ops_path)
                round_row["flashmla_ops_sha256"] = flashmla_ops_sha256
            rows.append(row)
            round_rows.extend(case_round_rows)
            print(
                f"{variant.name:10s} B={batch_size:2d} "
                f"{row['fused_provider']}={row['fused_median_us']:.2f} us "
                f"{row['fallback_provider']}="
                f"{row['fallback_median_us']:.2f} us "
                f"speedup={row['fused_speedup']:.3f}x "
                f"round_cv=({row['fused_round_median_cv']:.2%},"
                f"{row['fallback_round_median_cv']:.2%}) "
                f"cache={row['cache_mode']} "
                f"acceptance={row['acceptance_role']} "
                f"context={row['context_tokens_per_request']} "
                f"extra_range={row['extra_physical_tokens_per_request']} "
                f"extra_active={row['extra_active_length']} "
                f"max_abs={row['max_abs_diff']:.6f}",
                flush=True,
            )
            case_idx += 1
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    _write_csv(args.output_csv, rows)
    rounds_csv = args.output_rounds_csv
    if rounds_csv is None:
        suffix = args.output_csv.suffix or ".csv"
        stem = (
            args.output_csv.stem
            if args.output_csv.suffix
            else args.output_csv.name
        )
        rounds_csv = args.output_csv.with_name(f"{stem}.rounds{suffix}")
    _write_csv(rounds_csv, round_rows)
    print(f"wrote {args.output_csv}")
    print(f"wrote {rounds_csv}")


if __name__ == "__main__":
    main()
