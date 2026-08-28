"""Benchmark Qwen3.8-Flash-Next QSA sparse prefill geometries on SM120.

Pass equal-length ``--q-lens`` and ``--kv-lens`` lists to set each request's
query and KV lengths independently. Without ``--q-lens``, ``--kv-lens`` is the
single-request KV-length sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton

from sglang.srt.layers.attention.qsa.sparse_attn import (
    _get_prefill_config,
    _get_table_prefill_config,
    _sparse_gqa_chunk_prefill,
    _sparse_gqa_prefill,
)
from sglang.srt.utils import is_sm120_supported, is_sm121

DEFAULT_CHUNK = 8192
KV_LENS = (8192, 32768, 131072)
Q_HEADS = (6, 12)
DEFAULT_KV_HEADS = 1
HEAD_DIM = 256
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1
SCALE = HEAD_DIM**-0.5
GUARD_ELEMENTS = 4096


def _command_output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        command, check=True, text=True, capture_output=True, cwd=cwd
    ).stdout.strip()


def _resolve_nvcc() -> str | None:
    candidate = os.environ.get("CUDACXX") or os.environ.get("NVCC") or "nvcc"
    return shutil.which(candidate)


def _environment() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[5]
    driver = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    ).splitlines()[0]
    nvcc = _resolve_nvcc()
    toolkit = (
        "nvcc: not found"
        if nvcc is None
        else _command_output([nvcc, "--version"]).splitlines()[-1]
    )
    commit = _command_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver": driver,
        "cuda_runtime": torch.version.cuda,
        "cuda_toolkit": toolkit,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "host": platform.node(),
        "sglang_commit": commit,
    }


def _environment_line(environment: dict[str, object], args) -> str:
    batch = len(args.q_lens) if args.q_lens else 1
    return (
        f"gpu={environment['gpu']} cc={environment['compute_capability']} "
        f"driver={environment['driver']} cuda={environment['cuda_runtime']} "
        f"toolkit={environment['cuda_toolkit']!r} torch={environment['torch']} "
        f"triton={environment['triton']} python={environment['python']} "
        f"sglang_commit={environment['sglang_commit']} "
        "model=Qwen3.8-Flash-Next "
        f"kernel={args.kernel} batch={batch} chunk={args.chunk} "
        f"q_lens={args.q_lens} kv_lens={args.kv_lens} "
        f"kv_heads={args.kv_heads} head_dim={HEAD_DIM} topk={TOPK} dtype=bfloat16"
    )


def _timed(fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    ends[-1].synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _timed_geometries(case, geometries, providers, warmup, iterations):
    for step in range(warmup):
        order = providers if step % 2 == 0 else reversed(providers)
        for provider in order:
            case.launch(geometries[provider])
    torch.cuda.synchronize()
    events = {provider: [] for provider in providers}
    last_outputs = {}
    for step in range(iterations):
        order = providers if step % 2 == 0 else reversed(providers)
        for provider in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = case.launch(geometries[provider])
            end.record()
            if step == iterations - 1:
                last_outputs[provider] = output.clone()
            events[provider].append((start, end))
    torch.cuda.synchronize()
    samples = {
        provider: [start.elapsed_time(end) for start, end in pairs]
        for provider, pairs in events.items()
    }
    timings = {
        provider: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for provider, values in samples.items()
    }
    return timings, last_outputs


def _selected_indices(q_lens: list[int], prefix_lens: list[int]) -> torch.Tensor:
    pieces = []
    columns = torch.arange(TOPK, dtype=torch.int64, device="cuda")[None, :]
    row_offset = 0
    for q_len, prefix in zip(q_lens, prefix_lens):
        rows = torch.arange(q_len, dtype=torch.int64, device="cuda")[:, None]
        visible = prefix + rows + 1
        # The affine walk is deterministic and spreads selected tokens across
        # each visible prefix without building a randperm for every row.
        selected = (columns * 104729 + (rows + row_offset) * 13007 + 8191) % visible
        selected = selected.to(torch.int32)
        selected.masked_fill_(columns >= visible, -1)
        pieces.append(selected)
        row_offset += q_len
    return torch.cat(pieces)


def _randomized_indices(
    kv_len: int, query_rows: int, *, include_invalid: bool
) -> torch.Tensor:
    prefix = kv_len - query_rows
    rows = torch.arange(query_rows, dtype=torch.int32, device="cuda")[:, None]
    visible = prefix + rows + 1
    selected = torch.randint(
        0,
        2**31 - 1,
        (query_rows, TOPK),
        dtype=torch.int32,
        device="cuda",
    ).remainder_(visible)
    if include_invalid:
        selected[0, :4] = torch.tensor(
            [-7, -1, kv_len, kv_len + 19], dtype=torch.int32, device="cuda"
        )
    return selected


def _prefix_lens(q_lens: list[int], kv_lens: list[int]) -> list[int]:
    return [kv_len - q_len for q_len, kv_len in zip(q_lens, kv_lens)]


def _launch_sparse_prefill(
    kernel: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    geometry: tuple[int, int, int, int],
    max_q: int,
) -> torch.Tensor:
    block_m, block_n, warps, stages = geometry
    kv_heads = k.shape[1]
    group_size = q.shape[1] // kv_heads
    common = (q, k, v, out, indices)
    strides = (
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        0,
        indices.stride(1),
    )
    if kernel == "chunk":
        _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * kv_heads)](
            *common,
            cu_q,
            cu_k,
            kv_lens,
            SCALE,
            TOPK,
            *strides,
            NUM_KV_HEADS=kv_heads,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=HEAD_DIM,
            num_warps=warps,
            num_stages=stages,
        )
    else:
        _sparse_gqa_prefill[(max_q, (cu_q.shape[0] - 1) * kv_heads)](
            *common,
            cu_q,
            SCALE,
            TOPK,
            *strides,
            NUM_KV_HEADS=kv_heads,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=HEAD_DIM,
            num_warps=warps,
            num_stages=stages,
        )
    return out


class AttentionCase:
    def __init__(
        self,
        request_kv_lens: list[int],
        q_heads: int,
        kv_heads: int,
        q_lens: list[int],
        kernel: str,
    ):
        query_rows = sum(q_lens)
        total_k = sum(request_kv_lens)
        torch.manual_seed(20260828 + total_k + q_heads + kv_heads + query_rows)
        self.kv_len = total_k
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.group_size = q_heads // kv_heads
        self.query_rows = query_rows
        self.q_lens = q_lens
        self.request_kv_lens = request_kv_lens
        self.prefix_lens = _prefix_lens(q_lens, request_kv_lens)
        self.max_q = max(q_lens)
        self.kernel = kernel
        self.q = torch.randn(
            query_rows, q_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        self.k = torch.randn(
            total_k, kv_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        self.v = torch.randn_like(self.k)
        self.indices = _selected_indices(self.q_lens, self.prefix_lens)
        self.cu_q = torch.tensor(
            [0, *torch.tensor(self.q_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        self.cu_k = torch.tensor(
            [0, *torch.tensor(self.request_kv_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        self.kv_lens = torch.tensor(
            self.request_kv_lens, dtype=torch.int32, device="cuda"
        )
        self.out = torch.empty_like(self.q)

    def launch(self, geometry: tuple[int, int, int, int]) -> torch.Tensor:
        return _launch_sparse_prefill(
            self.kernel,
            self.q,
            self.k,
            self.v,
            self.out,
            self.indices,
            self.cu_q,
            self.cu_k,
            self.kv_lens,
            geometry,
            self.max_q,
        )

    def torch_reference(self, row_block: int = 64) -> torch.Tensor:
        output = torch.empty_like(self.q)
        for batch, kv_len in enumerate(self.request_kv_lens):
            q_start = int(self.cu_q[batch])
            q_end = int(self.cu_q[batch + 1])
            k_start = int(self.cu_k[batch])
            for begin in range(q_start, q_end, row_block):
                end = min(begin + row_block, q_end)
                selected = self.indices[begin:end]
                valid = (selected >= 0) & (selected < kv_len)
                safe = selected.clamp(0, kv_len - 1).long()
                for kv_head in range(self.kv_heads):
                    head_start = kv_head * self.group_size
                    head_end = head_start + self.group_size
                    request_k = self.k[k_start : k_start + kv_len, kv_head]
                    request_v = self.v[k_start : k_start + kv_len, kv_head]
                    keys = request_k.index_select(0, safe.flatten()).view(
                        end - begin, TOPK, HEAD_DIM
                    )
                    values = request_v.index_select(0, safe.flatten()).view_as(keys)
                    scores = torch.einsum(
                        "bhd,bkd->bhk",
                        self.q[begin:end, head_start:head_end].float(),
                        keys.float(),
                    )
                    scores.mul_(SCALE)
                    scores.masked_fill_(~valid[:, None, :], -float("inf"))
                    probabilities = torch.softmax(scores, dim=-1)
                    output[begin:end, head_start:head_end] = torch.einsum(
                        "bhk,bkd->bhd", probabilities, values.float()
                    ).to(output.dtype)
        return output


def _tuned_geometry(
    total_q: int,
    q_heads: int,
    kv_heads: int,
    num_requests: int,
    kernel: str,
    max_q: int,
) -> tuple[int, int, int, int]:
    return _get_prefill_config(
        total_q,
        q_heads // kv_heads,
        num_requests,
        HEAD_DIM,
        kernel="ordinary" if kernel == "non-chunk" else "chunk",
        topk=TOPK,
        num_kv_heads=kv_heads,
        max_q=max_q,
    )


def _previous_geometry(
    total_q: int, q_heads: int, kv_heads: int
) -> tuple[int, int, int, int]:
    return _get_table_prefill_config(total_q, q_heads // kv_heads)


def _provider_geometries(
    total_q: int,
    q_heads: int,
    kv_heads: int,
    forced_geometries,
    num_requests: int,
    kernel: str,
    max_q: int,
):
    geometries = {
        "tuned": _tuned_geometry(
            total_q, q_heads, kv_heads, num_requests, kernel, max_q
        ),
        "previous": _previous_geometry(total_q, q_heads, kv_heads),
        "block-m-only": (8, 16, 1, 2),
        "schedule-only": (16, 16, 4, 3),
    }
    geometries.update(forced_geometries)
    return geometries


def _resolve_flash_attn():
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        return None


class GatherFlashSynthetic:
    def __init__(self, case: AttentionCase):
        self.case = case
        self.fn = _resolve_flash_attn()
        if self.fn is None:
            raise ImportError("flash_attn varlen is unavailable")
        row_kv_lens = torch.repeat_interleave(
            case.kv_lens, torch.tensor(case.q_lens, dtype=torch.int64, device="cuda")
        )
        self.valid = (case.indices >= 0) & (case.indices < row_kv_lens[:, None])
        self.counts = self.valid.sum(dim=1, dtype=torch.int32)
        self.cu_q = torch.arange(case.query_rows + 1, dtype=torch.int32, device="cuda")
        self.cu_k = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                self.counts.cumsum(0, dtype=torch.int32),
            ]
        )

    def full(self) -> torch.Tensor:
        row_k_offsets = torch.repeat_interleave(
            self.case.cu_k[:-1],
            torch.tensor(self.case.q_lens, dtype=torch.int64, device="cuda"),
        )
        selected = (self.case.indices + row_k_offsets[:, None])[self.valid].long()
        packed_k = self.case.k.index_select(0, selected)
        packed_v = self.case.v.index_select(0, selected)
        return self.fn(
            q=self.case.q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=self.cu_q,
            cu_seqlens_k=self.cu_k,
            max_seqlen_q=1,
            max_seqlen_k=TOPK,
            softmax_scale=SCALE,
            causal=False,
        )


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
    }


def _case_q_lens(request_kv_lens: list[int], args) -> list[int]:
    if args.q_lens:
        return args.q_lens
    return [args.chunk if args.kernel == "chunk" else request_kv_lens[0]]


def _bench_case(request_kv_lens: list[int], q_heads: int, args) -> dict[str, object]:
    q_lens = _case_q_lens(request_kv_lens, args)
    query_rows = sum(q_lens)
    case = AttentionCase(request_kv_lens, q_heads, args.kv_heads, q_lens, args.kernel)
    geometries = _provider_geometries(
        query_rows,
        q_heads,
        args.kv_heads,
        args.forced_geometries,
        len(q_lens),
        args.kernel,
        max(q_lens),
    )
    tuned_geometry = geometries["tuned"]
    previous_geometry = geometries["previous"]
    tuned = case.launch(tuned_geometry).clone()
    previous = case.launch(previous_geometry).clone()
    torch.cuda.synchronize()
    result: dict[str, object] = {
        "kv_len": case.kv_len,
        "kv_lens": request_kv_lens,
        "q_heads": q_heads,
        "kv_heads": args.kv_heads,
        "kernel": args.kernel,
        "batch": len(q_lens),
        "q_lens": q_lens,
        "prefix_lens": case.prefix_lens,
        "total_q": query_rows,
        "tuned_geometry": tuned_geometry,
        "previous_geometry": previous_geometry,
        "provider_geometries": {
            provider: geometries[provider]
            for provider in args.providers
            if provider in geometries
        },
        "tuned_previous_error": _error(tuned, previous),
        "provider_errors": {},
        "providers": {},
        "skips": {},
    }
    providers = result["providers"]
    provider_errors = result["provider_errors"]
    assert isinstance(providers, dict)
    assert isinstance(provider_errors, dict)
    timed_providers = [
        provider for provider in args.providers if provider in geometries
    ]
    timings, timed_outputs = _timed_geometries(
        case,
        geometries,
        timed_providers,
        args.warmup,
        args.iterations,
    )
    providers.update(timings)
    expected = case.torch_reference(args.torch_row_block)
    torch.cuda.synchronize()
    result["torch_error"] = _error(timed_outputs.get("tuned", tuned), expected)
    providers["torch"] = _timed(
        lambda: case.torch_reference(args.torch_row_block),
        args.warmup,
        args.torch_iterations,
    )
    for provider, output in timed_outputs.items():
        provider_errors[provider] = _error(output, expected)
    if "gather-flash-synthetic" in args.providers:
        skips = result["skips"]
        assert isinstance(skips, dict)
        try:
            flash = GatherFlashSynthetic(case)
            actual = flash.full()
            torch.cuda.synchronize()
            result["flash_error"] = _error(tuned, actual)
            provider_errors["gather-flash-synthetic"] = _error(actual, expected)
            providers["gather-flash-synthetic"] = _timed(
                flash.full, args.warmup, args.flash_iterations
            )
        except (ImportError, torch.cuda.OutOfMemoryError) as exc:
            skips["gather-flash-synthetic"] = f"{type(exc).__name__}: {exc}"
    result["numeric_reference"] = "torch"
    result["passed"] = all(
        error["finite"] and error["max_abs"] <= args.max_abs_error
        for error in provider_errors.values()
    )
    torch.cuda.empty_cache()
    return result


def _guarded_tensor(shape, dtype, sentinel):
    elements = math.prod(shape)
    storage = torch.empty(elements + 2 * GUARD_ELEMENTS, dtype=dtype, device="cuda")
    storage[:GUARD_ELEMENTS].fill_(sentinel)
    storage[-GUARD_ELEMENTS:].fill_(sentinel)
    tensor = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + elements].view(shape)
    return storage, tensor


def _guard_is_clean(storage: torch.Tensor, sentinel) -> bool:
    return bool(
        torch.all(storage[:GUARD_ELEMENTS] == sentinel)
        and torch.all(storage[-GUARD_ELEMENTS:] == sentinel)
    )


def _sanitizer_case(
    kernel: str,
    kv_len: int,
    q_heads: int,
    kv_heads: int,
    *,
    include_invalid: bool,
) -> None:
    query_rows = DEFAULT_CHUNK if kernel == "chunk" else kv_len
    torch.manual_seed(20260829 + kv_len + q_heads + kv_heads)
    q_storage, q = _guarded_tensor((query_rows, q_heads, HEAD_DIM), torch.bfloat16, -97)
    k_storage, k = _guarded_tensor((kv_len, kv_heads, HEAD_DIM), torch.bfloat16, -101)
    v_storage, v = _guarded_tensor((kv_len, kv_heads, HEAD_DIM), torch.bfloat16, -103)
    i_storage, indices = _guarded_tensor((query_rows, TOPK), torch.int32, -107)
    # Leave the output body uninitialized so initcheck can detect unwritten rows.
    o_storage, output = _guarded_tensor(
        (query_rows, q_heads, HEAD_DIM), torch.bfloat16, -109
    )
    q.uniform_(-0.5, 0.5)
    k.uniform_(-0.5, 0.5)
    v.uniform_(-0.5, 0.5)
    indices.copy_(
        _randomized_indices(
            kv_len,
            query_rows,
            include_invalid=include_invalid,
        )
    )
    cu_q = torch.tensor([0, query_rows], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
    geometry = _tuned_geometry(query_rows, q_heads, kv_heads, 1)
    _launch_sparse_prefill(
        kernel,
        q,
        k,
        v,
        output,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        geometry,
        query_rows,
    )
    torch.cuda.synchronize()
    guards = (q_storage, k_storage, v_storage, i_storage, o_storage)
    sentinels = (-97, -101, -103, -107, -109)
    if not all(
        _guard_is_clean(storage, sentinel)
        for storage, sentinel in zip(guards, sentinels)
    ):
        raise RuntimeError(
            f"guard band changed for kernel={kernel} kv={kv_len} q_heads={q_heads}"
        )
    if not torch.isfinite(output).all():
        raise RuntimeError(
            f"nonfinite output for kernel={kernel} kv={kv_len} q_heads={q_heads}"
        )
    print(
        f"SANITIZE PASS kernel={kernel} kv={kv_len} "
        f"q_heads={q_heads} kv_heads={kv_heads} geometry={tuple(geometry)} "
        f"indices={'invalid-bounds' if include_invalid else 'all-in-range'}",
        flush=True,
    )


def _run_sanitizer_driver(args) -> None:
    print(
        "SANITIZER SCOPE both kernels at kv=8192 and kv=131072; "
        "output body starts uninitialized",
        flush=True,
    )
    all_in_range_case = ("non-chunk", 8192, min(args.q_heads))
    for kernel in ("non-chunk", "chunk"):
        for kv_len in (8192, 131072):
            for q_heads in args.q_heads:
                include_invalid = (kernel, kv_len, q_heads) != all_in_range_case
                _sanitizer_case(
                    kernel,
                    kv_len,
                    q_heads,
                    args.kv_heads,
                    include_invalid=include_invalid,
                )


def _run_smoke(args) -> None:
    providers = [
        "tuned",
        "previous",
        "block-m-only",
        "schedule-only",
        "torch",
        "gather-flash-synthetic",
    ]
    for kernel in ("non-chunk", "chunk"):
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.kernel = kernel
        smoke_args.kv_heads = 1
        smoke_args.q_lens = [3, 5]
        smoke_args.kv_lens = [3, 5] if kernel == "non-chunk" else [7, 9]
        smoke_args.providers = providers
        smoke_args.forced_geometries = {}
        smoke_args.warmup = 0
        smoke_args.iterations = 1
        smoke_args.torch_iterations = 1
        smoke_args.torch_row_block = 8
        smoke_args.flash_iterations = 1
        result = _bench_case(smoke_args.kv_lens, 6, smoke_args)
        missing = set(providers) - set(result["providers"])
        if missing or result["skips"] or not result["passed"]:
            raise RuntimeError(
                f"smoke failed for kernel={kernel}: missing={sorted(missing)} "
                f"skips={result['skips']} passed={result['passed']}"
            )
        print(f"SMOKE PASS kernel={kernel} providers={len(providers)}", flush=True)


def _value(provider: dict[str, object] | None) -> str:
    if provider is None:
        return "n/a"
    return f"{provider['median_ms']:.4f}"


def _error_value(error: dict[str, object] | None) -> str:
    if error is None:
        return "n/a"
    return f"{error['max_abs']:.6f}"


def _print_table(results: list[dict[str, object]], provider_names: list[str]) -> None:
    timing_providers = [
        provider
        for provider in provider_names
        if provider not in ("torch", "gather-flash-synthetic")
    ]
    columns = " | ".join(f"{provider} ms" for provider in timing_providers)
    print(
        f"| total Q | Q heads/KV heads | KV length | {columns} | "
        "torch ms | gather+flash synthetic ms | tuned max abs | "
        "synthetic max abs | result |"
    )
    print("|--:|:--|--:|" + "--:|" * (len(timing_providers) + 4) + ":--|")
    for result in results:
        providers = result["providers"]
        provider_errors = result["provider_errors"]
        timings = " | ".join(
            _value(providers.get(provider)) for provider in timing_providers
        )
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"| {result['total_q']} | {result['q_heads']}/{result['kv_heads']} | "
            f"{result['kv_len']} | {timings} | "
            f"{_value(providers.get('torch'))} | "
            f"{_value(providers.get('gather-flash-synthetic'))} | "
            f"{_error_value(provider_errors.get('tuned'))} | "
            f"{_error_value(provider_errors.get('gather-flash-synthetic'))} | "
            f"{status} |"
        )
        for provider, geometry in result["provider_geometries"].items():
            print(
                f"provider_geometry total_q={result['total_q']} "
                f"q_heads={result['q_heads']} kv={result['kv_len']} "
                f"provider={provider} geometry={tuple(geometry)}"
            )
        for provider, error in provider_errors.items():
            print(
                f"provider_validation total_q={result['total_q']} "
                f"q_heads={result['q_heads']} kv={result['kv_len']} "
                f"provider={provider} reference={result['numeric_reference']} "
                f"finite={error['finite']} max_abs={error['max_abs']:.6f}"
            )
        for provider, reason in result["skips"].items():
            print(
                f"provider_status q_heads={result['q_heads']} kv={result['kv_len']} "
                f"provider={provider} skipped={reason}"
            )


def _parse_forced_geometries(values, parser) -> dict[str, tuple[int, int, int, int]]:
    forced = {}
    reserved = {
        "tuned",
        "previous",
        "block-m-only",
        "schedule-only",
        "torch",
        "gather-flash-synthetic",
    }
    for name, *raw_geometry in values:
        if name in reserved or name in forced:
            parser.error(f"forced geometry name must be unique and unreserved: {name}")
        try:
            geometry = tuple(int(value) for value in raw_geometry)
        except ValueError:
            parser.error(
                f"forced geometry values must be integers: {name} {raw_geometry}"
            )
        if any(value <= 0 for value in geometry):
            parser.error(f"forced geometry values must be positive: {name} {geometry}")
        forced[name] = geometry
    return forced


def main() -> None:
    epilog = """
Query sweep for the group-6 sm120 family:
  for q in 32 64 128 256 512 1024 8192; do python %(prog)s --kernel chunk \
    --chunk $q --kv-lens 8192 --q-heads 6 --providers previous sm120-family torch \
    --force-geometry sm120-family 8 16 4 3 --warmup 10 --iterations 100 \
    --torch-iterations 3; done

Query sweep for the group-12 sm120 family:
  for q in 32 64 128 256 512 1024 8192; do python %(prog)s --kernel chunk \
    --chunk $q --kv-lens 8192 --q-heads 12 --providers previous sm120-family torch \
    --force-geometry sm120-family 16 16 2 3 --warmup 10 --iterations 100 \
    --torch-iterations 3; done

Non-chunk comparison:
  python %(prog)s --kernel non-chunk --kv-lens 8192 32768 131072 \
    --q-heads 6 12 --providers previous tuned torch --warmup 10 --iterations 100 \
    --torch-iterations 3

TP1 group-12 comparison:
  python %(prog)s --kernel chunk --chunk 8192 --kv-lens 8192 32768 131072 \
    --q-heads 24 --kv-heads 2 --providers previous tuned torch --warmup 10 \
    --iterations 100 --torch-iterations 3

Group-12 schedule ablation:
  python %(prog)s --kernel chunk --chunk 8192 --kv-lens 8192 32768 131072 \
    --q-heads 12 --providers previous tuned schedule-only torch --warmup 10 \
    --iterations 100 --torch-iterations 3

Group-6 geometry ablation:
  python %(prog)s --kernel chunk --chunk 8192 --kv-lens 8192 32768 131072 \
    --q-heads 6 --providers previous tuned block-m-only schedule-only torch \
    --warmup 10 --iterations 100 --torch-iterations 3

Many-short-rows comparison (the production selector takes the table here, so
force the tuned tuples to reproduce the regime measurement):
  python %(prog)s --kernel chunk --q-lens 64 64 64 64 64 64 64 64 \
    64 64 64 64 64 64 64 64 --kv-lens 512 512 512 512 512 512 512 512 \
    512 512 512 512 512 512 512 512 --q-heads 6 \
    --force-geometry g6 8 16 4 3 --providers previous g6 \
    --warmup 10 --iterations 100
  python %(prog)s --kernel chunk --q-lens 64 64 64 64 64 64 64 64 \
    64 64 64 64 64 64 64 64 --kv-lens 512 512 512 512 512 512 512 512 \
    512 512 512 512 512 512 512 512 --q-heads 12 \
    --force-geometry g12 16 16 2 3 --providers previous g12 \
    --warmup 10 --iterations 100
"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help=(
            "providers to time: tuned, previous, block-m-only, schedule-only, "
            "torch, gather-flash-synthetic, or a --force-geometry name"
        ),
        default=["tuned", "previous", "torch", "gather-flash-synthetic"],
    )
    parser.add_argument(
        "--force-geometry",
        action="append",
        nargs=5,
        default=[],
        metavar=("NAME", "BLOCK_M", "BLOCK_N", "WARPS", "STAGES"),
        help="add a named provider with an exact launch geometry; repeatable",
    )
    parser.add_argument("--kernel", choices=("chunk", "non-chunk"), default="chunk")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument(
        "--q-lens",
        nargs="+",
        type=int,
        help="per-request query lengths; requires a matching --kv-lens list",
    )
    parser.add_argument(
        "--kv-lens",
        nargs="+",
        type=int,
        default=list(KV_LENS),
        help=(
            "per-request KV lengths with --q-lens; otherwise single-request "
            "KV lengths to sweep"
        ),
    )
    parser.add_argument("--q-heads", nargs="+", type=int, default=list(Q_HEADS))
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--torch-iterations", type=int, default=3)
    parser.add_argument("--torch-row-block", type=int, default=64)
    parser.add_argument("--flash-iterations", type=int, default=3)
    parser.add_argument("--max-abs-error", type=float, default=2e-2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.sanitize:
        parser.error("--smoke and --sanitize are mutually exclusive")
    if args.chunk <= 0:
        parser.error("--chunk must be positive")
    if args.q_lens and any(q_len <= 0 for q_len in args.q_lens):
        parser.error("--q-lens values must be positive")
    if any(kv_len <= 0 for kv_len in args.kv_lens):
        parser.error("--kv-lens values must be positive")
    if args.q_lens and len(args.kv_lens) != len(args.q_lens):
        parser.error("--kv-lens must have one value for each --q-lens value")
    if args.kv_heads <= 0:
        parser.error("--kv-heads must be positive")
    invalid_q_heads = [
        q_heads
        for q_heads in args.q_heads
        if q_heads <= 0 or q_heads % args.kv_heads != 0
    ]
    if invalid_q_heads:
        parser.error(
            "--q-heads values must be positive multiples of --kv-heads: "
            f"{invalid_q_heads}"
        )
    args.forced_geometries = _parse_forced_geometries(args.force_geometry, parser)
    known_providers = {
        "tuned",
        "previous",
        "block-m-only",
        "schedule-only",
        "torch",
        "gather-flash-synthetic",
        *args.forced_geometries,
    }
    unknown_providers = set(args.providers) - known_providers
    if unknown_providers:
        parser.error(f"unknown providers: {sorted(unknown_providers)}")
    if "torch" not in args.providers:
        args.providers.append("torch")
    if args.smoke:
        _run_smoke(args)
        return
    request_kv_cases = (
        [args.kv_lens] if args.q_lens else [[kv_len] for kv_len in args.kv_lens]
    )
    for request_kv_lens in request_kv_cases:
        q_lens = _case_q_lens(request_kv_lens, args)
        total_q = sum(q_lens)
        for q_heads in args.q_heads:
            geometries = _provider_geometries(
                total_q,
                q_heads,
                args.kv_heads,
                args.forced_geometries,
                len(q_lens),
                args.kernel,
                max(q_lens),
            )
            group_size = q_heads // args.kv_heads
            for provider in args.providers:
                if provider in geometries and geometries[provider][0] < group_size:
                    parser.error(
                        f"provider {provider} has BLOCK_M={geometries[provider][0]} "
                        f"below the {group_size}-head group size"
                    )
    invalid_pairs = [
        (q_len, kv_len)
        for request_kv_lens in request_kv_cases
        for q_len, kv_len in zip(_case_q_lens(request_kv_lens, args), request_kv_lens)
        if kv_len < q_len
    ]
    if invalid_pairs:
        parser.error(f"each KV length must cover its query length: {invalid_pairs}")
    if args.kernel == "non-chunk" and args.q_lens and args.kv_lens != args.q_lens:
        parser.error("non-chunk requests require equal query and KV lengths")
    if args.sanitize:
        _run_sanitizer_driver(args)
        return
    environment = _environment()
    print(_environment_line(environment, args))
    results = [
        _bench_case(request_kv_lens, q_heads, args)
        for q_heads in args.q_heads
        for request_kv_lens in request_kv_cases
    ]
    _print_table(results, args.providers)
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "qsa_sparse_prefill_v1",
                    "environment": environment,
                    "model": {
                        "name": "Qwen3.8-Flash-Next",
                        "attention_dtype": "bfloat16",
                        "head_dim": HEAD_DIM,
                        "kv_heads": args.kv_heads,
                        "topk": TOPK,
                    },
                    "workload": {
                        "batch": len(args.q_lens) if args.q_lens else 1,
                        "kernel": args.kernel,
                        "chunk": args.chunk,
                        "q_lens": args.q_lens,
                        "kv_lens": args.kv_lens,
                        "q_heads": args.q_heads,
                    },
                    "timing": {
                        "method": "CUDA events; compilation excluded",
                        "warmup": args.warmup,
                        "iterations": args.iterations,
                        "torch_iterations": args.torch_iterations,
                        "flash_iterations": args.flash_iterations,
                    },
                    "providers": args.providers,
                    "forced_geometries": args.forced_geometries,
                    "max_abs_error": args.max_abs_error,
                    "results": results,
                }
            )
        )
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    if not torch.cuda.is_available() or not is_sm120_supported() or is_sm121():
        print("[skip] QSA sparse prefill benchmark requires SM120 CUDA.")
        sys.exit(0)
    main()
