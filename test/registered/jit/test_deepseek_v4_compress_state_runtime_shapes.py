# DeepSeek V4 compress-state test and benchmark entry.
#
# What this covers:
# - Synthetic Flash/Pro C4/C128 decode/prefill shapes for broad operator
#   performance coverage.
# - Replays 84 compress shapes captured from zc01 DeepSeek-V4-Flash serving.
# - The capture used EAGLE, so runtime compress plans were prefill-style plans
#   for both EXTEND and TARGET_VERIFY. Some short C128 cases have out_shape=0.
# - This file keeps the out_shape=0 diff handling local to the runtime replay
#   benchmark path.
#
# Test command:
#   python3 -m pytest -q \
#     test/registered/jit/test_deepseek_v4_compress_state_runtime_shapes.py
#
# Runtime-shape benchmark command:
#   python3 test/registered/jit/test_deepseek_v4_compress_state_runtime_shapes.py \
#     --benchmark \
#     --shape-source runtime \
#     --warmup 20 \
#     --iters 100 \
#     --csv /data00/eval_results/operator_bench/runtime_shape_bench.csv
#
# Synthetic Flash/Pro shape benchmark command:
#   python3 test/registered/jit/test_deepseek_v4_compress_state_runtime_shapes.py \
#     --benchmark \
#     --shape-source preset \
#     --shape-presets all \
#     --shape-tier smoke \
#     --warmup 10 \
#     --iters 30 \
#     --csv /data00/eval_results/operator_bench/preset_shape_bench.csv
#
# Service-level scenario where BF16 state compression is more likely to help:
# - Disable speculative/EAGLE so the workload is not dominated by small
#   TARGET_VERIFY shapes.
# - Prefer TP-only or DP=1 first; DP attention may reduce chunked prefill from
#   4096 to 512, which makes fixed kernel overhead dominate.
# - Use long random prompts and short outputs to make prefill dominate:
#   SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16 \
#   SGLANG_SHARED_EXPERT_TP1=1 \
#   SGLANG_ENABLE_THINKING=1 \
#   SGLANG_DSV4_FP4_EXPERTS=1 \
#   SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 \
#   sglang serve \
#     --trust-remote-code \
#     --model-path /data00/models/DeepSeek-V4-Flash \
#     --tp 8 \
#     --host 0.0.0.0 \
#     --port 8080 \
#     --mem-fraction-static 0.9 \
#     --moe-runner-backend marlin \
#     --chunked-prefill-size 4096 \
#     --max-prefill-tokens 16384 \
#     --max-running-requests 32 \
#     --cuda-graph-max-bs-decode 16 \
#     --enable-metrics \
#     --disable-radix-cache
#
# Workload for the service-level scenario:
#   HF_ENDPOINT=https://hf-mirror.com \
#   python3 -m sglang.benchmark.serving \
#     --host localhost \
#     --port 8080 \
#     --model /data00/models/DeepSeek-V4-Flash \
#     --dataset-name random \
#     --random-input-len 8192 \
#     --random-output-len 8 \
#     --random-range-ratio 1 \
#     --num-prompts 128 \
#     --max-concurrency 16 \
#     --request-rate 16
#
# Compare with the same service command without
# SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16. A visible service-level gain requires
# large prefill shapes and a non-trivial compress-kernel share in the profile.

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Literal, Optional

import pytest
import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
)
from sglang.jit_kernel.tests.deepseek_v4.common import (
    make_legacy_context,
    to_seq_extend,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_amd_ci(est_time=25, suite="nightly-amd-kernel-1-gpu", nightly=True)

Mode = Literal["decode", "prefill"]
ShapeTier = Literal["ci", "smoke", "full"]
MAX_PRESET_PREFILL_Q_TOKENS = 32768


@dataclass(frozen=True)
class ShapePreset:
    name: str
    hidden_size: int
    num_attention_heads: int
    index_topk: int
    ratios: tuple[Literal[4, 128], ...]
    decode_batch_sizes: dict[ShapeTier, tuple[int, ...]]
    prefill_batch_sizes: dict[ShapeTier, tuple[int, ...]]
    decode_seq_lens: dict[ShapeTier, tuple[int, ...]]
    prefill_extend_lens: dict[ShapeTier, dict[Literal[4, 128], tuple[int, ...]]]


SHAPE_PRESETS: dict[str, ShapePreset] = {
    # DeepSeek-V4-Flash config on zc01:
    # hidden_size=4096, num_attention_heads=64, index_topk=512,
    # compress_ratios=[0, 0, 4, 128, ... alternating ..., 0].
    # The compress kernel itself uses head_dim=512
    # (= qk_nope_head_dim 448 + qk_rope_head_dim 64).
    "flash": ShapePreset(
        name="flash",
        hidden_size=4096,
        num_attention_heads=64,
        index_topk=512,
        ratios=(4, 128),
        decode_batch_sizes={
            "ci": (16,),
            "smoke": (16, 128),
            "full": (1, 2, 4, 8, 16, 32, 64, 128),
        },
        prefill_batch_sizes={
            "ci": (1,),
            "smoke": (1, 16),
            "full": (1, 2, 4, 8, 16, 32),
        },
        decode_seq_lens={
            "ci": (128,),
            "smoke": (128,),
            "full": (128, 256, 4096),
        },
        prefill_extend_lens={
            "ci": {4: (128,), 128: (128,)},
            "smoke": {4: (16, 128, 4096), 128: (128, 4096)},
            "full": {4: (16, 128, 4096), 128: (128, 4096)},
        },
    ),
    # DeepSeek-V4-Pro config on zc01:
    # hidden_size=7168, num_attention_heads=128, index_topk=1024,
    # compress_ratios=[128, 128, 4, 128, ... alternating ..., 0].
    "pro": ShapePreset(
        name="pro",
        hidden_size=7168,
        num_attention_heads=128,
        index_topk=1024,
        ratios=(4, 128),
        decode_batch_sizes={
            "ci": (16,),
            "smoke": (16, 32),
            "full": (1, 2, 4, 8, 16, 32),
        },
        prefill_batch_sizes={
            "ci": (1,),
            "smoke": (1, 16),
            "full": (1, 2, 4, 8, 16, 32),
        },
        decode_seq_lens={
            "ci": (128,),
            "smoke": (128,),
            "full": (128, 256, 4096),
        },
        prefill_extend_lens={
            "ci": {4: (128,), 128: (128,)},
            "smoke": {4: (16, 128, 4096), 128: (128, 4096)},
            "full": {4: (16, 128, 4096), 128: (128, 4096)},
        },
    ),
}


@dataclass(frozen=True)
class BenchSpec:
    shape_name: str
    ratio: Literal[4, 128]
    mode: Mode
    batch_size: int
    head_dim: int
    tokens_per_req: int


@dataclass
class BenchInput:
    state_pool: torch.Tensor
    kv_score_input: torch.Tensor
    ape: torch.Tensor
    out: torch.Tensor
    plan: CompressorDecodePlan | CompressorPrefillPlan
    spec: BenchSpec

    @property
    def effective_bytes(self) -> int:
        tensors = (self.state_pool, self.kv_score_input, self.ape, self.out)
        return sum(t.numel() * t.element_size() for t in tensors)

    def run(self) -> torch.Tensor:
        return compress_forward(
            kv_score_buffer=self.state_pool,
            kv_score_input=self.kv_score_input,
            ape=self.ape,
            plan=self.plan,
            head_dim=self.spec.head_dim,
            compress_ratio=self.spec.ratio,
            out=self.out,
        )


@dataclass
class BenchResult:
    shape_name: str
    ratio: int
    mode: str
    batch_size: int
    tokens_per_req: int
    head_dim: int
    state_shape: str
    input_shape: str
    ape_shape: str
    out_shape: str
    fp32_state_mib: float
    bf16_state_mib: float
    input_mib: float
    ape_mib: float
    out_mib: float
    fp32_effective_mib: float
    bf16_effective_mib: float
    fp32_us: float
    fp32_gbps: float
    bf16_us: float
    bf16_gbps: float
    out_diff: float
    state_diff: float

    @property
    def speedup(self) -> float:
        return self.fp32_us / self.bf16_us


def _last_dim(ratio: int, head_dim: int) -> int:
    return head_dim * (4 if ratio == 4 else 2)


def _ape_len(ratio: int) -> int:
    return 8 if ratio == 4 else 128


def _shape_str(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


def _mib(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024


def _tensor_mib(tensor: torch.Tensor) -> float:
    return _mib(tensor.numel() * tensor.element_size())


def _make_plan(spec: BenchSpec):
    ctx = make_legacy_context(
        bs=spec.batch_size,
        compress_ratio=spec.ratio,
        head_dim=spec.head_dim,
    )
    if spec.mode == "decode":
        seq_lens = torch.full(
            (spec.batch_size,),
            spec.tokens_per_req,
            dtype=torch.int64,
            device="cuda",
        )
        plan = ctx.make_decode_plan(seq_lens)
        num_output_tokens = spec.batch_size
    else:
        seq_lens_cpu, extend_lens_cpu, num_q_tokens = to_seq_extend(
            [(spec.tokens_per_req, spec.tokens_per_req)] * spec.batch_size
        )
        plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q_tokens)
        num_output_tokens = int(plan.plan_c.shape[0])
    return ctx, plan, num_output_tokens


def _make_bench_input(
    spec: BenchSpec,
    state_dtype: torch.dtype,
    base_state: torch.Tensor,
    base_input: torch.Tensor,
    base_ape: torch.Tensor,
    plan: CompressorDecodePlan | CompressorPrefillPlan,
    num_q_tokens: int,
) -> BenchInput:
    state_pool = base_state.to(state_dtype)
    kv_score_input = base_input.clone()
    ape = base_ape.clone()
    out = torch.empty(
        (num_q_tokens, spec.head_dim),
        dtype=kv_score_input.dtype,
        device=kv_score_input.device,
    )
    return BenchInput(
        state_pool=state_pool,
        kv_score_input=kv_score_input,
        ape=ape,
        out=out,
        plan=plan,
        spec=spec,
    )


def _make_pair(spec: BenchSpec, seed: int) -> tuple[BenchInput, BenchInput]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ctx, plan, num_q_tokens = _make_plan(spec)
    last_dim = _last_dim(spec.ratio, spec.head_dim)

    if spec.mode == "decode":
        base_state = torch.randn(
            (ctx.num_pages, spec.ratio, last_dim),
            dtype=torch.float32,
            device="cuda",
        )
        input_rows = spec.batch_size
    else:
        base_state = torch.zeros(
            (ctx.num_pages, spec.ratio, last_dim),
            dtype=torch.float32,
            device="cuda",
        )
        input_rows = spec.batch_size * spec.tokens_per_req

    base_input = torch.randn(
        (input_rows, last_dim),
        dtype=torch.float32,
        device="cuda",
    )
    base_ape = torch.randn(
        (_ape_len(spec.ratio), spec.head_dim),
        dtype=torch.float32,
        device="cuda",
    )

    return (
        _make_bench_input(
            spec, torch.float32, base_state, base_input, base_ape, plan, num_q_tokens
        ),
        _make_bench_input(
            spec, torch.bfloat16, base_state, base_input, base_ape, plan, num_q_tokens
        ),
    )


def _time_us(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / iters


def _gbps(effective_bytes: int, elapsed_us: float) -> float:
    return effective_bytes / (elapsed_us * 1e-6) / 1e9


def _benchmark_spec(
    spec: BenchSpec,
    warmup: int,
    iters: int,
    seed: int,
) -> BenchResult:
    fp32_case, bf16_case = _make_pair(spec, seed)
    fp32_case.run()
    bf16_case.run()
    torch.cuda.synchronize()

    fp32_us = _time_us(fp32_case.run, warmup=warmup, iters=iters)
    bf16_us = _time_us(bf16_case.run, warmup=warmup, iters=iters)

    # Recreate once for diff so the timed loop's repeated in-place state writes
    # do not affect the comparison.
    fp32_diff_case, bf16_diff_case = _make_pair(spec, seed)
    fp32_out = fp32_diff_case.run()
    bf16_out = bf16_diff_case.run()
    torch.cuda.synchronize()

    out_diff = (fp32_out.float() - bf16_out.float()).abs().max().item()
    state_diff = (
        (fp32_diff_case.state_pool.float() - bf16_diff_case.state_pool.float())
        .abs()
        .max()
        .item()
    )

    return BenchResult(
        shape_name=spec.shape_name,
        ratio=spec.ratio,
        mode=spec.mode,
        batch_size=spec.batch_size,
        tokens_per_req=spec.tokens_per_req,
        head_dim=spec.head_dim,
        state_shape=_shape_str(fp32_case.state_pool),
        input_shape=_shape_str(fp32_case.kv_score_input),
        ape_shape=_shape_str(fp32_case.ape),
        out_shape=_shape_str(fp32_case.out),
        fp32_state_mib=_tensor_mib(fp32_case.state_pool),
        bf16_state_mib=_tensor_mib(bf16_case.state_pool),
        input_mib=_tensor_mib(fp32_case.kv_score_input),
        ape_mib=_tensor_mib(fp32_case.ape),
        out_mib=_tensor_mib(fp32_case.out),
        fp32_effective_mib=_mib(fp32_case.effective_bytes),
        bf16_effective_mib=_mib(bf16_case.effective_bytes),
        fp32_us=fp32_us,
        fp32_gbps=_gbps(fp32_case.effective_bytes, fp32_us),
        bf16_us=bf16_us,
        bf16_gbps=_gbps(bf16_case.effective_bytes, bf16_us),
        out_diff=out_diff,
        state_diff=state_diff,
    )


def _expand_shape_names(shape_presets: list[str]) -> list[str]:
    if "all" in shape_presets:
        return ["flash", "pro"]
    if "custom" in shape_presets and len(shape_presets) > 1:
        raise ValueError("--shape-presets custom cannot be mixed with shape presets.")
    return shape_presets


def _one_or_many(
    value: Optional[list[int]], fallback: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(value) if value is not None else fallback


def _validate_tokens(mode: Mode, ratio: int, tokens_per_req: int) -> None:
    if tokens_per_req % ratio != 0:
        raise ValueError(
            f"{mode} tokens_per_req={tokens_per_req} must be a multiple "
            f"of ratio={ratio}"
        )


def _make_shape_specs(args: argparse.Namespace) -> Iterable[BenchSpec]:
    shape_names = _expand_shape_names(args.shape_presets)
    for shape_name in shape_names:
        if shape_name == "custom":
            yield from _make_custom_specs(args)
            continue

        preset = SHAPE_PRESETS[shape_name]
        ratios = tuple(args.ratios) if args.ratios is not None else preset.ratios
        modes = tuple(args.modes) if args.modes is not None else ("decode", "prefill")
        decode_batch_sizes = _one_or_many(
            args.decode_batch_sizes or args.batch_sizes,
            preset.decode_batch_sizes[args.shape_tier],
        )
        prefill_batch_sizes = _one_or_many(
            args.prefill_batch_sizes or args.batch_sizes,
            preset.prefill_batch_sizes[args.shape_tier],
        )
        decode_seq_lens = _one_or_many(
            args.decode_seq_lens,
            preset.decode_seq_lens[args.shape_tier],
        )

        for ratio in ratios:
            if ratio not in preset.ratios:
                continue
            if "decode" in modes:
                for tokens_per_req in decode_seq_lens:
                    _validate_tokens("decode", ratio, tokens_per_req)
                    for batch_size in decode_batch_sizes:
                        yield BenchSpec(
                            shape_name=shape_name,
                            ratio=ratio,
                            mode="decode",
                            batch_size=batch_size,
                            head_dim=args.head_dim,
                            tokens_per_req=tokens_per_req,
                        )

            if "prefill" in modes:
                prefill_extend_lens = _one_or_many(
                    args.prefill_extend_lens,
                    preset.prefill_extend_lens[args.shape_tier][ratio],
                )
                for tokens_per_req in prefill_extend_lens:
                    _validate_tokens("prefill", ratio, tokens_per_req)
                    for batch_size in prefill_batch_sizes:
                        # The legacy prefill planner rejects very large
                        # synthetic q-token grids. Keep long-prefill coverage,
                        # but do not form unsupported cross-product cases.
                        if batch_size * tokens_per_req > MAX_PRESET_PREFILL_Q_TOKENS:
                            continue
                        yield BenchSpec(
                            shape_name=shape_name,
                            ratio=ratio,
                            mode="prefill",
                            batch_size=batch_size,
                            head_dim=args.head_dim,
                            tokens_per_req=tokens_per_req,
                        )


def _make_custom_specs(args: argparse.Namespace) -> Iterable[BenchSpec]:
    ratios = tuple(args.ratios) if args.ratios is not None else (4, 128)
    modes = tuple(args.modes) if args.modes is not None else ("decode", "prefill")
    batch_sizes = tuple(args.batch_sizes or [16, 32, 64, 128, 256])
    decode_seq_lens = tuple(args.decode_seq_lens or [128])
    prefill_extend_lens = tuple(args.prefill_extend_lens or [128])

    for ratio in ratios:
        for mode in modes:
            seq_lens = decode_seq_lens if mode == "decode" else prefill_extend_lens
            for tokens_per_req in seq_lens:
                _validate_tokens(mode, ratio, tokens_per_req)
                for batch_size in batch_sizes:
                    yield BenchSpec(
                        shape_name="custom",
                        ratio=ratio,
                        mode=mode,
                        batch_size=batch_size,
                        head_dim=args.head_dim,
                        tokens_per_req=tokens_per_req,
                    )


def _format_table(results: list[BenchResult]) -> str:
    if not results:
        return "No benchmark cases selected."

    headers = [
        "shape",
        "ratio",
        "mode",
        "bs",
        "tok/req",
        "head",
        "state",
        "input",
        "ape",
        "out",
        "state MiB",
        "input MiB",
        "ape MiB",
        "out MiB",
        "FP32 MiB",
        "BF16 MiB",
        "FP32 us",
        "FP32 GB/s",
        "BF16 us",
        "BF16 GB/s",
        "out diff",
        "state diff",
        "Speedup",
    ]
    rows = [
        [
            r.shape_name,
            str(r.ratio),
            r.mode,
            str(r.batch_size),
            str(r.tokens_per_req),
            str(r.head_dim),
            r.state_shape,
            r.input_shape,
            r.ape_shape,
            r.out_shape,
            f"{r.fp32_state_mib:.2f}/{r.bf16_state_mib:.2f}",
            f"{r.input_mib:.2f}",
            f"{r.ape_mib:.3f}",
            f"{r.out_mib:.2f}",
            f"{r.fp32_effective_mib:.2f}",
            f"{r.bf16_effective_mib:.2f}",
            f"{r.fp32_us:.2f}",
            f"{r.fp32_gbps:.1f}",
            f"{r.bf16_us:.2f}",
            f"{r.bf16_gbps:.1f}",
            f"{r.out_diff:.4g}",
            f"{r.state_diff:.4g}",
            f"{r.speedup:.2f}x",
        ]
        for r in results
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    line = "  ".join(headers[i].rjust(widths[i]) for i in range(len(headers)))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = [
        "  ".join(row[i].rjust(widths[i]) for i in range(len(headers))) for row in rows
    ]
    return "\n".join([line, sep, *body])


def _write_csv(path: Path, results: list[BenchResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ratio",
                "shape",
                "mode",
                "batch_size",
                "tokens_per_req",
                "head_dim",
                "state_shape",
                "input_shape",
                "ape_shape",
                "out_shape",
                "fp32_state_mib",
                "bf16_state_mib",
                "input_mib",
                "ape_mib",
                "out_mib",
                "fp32_effective_mib",
                "bf16_effective_mib",
                "fp32_us",
                "fp32_gbps",
                "bf16_us",
                "bf16_gbps",
                "out_diff",
                "state_diff",
                "speedup",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "ratio": r.ratio,
                    "shape": r.shape_name,
                    "mode": r.mode,
                    "batch_size": r.batch_size,
                    "tokens_per_req": r.tokens_per_req,
                    "head_dim": r.head_dim,
                    "state_shape": r.state_shape,
                    "input_shape": r.input_shape,
                    "ape_shape": r.ape_shape,
                    "out_shape": r.out_shape,
                    "fp32_state_mib": r.fp32_state_mib,
                    "bf16_state_mib": r.bf16_state_mib,
                    "input_mib": r.input_mib,
                    "ape_mib": r.ape_mib,
                    "out_mib": r.out_mib,
                    "fp32_effective_mib": r.fp32_effective_mib,
                    "bf16_effective_mib": r.bf16_effective_mib,
                    "fp32_us": r.fp32_us,
                    "fp32_gbps": r.fp32_gbps,
                    "bf16_us": r.bf16_us,
                    "bf16_gbps": r.bf16_gbps,
                    "out_diff": r.out_diff,
                    "state_diff": r.state_diff,
                    "speedup": r.speedup,
                }
            )


def _preset_args(**overrides):
    args = dict(
        shape_presets=["all"],
        model_shapes=None,
        shape_tier="ci",
        ratios=None,
        modes=None,
        batch_sizes=None,
        decode_batch_sizes=None,
        prefill_batch_sizes=None,
        decode_seq_lens=None,
        prefill_extend_lens=None,
        head_dim=512,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_flash_pro_shape_presets_cover_compress_paths() -> None:
    specs = list(_make_shape_specs(_preset_args()))
    keys = {
        (s.shape_name, s.ratio, s.mode, s.batch_size, s.tokens_per_req) for s in specs
    }

    for shape_name in ("flash", "pro"):
        assert (shape_name, 4, "decode", 16, 128) in keys
        assert (shape_name, 4, "prefill", 1, 128) in keys
        assert (shape_name, 128, "decode", 16, 128) in keys
        assert (shape_name, 128, "prefill", 1, 128) in keys

    assert all(s.head_dim == 512 for s in specs)
    assert all(s.tokens_per_req % s.ratio == 0 for s in specs)


@pytest.mark.parametrize(
    "spec",
    [
        BenchSpec("flash", 4, "decode", 2, 512, 128),
        BenchSpec("flash", 4, "prefill", 1, 512, 16),
        BenchSpec("pro", 128, "decode", 2, 512, 128),
        BenchSpec("pro", 128, "prefill", 1, 512, 128),
    ],
)
def test_compress_state_benchmark_cuda_smoke(spec: BenchSpec) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepSeek V4 compress benchmark smoke.")

    result = _benchmark_spec(spec, warmup=1, iters=1, seed=20260603)
    assert math.isfinite(result.fp32_us) and result.fp32_us > 0
    assert math.isfinite(result.bf16_us) and result.bf16_us > 0
    assert result.out_diff < 0.1
    assert result.state_diff < 0.1


# Captured from DeepSeek-V4-Flash runtime on zc01 with EAGLE enabled:
# /data00/eval_results/operator_bench/runtime_shape_20260603/shapes.jsonl
# The service used prefill-style compress plans for both EXTEND and TARGET_VERIFY.
CAPTURED_FLASH_RUNTIME_SPECS = [
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 1),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 6),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 28),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 29),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 30),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 284),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 285),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 286),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 287),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 288),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 289),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 1, 128, 512),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 142),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 143),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 144),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 2, 128, 145),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 3, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 4, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 5, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 6, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 7, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 8, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 10, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 12, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 14, 128, 4),
    BenchSpec("flash-runtime-indexer", 4, "prefill", 16, 128, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 1),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 6),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 28),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 29),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 30),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 284),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 285),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 286),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 287),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 288),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 289),
    BenchSpec("flash-runtime-core", 4, "prefill", 1, 512, 512),
    BenchSpec("flash-runtime-core", 4, "prefill", 2, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 2, 512, 142),
    BenchSpec("flash-runtime-core", 4, "prefill", 2, 512, 143),
    BenchSpec("flash-runtime-core", 4, "prefill", 2, 512, 144),
    BenchSpec("flash-runtime-core", 4, "prefill", 2, 512, 145),
    BenchSpec("flash-runtime-core", 4, "prefill", 3, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 4, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 5, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 6, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 7, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 8, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 10, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 12, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 14, 512, 4),
    BenchSpec("flash-runtime-core", 4, "prefill", 16, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 1),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 6),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 28),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 29),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 30),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 284),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 285),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 286),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 287),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 288),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 289),
    BenchSpec("flash-runtime-core", 128, "prefill", 1, 512, 512),
    BenchSpec("flash-runtime-core", 128, "prefill", 2, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 2, 512, 142),
    BenchSpec("flash-runtime-core", 128, "prefill", 2, 512, 143),
    BenchSpec("flash-runtime-core", 128, "prefill", 2, 512, 144),
    BenchSpec("flash-runtime-core", 128, "prefill", 2, 512, 145),
    BenchSpec("flash-runtime-core", 128, "prefill", 3, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 4, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 5, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 6, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 7, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 8, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 10, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 12, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 14, 512, 4),
    BenchSpec("flash-runtime-core", 128, "prefill", 16, 512, 4),
]


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.numel() == 0:
        return 0.0
    return (lhs.float() - rhs.float()).abs().max().item()


def _benchmark_runtime_spec(
    spec: BenchSpec,
    warmup: int,
    iters: int,
    seed: int,
) -> BenchResult:
    fp32_case, bf16_case = _make_pair(spec, seed)
    fp32_case.run()
    bf16_case.run()
    torch.cuda.synchronize()

    fp32_us = _time_us(fp32_case.run, warmup=warmup, iters=iters)
    bf16_us = _time_us(bf16_case.run, warmup=warmup, iters=iters)

    fp32_diff_case, bf16_diff_case = _make_pair(spec, seed)
    fp32_out = fp32_diff_case.run()
    bf16_out = bf16_diff_case.run()
    torch.cuda.synchronize()

    out_diff = _max_abs_diff(fp32_out, bf16_out)
    state_diff = _max_abs_diff(
        fp32_diff_case.state_pool,
        bf16_diff_case.state_pool,
    )

    return BenchResult(
        shape_name=spec.shape_name,
        ratio=spec.ratio,
        mode=spec.mode,
        batch_size=spec.batch_size,
        tokens_per_req=spec.tokens_per_req,
        head_dim=spec.head_dim,
        state_shape=_shape_str(fp32_case.state_pool),
        input_shape=_shape_str(fp32_case.kv_score_input),
        ape_shape=_shape_str(fp32_case.ape),
        out_shape=_shape_str(fp32_case.out),
        fp32_state_mib=_tensor_mib(fp32_case.state_pool),
        bf16_state_mib=_tensor_mib(bf16_case.state_pool),
        input_mib=_tensor_mib(fp32_case.kv_score_input),
        ape_mib=_tensor_mib(fp32_case.ape),
        out_mib=_tensor_mib(fp32_case.out),
        fp32_effective_mib=_mib(fp32_case.effective_bytes),
        bf16_effective_mib=_mib(bf16_case.effective_bytes),
        fp32_us=fp32_us,
        fp32_gbps=_gbps(fp32_case.effective_bytes, fp32_us),
        bf16_us=bf16_us,
        bf16_gbps=_gbps(bf16_case.effective_bytes, bf16_us),
        out_diff=out_diff,
        state_diff=state_diff,
    )


def test_captured_flash_runtime_specs_are_unique() -> None:
    keys = {
        (s.shape_name, s.ratio, s.mode, s.batch_size, s.head_dim, s.tokens_per_req)
        for s in CAPTURED_FLASH_RUNTIME_SPECS
    }
    assert len(keys) == len(CAPTURED_FLASH_RUNTIME_SPECS)
    assert len(CAPTURED_FLASH_RUNTIME_SPECS) == 84
    assert any(s.head_dim == 128 and s.ratio == 4 for s in CAPTURED_FLASH_RUNTIME_SPECS)
    assert any(s.head_dim == 512 and s.ratio == 4 for s in CAPTURED_FLASH_RUNTIME_SPECS)
    assert any(
        s.head_dim == 512 and s.ratio == 128 for s in CAPTURED_FLASH_RUNTIME_SPECS
    )


@pytest.mark.parametrize("spec", CAPTURED_FLASH_RUNTIME_SPECS)
def test_compress_state_runtime_shape_cuda_smoke(spec: BenchSpec) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepSeek V4 runtime compress shape smoke.")

    result = _benchmark_runtime_spec(spec, warmup=1, iters=1, seed=20260603)
    assert math.isfinite(result.fp32_us) and result.fp32_us > 0
    assert math.isfinite(result.bf16_us) and result.bf16_us > 0
    assert result.out_diff < 0.1
    assert result.state_diff < 0.1


def _run_benchmark(args: argparse.Namespace) -> int:
    ci = is_in_ci()
    args.warmup = args.warmup if args.warmup is not None else (5 if ci else 10)
    args.iters = args.iters if args.iters is not None else (20 if ci else 30)

    if not torch.cuda.is_available():
        print("[skip] CUDA is required for this benchmark.")
        return 0
    if args.head_dim % 128 != 0:
        raise ValueError("--head-dim must be a multiple of 128.")

    print("DeepSeek V4 compress state dtype benchmark")
    print(
        "effective GB/s = "
        "state_pool + kv_score_input + ape + output footprint / kernel time"
    )

    if args.shape_source == "runtime":
        specs = CAPTURED_FLASH_RUNTIME_SPECS
        if args.limit is not None:
            specs = specs[: args.limit]
        print(
            f"config: shape_source=runtime, cases={len(specs)}, "
            f"warmup={args.warmup}, iters={args.iters}"
        )
        results = [
            _benchmark_runtime_spec(
                spec,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
            for spec in specs
        ]
    else:
        args.shape_presets = args.shape_presets or args.model_shapes or ["all"]
        args.shape_tier = args.shape_tier or ("ci" if ci else "smoke")
        if args.decode_seq_len is not None:
            args.decode_seq_lens = [args.decode_seq_len]
        if args.prefill_extend_len is not None:
            args.prefill_extend_lens = [args.prefill_extend_len]

        specs = list(_make_shape_specs(args))
        if args.limit is not None:
            specs = specs[: args.limit]
        print(
            f"config: shape_source=preset, shape_presets={args.shape_presets}, "
            f"shape_tier={args.shape_tier}, ratios={args.ratios}, "
            f"modes={args.modes}, batch_sizes={args.batch_sizes}, "
            f"decode_batch_sizes={args.decode_batch_sizes}, "
            f"prefill_batch_sizes={args.prefill_batch_sizes}, "
            f"head_dim={args.head_dim}, decode_seq_lens={args.decode_seq_lens}, "
            f"prefill_extend_lens={args.prefill_extend_lens}, "
            f"cases={len(specs)}, warmup={args.warmup}, iters={args.iters}"
        )
        results = [
            _benchmark_spec(
                spec,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
            for spec in specs
        ]

    print(_format_table(results))
    if args.csv:
        _write_csv(args.csv, results)
        print(f"\nWrote CSV: {args.csv}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepSeek V4 compress state tests, runtime-shape replay benchmark, "
            "or synthetic Flash/Pro shape benchmark."
        )
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--shape-source",
        choices=("runtime", "preset"),
        default="runtime",
        help=(
            "runtime replays captured serving shapes; preset runs synthetic "
            "Flash/Pro/custom shape grids."
        ),
    )
    parser.add_argument(
        "--shape-presets",
        nargs="+",
        choices=("flash", "pro", "all", "custom"),
        default=None,
        help=(
            "Preset shapes for --shape-source preset. 'all' covers Flash and Pro. "
            "Use 'custom' with manual shape args."
        ),
    )
    parser.add_argument(
        "--model-shapes",
        nargs="+",
        choices=("flash", "pro", "all", "custom"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shape-tier",
        choices=("ci", "smoke", "full"),
        default=None,
        help=(
            "Shape grid size for preset shapes. Defaults to ci in CI and smoke "
            "otherwise."
        ),
    )
    parser.add_argument("--ratios", type=int, nargs="+", choices=(4, 128), default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("decode", "prefill"),
        default=None,
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--decode-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--prefill-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument(
        "--decode-seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="Decode seq_len values. Overrides preset decode seq_len grid.",
    )
    parser.add_argument(
        "--prefill-extend-lens",
        type=int,
        nargs="+",
        default=None,
        help="Prefill extend_len values. Overrides preset prefill extend_len grid.",
    )
    parser.add_argument(
        "--decode-seq-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prefill-extend-len",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit benchmark cases after shape expansion.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        sys.exit(_run_benchmark(_parse_args(sys.argv[1:])))
    sys.exit(pytest.main([__file__, "-v"]))
