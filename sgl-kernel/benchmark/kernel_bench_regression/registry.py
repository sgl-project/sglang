"""Registry of kernel benchmark cases tracked for performance regression.

Each :class:`BenchCase` points at an existing ``bench_*.py`` file under
``sgl-kernel/benchmark/`` and the ``triton.testing.perf_report`` object inside it.
The runner imports that object, drives the kernel-under-test provider directly, and
records the median metric for every config. We deliberately reuse the existing
benchmarks instead of duplicating kernel-launch logic so the regression suite stays
in sync with whatever the benchmark authors maintain.

The suite is model-agnostic: adding a kernel is a one-line edit here. Tag each case
with the kernel categories it belongs to (``mla``, ``moe``, ``gemm``, ``fp8``,
``attention``, ...) so the suite can grow across model families without renaming.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class BenchCase:
    """One kernel benchmark tracked for performance regression.

    Attributes:
        case_id: Stable identifier used as the key in the ground-truth JSON. Never
            rename an existing one or the ground truth stops matching. By convention
            this mirrors the kernel's own name (e.g. the bench function it wraps).
        bench_file: File name (relative to ``sgl-kernel/benchmark/``) to import.
        mark_attr: Name of the ``triton.testing.perf_report`` object in that module.
        provider: The ``line_arg`` value of the kernel under test (NOT the display
            ``line_name``). The benchmark function branches on this string, e.g.
            ``"sgl-kernel"`` or ``"sglang"``.
        metric: Unit string for reporting, e.g. ``"us"``, ``"TFLOPs"``, ``"GB/s"``.
        higher_is_better: ``True`` for throughput (TFLOPs/GB-s), ``False`` for
            latency (us). Drives which direction counts as a regression.
        tags: Kernel categories for grouping/filtering, e.g. ``("mla", "gemm")``.
            Documentation/selection only -- never affects the comparison.
        component: Human-readable description of what the kernel computes.
        min_compute_capability: Skip the case when the current GPU is older than
            this ``(major, minor)``. e.g. CUTLASS MLA needs ``(10, 0)`` (Blackwell).
        extra_args: Constant keyword args forwarded to the benchmark function for
            every config (e.g. CUTLASS MLA's ``block_size`` / ``num_kv_splits``).
        configs_override: Optional explicit list of x-value tuples to override the
            benchmark's own ``x_vals``. Used to pin a small, representative and
            deterministic config set independent of the benchmark's CI shortcuts.
            Each tuple must line up with the benchmark's ``x_names`` order.
    """

    case_id: str
    bench_file: str
    mark_attr: str
    provider: str
    metric: str
    higher_is_better: bool
    tags: Tuple[str, ...]
    component: str
    min_compute_capability: Tuple[int, int] = (9, 0)
    extra_args: dict = field(default_factory=dict)
    configs_override: Optional[List[tuple]] = None


# NOTE: Keep this list small and high-signal. The goal is a fast regression gate
# over the kernels that dominate decode/prefill, not exhaustive coverage. More
# cases (and more model families) can be added incrementally; just append a
# BenchCase pointing at any existing bench_*.py perf_report object.
KERNEL_BENCH_CASES: List[BenchCase] = [
    BenchCase(
        case_id="dsv3_fused_a_gemm",
        bench_file="bench_dsv3_fused_a_gemm.py",
        mark_attr="benchmark",
        provider="sgl-kernel",
        metric="TFLOPs",
        higher_is_better=True,
        tags=("mla", "gemm"),
        component="MLA fused down-proj 'A' GEMM (7168x2112)",
        configs_override=[(1,), (8,), (16,)],
    ),
    BenchCase(
        case_id="dsv3_router_gemm_bf16_out",
        bench_file="bench_dsv3_router_gemm.py",
        mark_attr="benchmark_bf16_output",
        provider="sgl-kernel-256",
        metric="TFLOPs",
        higher_is_better=True,
        tags=("moe", "gemm"),
        component="MoE router GEMM, 256 experts, bf16 output",
        configs_override=[(1,), (8,), (16,)],
    ),
    BenchCase(
        case_id="dsv3_router_gemm_float_out",
        bench_file="bench_dsv3_router_gemm.py",
        mark_attr="benchmark_float_output",
        provider="sgl-kernel-256",
        metric="TFLOPs",
        higher_is_better=True,
        tags=("moe", "gemm"),
        component="MoE router GEMM, 256 experts, fp32 output",
        configs_override=[(1,), (8,), (16,)],
    ),
    BenchCase(
        case_id="moe_fused_gate",
        bench_file="bench_moe_fused_gate.py",
        mark_attr="benchmark",
        provider="kernel",
        metric="us",
        higher_is_better=False,
        tags=("moe", "gate"),
        component="MoE fused gate + grouped top-k selector",
    ),
    BenchCase(
        case_id="per_token_group_quant_8bit",
        bench_file="bench_per_token_group_quant_8bit.py",
        mark_attr="benchmark",
        provider="sglang",
        metric="us",
        higher_is_better=False,
        tags=("fp8", "quant"),
        component="FP8 blockwise per-token-group quant (blockwise GEMM input)",
    ),
    BenchCase(
        case_id="dsv4_q_norm_rope",
        bench_file="bench_dsv4_norm_rope.py",
        mark_attr="benchmark_q_norm_rope",
        provider="sglang",
        metric="us",
        higher_is_better=False,
        tags=("attention", "norm", "rope"),
        component="Fused Q RMSNorm + RoPE",
        configs_override=[(1, 8, 192), (16, 8, 192), (64, 16, 192)],
    ),
    BenchCase(
        case_id="cutlass_mla_decode",
        bench_file="bench_cutlass_mla.py",
        mark_attr="benchmark",
        provider="128 heads",
        metric="GB/s",
        higher_is_better=True,
        tags=("mla", "attention", "decode"),
        component="CUTLASS MLA decode attention (TP=1, 128 heads)",
        min_compute_capability=(10, 0),
        extra_args={"block_size": 128, "num_kv_splits": 1},
        configs_override=[(1, 1024), (8, 2048), (32, 4096)],
    ),
]


def get_cases(case_ids: Optional[List[str]] = None) -> List[BenchCase]:
    """Return all cases, or only those whose ``case_id`` is in ``case_ids``."""
    if not case_ids:
        return list(KERNEL_BENCH_CASES)
    wanted = set(case_ids)
    selected = [c for c in KERNEL_BENCH_CASES if c.case_id in wanted]
    known = {c.case_id for c in KERNEL_BENCH_CASES}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"unknown case id(s): {' '.join(unknown)}")
    return selected
