# SPDX-License-Identifier: Apache-2.0
"""Benchmark MiniMax H3's RMSNorm + packed-token indexed adaLN chain.

Run directly on an NVIDIA or AMD server::

    python3 test/registered/kernels/benchmark/diffusion/\
        bench_minimax_h3_indexed_rmsnorm_adaln.py

The ``split`` provider is H3's ``quality="lossless"`` operation: PyTorch
``RMSNorm`` followed by SGLang's in-place indexed BF16 modulation kernel. The
``fused`` provider is the Triton path mounted for ``quality="high"`` requests.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.testing

from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_scale_shift_bf16_,
)
from sglang.kernels.ops.diffusion.triton.indexed_rmsnorm_adaln import (
    fused_indexed_rmsnorm_adaln,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)
register_amd_ci(est_time=20, stage="jit-kernel-benchmark", runner_config="amd")

HIDDEN_SIZE = 5376
EPS = 1e-5
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class CaseSpec:
    name: str
    tokens: int
    adaln_rows: int
    index_pattern: str


FULL_CASES = (
    CaseSpec("h3_t128_m3_grouped", 128, 3, "grouped"),
    CaseSpec("h3_t1024_m6_grouped", 1024, 6, "grouped"),
    CaseSpec("h3_t4096_m12_grouped", 4096, 12, "grouped"),
    CaseSpec("h3_t8192_m24_grouped", 8192, 24, "grouped"),
    CaseSpec("h3_t4096_m12_random", 4096, 12, "random"),
)
CASE_BY_NAME = {case.name: case for case in FULL_CASES}
CASE_NAMES = get_benchmark_range(
    full_range=[case.name for case in FULL_CASES],
    ci_range=["h3_t1024_m6_grouped"],
)

LINE_VALS = ["split", "fused"]
LINE_NAMES = ["H3 RMSNorm + indexed adaLN", "Triton fused"]
STYLES = [("red", "-"), ("blue", "--")]


def _make_indices(case: CaseSpec, generator: torch.Generator) -> torch.Tensor:
    if case.index_pattern == "random":
        return torch.randint(
            case.adaln_rows,
            (case.tokens,),
            device=DEFAULT_DEVICE,
            dtype=torch.int64,
            generator=generator,
        )
    if case.index_pattern == "grouped":
        repeats = triton.cdiv(case.tokens, case.adaln_rows)
        return torch.arange(
            case.adaln_rows,
            device=DEFAULT_DEVICE,
            dtype=torch.int64,
        ).repeat_interleave(repeats)[: case.tokens]
    raise ValueError(f"unknown index pattern: {case.index_pattern}")


def make_inputs(case: CaseSpec) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEFAULT_DEVICE)
    generator.manual_seed(case.tokens * 8191 + case.adaln_rows * 127)
    shape = (case.tokens, HIDDEN_SIZE)
    modulation_shape = (case.adaln_rows, HIDDEN_SIZE)
    return {
        "x": torch.randn(
            shape,
            device=DEFAULT_DEVICE,
            dtype=DTYPE,
            generator=generator,
        ),
        "weight": torch.randn(
            HIDDEN_SIZE,
            device=DEFAULT_DEVICE,
            dtype=DTYPE,
            generator=generator,
        ),
        "shift": torch.randn(
            modulation_shape,
            device=DEFAULT_DEVICE,
            dtype=DTYPE,
            generator=generator,
        )
        * 0.1,
        "scale": torch.randn(
            modulation_shape,
            device=DEFAULT_DEVICE,
            dtype=DTYPE,
            generator=generator,
        )
        * 0.1,
        "indices": _make_indices(case, generator),
    }


def h3_split(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Current H3 operation, including its BF16 modulation boundaries."""
    normalized = F.rms_norm(
        inputs["x"],
        (HIDDEN_SIZE,),
        inputs["weight"],
        EPS,
    )
    return indexed_scale_shift_bf16_(
        normalized,
        inputs["shift"],
        inputs["scale"],
        inputs["indices"],
    )


def h3_fused(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return fused_indexed_rmsnorm_adaln(
        inputs["x"],
        inputs["weight"],
        inputs["shift"],
        inputs["scale"],
        inputs["indices"],
        EPS,
    )


def validate(case: CaseSpec) -> None:
    """Report exactness and fail only when the numerical error is material."""
    inputs = make_inputs(case)
    expected = h3_split(inputs)
    actual = h3_fused(inputs)
    exact = torch.equal(actual, expected)
    difference = (actual.float() - expected.float()).abs()
    max_abs = difference.max().item()
    mismatch_rate = (actual != expected).float().mean().item()
    print(
        f"{case.name}: exact={exact}, max_abs={max_abs:.6g}, "
        f"mismatch_rate={mismatch_rate:.6%}"
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case_name"],
        x_vals=CASE_NAMES,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="minimax-h3-indexed-rmsnorm-adaln",
        args={},
    )
)
def benchmark(case_name: str, provider: str) -> Tuple[float, float, float]:
    case = CASE_BY_NAME[case_name]
    inputs = make_inputs(case)
    fn = h3_split if provider == "split" else h3_fused
    return run_benchmark_no_cudagraph(lambda: fn(inputs))


if __name__ == "__main__":
    print("Validating MiniMax H3 indexed RMSNorm + adaLN fused kernel...")
    for case_name in CASE_NAMES:
        validate(CASE_BY_NAME[case_name])
    print("Running MiniMax H3 indexed RMSNorm + adaLN benchmark...")
    benchmark.run(print_data=True)
