from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=13, suite="stage-b-kernel-benchmark-1-gpu-large")

MAX_SEQ_LEN = 131072
ROPE_BASE = 10000.0


@dataclass(frozen=True)
class CaseSpec:
    name: str
    batch_size: int
    num_tokens: int
    num_heads: int
    head_dim: int
    rope_dim: int
    is_neox: bool


BENCH_CASES = (
    CaseSpec("flux_1024", 1, 4096, 24, 128, 128, False),
    CaseSpec("qwen_image_1024", 1, 4096, 32, 128, 128, False),
    CaseSpec("qwen_image_partial", 1, 4096, 32, 128, 64, False),
    # Z-Image-Turbo default 1024x1024 config: dim=3840, num_heads=30 -> head_dim=128.
    CaseSpec("zimage_1024", 1, 4096, 30, 128, 128, False),
    CaseSpec("batch2_medium", 2, 2048, 24, 128, 128, False),
)
CASE_BY_NAME = {case.name: case for case in BENCH_CASES}
CASE_NAMES = get_benchmark_range(
    full_range=[case.name for case in BENCH_CASES],
    ci_range=[case.name for case in BENCH_CASES],
)
LINE_VALS = ["split", "fused"]
LINE_NAMES = ["JIT QKNorm + FlashInfer RoPE", "SGL JIT Fused QKNorm+RoPE"]
STYLES = [("red", "-"), ("blue", "--")]


def create_cos_sin_cache(
    rotary_dim: int,
    max_position: int = MAX_SEQ_LEN,
    base: float = ROPE_BASE,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEFAULT_DEVICE)
            / rotary_dim
        )
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEFAULT_DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def make_inputs(case: CaseSpec) -> dict[str, torch.Tensor | bool]:
    seed = (
        case.batch_size * 1_000_003
        + case.num_tokens * 8191
        + case.num_heads * 127
        + case.head_dim * 17
        + case.rope_dim
    )
    generator = torch.Generator(device=DEFAULT_DEVICE)
    generator.manual_seed(seed)
    return {
        "q": torch.randn(
            case.batch_size * case.num_tokens,
            case.num_heads,
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "k": torch.randn(
            case.batch_size * case.num_tokens,
            case.num_heads,
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "q_weight": torch.randn(
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "k_weight": torch.randn(
            case.head_dim,
            device=DEFAULT_DEVICE,
            dtype=DEFAULT_DTYPE,
            generator=generator,
        ),
        "positions": torch.randint(
            0,
            MAX_SEQ_LEN,
            (case.batch_size * case.num_tokens,),
            device=DEFAULT_DEVICE,
            dtype=torch.int64,
            generator=generator,
        ),
        "cos_sin_cache": create_cos_sin_cache(case.rope_dim),
        "is_neox": case.is_neox,
    }


def clone_inputs(
    inputs: dict[str, torch.Tensor | bool],
) -> dict[str, torch.Tensor | bool]:
    out: dict[str, torch.Tensor | bool] = {}
    for key, value in inputs.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def split_qknorm_rope(inputs: dict[str, torch.Tensor | bool]) -> None:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    from sglang.jit_kernel.norm import fused_inplace_qknorm

    q = inputs["q"]
    k = inputs["k"]
    q_weight = inputs["q_weight"]
    k_weight = inputs["k_weight"]
    positions = inputs["positions"]
    cos_sin_cache = inputs["cos_sin_cache"]
    is_neox = bool(inputs["is_neox"])

    fused_inplace_qknorm(q, k, q_weight, k_weight)
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q.view(q.shape[0], -1),
        key=k.view(k.shape[0], -1),
        head_size=q.shape[-1],
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def fused_qknorm_rope(inputs: dict[str, torch.Tensor | bool]) -> None:
    from sglang.jit_kernel.diffusion.qknorm_rope import fused_inplace_qknorm_rope

    fused_inplace_qknorm_rope(
        inputs["q"],
        inputs["k"],
        inputs["q_weight"],
        inputs["k_weight"],
        inputs["cos_sin_cache"],
        inputs["positions"],
        is_neox=bool(inputs["is_neox"]),
        rope_dim=inputs["cos_sin_cache"].shape[-1],
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case_name"],
        x_vals=CASE_NAMES,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="diffusion-qknorm-rope-performance",
        args={},
    )
)
def benchmark(case_name: str, provider: str) -> Tuple[float, float, float]:
    case = CASE_BY_NAME[case_name]
    inputs = make_inputs(case)
    fn = split_qknorm_rope if provider == "split" else fused_qknorm_rope
    return run_benchmark_no_cudagraph(lambda: fn(inputs))


if __name__ == "__main__":
    print("Running diffusion qknorm + rope performance benchmark...")
    benchmark.run(print_data=True)
