from dataclasses import dataclass

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fused_ltx25_decoder_rope
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


@dataclass(frozen=True)
class Case:
    name: str
    frames: int
    height: int
    width: int
    heads: int


CASES = {
    case.name: case
    for case in (
        Case("stage0", 18, 17, 30, 32),
        Case("stage4_tile", 16, 68, 96, 8),
        Case("stage5_tile", 31, 136, 192, 4),
    )
}
DIM_SPLIT = (16, 24, 24)


def make_tables(case: Case) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    tables = []
    for length, dim in zip(
        (case.frames, case.height, case.width), DIM_SPLIT, strict=True
    ):
        exponents = torch.arange(0, dim, 2, dtype=torch.float64, device="cuda") / dim
        inv_freqs = (1.0 / 10000.0**exponents).to(torch.float32)
        positions = torch.arange(length, dtype=torch.float32, device="cuda")
        angles = positions[:, None] * inv_freqs[None, :]
        tables.append((angles.cos(), angles.sin()))
    return tuple(tables)


def eager_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    def apply(hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = []
        offset = 0
        for axis, (dim, (cos, sin)) in enumerate(
            zip(DIM_SPLIT, tables, strict=True), 1
        ):
            chunk = hidden_states[..., offset : offset + dim]
            pairs = chunk.reshape(*chunk.shape[:-1], dim // 2, 2)
            even = pairs[..., 0].float()
            odd = pairs[..., 1].float()
            shape = [1, 1, 1, 1, 1, dim // 2]
            shape[axis] = cos.shape[0]
            cos_view = cos.reshape(shape)
            sin_view = sin.reshape(shape)
            rotated = torch.stack(
                [
                    even * cos_view - odd * sin_view,
                    even * sin_view + odd * cos_view,
                ],
                dim=-1,
            )
            outputs.append(rotated.reshape(chunk.shape).to(hidden_states.dtype))
            offset += dim
        return torch.cat(outputs, dim=-1)

    return apply(q), apply(k)


@marker.parametrize("case_name", list(CASES), ci_vals=["stage5_tile"])
@marker.benchmark("impl", ["eager", "jit"], unit="ms")
def benchmark(case_name: str, impl: str) -> marker.BenchResult:
    case = CASES[case_name]
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(
        1,
        case.frames,
        case.height,
        case.width,
        case.heads,
        64,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn_like(q)
    tables = make_tables(case)
    if impl == "eager":
        fn = eager_rope
    else:
        fn = lambda q, k, tables: fused_ltx25_decoder_rope(
            q,
            k,
            *tables[0],
            *tables[1],
            *tables[2],
            DIM_SPLIT[0],
            DIM_SPLIT[1],
        )
    return marker.do_bench(
        fn,
        input_args=(q, k, tables),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
