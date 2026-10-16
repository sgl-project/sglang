import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fused_inplace_helios_qk_rope
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)


def _split(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> None:
    def apply(value: torch.Tensor) -> torch.Tensor:
        x_1, x_2 = value.unflatten(-1, (-1, 2)).unbind(-1)
        cos, sin = freqs.unsqueeze(-2).chunk(2, dim=-1)
        out = torch.empty_like(value)
        out[..., 0::2] = x_1 * cos[..., 0::2] - x_2 * sin[..., 1::2]
        out[..., 1::2] = x_1 * sin[..., 1::2] + x_2 * cos[..., 0::2]
        return out.type_as(value)

    apply(q)
    apply(k)


FN_MAP = {
    "eager": _split,
    "jit": fused_inplace_helios_qk_rope,
}


@marker.parametrize("tokens", [2160, 8640], [8640])
@marker.benchmark("impl", ["eager", "jit"])
def benchmark(tokens: int, impl: str):
    generator = torch.Generator(device="cuda").manual_seed(20260826)
    q = torch.randn(
        tokens,
        40,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k = torch.randn_like(q)
    freqs = torch.randn(
        tokens, 256, device="cuda", dtype=torch.float32, generator=generator
    )
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(q, k, freqs),
        memory_args=(q, k, freqs),
        memory_output=None,
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
