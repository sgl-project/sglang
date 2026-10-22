import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.attention.linear.kernels.kda_aiter import (
    AiterKDAPrefillKernel,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_TOTAL_TOKENS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
_NUM_SEQUENCES = (1, 2, 3, 4, 8)


def _balanced_lengths(total: int, num_sequences: int) -> list[int]:
    quotient, remainder = divmod(total, num_sequences)
    return [
        quotient + (1 if index < remainder else 0) for index in range(num_sequences)
    ]


CASES = {
    f"h{heads}_n{num_sequences}_t{total}": (
        heads,
        _balanced_lengths(total, num_sequences),
    )
    for heads in (16, 8)
    for num_sequences in _NUM_SEQUENCES
    for total in _TOTAL_TOKENS
}


@marker.parametrize(
    "case",
    list(CASES),
    [
        "h16_n1_t8192",
        "h16_n3_t16384",
        "h8_n2_t16384",
        "h8_n3_t16384",
        "h8_n4_t16384",
    ],
)
@marker.benchmark("implementation", ["current", "aiter"])
def benchmark_kda_aiter_prefill(case: str, implementation: str):
    heads, lengths = CASES[case]
    total = sum(lengths)
    generator = torch.Generator(device="cuda").manual_seed(total + heads)
    shape = (1, total, heads, 128)

    def randn(*size, dtype=torch.bfloat16, scale=0.02):
        return (
            torch.randn(
                *size,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    q = randn(*shape)
    k = randn(*shape)
    v = randn(*shape)
    g = randn(*shape)
    beta = randn(1, total, heads)
    A_log = torch.log(
        torch.empty(heads, device="cuda").uniform_(1, 16, generator=generator)
    )
    dt_bias = randn(heads * 128, dtype=torch.float32)
    ssm_states = randn(
        len(lengths),
        heads,
        128,
        128,
        scale=0.01,
    )
    cache_indices = torch.arange(
        len(lengths),
        device="cuda",
        dtype=torch.int64,
    )
    query_start_loc = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device="cuda",
        dtype=torch.int64,
    )
    triton_kernel = TritonKDAKernel()
    kernel = (
        triton_kernel
        if implementation == "current"
        else AiterKDAPrefillKernel(triton_fallback=triton_kernel)
    )

    def fn():
        return kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
            beta_is_raw=True,
        )

    return marker.do_bench(
        fn,
        input_args=(),
        graph_clone_args=(),
        graph_clone_kwargs=(),
        memory_args=None,
        memory_output=None,
    )


if __name__ == "__main__":
    benchmark_kda_aiter_prefill.run()
