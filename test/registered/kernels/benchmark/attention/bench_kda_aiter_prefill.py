import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.chunk_kda_aiter import aiter_chunk_kda_fwd
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="jit-kernel-benchmark-test-amd")

CASES = {
    "tp4_n3": (16, [5462, 5461, 5461]),
    "tp8_n2": (8, [8192, 8192]),
    "tp8_n3": (8, [5462, 5461, 5461]),
    "tp8_n4": (8, [4096, 4096, 4096, 4096]),
}


@marker.parametrize("case", list(CASES), ["tp4_n3", "tp8_n3"])
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

    inputs = {
        "q": randn(*shape),
        "k": randn(*shape),
        "v": randn(*shape),
        "g": randn(*shape),
        "beta": randn(1, total, heads),
        "A_log": torch.log(
            torch.empty(heads, device="cuda").uniform_(1, 16, generator=generator)
        ),
        "dt_bias": randn(heads * 128, dtype=torch.float32),
        "initial_state": randn(
            len(lengths),
            heads,
            128,
            128,
            scale=0.01,
        ),
        "initial_state_indices": torch.arange(
            len(lengths),
            device="cuda",
            dtype=torch.int64,
        ),
        "cu_seqlens": torch.tensor(
            [0] + torch.tensor(lengths).cumsum(0).tolist(),
            device="cuda",
            dtype=torch.int64,
        ),
    }

    if implementation == "current":

        def fn():
            return chunk_kda(
                **inputs,
                lower_bound=-5.0,
                beta_is_raw=True,
                use_qk_l2norm_in_kernel=True,
            )

    else:

        def fn():
            return aiter_chunk_kda_fwd(
                **inputs,
                lower_bound=-5.0,
                scale=128**-0.5,
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
