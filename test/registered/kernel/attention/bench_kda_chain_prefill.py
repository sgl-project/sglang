import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.chunk_kda_chain_gfx950 import (
    chunk_kda_chain_fwd,
)
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.kda import chunk_gla_fwd_o_gk
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="jit-kernel-benchmark-test-amd")


@marker.parametrize(
    "lengths",
    [
        (8192,),
        (8192, 8192),
        (5462, 5461, 5461),
        (4096, 4096, 4096, 4096),
    ],
)
@marker.parametrize("heads", [8, 16])
@marker.benchmark("implementation", ["current", "chain"])
def benchmark_kda_chain_prefill(lengths, heads, implementation):
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

    q, k, v, w = (randn(*shape) for _ in range(4))
    gk = randn(*shape, dtype=torch.float32, scale=0.001)
    A = randn(1, total, heads, 64)
    state = randn(len(lengths), heads, 128, 128, scale=0.01)
    state_indices = torch.arange(
        len(lengths),
        device="cuda",
        dtype=torch.int64,
    )
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(),
        device="cuda",
        dtype=torch.int64,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    if implementation == "current":

        def fn():
            h, v_new = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=v,
                gk=gk,
                initial_state=state,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
            return chunk_gla_fwd_o_gk(
                q=q,
                v=v_new,
                g=gk,
                A=A,
                h=h,
                o=v,
                scale=128**-0.5,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

    else:

        def fn():
            return chunk_kda_chain_fwd(
                q=q,
                k=k,
                v=v,
                w=w,
                gk=gk,
                A=A,
                scale=128**-0.5,
                initial_state=state,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
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
    benchmark_kda_chain_prefill.run()
