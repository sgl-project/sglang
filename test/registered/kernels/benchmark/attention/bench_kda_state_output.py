import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla import chunk_delta_h
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_o_128,
)
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.kda import chunk_gla_fwd_o_gk, chunk_kda
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="jit-kernel-benchmark-test-amd")


@marker.parametrize("tokens", [128, 256, 512, 768, 1024, 8192], [128, 512, 768])
@marker.parametrize("heads", [8, 16])
@marker.benchmark("implementation", ["current", "fused"])
def benchmark_kda_state_output(tokens: int, heads: int, implementation: str):
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    w = torch.randn_like(q)
    g = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.001
    A = torch.randn(
        (1, tokens, heads, 64),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        (1, heads, 128, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int64)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    if implementation == "current":

        def fn():
            h, v_new = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=v,
                gk=g,
                initial_state=state,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
            return chunk_gla_fwd_o_gk(
                q=q,
                v=v_new,
                g=g,
                A=A,
                h=h,
                o=v,
                scale=128**-0.5,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

    else:

        def fn():
            return chunk_gated_delta_rule_fwd_o_128(
                q=q,
                k=k,
                v=v,
                w=w,
                gk=g,
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


@marker.parametrize("tokens", [128, 256, 512, 768, 8192], [128, 256])
@marker.parametrize("heads", [8, 16])
@marker.benchmark("implementation", ["current", "fused"])
def benchmark_complete_kda(tokens: int, heads: int, implementation: str):
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
    k = torch.randn_like(q) * 0.02
    v = torch.randn_like(q) * 0.02
    g = torch.randn_like(q) * 0.02
    beta = torch.randn(
        (1, tokens, heads),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        (1, heads, 128, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int64)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)
    A_log = torch.randn(
        (1, 1, heads, 1),
        device="cuda",
        dtype=torch.float32,
    )
    dt_bias = torch.randn(heads * 128, device="cuda", dtype=torch.float32)

    def fn():
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            initial_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    original = chunk_delta_h.is_gfx95_supported
    chunk_delta_h.is_gfx95_supported = lambda: implementation == "fused"
    try:
        return marker.do_bench(
            fn,
            input_args=(),
            graph_clone_args=(),
            graph_clone_kwargs=(),
            memory_args=None,
            memory_output=None,
        )
    finally:
        chunk_delta_h.is_gfx95_supported = original


if __name__ == "__main__":
    benchmark_kda_state_output.run()
    benchmark_complete_kda.run()
