"""The q / k / v split + chunk_gated_delta_rule step of an eager GDN extend.

Inputs are the row-major [T, q | k | v] conv output and a ragged varlen batch.
- copies: q / k / v are copied first: one fused split copy when
  q + k + v <= MAX_FUSED_QKV_SPLIT_DIM columns, else the three contiguous
  copies FLA's input_guard makes.
- views: the split views go straight in (strided q / k / v reads).
- no_pad: views, plus no_padding_rows=True (no output zero fill).
Timed without a CUDA graph (prepare_chunk_indices reads cu_seqlens on the host).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.srt.layers.attention.linear.gdn_backend import MAX_FUSED_QKV_SPLIT_DIM
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEAD_DIM = 128


def make_step(impl: str, num_heads: int, num_v_heads: int, num_tokens: int):
    """Returns the step of ``impl`` and the state pool it updates in place."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    num_seqs = max(1, min(8, num_tokens // 2048))
    weights = [1.0 + 0.5 * (i % 3) for i in range(num_seqs)]
    lens = [int(num_tokens * w / sum(weights)) for w in weights]
    lens[-1] = num_tokens - sum(lens[:-1])
    dims = [num_heads * HEAD_DIM, num_heads * HEAD_DIM, num_v_heads * HEAD_DIM]
    mixed = torch.randn(num_tokens, sum(dims), generator=gen, device="cuda")
    a = torch.randn(num_tokens, num_v_heads, generator=gen, device="cuda")
    b = torch.randn(num_tokens, num_v_heads, generator=gen, device="cuda")
    mixed, a, b = (t.to(torch.bfloat16) for t in (mixed, a, b))
    A_log = torch.log(torch.rand(num_v_heads, generator=gen, device="cuda") * 15 + 1)
    dt_bias = torch.randn(num_v_heads, generator=gen, device="cuda")
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    num_slots = num_seqs + 8
    state = torch.randn(
        num_slots, num_v_heads, HEAD_DIM, HEAD_DIM, generator=gen, device="cuda"
    )
    state *= 0.05
    indices = torch.randperm(num_slots, generator=gen, device="cuda")[:num_seqs].to(
        torch.int32
    )
    cu_seqlens = torch.tensor(
        [sum(lens[:i]) for i in range(num_seqs + 1)], dtype=torch.int32, device="cuda"
    )
    shapes = [(num_heads, HEAD_DIM), (num_heads, HEAD_DIM), (num_v_heads, HEAD_DIM)]

    def views():
        parts = torch.split(mixed, dims, dim=-1)
        return [p.view(1, num_tokens, *s) for p, s in zip(parts, shapes)]

    def copies():
        if sum(dims) <= MAX_FUSED_QKV_SPLIT_DIM:
            return fused_qkv_split_gdn_prefill(
                mixed, num_heads, num_heads, num_v_heads, *[HEAD_DIM] * 3
            )
        return [t.contiguous() for t in views()]

    def step():
        q, k, v = copies() if impl == "copies" else views()
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=state,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            no_padding_rows=impl == "no_pad",
        )[0]

    return step, state


@marker.parametrize(
    "num_heads,num_v_heads",
    [(16, 32), (8, 16), (16, 48), (16, 64)],
    [(16, 32), (16, 48)],
)
@marker.parametrize("num_tokens", [512, 4096, 16384], [4096])
@marker.benchmark("impl", ["copies", "views", "no_pad"])
def benchmark(num_heads: int, num_v_heads: int, num_tokens: int, impl: str):
    step, _ = make_step(impl, num_heads, num_v_heads, num_tokens)
    return marker.do_bench(
        step, use_cuda_graph=False, disable_log_bandwidth=True, estimated_time_ms=100
    )


if __name__ == "__main__":
    benchmark.run()
