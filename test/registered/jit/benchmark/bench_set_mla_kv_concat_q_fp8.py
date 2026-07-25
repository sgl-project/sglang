"""Fused fp8 quantize + KV-scatter + q-concat vs the aten chain it replaces."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.set_mla_kv_concat_q import set_mla_kv_concat_q_fp8
from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NOPE, ROPE, TOTAL, PAGES = 512, 64, 576, 8192


def _aten_chain(pool, loc, k_nope, k_rope, q_nope, q_rope):
    q_fp8 = concat_mla_absorb_q_general(q_nope, q_rope).to(torch.float8_e4m3fn)
    row = torch.cat(
        [k_nope.to(torch.float8_e4m3fn), k_rope.to(torch.float8_e4m3fn)], dim=-1
    )
    pool[loc] = row
    return q_fp8


FN_MAP = {"fused": set_mla_kv_concat_q_fp8, "aten_chain": _aten_chain}


@marker.parametrize("bs", [1, 8, 64, 256], [1, 64])
@marker.parametrize("heads", [8], [8])
@marker.benchmark("impl", ["fused", "aten_chain"])
def benchmark(bs: int, heads: int, impl: str):
    if torch.cuda.get_device_capability()[0] < 9:
        marker.skip("requires SM90+ (TMA bulk store)")

    gen = torch.Generator(device="cuda").manual_seed(bs)
    pool = torch.zeros(PAGES, TOTAL, device="cuda", dtype=torch.float8_e4m3fn)
    latent = (
        torch.randn(bs, TOTAL, generator=gen, device="cuda", dtype=torch.float32)
        .mul(0.1)
        .to(torch.bfloat16)
    )
    q_all = (
        torch.randn(bs, heads, TOTAL, generator=gen, device="cuda", dtype=torch.float32)
        .mul(0.1)
        .to(torch.bfloat16)
    )
    loc = torch.randperm(PAGES, generator=gen, device="cuda")[:bs]
    args = (
        pool,
        loc,
        latent[:, :NOPE],
        latent[:, NOPE:],
        q_all[..., :NOPE],
        q_all[..., NOPE:],
    )
    return marker.do_bench(
        FN_MAP[impl],
        input_args=args,
        graph_clone_args=(0, 1, 2, 3, 4, 5),
        memory_args=(latent, q_all, loc),
    )


if __name__ == "__main__":
    benchmark.run()
