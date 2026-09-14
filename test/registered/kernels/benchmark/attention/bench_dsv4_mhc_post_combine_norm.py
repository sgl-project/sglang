"""DSV4.1 mHC sublayer boundary at decode batch sizes: the fused JIT kernel
(cluster size x vector width sweep) against the production Triton pair, the
TileLang post kernel, FlashInfer's post kernel and a torch reference.

Every implementation reads x [m, 5120], residual [m, 4, 5120], the fp32
coefficients and the norm weight, and writes the new residual plus y, so the
GB/s column uses one common footprint (the fused kernel writes residual in
place; the baselines allocate a fresh one, as production does today).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.mhc import (
    HC_WIDTH,
    HIDDEN_DIM,
    mhc_post_combine_norm,
    mhc_post_combine_norm_reference,
)
from sglang.kernels.ops.layernorm import rmsnorm
from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
from sglang.kernels.ops.layernorm.mhc import hc_combine, mhc_post
from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="4-gpu-b200"
)

EPS = 1e-6
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 148, 152, 192, 256]


def make_inputs(m: int):
    dev = "cuda"
    x = torch.randn(m, HIDDEN_DIM, device=dev, dtype=torch.bfloat16)
    residual = torch.randn(m, HC_WIDTH, HIDDEN_DIM, device=dev, dtype=torch.bfloat16)
    post = 2.0 * torch.sigmoid(torch.randn(m, HC_WIDTH, device=dev))
    comb = torch.softmax(torch.randn(m, HC_WIDTH, HC_WIDTH, device=dev), dim=-1)
    pre = torch.sigmoid(torch.randn(m, HC_WIDTH, device=dev))
    weight = (torch.rand(HIDDEN_DIM, device=dev) + 0.5).to(torch.bfloat16)
    return x, residual, post, comb, pre, weight


def make_fused(cluster_size: int, vec_size: int):
    def fused(x, residual, post, comb, pre, weight):
        return mhc_post_combine_norm(
            x,
            residual,
            post,
            comb,
            pre,
            weight,
            EPS,
            cluster_size=cluster_size,
            vec_size=vec_size,
        )

    return fused


def fused_auto(x, residual, post, comb, pre, weight):
    return mhc_post_combine_norm(x, residual, post, comb, pre, weight, EPS)


def combine_norm(new_residual, pre, weight):
    """Production combine + norm: the fused Triton kernel up to 8 tokens, else
    the Triton combine followed by the AOT rmsnorm."""
    m = new_residual.shape[0]
    flat = new_residual.view(m, -1)
    if m <= 8:
        return hc_combine_norm(flat, pre, weight, EPS)
    return rmsnorm(hc_combine(flat, pre, HC_WIDTH, new_residual.dtype), weight, EPS)


def prod_pair(x, residual, post, comb, pre, weight):
    return combine_norm(mhc_post_split_h(x, residual, post, comb), pre, weight)


def post_only(x, residual, post, comb, pre, weight):
    return mhc_post_split_h(x, residual, post, comb)


def combine_only(x, residual, post, comb, pre, weight):
    return combine_norm(residual, pre, weight)


def tilelang_pair(x, residual, post, comb, pre, weight):
    return combine_norm(mhc_post(x, residual, post, comb), pre, weight)


def flashinfer_pair(x, residual, post, comb, pre, weight):
    from flashinfer.mhc import mhc_post as fi_mhc_post

    return combine_norm(fi_mhc_post(x, residual, post, comb), pre, weight)


def torch_reference(x, residual, post, comb, pre, weight):
    return mhc_post_combine_norm_reference(x, residual, post, comb, pre, weight, EPS)[1]


BASELINES = {
    "prod_pair": prod_pair,
    "post_only": post_only,
    "combine_only": combine_only,
    "tilelang_pair": tilelang_pair,
    "flashinfer_pair": flashinfer_pair,
    "torch": torch_reference,
}


def run(fn, m: int):
    x, residual, post, comb, pre, weight = make_inputs(m)
    footprint = (
        2 * residual.nbytes
        + 2 * x.nbytes
        + post.nbytes
        + comb.nbytes
        + pre.nbytes
        + weight.nbytes
    )
    return marker.do_bench(
        fn,
        input_args=(x, residual, post, comb, pre, weight),
        # every argument is read; residual is also written in place by the fused kernel
        graph_clone_args="all",
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=footprint,
    )


@marker.parametrize("m", BATCH_SIZES, [1, 2, 4, 8, 16, 33, 34, 48, 64])
@marker.benchmark("impl", ["fused_auto", "fused_cga5_v8", "fused_cga1_v8", *BASELINES])
def benchmark_vs_baselines(m: int, impl: str):
    if impl == "fused_auto":
        fn = fused_auto
    elif impl.startswith("fused_"):
        _, cga, vec = impl.split("_")  # fused_cga1_v8
        fn = make_fused(int(cga[3:]), int(vec[1:]))
    else:
        fn = BASELINES[impl]
        if impl == "flashinfer_pair":
            try:
                import flashinfer.mhc  # noqa: F401
            except ImportError:
                marker.skip("flashinfer.mhc not installed")
    return run(fn, m)


if __name__ == "__main__":
    # The TileLang post kernel consults the CP layout; pin a single-rank layout.
    with get_parallel().override(attn_cp_size=1):
        benchmark_vs_baselines.run()
