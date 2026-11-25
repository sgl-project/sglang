"""Hy4-Flash iHC post + pre + RMSNorm at decode / DFly verify token counts: the
thread-block-cluster kernel vs the single-pass Triton kernels it takes over from,
and its fused fp8 output vs the separate per-token-group quant op."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.layernorm import hy4_ihc_cluster as cl
from sglang.kernels.ops.layernorm import hy4_ihc_single_pass as sp
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HC, HIDDEN = 4, 3072
EPS = dict(magnitude=2.0, norm_eps=1e-5, hc_eps=1e-6, rms_eps=1e-5)


def _inputs(T, has_out):
    g = torch.Generator(device="cuda").manual_seed(T)

    def randn(*shape):
        return torch.randn(shape, generator=g, device="cuda")

    return dict(
        output=randn(T, HIDDEN).bfloat16() if has_out else None,
        residual=randn(T, HC, HIDDEN).bfloat16(),
        post=(torch.sigmoid(randn(T, HC)) * 2.0) if has_out else None,
        hc_fn=randn(2 * HC, HC * HIDDEN) * 0.02,
        hc_scale=torch.tensor([0.4, 0.2], device="cuda"),
        hc_base=randn(2 * HC) * 0.5,
        rms_weight=(1.0 + 0.1 * randn(HIDDEN)).bfloat16(),
        out_norm_weight=(1.0 + 0.1 * randn(HIDDEN)).bfloat16() if has_out else None,
    )


def _cluster(quant):
    def run(output, residual, post, hc_fn, hc_scale, hc_base, rms_w, out_norm_w):
        return cl.hy4_ihc_cluster(
            output,
            residual,
            post,
            hc_fn,
            hc_scale,
            hc_base,
            rms_weight=rms_w,
            out_norm_weight=out_norm_w,
            out_norm_eps=1e-5 if out_norm_w is not None else 0.0,
            quant=quant,
            **EPS,
        )

    return run


def _cluster_then_quant(
    output, residual, post, hc_fn, hc_scale, hc_base, rms_w, out_norm_w
):
    y = _cluster(False)(
        output, residual, post, hc_fn, hc_scale, hc_base, rms_w, out_norm_w
    )[1]
    return sglang_per_token_group_quant_fp8(
        y, 128, column_major_scales=True, scale_tma_aligned=True
    )


def _single_pass(output, residual, post, hc_fn, hc_scale, hc_base, rms_w, out_norm_w):
    return sp.hy4_ihc_single_pass(
        output,
        residual,
        post,
        hc_fn,
        hc_scale,
        hc_base,
        rms_weight=rms_w,
        out_norm_weight=out_norm_w,
        out_norm_eps=1e-5 if out_norm_w is not None else 0.0,
        **EPS,
    )


FN_MAP = {
    "cluster": _cluster(False),
    "single_pass": _single_pass,
    "cluster_fp8": _cluster(True),
    "cluster_then_quant": _cluster_then_quant,
}
KEYS = ("output", "residual", "post", "hc_fn", "hc_scale", "hc_base", "rms_weight")


@marker.parametrize("T", [1, 2, 6, 8, 16, 32, 48, 64, 78, 90, 114, 128], [1, 32, 78])
@marker.parametrize("has_out", [True, False], [True])
@marker.benchmark("impl", list(FN_MAP), gate=["cluster", "cluster_fp8"])
def benchmark(T: int, has_out: bool, impl: str):
    device = torch.device("cuda")
    sp.prepare_hy4_ihc_single_pass(device=device, out_norm_variants=(True,))
    max_t = cl.hy4_ihc_cluster_max_tokens(
        hc_mult=HC,
        hidden_size=HIDDEN,
        has_output=has_out,
        out_norm=has_out,
        quant=impl == "cluster_fp8",
        device=device,
    )
    if T > max_t:
        marker.skip(f"T={T} is past one wave of clusters ({max_t})")
    x = _inputs(T, has_out)
    args = tuple(x[k] for k in KEYS) + (x["out_norm_weight"],)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=args,
        graph_clone_args=[i for i, a in enumerate(args) if a is not None],
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
