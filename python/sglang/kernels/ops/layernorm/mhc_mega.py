"""SGLang DSv4.1 prefill integration of Mega mHC.

Credit: DeepSeek's DeepGEMM project provides the fused Mega mHC kernel.
https://github.com/deepseek-ai/DeepGEMM
"""

import torch

from sglang.srt.utils import is_cuda

if is_cuda():
    import deep_gemm


def mhc_mega_boundary(
    x,
    residual,
    pre,
    post,
    comb,
    fn,
    scale,
    base,
    weight,
    rms_eps,
    hc_eps,
    norm_eps,
    sinkhorn_iters,
    emit_mxfp8: bool = False,
):
    """Update the residual, collapse with carried pre, and compute next stats.

    The collapse uses ``pre`` from the preceding sublayer, not the new pre
    produced from the updated residual. Preserve this DSv4.1 shifted ordering.
    Used by the opt-in SM10x prefill path. Before CUDA graph capture, warm up
    on the capture stream to initialize DeepGEMM's per-stream barriers.
    """
    rows, hidden = x.shape
    hc = residual.shape[1]
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    next_pre = torch.empty((rows, hc, 1), device=x.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    y_fp8 = y_gemm_sf = None
    if emit_mxfp8:
        y_fp8 = torch.empty((rows, hidden), device=x.device, dtype=torch.float8_e4m3fn)
        rows_aligned = (rows + 3) // 4 * 4
        y_gemm_sf = torch.empty(
            (hidden // 128, rows_aligned), device=x.device, dtype=torch.int32
        ).mT[:rows]
    deep_gemm.mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=pre.unsqueeze(-1),
        post_mix=post.unsqueeze(-1),
        comb_res_mix=comb,
        fn=fn,
        mix_scales=scale,
        mix_bases=base,
        hc_mult=hc,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=hc_eps,
        num_sinkhorn_iters=sinkhorn_iters,
        rmsnorm_weight=weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=updated,
        new_prev_mix=next_pre,
        new_post_mix=next_post,
        new_comb_res_mix=next_comb,
        y_bf16=normalized,
        y_fp8=y_fp8,
        y_gemm_sf=y_gemm_sf,
    )
    stats = (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)
    if emit_mxfp8:
        return updated, normalized, stats, (y_fp8, y_gemm_sf)
    return updated, normalized, stats
