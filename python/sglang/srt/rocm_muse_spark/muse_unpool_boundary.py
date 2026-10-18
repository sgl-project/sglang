"""BF16 unpooler split-K reduction fused with Muse's existing FFN boundary.

This path uses the Triton split-K dense producer. It does not
change expert normalization or communication, and introduces no assembly.
"""

import torch
import triton as tr
import triton.language as tl

from sglang.srt.rocm_muse_spark.dense_kernels import _dense_partial


def can_fuse_unpool(layer, x, boundary):
    if not layer.is_moe_layer or boundary is None:
        return False
    moe = layer.mlp
    weight = moe.post_expert_proj.weight if moe.post_expert_proj is not None else None
    return (
        torch.version.hip is not None
        and moe.tp_size == 8
        and moe.ep_size == 8
        and 0 < x.shape[0] <= 16
        and x.dtype == torch.bfloat16
        and x.shape[1] == 8192
        and x.is_contiguous()
        and weight is not None
        and weight.shape == (8192, 4096)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and moe.post_expert_proj.bias is None
        and boundary._post_norm_is_input_scale
        and boundary._use_residual_gate
        and boundary._post_norm.has_weight
        and boundary._pre_norm.has_weight
        and boundary._residual_add.gate.numel() == 8192
    )


@tr.jit
def _unpool_reduce_boundary(
    PARTIAL,
    STREAM,
    OUTPUT,
    NEW_STREAM,
    POST_WEIGHT,
    PRE_WEIGHT,
    ALPHA,
    BETA,
    M: tl.constexpr,
    H: tl.constexpr,
    SK: tl.constexpr,
    POST_EPS: tl.constexpr,
    PRE_EPS: tl.constexpr,
    PRE_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.arange(0, H)
    split = tl.arange(0, SK)
    partial = tl.load(PARTIAL + split[:, None] * M * H + row * H + h[None, :])
    # The original GEMM stores BF16 before the input-scale norm. Keep both
    # BF16 rounding boundaries; moving the norm into FP32 GEMM would differ.
    branch = tl.sum(partial, axis=0).to(tl.bfloat16).to(tl.float32)
    post_weight = tl.load(POST_WEIGHT + h).to(tl.float32)
    post_input = (branch * post_weight).to(tl.bfloat16).to(tl.float32)
    post_rstd = tl.rsqrt(tl.sum(post_input * post_input, axis=0) / H + POST_EPS)
    normed_branch = (post_input * post_rstd).to(tl.bfloat16).to(tl.float32)
    stream = tl.load(STREAM + row * H + h)
    alpha, beta = tl.load(ALPHA + h), tl.load(BETA + h)
    new_stream = alpha * stream + beta * normed_branch
    tl.store(NEW_STREAM + row * H + h, new_stream)
    pre_input = new_stream
    if PRE_BF16:
        pre_input = pre_input.to(tl.bfloat16).to(tl.float32)
    pre_rstd = tl.rsqrt(tl.sum(pre_input * pre_input, axis=0) / H + PRE_EPS)
    pre_weight = tl.load(PRE_WEIGHT + h).to(tl.float32)
    normalized = pre_input * pre_rstd * pre_weight
    tl.store(OUTPUT + row * H + h, normalized)


def unpool_boundary(x, weight, stream, boundary, cfg, *, pre_bf16=False, warps=8):
    m, k = x.shape
    h = weight.shape[0]
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous() and stream.is_contiguous()
    assert k == 4096 and h == 8192 and 0 < m <= 64
    assert stream.dtype == torch.float32 and stream.shape == (m, h)
    assert boundary._post_norm_is_input_scale and boundary._use_residual_gate
    assert boundary._post_norm.has_weight and boundary._pre_norm.has_weight
    assert boundary._residual_add.gate.numel() == h
    boundary._ensure_cached_weights(torch.bfloat16)
    sk = cfg["sk"]
    assert sk in (2, 4, 8)
    partial = torch.empty((sk, m, h), dtype=torch.float32, device=x.device)
    output = torch.empty((m, h), dtype=torch.bfloat16, device=x.device)
    new_stream = torch.empty_like(stream)
    _dense_partial[(h // cfg["bn"], tr.cdiv(m, 16), sk)](
        x,
        weight,
        partial,
        m,
        h,
        k,
        16,
        cfg["bn"],
        cfg["bk"],
        sk,
        cfg.get("cache", ".cg"),
        num_warps=cfg.get("warps", 4),
        num_stages=cfg.get("stages", 2),
        matrix_instr_nonkdim=16,
    )
    _unpool_reduce_boundary[(m,)](
        partial,
        stream,
        output,
        new_stream,
        boundary._cc_post_weight,
        boundary._cc_pre_weight,
        boundary._cc_alpha,
        boundary._cc_beta,
        m,
        h,
        sk,
        boundary._post_norm.eps,
        boundary._pre_norm.eps,
        pre_bf16,
        num_warps=warps,
        num_stages=1,
    )
    return output, new_stream
