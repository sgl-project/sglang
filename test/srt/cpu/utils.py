import math

import torch
import torch.nn.functional as F

precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}


BLOCK_N, BLOCK_K = 64, 128
factor_for_scale = 1e-3
fp8_max, fp8_min = 400, -400


def SiluAndMul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def GeluAndMul(x: torch.Tensor, approximate="tanh") -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate=approximate) * x[..., d:]


def per_token_quant_int8(x):
    x = x.float()
    absmax = x.abs().max(dim=-1).values
    absmax = absmax.clamp_min(1e-10).unsqueeze(-1)
    scale_x = absmax / 127
    x_q = x.mul(127 / absmax)
    x_q = torch.round(x_q).to(torch.int8)

    return x_q, scale_x


def convert_weight(weight, scale_block_size, A_dtype):
    N, K = weight.size()
    fp8_max = 448.0
    scale_block_size_N, scale_block_size_K = scale_block_size  # (128, 128)

    pad_N = (scale_block_size_N - (N % scale_block_size_N)) % scale_block_size_N
    pad_K = (scale_block_size_K - (K % scale_block_size_K)) % scale_block_size_K

    if pad_N > 0 or pad_K > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))

    weight_blocks = weight.view(
        math.ceil(N / scale_block_size_N),
        scale_block_size_N,
        math.ceil(K / scale_block_size_K),
        scale_block_size_K,
    )  # (8, 128, 8, 128)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3).contiguous()  # (8, 8, 128, 128)

    # Step 2: compute per-block max abs values → scale
    abs_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True)  # (8, 8, 1, 1)
    scales = abs_max / fp8_max
    scales = torch.where(
        scales == 0, torch.ones_like(scales), scales
    )  # avoid division by zero

    q_fp8 = (weight_blocks / scales).to(torch.float8_e4m3fn)
    q_fp8_reshape = q_fp8.permute(0, 2, 1, 3).contiguous()

    if pad_N > 0 or pad_K > 0:
        q_fp8_reshape = q_fp8_reshape.view(N + pad_N, K + pad_K)
        q_fp8_reshape = q_fp8_reshape[:N, :K].contiguous()
    else:
        q_fp8_reshape = q_fp8_reshape.view(N, K)

    dq_weight = q_fp8.float() * scales
    dq_weight = dq_weight.permute(0, 2, 1, 3).contiguous()  # (8, 128, 8, 128)

    if pad_N > 0 or pad_K > 0:
        w_dq = dq_weight.view(N + pad_N, K + pad_K).to(A_dtype)
        w_dq = w_dq[:N, :K].contiguous()
    else:
        w_dq = dq_weight.view(N, K).to(A_dtype)

    scales = scales.view(
        math.ceil(N / scale_block_size_N), math.ceil(K / scale_block_size_K)
    )

    return q_fp8_reshape, scales, w_dq


def native_w8a8_per_token_matmul(A, B, As, Bs, bias, output_dtype=torch.bfloat16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    if bias is not None:
        C.add_(bias.view(1, -1))

    return C.reshape(origin_C_shape).to(output_dtype)


def torch_naive_moe(a, w1, w2, b, routed_scaling_factor):

    ic1 = torch.matmul(a, w1.transpose(0, 1))
    ic2 = SiluAndMul(ic1)
    ic3 = torch.matmul(ic2, w2.transpose(0, 1))

    return ic3 + b * routed_scaling_factor


def torch_w8a8_per_column_moe(a, w1_q, w2_q, w1_s, w2_s, b, routed_scaling_factor):

    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8(a)

    ic1 = native_w8a8_per_token_matmul(
        a_q, w1_q, a_s, w1_s, bias=None, output_dtype=torch.float32
    )
    ic2 = SiluAndMul(ic1)

    a1_q, a1_s = per_token_quant_int8(ic2)
    ic3 = native_w8a8_per_token_matmul(
        a1_q, w2_q, a1_s, w2_s, bias=None, output_dtype=torch.float32
    )

    return ic3 + b * routed_scaling_factor


def scaled_weight(weight, scales):
    E, N, K = weight.shape
    pad_N = (BLOCK_N - (N % BLOCK_N)) % BLOCK_N
    pad_K = (BLOCK_K - (K % BLOCK_K)) % BLOCK_K

    if pad_N > 0 or pad_K > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))

    weight_block = (
        weight.view(E, math.ceil(N / BLOCK_N), BLOCK_N, math.ceil(K / BLOCK_K), BLOCK_K)
        .permute(0, 1, 3, 2, 4)
        .float()
        .contiguous()
    )

    weight_scaled = (
        (
            weight_block
            * scales.view(E, math.ceil(N / BLOCK_N), math.ceil(K / BLOCK_K), 1, 1)
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if pad_N > 0 or pad_K > 0:
        weight_scaled = weight_scaled.view(E, N + pad_N, K + pad_K)
        weight_scaled = weight_scaled[..., :N, :K].contiguous()
    else:
        weight_scaled = weight_scaled.view(E, N, K)
    return weight_scaled


def torch_naive_fused_moe(a, w1, w2, score, topk, renormalize):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def torch_w8a8_per_column_fused_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk):
    """This function performs fused moe with per-column int8 quantization using native torch."""

    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8(a)
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # Also repeat the scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=torch.float32, device=a.device)

    # Calculate routing
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # First MLP layer: note that a_s is now per-token
            inter_out = native_w8a8_per_token_matmul(
                a_q[mask],
                w1[i],
                a_s[mask],
                w1_s[i],
                bias=None,
                output_dtype=torch.float32,
            )
            # Activation function
            act_out = SiluAndMul(inter_out)
            # Quantize activation output with per-token
            act_out_q, act_out_s = per_token_quant_int8(act_out)
            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q,
                w2[i],
                act_out_s,
                w2_s[i],
                bias=None,
                output_dtype=torch.float32,
            )
    # Apply routing weights and sum
    return (
        (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype))
        .sum(dim=1)
        .to(a.dtype)
    )


def native_fp8_fused_moe(a, w1, w2, topk_weight, topk_ids, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D).float()
    out = torch.zeros(B * topk, w2.shape[1], dtype=torch.float32, device=a.device)

    # Calculate routing
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            ic0 = torch.matmul(a[mask], w1[i].transpose(0, 1))
            ic1 = SiluAndMul(ic0)
            out[mask] = torch.matmul(ic1, w2[i].transpose(0, 1))

    return (
        (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype))
        .sum(dim=1)
        .to(a.dtype)
    )


def make_non_contiguous(x: torch.Tensor) -> torch.Tensor:
    """
    Make a tensor non-contiguous by slicing it via last dimension.
    """
    last_dim = x.shape[-1]
    return x[..., : last_dim // 2] if x.is_contiguous() else x
