import math

import torch

precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}


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
