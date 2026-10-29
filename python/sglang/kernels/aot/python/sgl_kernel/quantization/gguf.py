import torch


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, M: int, N: int, dtype: torch.dtype
):
    assert M > 0 and N > 0, "GGUF weight Input shape must be of positive dimensions"
    return torch.ops.sgl_kernel.ggml_dequantize.default(weight, quant_type, M, N, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    return torch.ops.sgl_kernel.ggml_mul_mat_vec_a8.default(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    return torch.ops.sgl_kernel.ggml_mul_mat_a8.default(weight, x, quant_type, row)


def ggml_moe_a8(
    input: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_token_post_padded: torch.Tensor,
    type: int,
    row: int,
    topk: int,
    tokens: int,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.ggml_moe_a8.default(
        input,
        weight,
        sorted_token_ids,
        expert_ids,
        num_token_post_padded,
        type,
        row,
        topk,
        tokens,
    )


def ggml_moe_a8_vec(
    input: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.ggml_moe_a8_vec.default(
        input, weight, topk_ids, top_k, type, row, tokens
    )


def ggml_moe_get_block_size(type: int) -> int:
    return torch.ops.sgl_kernel.ggml_moe_get_block_size.default(type)


def ggml_moe_mmq(
    input: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    type: int,
    row: int,
    x_per_slot: bool,
) -> torch.Tensor:
    """MoE matmul with llama.cpp's MMQ (CUDA): returns fp32 [T, top_k, row] with
    out[t, s] = weight[topk_ids[t, s], :row] @ x(t, s), where x(t, s) is input[t]
    (gate / up) or input[t * top_k + s] (x_per_slot, the down projection). Slots
    with an expert id < 0 (or >= the number of experts) give zero rows; the
    experts of a token must be distinct (a repeated one also gives a zero row)."""
    return torch.ops.sgl_kernel.ggml_moe_mmq.default(
        input, weight, topk_ids, type, row, x_per_slot
    )
