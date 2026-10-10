# SPDX-License-Identifier: Apache-2.0

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import (
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    GGUFReader,
    ReaderTensor,
    dequantize,
)
from huggingface_hub import snapshot_download
from sgl_kernel import (
    ggml_dequantize,
    ggml_moe_a8,
    ggml_moe_a8_vec,
    ggml_moe_get_block_size,
    ggml_mul_mat_a8,
    ggml_mul_mat_vec_a8,
    ggml_supports_iq_mmq,
    moe_align_block_size,
    silu_and_mul,
)

GGUF_SAMPLE = snapshot_download(
    "Isotr0py/test-gguf-sample",
    revision="d82b8773934ef260d8d8a896a7c197bc69a0fac1",
    allow_patterns="*.gguf",
)
GGUF_SAMPLE_MOE = snapshot_download(
    "SzymonOzog/test-gguf-moe-sample",
    revision="2b77eb27ae2ea4b3f68cf3042961509a9f5847b8",
    allow_patterns=[
        "Quant_IQ1_S_512.gguf",
        "Quant_IQ2_XXS_512.gguf",
        "Quant_IQ2_XS_512.gguf",
        "Quant_IQ2_S_512.gguf",
        "Quant_IQ3_XXS_512.gguf",
        "Quant_IQ3_S_512.gguf",
        "Quant_IQ4_NL_512.gguf",
        "Quant_IQ4_XS_512.gguf",
        "Quant_Q5_0_512.gguf",
    ],
)


def get_gguf_sample_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


def get_gguf_MoE_tensors(
    hidden_size: int, quant_type: GGMLQuantizationType
) -> list[ReaderTensor]:
    sample_dir = GGUF_SAMPLE_MOE
    filename = f"Quant_{quant_type.name}_{hidden_size}.gguf"
    sample_file = Path(sample_dir) / filename
    return GGUFReader(sample_file).tensors


DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
# Hidden_size for testing, must match the sample file in HF repo,
# we have `hidden_size = 256, 1024` for test in HF repo currently.
HIDDEN_SIZES = [256, 1024]
NUM_TOKENS = [7, 2050]  # Arbitrary values for testing
SEEDS = [0]
QUANT_TYPES = [
    # i-matrix
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
    # k-quants
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    # standard quantization
    GGMLQuantizationType.Q4_0,
    GGMLQuantizationType.Q5_0,
    GGMLQuantizationType.Q8_0,
]
MMQ_IMATRIX_QUANT_TYPES = [
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
]
MMQ_MOE_TEST_TYPES = MMQ_IMATRIX_QUANT_TYPES + [GGMLQuantizationType.Q5_0]
MMQ_MOE_TEST_TYPE_PAIRS = [
    (quant_type, quant_type) for quant_type in MMQ_MOE_TEST_TYPES
] + [
    (GGMLQuantizationType.IQ3_S, GGMLQuantizationType.Q5_0),
    (GGMLQuantizationType.Q5_0, GGMLQuantizationType.IQ3_S),
]
MMQ_K_ALIGNMENTS = {
    GGMLQuantizationType.Q4_0: 256,
    GGMLQuantizationType.Q5_0: 256,
    GGMLQuantizationType.Q8_0: 128,
    GGMLQuantizationType.Q2_K: 512,
    GGMLQuantizationType.Q3_K: 512,
    GGMLQuantizationType.Q4_K: 256,
    GGMLQuantizationType.Q5_K: 256,
    GGMLQuantizationType.Q6_K: 256,
    **{quant_type: 256 for quant_type in MMQ_IMATRIX_QUANT_TYPES},
}


def _moe_align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_token_ids = torch.empty(
        max_num_tokens_padded, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (max_num_tokens_padded + block_size - 1) // block_size,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty(
        num_experts + 2, dtype=torch.int32, device=topk_ids.device
    )
    moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_dequantize(
    hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType
):
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape_str = tensor.name.split("_")[-1]
        shape = map(int, shape_str.split("x"))

        ref_output = torch.tensor(
            dequantize(tensor.data, quant_type), device="cuda"
        ).to(dtype)
        output = ggml_dequantize(
            torch.tensor(tensor.data, device="cuda"), quant_type, *list(shape), dtype
        )

        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=4e-2)


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_mmvq(hidden_size: int, dtype: torch.dtype, quant_type: GGMLQuantizationType):

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((1, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ggml_mul_mat_vec_a8(qweight, x, quant_type, qweight.shape[0]).to(dtype)

        # NOTE(FlamingoPg): There can be occasional errors, Loosen the granularity of gguf bf16 verification.
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1}
        rtols = {torch.half: 1e-1, torch.bfloat16: 3e1, torch.float: 1e-1}

        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "quant_type",
    MMQ_IMATRIX_QUANT_TYPES
    + [
        # k-quants
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        # standard quants
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q5_0,
        GGMLQuantizationType.Q8_0,
    ],
)
@torch.inference_mode()
def test_mmq(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_type: GGMLQuantizationType,
):
    torch.manual_seed(0)
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        alignment = MMQ_K_ALIGNMENTS[quant_type]
        if hidden_size % alignment != 0:
            with pytest.raises(RuntimeError, match="requires an input size"):
                ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])
            continue
        output = ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])
        if quant_type in MMQ_IMATRIX_QUANT_TYPES:
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, ref_output, atol=1.5, rtol=1e-1)
            continue
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


def _reference_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty((x.shape[0], w2.shape[1]), dtype=x.dtype, device=x.device)
    for token_idx in range(x.shape[0]):
        expert_indices = topk_ids[token_idx].long()
        gate_up = torch.einsum("h,enh->en", x[token_idx], w13[expert_indices])
        activated = silu_and_mul(gate_up)
        expert_output = torch.einsum("ei,eoi->eo", activated, w2[expert_indices])
        output[token_idx] = torch.sum(
            expert_output * topk_weights[token_idx, :, None], dim=0
        )
    return output


def test_moe_block_size_dispatch():
    assert ggml_moe_get_block_size(GGMLQuantizationType.Q5_0) > 0


def test_iq_mmq_capability_dispatch():
    assert ggml_supports_iq_mmq()


def test_q8_1_is_rejected_by_unsupported_low_level_paths():
    qweight = torch.empty((1, 32), dtype=torch.uint8, device="cuda")
    x = torch.empty((1, 256), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match="Unsupported GGUF dequantization type"):
        ggml_dequantize(qweight, GGMLQuantizationType.Q8_1, 1, 256, torch.bfloat16)
    with pytest.raises(RuntimeError, match="Unsupported GGUF MMQ quantization type"):
        ggml_mul_mat_a8(qweight, x, GGMLQuantizationType.Q8_1, 1)
    with pytest.raises(RuntimeError, match="Unsupported GGUF MMVQ quantization type"):
        ggml_mul_mat_vec_a8(qweight, x, GGMLQuantizationType.Q8_1, 1)

    topk_ids = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    with pytest.raises(
        RuntimeError, match="Unsupported GGUF MoE vector quantization type"
    ):
        ggml_moe_a8_vec(
            x, qweight.unsqueeze(0), topk_ids, 1, GGMLQuantizationType.Q8_1, 1, 1
        )


@pytest.mark.parametrize(
    "quant_type,hidden_size",
    [
        (GGMLQuantizationType.Q5_0, 32),
        (GGMLQuantizationType.Q8_0, 96),
        (GGMLQuantizationType.Q2_K, 256),
        (GGMLQuantizationType.Q3_K, 256),
        (GGMLQuantizationType.IQ4_NL, 32),
        (GGMLQuantizationType.IQ4_NL, 96),
    ],
)
def test_mmq_rejects_unaligned_k(quant_type: GGMLQuantizationType, hidden_size: int):
    torch.manual_seed(0)
    tensor = get_gguf_sample_tensors(256, quant_type)[0]
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    packed_cols = hidden_size // block_size * type_size
    qweight = torch.tensor(tensor.data[:, :packed_cols], device="cuda")
    x = torch.rand((9, hidden_size), dtype=torch.bfloat16, device="cuda")

    error = "requires an input size divisible by"
    with pytest.raises(RuntimeError, match=error):
        ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])

    sorted_token_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    expert_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    num_tokens_post_padded = torch.ones(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match=error):
        ggml_moe_a8(
            x[:1],
            qweight.unsqueeze(0),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            quant_type,
            qweight.shape[0],
            1,
            1,
        )


def test_moe_mmq_invalid_expert_writes_zero():
    torch.manual_seed(0)
    quant_type = GGMLQuantizationType.IQ3_S
    tensor = get_gguf_MoE_tensors(512, quant_type)[0]
    qweight = torch.tensor(tensor.data, device="cuda")
    weight = dequantize(tensor.data, quant_type)
    x = torch.rand((1, weight.shape[-1]), dtype=torch.bfloat16, device="cuda")
    sorted_token_ids = torch.tensor([0, 1, 1, 1], dtype=torch.int32, device="cuda")
    expert_ids = torch.tensor([-1], dtype=torch.int32, device="cuda")
    num_tokens_post_padded = torch.tensor([4], dtype=torch.int32, device="cuda")

    output = ggml_moe_a8(
        x,
        qweight,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        weight.shape[-2],
        1,
        1,
    )
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)

    vector_output = ggml_moe_a8_vec(
        x,
        qweight,
        torch.tensor([[-1]], dtype=torch.int32, device="cuda"),
        1,
        quant_type,
        weight.shape[-2],
        1,
    )
    torch.testing.assert_close(
        vector_output, torch.zeros_like(vector_output), atol=0, rtol=0
    )


@pytest.mark.parametrize("quant_type,quant_type2", MMQ_MOE_TEST_TYPE_PAIRS)
@torch.inference_mode()
def test_moe_mmq(quant_type: GGMLQuantizationType, quant_type2: GGMLQuantizationType):
    torch.manual_seed(0)
    num_tokens = 65 if quant_type == quant_type2 == GGMLQuantizationType.Q5_0 else 128
    top_k = 4
    w13_tensor = get_gguf_MoE_tensors(512, quant_type)[0]
    w2_tensor = get_gguf_MoE_tensors(512, quant_type2)[1]
    w13 = torch.tensor(w13_tensor.data, device="cuda")
    w2 = torch.tensor(w2_tensor.data, device="cuda")
    w13_dequant = torch.tensor(
        dequantize(w13_tensor.data, quant_type), device="cuda"
    ).to(torch.bfloat16)
    w2_dequant = torch.tensor(
        dequantize(w2_tensor.data, quant_type2), device="cuda"
    ).to(torch.bfloat16)

    x = torch.rand(
        (num_tokens, w13_dequant.shape[-1]),
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_weights = torch.rand((num_tokens, top_k), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(
        0,
        w13.shape[0],
        (num_tokens, top_k),
        dtype=torch.int32,
        device="cuda",
    )

    block_size = ggml_moe_get_block_size(quant_type)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align(
        topk_ids, block_size, w13.shape[0]
    )
    output = ggml_moe_a8(
        x,
        w13,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        w13.shape[1],
        top_k,
        num_tokens,
    )
    output = silu_and_mul(output)
    output = ggml_moe_a8(
        output,
        w2,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type2,
        w2.shape[1],
        1,
        num_tokens * top_k,
    )
    output = output.reshape(num_tokens, top_k, w2.shape[1])
    output = torch.sum(output * topk_weights[:, :, None], dim=1)

    ref_output = _reference_moe(x, w13_dequant, w2_dequant, topk_weights, topk_ids)
    torch.testing.assert_close(output, ref_output, atol=1, rtol=1e-1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
