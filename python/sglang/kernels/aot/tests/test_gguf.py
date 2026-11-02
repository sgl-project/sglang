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
)

GGUF_SAMPLE = snapshot_download("Isotr0py/test-gguf-sample")
GGUF_SAMPLE_MOE = snapshot_download("SzymonOzog/test-gguf-moe-sample")


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
# Relative L2 limits against gguf-py's dequantization (fp32). The dequant kernels
# match it to ~1e-3 (half-precision scale arithmetic); MMVQ and MMQ add the q8_1
# activation rounding (~6e-3).
DEQUANT_REL_L2 = 2e-3
MATMUL_REL_L2 = 1e-2
# ggml_mul_mat_a8 (llama.cpp's MMQ) supports every type here; the sample repo has
# no Q4_1 / Q5_1 files, the synthetic tests below cover them.
MMQ_QUANT_TYPES = [
    # i-matrix
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
    GGMLQuantizationType.IQ2_XXS,
    GGMLQuantizationType.IQ2_XS,
    GGMLQuantizationType.IQ2_S,
    GGMLQuantizationType.IQ3_XXS,
    GGMLQuantizationType.IQ3_S,
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
MMQ_SYNTHETIC_QUANT_TYPES = MMQ_QUANT_TYPES + [
    GGMLQuantizationType.Q4_1,
    GGMLQuantizationType.Q5_1,
]
# Covers J tiles of 8 / 16 / 72 / 128 / 136 columns and a partial last tile.
MMQ_NUM_TOKENS = [7, 8, 9, 65, 128, 129, 2050]
QUANT_TYPES = [
    # i-matrix
    GGMLQuantizationType.IQ1_M,
    GGMLQuantizationType.IQ1_S,
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
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_dequantize_matches_gguf_py(hidden_size: int, quant_type: GGMLQuantizationType):
    # In fp32 the dequantized weights must reproduce gguf-py's; the element-wise
    # tolerances of test_dequantize let a wrong codebook through (IQ3_S used a
    # pre-release grid: 1.4e-2 relative L2 on these samples).
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape = list(map(int, tensor.name.split("_")[-1].split("x")))
        ref = torch.tensor(
            dequantize(tensor.data, quant_type), device="cuda", dtype=torch.float32
        )
        out = ggml_dequantize(
            torch.tensor(tensor.data, device="cuda"), quant_type, *shape, torch.float32
        )
        rel = (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()
        assert rel < DEQUANT_REL_L2, f"{quant_type.name}: relative L2 {rel:.2e}"


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@torch.inference_mode()
def test_mmvq_matches_gguf_py(hidden_size: int, quant_type: GGMLQuantizationType):
    # MMVQ quantizes the activations to q8_1, so it is compared in relative L2
    # against an fp32 product with gguf-py's weights.
    torch.manual_seed(0)
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape = list(map(int, tensor.name.split("_")[-1].split("x")))
        weight = torch.tensor(
            dequantize(tensor.data, quant_type), device="cuda", dtype=torch.float32
        )
        x = torch.randn(7, shape[1], device="cuda", dtype=torch.bfloat16)
        out = ggml_mul_mat_vec_a8(
            torch.tensor(tensor.data, device="cuda"), x, quant_type, shape[0]
        ).float()
        ref = x.float() @ weight.T
        rel = (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()
        assert rel < MATMUL_REL_L2, f"{quant_type.name}: relative L2 {rel:.2e}"


@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("quant_type", MMQ_QUANT_TYPES)
@torch.inference_mode()
def test_mmq_matches_gguf_py(hidden_size: int, quant_type: GGMLQuantizationType):
    # Same check for MMQ (the batched path): its block minimum / offset terms read
    # the q8_1 block sums, which must cover the quantized activations. The sample
    # weights have no row padding (hidden_size 256 needs it), so this also covers
    # the copy that ggml_mul_mat_a8 makes for such weights.
    torch.manual_seed(0)
    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    for tensor in tensors:
        shape = list(map(int, tensor.name.split("_")[-1].split("x")))
        weight = torch.tensor(
            dequantize(tensor.data, quant_type), device="cuda", dtype=torch.float32
        )
        qweight = torch.tensor(tensor.data, device="cuda")
        for num_tokens in MMQ_NUM_TOKENS:
            x = torch.randn(num_tokens, shape[1], device="cuda", dtype=torch.bfloat16)
            out = ggml_mul_mat_a8(qweight, x, quant_type, shape[0])
            assert out.dtype == x.dtype and out.shape == (num_tokens, shape[0])
            ref = x.float() @ weight.T
            rel = (torch.linalg.norm(out.float() - ref) / torch.linalg.norm(ref)).item()
            assert rel < MATMUL_REL_L2, (
                f"{quant_type.name} {tensor.name} M={num_tokens}: relative L2 {rel:.2e}"
            )


_FP16_SCALE = torch.tensor([2.0**-8], dtype=torch.float16).view(torch.uint8)


def random_gguf_blocks(
    quant_type: GGMLQuantizationType, rows: int, cols: int, seed: int
) -> torch.Tensor:
    """Random GGUF blocks with small finite fp16 scale fields (ggml-common.h layouts)."""
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    gen = torch.Generator().manual_seed(seed)
    b = torch.randint(
        0, 256, (rows * cols // block_size, type_size), dtype=torch.uint8, generator=gen
    )
    if quant_type == GGMLQuantizationType.Q2_K:  # scales, qs, d, dmin
        b[:, -4:-2] = _FP16_SCALE
        b[:, -2:] = _FP16_SCALE
    elif quant_type in (GGMLQuantizationType.Q3_K, GGMLQuantizationType.Q6_K):
        b[:, -2:] = _FP16_SCALE  # d last
    elif quant_type == GGMLQuantizationType.IQ1_M:
        # qs[32], qh[16], scales[8]: the fp16 d is the top nibbles of the four
        # uint16 scale words, lowest nibble in the first
        d = int(_FP16_SCALE[0]) | int(_FP16_SCALE[1]) << 8
        for i in range(4):
            hi = 32 + 16 + 2 * i + 1
            b[:, hi] = (b[:, hi] & 0x0F) | (((d >> (4 * i)) & 0x0F) << 4)
    else:  # d first: 32-value blocks, Q4_K / Q5_K and all i-quants
        b[:, :2] = _FP16_SCALE
        if quant_type in (
            GGMLQuantizationType.Q4_1,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
        ):
            b[:, 2:4] = _FP16_SCALE  # m / dmin
    return b.view(rows, -1)


def padded_gguf_weight(qweight: torch.Tensor, quant_type: GGMLQuantizationType):
    """qweight on CUDA in a storage that extends past the last row like ggml's
    buffers (to a multiple of 512 values per row), so no copy is made per call."""
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    cols = qweight.shape[1] // type_size * block_size
    tail = (-cols % 512) // block_size * type_size
    storage = torch.zeros(qweight.numel() + tail, dtype=torch.uint8, device="cuda")
    out = storage[: qweight.numel()].view(qweight.shape)
    out.copy_(qweight)
    return out


def rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (torch.linalg.norm(out.float() - ref) / torch.linalg.norm(ref)).item()


# (rows, cols): rows % 128 != 0 runs the bounds-checked kernels, cols % 512 != 0
# makes the kernel read past the end of each row (into the row padding).
MMQ_SYNTHETIC_SHAPES = [(1000, 2304), (1024, 2048)]


@pytest.mark.parametrize("shape", MMQ_SYNTHETIC_SHAPES)
@pytest.mark.parametrize("quant_type", MMQ_SYNTHETIC_QUANT_TYPES)
@torch.inference_mode()
def test_mmq_synthetic_matches_gguf_py(
    shape: tuple[int, int], quant_type: GGMLQuantizationType
):
    rows, cols = shape
    torch.manual_seed(0)
    qweight_cpu = random_gguf_blocks(quant_type, rows, cols, seed=int(quant_type))
    weight = torch.tensor(
        dequantize(qweight_cpu.numpy(), quant_type), device="cuda", dtype=torch.float32
    )
    assert torch.isfinite(weight).all()
    qweight = padded_gguf_weight(qweight_cpu, quant_type)
    for num_tokens in MMQ_NUM_TOKENS:
        x = torch.randn(num_tokens, cols, device="cuda", dtype=torch.bfloat16)
        out = ggml_mul_mat_a8(qweight, x, quant_type, rows)
        assert out.dtype == x.dtype and out.shape == (num_tokens, rows)
        rel = rel_l2(out, x.float() @ weight.T)
        assert rel < MATMUL_REL_L2, f"{quant_type.name} M={num_tokens}: {rel:.2e}"

    x = torch.randn(65, cols, device="cuda", dtype=torch.float32)
    for dtype in (torch.float32, torch.float16):
        # The activations are quantized from the input dtype's values.
        xd = x.to(dtype)
        out = ggml_mul_mat_a8(qweight, xd, quant_type, rows)
        assert out.dtype == dtype
        assert rel_l2(out, xd.float() @ weight.T) < MATMUL_REL_L2
    # A weight without row padding is copied per call: same result.
    torch.testing.assert_close(
        ggml_mul_mat_a8(qweight_cpu.cuda(), x, quant_type, rows),
        ggml_mul_mat_a8(qweight, x, quant_type, rows),
        rtol=0,
        atol=0,
    )
    # Non-contiguous activations (row stride > cols).
    x_view = torch.randn(65, cols + 64, device="cuda", dtype=torch.bfloat16)[:, :cols]
    torch.testing.assert_close(
        ggml_mul_mat_a8(qweight, x_view, quant_type, rows),
        ggml_mul_mat_a8(qweight, x_view.contiguous(), quant_type, rows),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "quant_type",
    [
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.IQ4_XS,
    ],
)
@torch.inference_mode()
def test_mmq_large_batch(quant_type: GGMLQuantizationType):
    rows, cols, num_tokens = 1000, 2304, 16384
    torch.manual_seed(0)
    qweight_cpu = random_gguf_blocks(quant_type, rows, cols, seed=int(quant_type))
    weight = torch.tensor(
        dequantize(qweight_cpu.numpy(), quant_type), device="cuda", dtype=torch.float32
    )
    qweight = padded_gguf_weight(qweight_cpu, quant_type)
    x = torch.randn(num_tokens, cols, device="cuda", dtype=torch.bfloat16)
    out = ggml_mul_mat_a8(qweight, x, quant_type, rows)
    assert rel_l2(out, x.float() @ weight.T) < MATMUL_REL_L2


@pytest.mark.parametrize("quant_type", [GGMLQuantizationType.Q8_1])
@torch.inference_mode()
def test_mmq_unsupported_type_raises(quant_type: GGMLQuantizationType):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    qweight = torch.zeros(128, 1024 // block_size * type_size, dtype=torch.uint8)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not supported"):
        ggml_mul_mat_a8(qweight.cuda(), x, quant_type, 128)


# (rows, cols, tokens): enough output tiles for the stream-k decomposition, whose
# tiles are split between thread blocks and summed by a fix-up kernel, on any GPU
# size; plus a partial last tile in both directions.
MMQ_HALF_OUTPUT_CASES = [
    (1000, 2304, 9),
    (1000, 2304, 129),
    (1000, 2304, 300),
    (4096, 2048, 100),
    (4096, 2048, 2050),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quant_type", MMQ_SYNTHETIC_QUANT_TYPES)
@torch.inference_mode()
def test_mmq_half_output_is_rounded_fp32(
    quant_type: GGMLQuantizationType, dtype: torch.dtype
):
    # The kernel stores bf16 / fp16 outputs directly: each value must be the fp32
    # result rounded once (to nearest even), bit for bit, including the outputs of
    # tiles that the stream-k decomposition splits between thread blocks. The fp32
    # result comes from the same call with x.float(): the q8_1 activations are
    # quantized from the same values either way.
    for rows, cols, num_tokens in MMQ_HALF_OUTPUT_CASES:
        qweight = padded_gguf_weight(
            random_gguf_blocks(quant_type, rows, cols, seed=int(quant_type)), quant_type
        )
        x = torch.randn(num_tokens, cols, device="cuda").to(dtype)
        out = ggml_mul_mat_a8(qweight, x, quant_type, rows)
        assert out.dtype == dtype and out.shape == (num_tokens, rows)
        ref = ggml_mul_mat_a8(qweight, x.float(), quant_type, rows).to(dtype)
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (
            f"{quant_type.name} {rows}x{cols} M={num_tokens} {dtype}: "
            f"{(out.view(torch.int16) != ref.view(torch.int16)).sum().item()} values differ"
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "quant_type", [GGMLQuantizationType.Q5_K, GGMLQuantizationType.IQ3_XXS]
)
@torch.inference_mode()
def test_mmq_cuda_graph(quant_type: GGMLQuantizationType, dtype: torch.dtype):
    # Captured without a warm-up call of this shape: the kernel set-up (shared
    # memory limit, stream-k scratch buffer) also runs under capture.
    rows, cols, num_tokens = 1000, 2304, 200
    qweight = padded_gguf_weight(
        random_gguf_blocks(quant_type, rows, cols, seed=1), quant_type
    )
    x = torch.randn(num_tokens, cols, device="cuda", dtype=dtype)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ggml_mul_mat_a8(qweight, x, quant_type, rows)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, ggml_mul_mat_a8(qweight, x, quant_type, rows), rtol=0, atol=0
        )
        torch.testing.assert_close(
            out,
            ggml_mul_mat_a8(qweight, x.float(), quant_type, rows).to(dtype),
            rtol=0,
            atol=0,
        )


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
    [
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

    tensors = get_gguf_sample_tensors(hidden_size, quant_type)
    x = torch.rand((num_tokens, hidden_size), dtype=dtype, device="cuda")
    for tensor in tensors:
        weight = torch.tensor(dequantize(tensor.data, quant_type), device="cuda").to(
            dtype
        )
        ref_output = x @ weight.T

        qweight = torch.tensor(tensor.data, device="cuda")
        output = ggml_mul_mat_a8(qweight, x, quant_type, qweight.shape[0])
        atols = {torch.half: 1, torch.bfloat16: 1.5, torch.float: 1.2}
        # test matrix has inputs centered around 0 and lower precision from
        # bfloat16 tends to accumulate and can greatly inflate rtol
        # since outputs are also very close to 0
        rtols = {torch.half: 1e-1, torch.bfloat16: 1e4, torch.float: 2e1}
        torch.testing.assert_close(
            output, ref_output, atol=atols[dtype], rtol=rtols[dtype]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
