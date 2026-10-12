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
    ggml_moe_mmq,
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


# ---------------------------------------------------------------------------
# MoE: ggml_moe_mmq (llama.cpp's MMQ, MUL_MAT_ID path)


def moe_reference(
    x: torch.Tensor, weight: torch.Tensor, topk_ids: torch.Tensor, x_per_slot: bool
) -> torch.Tensor:
    """fp32 [T, top_k, rows]: per-expert loop over dequantized weights [E, rows, K];
    slots with an expert id < 0 are zero."""
    num_tokens, top_k = topk_ids.shape
    xs = x.float() if x_per_slot else x.float().repeat_interleave(top_k, 0)
    flat = topk_ids.reshape(-1)
    out = torch.zeros(num_tokens * top_k, weight.shape[1], device="cuda")
    for e in range(weight.shape[0]):
        sel = (flat == e).nonzero().squeeze(1)
        if sel.numel():
            out[sel] = xs[sel] @ weight[e].T
    return out.view(num_tokens, top_k, -1)


def random_topk_ids(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    unused: tuple,
    seed: int,
    masked: bool = True,
) -> torch.Tensor:
    """Distinct experts per token, never the experts in `unused`; with `masked`,
    every 5th token's first route is -1 (as with expert parallelism)."""
    gen = torch.Generator().manual_seed(seed)
    allowed = torch.tensor([e for e in range(num_experts) if e not in unused])
    ids = torch.stack(
        [
            allowed[torch.randperm(len(allowed), generator=gen)[:top_k]]
            for _ in range(num_tokens)
        ]
    )
    if masked:
        ids[::5, 0] = -1
    return ids.cuda()


_MOE_WEIGHTS = {}


def moe_synthetic_weight(quant_type: GGMLQuantizationType, rows: int, cols: int):
    """128 random experts (padded storage, gguf-py dequantization), cached."""
    key = (quant_type, rows, cols)
    if key not in _MOE_WEIGHTS:
        q = random_gguf_blocks(
            quant_type, 128 * rows, cols, seed=int(quant_type) + rows
        )
        w = torch.tensor(
            dequantize(q.numpy(), quant_type), device="cuda", dtype=torch.float32
        )
        q = padded_gguf_weight(q, quant_type)
        _MOE_WEIGHTS.clear()
        _MOE_WEIGHTS[key] = (q.view(128, rows, -1), w.view(128, rows, cols))
    return _MOE_WEIGHTS[key]


# (num_experts, top_k, num_tokens)
MOE_CONFIGS = [(8, 2, 65), (64, 8, 512), (128, 6, 9), (64, 1, 4096), (8, 8, 9)]


@pytest.mark.parametrize("quant_type", MMQ_SYNTHETIC_QUANT_TYPES)
@torch.inference_mode()
def test_moe_mmq_synthetic_matches_gguf_py(quant_type: GGMLQuantizationType):
    # rows % 128 != 0 (bounds-checked tiles), cols % 512 != 0 (row padding read).
    rows, cols = 320, 768
    qweight_all, weight_all = moe_synthetic_weight(quant_type, rows, cols)
    for i, (num_experts, top_k, num_tokens) in enumerate(MOE_CONFIGS):
        qweight, weight = qweight_all[:num_experts], weight_all[:num_experts]
        # experts 1 and E-1 get no tokens (unless every expert is routed to)
        unused = (1, num_experts - 1) if num_experts - 2 >= top_k else ()
        ids = random_topk_ids(num_tokens, num_experts, top_k, unused, seed=i)
        if i % 2:
            ids = ids.int()
        x = torch.randn(num_tokens, cols, device="cuda", dtype=torch.bfloat16)
        out = ggml_moe_mmq(x, qweight, ids, quant_type, rows, False)
        assert out.dtype == torch.float32 and out.shape == (num_tokens, top_k, rows)
        masked = ids.reshape(-1) < 0
        assert (out.view(-1, rows)[masked] == 0).all()
        rel = rel_l2(out, moe_reference(x, weight, ids, False))
        assert rel < MATMUL_REL_L2, (
            f"{quant_type.name} {num_experts=} {top_k=} {num_tokens=}: {rel:.2e}"
        )

        # Down projection: one input row per route.
        xs = torch.randn(num_tokens * top_k, cols, device="cuda", dtype=torch.float32)
        out = ggml_moe_mmq(xs, qweight, ids, quant_type, rows, True)
        assert (out.view(-1, rows)[masked] == 0).all()
        rel = rel_l2(out, moe_reference(xs, weight, ids, True))
        assert rel < MATMUL_REL_L2, (
            f"{quant_type.name} down {num_experts=} {top_k=}: {rel:.2e}"
        )


@pytest.mark.parametrize("quant_type", MMQ_QUANT_TYPES)
@torch.inference_mode()
def test_moe_mmq_sample_matches_gguf_py(quant_type: GGMLQuantizationType):
    # The first 16 of the sample's 256 experts.
    num_experts, top_k, num_tokens = 16, 8, 65
    for tensor in get_gguf_MoE_tensors(512, quant_type):
        data = np.ascontiguousarray(tensor.data[:num_experts])
        weight = torch.tensor(
            dequantize(data, quant_type), device="cuda", dtype=torch.float32
        )
        rows, cols = weight.shape[1], weight.shape[2]
        qweight = torch.tensor(data, device="cuda")  # no row padding: copied per call
        ids = random_topk_ids(num_tokens, num_experts, top_k, (3,), seed=0)
        x = torch.randn(num_tokens, cols, device="cuda", dtype=torch.bfloat16)
        out = ggml_moe_mmq(x, qweight, ids, quant_type, rows, False)
        rel = rel_l2(out, moe_reference(x, weight, ids, False))
        assert rel < MATMUL_REL_L2, f"{quant_type.name} {tensor.name}: {rel:.2e}"
        xs = torch.randn(num_tokens * top_k, cols, device="cuda", dtype=torch.bfloat16)
        out = ggml_moe_mmq(xs, qweight, ids, quant_type, rows, True)
        rel = rel_l2(out, moe_reference(xs, weight, ids, True))
        assert rel < MATMUL_REL_L2, f"{quant_type.name} {tensor.name} down: {rel:.2e}"


@pytest.mark.parametrize(
    "quant_type", [GGMLQuantizationType.Q8_0, GGMLQuantizationType.IQ4_XS]
)
@torch.inference_mode()
def test_moe_mmq_token_chunks(quant_type: GGMLQuantizationType):
    # More tokens than one call takes (the expert sort keeps 4 bytes per token in
    # shared memory): the op splits the tokens.
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_tokens = props.shared_memory_per_block_optin // 4 + 37
    num_experts, top_k, rows, cols = 8, 2, 128, 256
    qweight_cpu = random_gguf_blocks(quant_type, num_experts * rows, cols, seed=5)
    weight = torch.tensor(
        dequantize(qweight_cpu.numpy(), quant_type), device="cuda", dtype=torch.float32
    ).view(num_experts, rows, cols)
    qweight = padded_gguf_weight(qweight_cpu, quant_type).view(num_experts, rows, -1)
    ids = random_topk_ids(num_tokens, num_experts, top_k, (), seed=1)
    x = torch.randn(num_tokens, cols, device="cuda", dtype=torch.bfloat16)
    out = ggml_moe_mmq(x, qweight, ids, quant_type, rows, False)
    assert rel_l2(out, moe_reference(x, weight, ids, False)) < MATMUL_REL_L2


@torch.inference_mode()
def test_moe_mmq_repeated_expert_is_masked():
    # Top-k gives distinct experts per token; a repeated one must not corrupt
    # memory: it is treated like a masked route (zero row).
    quant_type = GGMLQuantizationType.Q4_K
    num_experts, top_k, num_tokens, rows, cols = 16, 4, 65, 320, 768
    qweight, weight = moe_synthetic_weight(quant_type, rows, cols)
    qweight, weight = qweight[:num_experts], weight[:num_experts]
    ids = random_topk_ids(num_tokens, num_experts, top_k, (), seed=7, masked=False)
    ids[::3, 2] = ids[::3, 0]
    expected_ids = ids.clone()
    expected_ids[::3, 2] = -1
    for x_per_slot in (False, True):
        x = torch.randn(num_tokens * (top_k if x_per_slot else 1), cols, device="cuda")
        out = ggml_moe_mmq(x, qweight, ids, quant_type, rows, x_per_slot)
        ref = moe_reference(x, weight, expected_ids, x_per_slot)
        assert (out[::3, 2] == 0).all()
        assert rel_l2(out, ref) < MATMUL_REL_L2


@torch.inference_mode()
def test_moe_a8_more_than_256_experts():
    # ggml_moe_a8 (the dp4a MoE kernel) used to skip every slot whose expert id
    # was above 255, so models with more experts lost those routes (zero rows).
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    quant_type = GGMLQuantizationType.Q4_0
    num_experts, top_k, num_tokens, rows, cols = 300, 4, 96, 256, 512
    qweight, weight = moe_synthetic_weight(quant_type, rows, cols)
    # the 128 cached experts tiled to 300: expert e is weight e % 128
    reps = (num_experts + 127) // 128
    qweight = qweight.repeat(reps, 1, 1)[:num_experts].contiguous()
    weight = weight.repeat(reps, 1, 1)[:num_experts]
    # every route goes to an expert >= 256
    ids = (
        random_topk_ids(num_tokens, num_experts - 256, top_k, (), seed=3, masked=False)
        + 256
    )
    x = torch.randn(num_tokens, cols, device="cuda", dtype=torch.float16)
    block_size = ggml_moe_get_block_size(quant_type)
    sorted_ids, expert_ids, num_post_pad = moe_align_block_size(
        ids, block_size, num_experts
    )
    out = ggml_moe_a8(
        x,
        qweight,
        sorted_ids,
        expert_ids,
        num_post_pad,
        quant_type,
        rows,
        top_k,
        num_tokens,
    ).view(num_tokens, top_k, rows)
    ref = moe_reference(x, weight, ids, False)
    assert (out.float().abs().sum(-1) > 0).all()
    assert rel_l2(out.float(), ref) < MATMUL_REL_L2


@torch.inference_mode()
def test_moe_mmq_unsupported_type_raises():
    block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q8_1]
    qweight = torch.zeros(4, 128, 1024 // block_size * type_size, dtype=torch.uint8)
    x = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(16, 2, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="not supported"):
        ggml_moe_mmq(x, qweight.cuda(), ids, GGMLQuantizationType.Q8_1, 128, False)


@pytest.mark.parametrize(
    "quant_type", [GGMLQuantizationType.Q4_K, GGMLQuantizationType.IQ3_S]
)
@torch.inference_mode()
def test_moe_mmq_cuda_graph(quant_type: GGMLQuantizationType):
    num_experts, top_k, num_tokens, rows, cols = 64, 8, 100, 320, 768
    qweight = moe_synthetic_weight(quant_type, rows, cols)[0][:num_experts]
    ids = random_topk_ids(num_tokens, num_experts, top_k, (2,), seed=3)
    x = torch.randn(num_tokens, cols, device="cuda", dtype=torch.bfloat16)
    xs = torch.randn(num_tokens * top_k, cols, device="cuda", dtype=torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ggml_moe_mmq(x, qweight, ids, quant_type, rows, False)
        out_down = ggml_moe_mmq(xs, qweight, ids, quant_type, rows, True)
    for i in range(3):
        # new activations and new routes
        x.copy_(torch.randn_like(x))
        xs.copy_(torch.randn_like(xs))
        ids.copy_(random_topk_ids(num_tokens, num_experts, top_k, (i,), seed=10 + i))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, ggml_moe_mmq(x, qweight, ids, quant_type, rows, False), rtol=0, atol=0
        )
        torch.testing.assert_close(
            out_down,
            ggml_moe_mmq(xs, qweight, ids, quant_type, rows, True),
            rtol=0,
            atol=0,
        )


# fused_moe_gguf (SGLang's GGUF MoE layer): gate/up, SiLU, down, weighted sum.
FUSED_MOE_REL_L2 = 1.5e-2  # two q8_1-quantized products


@pytest.mark.parametrize(
    "quant_types",
    [
        (GGMLQuantizationType.IQ3_S, GGMLQuantizationType.IQ4_NL),
        (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q6_K),
    ],
)
@pytest.mark.parametrize("num_tokens", [4, 65, 512])
@torch.inference_mode()
def test_fused_moe_gguf(quant_types, num_tokens: int):
    gguf_layer = pytest.importorskip("sglang.srt.layers.quantization.gguf")
    t1, t2 = quant_types
    num_experts, top_k, hidden, inter = 64, 8, 512, 256
    q1 = random_gguf_blocks(t1, num_experts * 2 * inter, hidden, seed=11)
    q2 = random_gguf_blocks(t2, num_experts * hidden, inter, seed=12)
    w1 = torch.tensor(dequantize(q1.numpy(), t1), device="cuda").view(
        num_experts, 2 * inter, hidden
    )
    w2 = torch.tensor(dequantize(q2.numpy(), t2), device="cuda").view(
        num_experts, hidden, inter
    )
    q1 = padded_gguf_weight(q1, t1).view(num_experts, 2 * inter, -1)
    q2 = padded_gguf_weight(q2, t2).view(num_experts, hidden, -1)
    # int32 as SGLang's top-k returns them (ggml_moe_a8_vec reads int32)
    ids = random_topk_ids(num_tokens, num_experts, top_k, (5,), seed=4, masked=False)
    ids = ids.int()
    torch.manual_seed(0)
    topk_weights = torch.rand(num_tokens, top_k, device="cuda")
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    out = gguf_layer.fused_moe_gguf(x, q1, q2, topk_weights, ids, t1, t2, "silu")
    assert out.dtype == x.dtype and out.shape == x.shape

    h = moe_reference(x, w1, ids, False)
    h = torch.nn.functional.silu(h[..., :inter]) * h[..., inter:]
    y = moe_reference(h.view(num_tokens * top_k, inter), w2, ids, True)
    ref = (y * topk_weights.unsqueeze(-1)).sum(1)
    rel = rel_l2(out, ref)
    assert rel < FUSED_MOE_REL_L2, f"{t1.name}/{t2.name} T={num_tokens}: {rel:.2e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
