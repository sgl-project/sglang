import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch


def _load_codec_module():
    codec_path = (
        Path(__file__).resolve().parents[3]
        / "srt/layers/attention/dsa/nvfp4_k_cache.py"
    )
    spec = importlib.util.spec_from_file_location("dsa_nvfp4_k_cache_test", codec_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CODEC = _load_codec_module()
NVFP4_BYTES_PER_TOKEN = _CODEC.NVFP4_BYTES_PER_TOKEN
dequantize_nvfp4_k_cache_paged_reference = (
    _CODEC.dequantize_nvfp4_k_cache_paged_reference
)
quantize_nvfp4_k_cache_into = _CODEC.quantize_nvfp4_k_cache_into


LOCAL_Q_HEADS = 64
QK_DIM = 576
V_DIM = 512
PAGE_SIZE = 64
TOPK = 2048
CONTEXT_LENGTH = 200_000


def _is_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _load_flashmla_extension() -> None:
    explicit_library = os.environ.get("SGLANG_FLASHMLA_LIBRARY")
    if explicit_library:
        torch.ops.load_library(explicit_library)
        return
    from sgl_kernel import flashmla_ops  # noqa: F401


def _reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    global_scale: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor,
    sink: torch.Tensor | None,
    sm_scale: float,
    scheduler_metadata: torch.Tensor,
):
    b, sq, h, _ = q.shape
    rows = b * sq
    flat_indices = indices.reshape(rows, TOPK)
    row_lengths = topk_length.reshape(b, 1).expand(b, sq).reshape(rows)
    physical_tokens = cache.numel() // NVFP4_BYTES_PER_TOKEN
    position = torch.arange(TOPK, device=q.device).view(1, TOPK)
    valid = (
        (position < row_lengths.reshape(rows, 1))
        & (flat_indices >= 0)
        & (flat_indices < physical_tokens)
    )
    safe_indices = flat_indices.masked_fill(~valid, 0)
    decoded = (
        dequantize_nvfp4_k_cache_paged_reference(
            cache, safe_indices, global_scale, torch.bfloat16
        )
        .reshape(rows, TOPK, QK_DIM)
        .float()
    )

    # Match the open FlashMLA FP8-cache contract: the producer materializes
    # BF16 latent/RoPE tiles, QK and PV consume BF16 operands, and their
    # accumulators plus online-softmax state remain FP32.
    q_mma = q.reshape(rows, h, QK_DIM).float()
    latent_mma = decoded[..., :V_DIM]
    kv_mma = decoded
    logits = torch.einsum("rhd,rkd->rhk", q_mma, kv_mma) * sm_scale
    logits.masked_fill_(~valid[:, None, :], float("-inf"))
    # FlashMLA's scheduler can split one request across many CTAs.  Each split
    # runs its own online softmax and quantizes the *unnormalized* probability
    # tile to BF16 before PV; the combine kernel then merges the normalized
    # partial outputs using their LSEs.  Reconstruct the exact block ranges from
    # DecodingSchedMeta instead of treating all 2048 candidates as one stream.
    metadata_cpu = scheduler_metadata.detach().cpu()
    request_split_ranges: list[list[tuple[int, int]]] = [[] for _ in range(b)]
    num_blocks_per_request = [
        max(1, math.ceil(int(length) / 64)) for length in topk_length.tolist()
    ]
    for meta in metadata_cpu:
        begin_req, end_req, begin_block, end_block = map(int, meta[:4])
        if begin_req >= b:
            continue
        for request in range(begin_req, min(end_req, b - 1) + 1):
            start = begin_block if request == begin_req else 0
            end = end_block if request == end_req else num_blocks_per_request[request]
            if end > start:
                request_split_ranges[request].append((start, end))
    split_ranges = [
        request_split_ranges[request] for request in range(b) for _ in range(sq)
    ]

    out = torch.empty(rows, h, V_DIM, dtype=torch.float32, device=q.device)
    combined_lse = torch.empty(rows, h, dtype=torch.float32, device=q.device)
    for request, ranges in enumerate(split_ranges):
        assert ranges, f"scheduler produced no split for request {request}"
        partial_outputs = []
        partial_lses = []
        for begin_block, end_block in ranges:
            running_max = torch.full(
                (h,), -1.0e30, dtype=torch.float32, device=q.device
            )
            running_sum = torch.zeros_like(running_max)
            numerator = torch.zeros(h, V_DIM, dtype=torch.float32, device=q.device)
            for block in range(begin_block, end_block):
                start = block * 64
                end = start + 64
                block_logits = logits[request, :, start:end]
                block_max = block_logits.max(dim=-1).values
                # The H64 kernel uses ``__any_sync`` per warp: a trigger from
                # any of 32 heads rescales that half of the head tile, while
                # the other half remains independent.
                should_rescale = (
                    ((block_max - running_max) > 6.0)
                    .view(-1, 32)
                    .any(dim=-1, keepdim=True)
                    .expand(-1, 32)
                    .reshape(h)
                )
                new_max = torch.where(
                    should_rescale,
                    torch.maximum(running_max, block_max),
                    running_max,
                )
                old_scale = torch.exp(running_max - new_max)
                block_probability = torch.exp(block_logits - new_max.unsqueeze(-1))
                block_probability = torch.nan_to_num(block_probability, nan=0.0)
                numerator.mul_(old_scale.unsqueeze(-1))
                numerator.add_(
                    torch.einsum(
                        "hk,kd->hd",
                        block_probability.to(torch.bfloat16).float(),
                        latent_mma[request, start:end],
                    )
                )
                running_sum.mul_(old_scale).add_(block_probability.sum(dim=-1))
                running_max = new_max
            partial_outputs.append(numerator / running_sum.unsqueeze(-1))
            partial_lses.append(torch.log(running_sum) + running_max)

        split_lse = torch.stack(partial_lses, dim=0)
        split_out = torch.stack(partial_outputs, dim=0)
        if sink is not None:
            combine_lse = torch.cat((split_lse, sink.view(1, h)), dim=0)
            weights = torch.softmax(combine_lse, dim=0)[:-1]
            combined_lse[request] = torch.logsumexp(combine_lse, dim=0)
        else:
            weights = torch.softmax(split_lse, dim=0)
            combined_lse[request] = torch.logsumexp(split_lse, dim=0)
        out[request] = torch.einsum("sh,shd->hd", weights, split_out)
    return (
        out.reshape(b, sq, h, V_DIM).to(torch.bfloat16),
        combined_lse.reshape(b, sq, h).transpose(1, 2).contiguous(),
    )


def _make_case(b: int, sq: int, global_scale_value: float = 1.375):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810 + b * 10 + sq)
    physical_tokens = b * CONTEXT_LENGTH
    cache = torch.zeros(
        math.ceil(physical_tokens / PAGE_SIZE),
        PAGE_SIZE,
        1,
        NVFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device=device,
    )
    q = (
        torch.randn(
            b,
            sq,
            LOCAL_Q_HEADS,
            QK_DIM,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    indices = torch.empty(b, sq, TOPK, dtype=torch.int32, device=device)
    for request in range(b):
        request_start = request * CONTEXT_LENGTH
        for query in range(sq):
            row = torch.randperm(
                CONTEXT_LENGTH, generator=generator, device=device, dtype=torch.int64
            )[:TOPK].to(torch.int32)
            indices[request, query] = row + request_start

    # Exercise page boundaries, last valid row, negative/positive OOB values,
    # and duplicates.  The last element is also masked by topk_length.
    boundary_values = torch.tensor(
        [
            1,
            63,
            64,
            65,
            2047,
            2048,
            199_999,
            200_000,
            200_001,
            201_999,
            202_000,
        ],
        dtype=torch.int32,
        device=device,
    )
    indices[0, 0, : boundary_values.numel()] = boundary_values
    indices[0, 0, 20] = -1
    indices[0, 0, 21] = indices[0, 0, 22]
    topk_length = torch.full((b,), TOPK, dtype=torch.int32, device=device)
    topk_length[-1] = TOPK - 1

    valid_locations = torch.unique(
        indices[(indices >= 0) & (indices < physical_tokens)].to(torch.int64)
    )
    latent = (
        torch.randn(
            valid_locations.numel(),
            1,
            V_DIM,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    rope = (
        torch.randn(
            valid_locations.numel(),
            1,
            QK_DIM - V_DIM,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    global_scale = torch.tensor(
        [global_scale_value], dtype=torch.float32, device=device
    )
    quantize_nvfp4_k_cache_into(latent, rope, cache, valid_locations, global_scale)
    sink = torch.linspace(-2.0, -0.25, LOCAL_Q_HEADS, device=device)
    return q, cache, global_scale, indices, topk_length, sink


@pytest.mark.skipif(not _is_sm100(), reason="SM100 is required")
@torch.inference_mode()
def test_glm52_nvfp4_repeated_single_kv_row():
    _load_flashmla_extension()
    device = torch.device("cuda")
    cache = torch.zeros(
        1,
        PAGE_SIZE,
        1,
        NVFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device=device,
    )
    # Every dimension has a deterministic sign/magnitude pattern, while each
    # block remains exactly representable by the NVFP4->E4M3 contract.
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=device,
        dtype=torch.float32,
    )
    latent = values.repeat(V_DIM // values.numel())
    latent[1::2].neg_()
    latent = latent.mul(0.125).to(torch.bfloat16).view(1, 1, V_DIM)
    rope = torch.zeros(1, 1, QK_DIM - V_DIM, dtype=torch.bfloat16, device=device)
    global_scale = torch.tensor([0.125], dtype=torch.float32, device=device)
    quantize_nvfp4_k_cache_into(
        latent,
        rope,
        cache,
        torch.zeros(1, dtype=torch.int64, device=device),
        global_scale,
    )
    decoded = dequantize_nvfp4_k_cache_paged_reference(
        cache,
        torch.zeros(1, dtype=torch.int64, device=device),
        global_scale,
        torch.bfloat16,
    )[0, 0, :V_DIM].float()
    q = torch.zeros(1, 1, LOCAL_Q_HEADS, QK_DIM, dtype=torch.bfloat16, device=device)
    indices = torch.zeros(1, 1, TOPK, dtype=torch.int32, device=device)
    lengths = torch.full((1,), TOPK, dtype=torch.int32, device=device)
    out, _, _, _ = torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
        q, cache, global_scale, indices, lengths, None, None, None, V_DIM, 1.0
    )
    actual = out.float()[0, 0, 0]
    torch.testing.assert_close(actual, decoded, atol=8e-4, rtol=2.01 / 128)


@pytest.mark.skipif(not _is_sm100(), reason="SM100 is required")
@torch.inference_mode()
def test_glm52_nvfp4_all_invalid_indices_return_zero_output():
    _load_flashmla_extension()
    device = torch.device("cuda")
    q = torch.zeros(
        1, 1, LOCAL_Q_HEADS, QK_DIM, dtype=torch.bfloat16, device=device
    )
    cache = torch.zeros(
        1,
        PAGE_SIZE,
        1,
        NVFP4_BYTES_PER_TOKEN,
        dtype=torch.uint8,
        device=device,
    )
    global_scale = torch.ones(1, dtype=torch.float32, device=device)
    indices = torch.full((1, 1, TOPK), -1, dtype=torch.int32, device=device)
    lengths = torch.full((1,), TOPK, dtype=torch.int32, device=device)
    sink = torch.zeros(LOCAL_Q_HEADS, dtype=torch.float32, device=device)

    out, lse, _, _ = torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
        q, cache, global_scale, indices, lengths, sink, None, None, V_DIM, 1.0
    )

    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
    # Match FlashMLA's existing lonely-query convention: a query with no
    # attendable key has a zero output and uses +inf as its LSE sentinel.
    assert torch.isposinf(lse).all()


@pytest.mark.skipif(not _is_sm100(), reason="SM100 is required")
@pytest.mark.parametrize("b,sq", [(1, 1), (2, 1), (1, 6), (4, 6)])
@torch.inference_mode()
def test_glm52_nvfp4_sparse_decode(b: int, sq: int):
    _load_flashmla_extension()
    q, cache, global_scale, indices, topk_length, sink = _make_case(b, sq)
    sm_scale = 1.0 / math.sqrt(QK_DIM)
    out, lse, metadata, _num_splits = (
        torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
            q,
            cache,
            global_scale,
            indices,
            topk_length,
            sink,
            None,
            None,
            V_DIM,
            sm_scale,
        )
    )
    ref_out, ref_lse = _reference(
        q,
        cache,
        global_scale,
        indices,
        topk_length,
        sink,
        sm_scale,
        metadata,
    )
    torch.testing.assert_close(out, ref_out, atol=8e-4, rtol=2.01 / 128)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=8.01 / 65536)


@pytest.mark.skipif(not _is_sm100(), reason="SM100 is required")
@pytest.mark.parametrize("b,sq", [(1, 1), (2, 1), (4, 1), (1, 6), (2, 6), (4, 6)])
@torch.inference_mode()
def test_glm52_nvfp4_sparse_decode_cuda_graph_replay(b: int, sq: int):
    _load_flashmla_extension()
    q, cache, global_scale, indices, topk_length, _ = _make_case(b, sq)
    out_address = None
    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
            q, cache, global_scale, indices, topk_length, None, None, None, V_DIM, 1.0
        )
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out, lse, _, _ = torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
            q, cache, global_scale, indices, topk_length, None, None, None, V_DIM, 1.0
        )
    out_address = out.data_ptr()
    first = out.clone()
    q.copy_(torch.zeros_like(q))
    indices[..., 0].fill_(-1)
    topk_length.fill_(TOPK - 1)
    global_scale.fill_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert out.data_ptr() == out_address
    assert torch.isfinite(lse).all()
    assert not torch.equal(first, out)


@pytest.mark.skipif(not _is_sm100(), reason="SM100 is required")
@pytest.mark.parametrize("b,sq", [(1, 1), (2, 1), (4, 1), (1, 6), (4, 6)])
@torch.inference_mode()
def test_glm52_nvfp4_sparse_decode_cuda_graph_external_metadata(b: int, sq: int):
    """Match SGLang graph capture, which supplies stock FlashMLA metadata."""
    _load_flashmla_extension()

    q, cache, global_scale, indices, topk_length, _ = _make_case(b, sq)
    rows = b * sq
    # The DSA backend flattens EAGLE's logical [B, Sq] into an effective
    # [B * Sq, 1] sparse-decode batch before calling FlashMLA.
    q_kernel = q.view(rows, 1, LOCAL_Q_HEADS, QK_DIM)
    indices_kernel = indices.view(rows, 1, TOPK)
    # Graph capture uses dummy short sequence lengths (1..Sq) even though the
    # fixed sparse page table and scheduler retain the production top-k width.
    row_lengths = torch.arange(
        1, sq + 1, dtype=torch.int32, device=q.device
    ).repeat(b)
    scheduler_metadata, num_splits = (
        torch.ops.sgl_kernel.get_mla_decoding_metadata.default(
            row_lengths,
            LOCAL_Q_HEADS,
            1,
            LOCAL_Q_HEADS,
            True,
            TOPK,
        )
    )
    assert scheduler_metadata.shape[0] == 148
    for _ in range(3):
        torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
            q_kernel,
            cache,
            global_scale,
            indices_kernel,
            None,
            None,
            scheduler_metadata,
            num_splits,
            V_DIM,
            1.0,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse, _, _ = torch.ops.sgl_kernel.sparse_decode_fwd_nvfp4.default(
            q_kernel,
            cache,
            global_scale,
            indices_kernel,
            None,
            None,
            scheduler_metadata,
            num_splits,
            V_DIM,
            1.0,
        )
    out_address = out.data_ptr()
    first = out.clone()
    q_kernel.zero_()
    indices_kernel[..., 0].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert out.data_ptr() == out_address
    assert torch.isfinite(lse).all()
    assert not torch.equal(first, out)
