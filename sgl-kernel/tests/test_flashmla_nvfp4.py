"""Standalone correctness coverage for the SM90 sparse NVFP4 MLA op.

The native kernel consumes physical token indices rather than a logical block
table.  This test deliberately stores the logical sequence through a
non-identity page mapping, passes the corresponding physical top-k indices,
and compares against attention over the exactly dequantized 416-byte cache.
"""

import math
from typing import Callable, Tuple

import pytest
import torch

try:
    import sgl_kernel.flash_mla as _flash_mla
except Exception as exc:
    _flash_mla = None
    _flash_mla_import_error = exc
else:
    _flash_mla_import_error = None

try:
    from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
        NVFP4_BYTES_PER_TOKEN,
        NVFP4_LATENT_DIM,
        NVFP4_ROPE_DIM,
        dequantize_nvfp4_k_cache_paged,
        quantize_nvfp4_k_cache_into,
    )
except Exception as exc:
    NVFP4_BYTES_PER_TOKEN = 416
    NVFP4_LATENT_DIM = 512
    NVFP4_ROPE_DIM = 64
    dequantize_nvfp4_k_cache_paged = None
    quantize_nvfp4_k_cache_into = None
    _codec_import_error = exc
else:
    _codec_import_error = None


_PAGE_SIZE = 64
_HEAD_DIM = NVFP4_LATENT_DIM + NVFP4_ROPE_DIM
_NUM_Q_HEADS = 128
_PROFILE_TOKENS_PER_REQUEST = 32 * 1024
_PROFILE_PAGES_PER_REQUEST = _PROFILE_TOKENS_PER_REQUEST // _PAGE_SIZE


def _is_sm90_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (9, 0):
        return False
    if torch.version.cuda is None:
        return False
    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    return cuda_version >= (12, 5)


def _get_native_entry_points() -> Tuple[Callable, Callable]:
    if _flash_mla is None:
        pytest.fail(f"sgl_kernel.flash_mla is unavailable: {_flash_mla_import_error}")

    native_fwd = getattr(_flash_mla, "flash_mla_with_kvcache_nvfp4", None)
    get_metadata = getattr(_flash_mla, "get_mla_metadata", None)
    if native_fwd is None:
        pytest.fail("flash_mla_with_kvcache_nvfp4 is not implemented")
    if get_metadata is None:
        pytest.fail("get_mla_metadata is unavailable")

    # The Python source may be newer than the installed extension.  Treat that
    # as an unavailable native implementation rather than failing collection.
    if not hasattr(torch.ops.sgl_kernel, "fwd_kvcache_mla_nvfp4"):
        pytest.fail("sgl_kernel was built without fwd_kvcache_mla_nvfp4")
    return native_fwd, get_metadata


def _require_codec() -> Tuple[Callable, Callable]:
    if quantize_nvfp4_k_cache_into is None or dequantize_nvfp4_k_cache_paged is None:
        pytest.fail(f"NSA NVFP4 codec is unavailable: {_codec_import_error}")
    return quantize_nvfp4_k_cache_into, dequantize_nvfp4_k_cache_paged


def _logical_to_physical(
    logical_indices: torch.Tensor, page_order: torch.Tensor
) -> torch.Tensor:
    """Map logical token positions through a permuted physical page table."""

    logical_indices = logical_indices.to(torch.int64)
    logical_pages = torch.div(logical_indices, _PAGE_SIZE, rounding_mode="floor")
    offsets = logical_indices.remainder(_PAGE_SIZE)
    return page_order[logical_pages] * _PAGE_SIZE + offsets


def _make_physical_topk(
    *,
    seq_len: int,
    seq_len_q: int,
    topk: int,
    page_order: torch.Tensor,
    token_capacity: int,
    generator: torch.Generator,
    num_invalid: int = 8,
) -> torch.Tensor:
    """Create shuffled physical top-k rows with invalid entries mixed in."""

    assert 0 <= num_invalid <= topk
    num_negative = num_invalid // 2
    num_valid = topk - num_invalid
    assert num_valid >= page_order.numel()
    assert num_valid <= seq_len

    # Force every logical page to participate, then sample the remaining rows.
    required = torch.arange(page_order.numel(), device=page_order.device) * _PAGE_SIZE
    candidates = torch.arange(seq_len, device=page_order.device)
    keep = torch.ones(seq_len, dtype=torch.bool, device=page_order.device)
    keep[required] = False
    candidates = candidates[keep]

    rows = []
    for _ in range(seq_len_q):
        sample = candidates[
            torch.randperm(
                candidates.numel(), device=candidates.device, generator=generator
            )[: num_valid - required.numel()]
        ]
        logical = torch.cat((required, sample))
        physical = _logical_to_physical(logical, page_order).to(torch.int32)
        physical = torch.cat(
            (
                physical,
                -torch.arange(
                    1,
                    num_negative + 1,
                    dtype=torch.int32,
                    device=physical.device,
                ),
                torch.arange(
                    token_capacity,
                    token_capacity + num_invalid - num_negative,
                    dtype=torch.int32,
                    device=physical.device,
                ),
            )
        )
        physical = physical[
            torch.randperm(topk, device=physical.device, generator=generator)
        ]
        rows.append(physical)
    return torch.stack(rows).unsqueeze(0).contiguous()


def _reference_sparse_attention(
    q: torch.Tensor,
    dequantized_physical_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference the native op's output and natural-log LSE in FP32."""

    batch, seq_len_q, num_heads, head_dim = q.shape
    assert num_heads > 0
    assert head_dim == _HEAD_DIM

    physical_cache = dequantized_physical_cache.view(-1, _HEAD_DIM).float()
    out = torch.empty(
        (batch, seq_len_q, num_heads, NVFP4_LATENT_DIM),
        dtype=torch.float32,
        device=q.device,
    )
    lse = torch.empty(
        (batch, num_heads, seq_len_q), dtype=torch.float32, device=q.device
    )

    for batch_idx in range(batch):
        for query_idx in range(seq_len_q):
            indices = physical_indices[batch_idx, query_idx]
            indices = indices[
                (indices >= 0) & (indices < physical_cache.shape[0])
            ].long()
            selected_kv = physical_cache.index_select(0, indices)
            logits = q[batch_idx, query_idx].float() @ selected_kv.transpose(0, 1)
            logits *= softmax_scale
            lse[batch_idx, :, query_idx] = torch.logsumexp(logits, dim=-1)
            probability = torch.softmax(logits, dim=-1, dtype=torch.float32)
            out[batch_idx, query_idx] = probability @ selected_kv[:, :NVFP4_LATENT_DIM]
    return out.to(torch.bfloat16), lse


def _reference_sparse_attention_selected(
    q: torch.Tensor,
    selected_kv: torch.Tensor,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference attention when selected physical rows are already gathered."""

    batch, seq_len_q, num_heads, head_dim = q.shape
    assert selected_kv.ndim == 4
    assert selected_kv.shape[:2] == (batch, seq_len_q)
    assert selected_kv.shape[-1] == head_dim == _HEAD_DIM

    selected_kv = selected_kv.float()
    logits = torch.matmul(q.float(), selected_kv.transpose(-1, -2))
    logits *= softmax_scale
    lse = torch.logsumexp(logits, dim=-1).transpose(1, 2).contiguous()
    probability = torch.softmax(logits, dim=-1, dtype=torch.float32)
    out = torch.matmul(probability, selected_kv[..., :NVFP4_LATENT_DIM])
    assert out.shape == (batch, seq_len_q, num_heads, NVFP4_LATENT_DIM)
    return out.to(torch.bfloat16), lse


def _make_glm52_tp8_case(
    quantize: Callable,
    dequantize: Callable,
    get_metadata: Callable,
    batch_size: int = 1,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Build the GLM-5/5.2 TP8 decode shape: Hq=8 and top-k=2048."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20262708)
    seq_len = 2053
    seq_len_q = 1
    num_q_heads = 8
    topk = 2048
    num_logical_pages = math.ceil(seq_len / _PAGE_SIZE)
    num_physical_pages = num_logical_pages + 2
    page_order = torch.roll(
        torch.arange(num_logical_pages, dtype=torch.int64, device=device),
        shifts=7,
    )
    assert not torch.equal(page_order, torch.arange(num_logical_pages, device=device))

    q = (
        torch.randn(
            (batch_size, seq_len_q, num_q_heads, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    logical_kv = (
        torch.randn(
            (seq_len, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    global_scale = torch.tensor([1.375], dtype=torch.float32, device=device)
    k_cache = torch.zeros(
        (num_physical_pages, _PAGE_SIZE, 1, NVFP4_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    logical_rows = torch.arange(seq_len, dtype=torch.int64, device=device)
    physical_rows = _logical_to_physical(logical_rows, page_order).to(torch.int32)
    quantize(
        logical_kv[:, :NVFP4_LATENT_DIM],
        logical_kv[:, NVFP4_LATENT_DIM:],
        k_cache,
        physical_rows,
        global_scale,
    )

    # This target shape needs 2048 genuinely valid selections.  Keep the
    # invalid-index behavior in the smaller tests instead of displacing rows.
    physical_topk = (
        _make_physical_topk(
            seq_len=seq_len,
            seq_len_q=seq_len_q,
            topk=topk,
            page_order=page_order,
            token_capacity=num_physical_pages * _PAGE_SIZE,
            generator=generator,
            num_invalid=0,
        )
        .expand(batch_size, seq_len_q, topk)
        .contiguous()
    )
    assert physical_topk[0].unique().numel() == topk
    assert bool((physical_topk >= 0).all())
    assert bool((physical_topk < num_physical_pages * _PAGE_SIZE).all())

    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    tile_scheduler_metadata, num_splits = get_metadata(
        cache_seqlens,
        seq_len_q * num_q_heads,
        1,
        num_q_heads,
        is_fp8_kvcache=True,
        topk=topk,
    )
    all_physical_rows = torch.arange(
        num_physical_pages * _PAGE_SIZE, dtype=torch.int32, device=device
    )
    dequantized = dequantize(
        k_cache, all_physical_rows, global_scale, dtype=torch.bfloat16
    )
    softmax_scale = 1.0 / math.sqrt(_HEAD_DIM)
    return (
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        dequantized,
        softmax_scale,
    )


def _make_disjoint_32k_case(
    quantize: Callable,
    dequantize: Callable,
    get_metadata: Callable,
    batch_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Build the profile shape with one disjoint 32K physical range per request."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260709 + batch_size)
    seq_len_q = 1
    num_q_heads = 8
    topk = 2048
    num_physical_pages = batch_size * _PROFILE_PAGES_PER_REQUEST
    token_capacity = num_physical_pages * _PAGE_SIZE

    q = (
        torch.randn(
            (batch_size, seq_len_q, num_q_heads, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    k_cache = torch.zeros(
        (num_physical_pages, _PAGE_SIZE, 1, NVFP4_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    global_scale = torch.tensor([1.375], dtype=torch.float32, device=device)

    # Four tokens from every page give exactly 2048 selections.  The request
    # base makes the flattened physical ranges pairwise disjoint, while the
    # per-request slot rotation prevents all requests from using identical
    # page offsets.
    page_ids = torch.arange(
        _PROFILE_PAGES_PER_REQUEST, dtype=torch.int32, device=device
    ).repeat_interleave(4)
    slots = torch.tensor([0, 17, 34, 51], dtype=torch.int32, device=device).repeat(
        _PROFILE_PAGES_PER_REQUEST
    )
    physical_rows = []
    for batch_idx in range(batch_size):
        request_start = batch_idx * _PROFILE_TOKENS_PER_REQUEST
        request_rows = (
            request_start
            + page_ids * _PAGE_SIZE
            + (slots + batch_idx * 7).remainder(_PAGE_SIZE)
        )
        request_rows = request_rows[
            torch.randperm(topk, device=device, generator=generator)
        ]
        physical_rows.append(request_rows)
    physical_topk = torch.stack(physical_rows).unsqueeze(1).contiguous()

    request_starts = (
        torch.arange(batch_size, dtype=torch.int32, device=device)
        * _PROFILE_TOKENS_PER_REQUEST
    )
    assert physical_topk.shape == (batch_size, seq_len_q, topk)
    assert bool((physical_topk[:, 0].amin(dim=1) >= request_starts).all())
    assert bool(
        (
            physical_topk[:, 0].amax(dim=1)
            < request_starts + _PROFILE_TOKENS_PER_REQUEST
        ).all()
    )
    assert physical_topk.unique().numel() == batch_size * topk
    assert bool((physical_topk >= 0).all())
    assert bool((physical_topk < token_capacity).all())
    assert all(
        (physical_topk[batch_idx, 0] - batch_idx * _PROFILE_TOKENS_PER_REQUEST)
        .div(_PAGE_SIZE, rounding_mode="floor")
        .unique()
        .numel()
        == _PROFILE_PAGES_PER_REQUEST
        for batch_idx in range(batch_size)
    )

    selected_kv = (
        torch.randn(
            (batch_size * topk, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    flat_physical_rows = physical_topk.reshape(-1)
    quantize(
        selected_kv[:, :NVFP4_LATENT_DIM],
        selected_kv[:, NVFP4_LATENT_DIM:],
        k_cache,
        flat_physical_rows,
        global_scale,
    )

    cache_seqlens = torch.full(
        (batch_size,),
        _PROFILE_TOKENS_PER_REQUEST,
        dtype=torch.int32,
        device=device,
    )
    tile_scheduler_metadata, num_splits = get_metadata(
        cache_seqlens,
        seq_len_q * num_q_heads,
        1,
        num_q_heads,
        is_fp8_kvcache=True,
        topk=topk,
    )
    dequantized_selected = dequantize(
        k_cache, flat_physical_rows, global_scale, dtype=torch.bfloat16
    ).view(batch_size, seq_len_q, topk, _HEAD_DIM)
    softmax_scale = 1.0 / math.sqrt(_HEAD_DIM)
    return (
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        dequantized_selected,
        softmax_scale,
    )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@pytest.mark.parametrize(
    ("topk", "global_scale_value"),
    [(64, 0.625), (128, 1.75)],
)
@torch.inference_mode()
def test_flashmla_sparse_nvfp4_physical_pages(
    topk: int, global_scale_value: float
) -> None:
    native_fwd, get_metadata = _get_native_entry_points()
    quantize, dequantize = _require_codec()

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260708 + topk)
    batch_size = 2
    seq_len = 173
    seq_len_q = 2
    num_physical_pages = 5

    # Three logical pages occupy physical pages 3, 0, and 4.  Pages 1 and 2
    # remain unused so an accidental logical-index interpretation is visible.
    page_order = torch.tensor([3, 0, 4], dtype=torch.int64, device=device)
    assert not torch.equal(page_order, torch.arange(page_order.numel(), device=device))

    q = (
        torch.randn(
            (batch_size, seq_len_q, _NUM_Q_HEADS, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    logical_kv = (
        torch.randn(
            (seq_len, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    global_scale = torch.tensor(
        [global_scale_value], dtype=torch.float32, device=device
    )
    assert global_scale_value != 1.0

    k_cache = torch.zeros(
        (
            num_physical_pages,
            _PAGE_SIZE,
            1,
            NVFP4_BYTES_PER_TOKEN,
        ),
        dtype=torch.uint8,
        device=device,
    )
    logical_rows = torch.arange(seq_len, dtype=torch.int64, device=device)
    physical_rows = _logical_to_physical(logical_rows, page_order).to(torch.int32)
    quantize(
        logical_kv[:, :NVFP4_LATENT_DIM],
        logical_kv[:, NVFP4_LATENT_DIM:],
        k_cache,
        physical_rows,
        global_scale,
    )

    physical_topk = torch.cat(
        [
            _make_physical_topk(
                seq_len=seq_len,
                seq_len_q=seq_len_q,
                topk=topk,
                page_order=page_order,
                token_capacity=num_physical_pages * _PAGE_SIZE,
                generator=generator,
            )
            for _ in range(batch_size)
        ],
        dim=0,
    )
    token_capacity = num_physical_pages * _PAGE_SIZE
    invalid_mask = (physical_topk < 0) | (physical_topk >= token_capacity)
    assert bool((physical_topk < -1).any())
    assert bool((physical_topk >= token_capacity).any())
    assert bool((physical_topk > token_capacity).any())
    assert torch.equal(
        invalid_mask.sum(dim=-1),
        torch.full((batch_size, seq_len_q), 8, dtype=torch.int64, device=device),
    )

    cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    tile_scheduler_metadata, num_splits = get_metadata(
        cache_seqlens,
        seq_len_q * _NUM_Q_HEADS,
        1,
        _NUM_Q_HEADS,
        is_fp8_kvcache=True,
        topk=topk,
    )
    softmax_scale = 1.0 / math.sqrt(_HEAD_DIM)
    out, lse = native_fwd(
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        head_dim_v=NVFP4_LATENT_DIM,
        softmax_scale=softmax_scale,
    )
    # The native contract masks both negative and positive out-of-capacity
    # values.  They must neither dereference the cache nor poison the result.
    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(lse).all())

    all_physical_rows = torch.arange(
        num_physical_pages * _PAGE_SIZE, dtype=torch.int32, device=device
    )
    dequantized = dequantize(
        k_cache, all_physical_rows, global_scale, dtype=torch.bfloat16
    )
    out_ref, lse_ref = _reference_sparse_attention(
        q, dequantized, physical_topk, softmax_scale
    )

    assert out.shape == out_ref.shape
    assert lse.shape == lse_ref.shape
    torch.testing.assert_close(out, out_ref, atol=8e-4, rtol=2.01 / 128)
    torch.testing.assert_close(lse, lse_ref, atol=2e-4, rtol=8.01 / 65536)


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_sparse_nvfp4_glm52_tp8_shape() -> None:
    native_fwd, get_metadata = _get_native_entry_points()
    quantize, dequantize = _require_codec()
    (
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        dequantized_cache,
        softmax_scale,
    ) = _make_glm52_tp8_case(quantize, dequantize, get_metadata, batch_size=4)

    # Four effective requests cover the MTP 3-1-4 expansion used by GLM-5.2.
    assert q.shape == (4, 1, 8, _HEAD_DIM)
    assert physical_topk.shape == (4, 1, 2048)
    out, lse = native_fwd(
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        head_dim_v=NVFP4_LATENT_DIM,
        softmax_scale=softmax_scale,
    )
    out_ref, lse_ref = _reference_sparse_attention(
        q, dequantized_cache, physical_topk, softmax_scale
    )

    assert out.shape == (4, 1, 8, NVFP4_LATENT_DIM)
    assert lse.shape == (4, 8, 1)
    torch.testing.assert_close(out, out_ref, atol=8e-4, rtol=2.01 / 128)
    torch.testing.assert_close(lse, lse_ref, atol=2e-4, rtol=8.01 / 65536)


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_sparse_nvfp4_rejects_invalid_contracts() -> None:
    native_fwd, get_metadata = _get_native_entry_points()
    quantize, dequantize = _require_codec()
    (
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        _,
        softmax_scale,
    ) = _make_glm52_tp8_case(quantize, dequantize, get_metadata)

    common_args = (
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
    )
    with pytest.raises(RuntimeError, match="q must have dtype bfloat16"):
        native_fwd(
            q.float(),
            *common_args,
            physical_topk,
            head_dim_v=NVFP4_LATENT_DIM,
            softmax_scale=softmax_scale,
        )
    with pytest.raises(RuntimeError, match="topk must be a multiple of 64"):
        native_fwd(
            q,
            *common_args,
            physical_topk[..., :-1].contiguous(),
            head_dim_v=NVFP4_LATENT_DIM,
            softmax_scale=softmax_scale,
        )
    bad_cache = torch.empty(
        (k_cache.shape[0], _PAGE_SIZE, 1, NVFP4_BYTES_PER_TOKEN - 1),
        dtype=torch.uint8,
        device=q.device,
    )
    with pytest.raises(RuntimeError, match="416-byte NVFP4 layout"):
        native_fwd(
            q,
            bad_cache,
            global_scale,
            cache_seqlens,
            tile_scheduler_metadata,
            num_splits,
            physical_topk,
            head_dim_v=NVFP4_LATENT_DIM,
            softmax_scale=softmax_scale,
        )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@pytest.mark.parametrize("batch_size", [12, 17])
@torch.inference_mode()
def test_flashmla_sparse_nvfp4_disjoint_32k_ranges(batch_size: int) -> None:
    native_fwd, get_metadata = _get_native_entry_points()
    quantize, dequantize = _require_codec()
    (
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        dequantized_selected,
        softmax_scale,
    ) = _make_disjoint_32k_case(
        quantize, dequantize, get_metadata, batch_size=batch_size
    )

    assert k_cache.shape == (
        batch_size * _PROFILE_PAGES_PER_REQUEST,
        _PAGE_SIZE,
        1,
        NVFP4_BYTES_PER_TOKEN,
    )
    assert torch.equal(
        cache_seqlens,
        torch.full_like(cache_seqlens, _PROFILE_TOKENS_PER_REQUEST),
    )
    out, lse = native_fwd(
        q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        head_dim_v=NVFP4_LATENT_DIM,
        softmax_scale=softmax_scale,
    )
    out_ref, lse_ref = _reference_sparse_attention_selected(
        q, dequantized_selected, softmax_scale
    )

    assert out.shape == (batch_size, 1, 8, NVFP4_LATENT_DIM)
    assert lse.shape == (batch_size, 8, 1)
    torch.testing.assert_close(out, out_ref, atol=8e-4, rtol=2.01 / 128)
    torch.testing.assert_close(lse, lse_ref, atol=2e-4, rtol=8.01 / 65536)


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_sparse_nvfp4_cuda_graph_replay() -> None:
    native_fwd, get_metadata = _get_native_entry_points()
    quantize, dequantize = _require_codec()
    (
        replay_q,
        k_cache,
        global_scale,
        cache_seqlens,
        tile_scheduler_metadata,
        num_splits,
        physical_topk,
        dequantized_cache,
        softmax_scale,
    ) = _make_glm52_tp8_case(quantize, dequantize, get_metadata)

    # Keep q's address stable across capture/replay.  Starting from zeros makes
    # the post-replay mutation visible even before comparing with the reference.
    static_q = torch.zeros_like(replay_q)
    assert static_q.shape == (1, 1, 8, _HEAD_DIM)
    assert physical_topk.shape == (1, 1, 2048)

    # Warm lazy CUDA state on a side stream so capture contains only the op.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        native_fwd(
            static_q,
            k_cache,
            global_scale,
            cache_seqlens,
            tile_scheduler_metadata,
            num_splits,
            physical_topk,
            head_dim_v=NVFP4_LATENT_DIM,
            softmax_scale=softmax_scale,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = native_fwd(
            static_q,
            k_cache,
            global_scale,
            cache_seqlens,
            tile_scheduler_metadata,
            num_splits,
            physical_topk,
            head_dim_v=NVFP4_LATENT_DIM,
            softmax_scale=softmax_scale,
        )

    captured_out = graph_out.clone()
    captured_lse = graph_lse.clone()
    static_q.copy_(replay_q)
    graph.replay()
    torch.cuda.synchronize()

    out_ref, lse_ref = _reference_sparse_attention(
        replay_q, dequantized_cache, physical_topk, softmax_scale
    )
    assert not torch.equal(graph_out, captured_out)
    assert not torch.equal(graph_lse, captured_lse)
    torch.testing.assert_close(graph_out, out_ref, atol=8e-4, rtol=2.01 / 128)
    torch.testing.assert_close(graph_lse, lse_ref, atol=2e-4, rtol=8.01 / 65536)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
