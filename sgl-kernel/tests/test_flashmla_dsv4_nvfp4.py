"""Correctness coverage for the SM90 DeepSeek V4 NVFP4 decode op.

The native kernel consumes the production 380-byte DSV4 cache row and must
apply one online softmax over the SWA cache, the optional compressed cache,
and the attention sink.  References in this file therefore start from the
exactly dequantized physical cache rather than the pre-quantization BF16 keys.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

try:
    from sgl_kernel.flash_mla import (
        FlashMLASchedMeta,
        flash_mla_with_kvcache_dsv4_nvfp4,
    )
except Exception as exc:
    FlashMLASchedMeta = None
    flash_mla_with_kvcache_dsv4_nvfp4 = None
    _flashmla_import_error = exc
else:
    _flashmla_import_error = None

try:
    from sglang.srt.layers.attention.dsv4.nvfp4_k_cache import (
        DSV4_NVFP4_BYTES_PER_TOKEN,
        dequantize_dsv4_nvfp4_k_cache_paged,
        quantize_dsv4_nvfp4_k_cache_into,
    )
except Exception as exc:
    DSV4_NVFP4_BYTES_PER_TOKEN = 380
    dequantize_dsv4_nvfp4_k_cache_paged = None
    quantize_dsv4_nvfp4_k_cache_into = None
    _codec_import_error = exc
else:
    _codec_import_error = None


_HEAD_DIM = 512
_SWA_PAGE_SIZE = 256
_SWA_TOPK = 128
_OUTPUT_ATOL = 8e-4
_OUTPUT_RTOL = 2.01 / 128
_LSE_ATOL = 2e-4
_LSE_RTOL = 8.01 / 65536


def _is_sm90_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (9, 0):
        return False
    if torch.version.cuda is None:
        return False
    version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    return version >= (12, 5)


def _require_native() -> Callable:
    if flash_mla_with_kvcache_dsv4_nvfp4 is None:
        pytest.fail(
            f"DSV4 NVFP4 FlashMLA wrapper is unavailable: {_flashmla_import_error}"
        )
    if not hasattr(torch.ops.sgl_kernel, "dsv4_sparse_decode_fwd_nvfp4"):
        pytest.fail("sglang-kernel was built without dsv4_sparse_decode_fwd_nvfp4")
    return flash_mla_with_kvcache_dsv4_nvfp4


def _require_codec() -> tuple[Callable, Callable]:
    if (
        quantize_dsv4_nvfp4_k_cache_into is None
        or dequantize_dsv4_nvfp4_k_cache_paged is None
    ):
        pytest.fail(f"DSV4 NVFP4 codec is unavailable: {_codec_import_error}")
    return (
        quantize_dsv4_nvfp4_k_cache_into,
        dequantize_dsv4_nvfp4_k_cache_paged,
    )


@dataclass
class _DecodeCase:
    q: torch.Tensor
    kv: torch.Tensor
    kv_scale: torch.Tensor
    indices: torch.Tensor
    topk_length: torch.Tensor
    attn_sink: torch.Tensor
    dequantized_kv: torch.Tensor
    sm_scale: float
    extra_kv: Optional[torch.Tensor] = None
    extra_scale: Optional[torch.Tensor] = None
    extra_indices: Optional[torch.Tensor] = None
    extra_topk_length: Optional[torch.Tensor] = None
    dequantized_extra_kv: Optional[torch.Tensor] = None


def _make_cache(
    *,
    page_size: int,
    minimum_tokens: int,
    global_scale_value: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantize, dequantize = _require_codec()
    device = torch.device("cuda")
    num_pages = math.ceil(minimum_tokens / page_size)
    capacity = num_pages * page_size
    raw = torch.zeros(
        (num_pages, page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    source = (
        torch.randn(
            (capacity, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    scale = torch.tensor([global_scale_value], dtype=torch.float32, device=device)
    physical_rows = torch.arange(capacity, dtype=torch.int32, device=device)
    quantize(
        cache_k=source,
        kv_buffer=raw,
        loc=physical_rows,
        page_size=page_size,
        global_scale=scale,
    )
    dequantized = dequantize(
        raw,
        physical_rows,
        page_size=page_size,
        global_scale=scale,
    ).squeeze(1)
    cache_4d = raw.view(num_pages, page_size, 1, DSV4_NVFP4_BYTES_PER_TOKEN)
    return cache_4d, scale, dequantized


def _make_indices(
    *,
    batch_size: int,
    width: int,
    capacity: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert width % 64 == 0
    assert capacity >= batch_size * width
    device = torch.device("cuda")
    rows = []
    lengths = []
    for batch_idx in range(batch_size):
        begin = batch_idx * width
        row = torch.arange(begin, begin + width, dtype=torch.int32, device=device)
        row = row[torch.randperm(width, device=device, generator=generator)]
        length = width - 1 if batch_idx == 0 else max(1, width * 3 // 5)

        # Invalid entries inside the active prefix test the physical-capacity
        # mask.  The inactive suffix is deliberately all out of range so the
        # topk-length mask must run before any cache dereference.
        row[1] = -1
        row[3] = capacity + 17
        row[length:] = capacity + 31
        rows.append(row)
        lengths.append(length)
    return (
        torch.stack(rows).unsqueeze(1).contiguous(),
        torch.tensor(lengths, dtype=torch.int32, device=device),
    )


def _build_case(
    *,
    h_q: int,
    batch_size: int = 2,
    extra_page_size: Optional[int] = None,
    extra_topk: int = 0,
    seed: int = 20260714,
) -> _DecodeCase:
    assert h_q in (64, 128)
    assert (extra_page_size is None) == (extra_topk == 0)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed + h_q + extra_topk)

    q = (
        torch.randn(
            (batch_size, 1, h_q, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 10
    )
    kv, kv_scale, dequantized_kv = _make_cache(
        page_size=_SWA_PAGE_SIZE,
        minimum_tokens=batch_size * _SWA_TOPK,
        global_scale_value=0.625,
        generator=generator,
    )
    kv_capacity = kv.shape[0] * kv.shape[1]
    indices, topk_length = _make_indices(
        batch_size=batch_size,
        width=_SWA_TOPK,
        capacity=kv_capacity,
        generator=generator,
    )
    attn_sink = torch.linspace(-0.75, 0.25, h_q, dtype=torch.float32, device=device)
    case = _DecodeCase(
        q=q,
        kv=kv,
        kv_scale=kv_scale,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        dequantized_kv=dequantized_kv,
        sm_scale=1.0 / math.sqrt(_HEAD_DIM),
    )

    if extra_page_size is not None:
        extra_kv, extra_scale, dequantized_extra = _make_cache(
            page_size=extra_page_size,
            minimum_tokens=batch_size * extra_topk,
            global_scale_value=1.75,
            generator=generator,
        )
        extra_capacity = extra_kv.shape[0] * extra_kv.shape[1]
        extra_indices, extra_topk_length = _make_indices(
            batch_size=batch_size,
            width=extra_topk,
            capacity=extra_capacity,
            generator=generator,
        )
        case.extra_kv = extra_kv
        case.extra_scale = extra_scale
        case.extra_indices = extra_indices
        case.extra_topk_length = extra_topk_length
        case.dequantized_extra_kv = dequantized_extra
    return case


def _selected_rows(
    cache: torch.Tensor, indices: torch.Tensor, length: int
) -> torch.Tensor:
    length = max(0, min(length, indices.numel()))
    active = indices[:length]
    valid = (active >= 0) & (active < cache.shape[0])
    return cache.index_select(0, active[valid].long())


def _reference(case: _DecodeCase) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, h_q, _ = case.q.shape
    out = torch.empty(
        (batch_size, 1, h_q, _HEAD_DIM),
        dtype=torch.float32,
        device=case.q.device,
    )
    lse = torch.empty((batch_size, h_q, 1), dtype=torch.float32, device=case.q.device)
    primary = case.dequantized_kv.float()
    extra = (
        case.dequantized_extra_kv.float()
        if case.dequantized_extra_kv is not None
        else None
    )

    for batch_idx in range(batch_size):
        selected = [
            _selected_rows(
                primary,
                case.indices[batch_idx, 0],
                int(case.topk_length[batch_idx].item()),
            )
        ]
        if extra is not None:
            assert case.extra_indices is not None
            assert case.extra_topk_length is not None
            selected.append(
                _selected_rows(
                    extra,
                    case.extra_indices[batch_idx, 0],
                    int(case.extra_topk_length[batch_idx].item()),
                )
            )
        selected_kv = torch.cat(selected, dim=0)
        logits = case.q[batch_idx, 0].float() @ selected_kv.transpose(0, 1)
        logits *= case.sm_scale
        logits_with_sink = torch.cat((logits, case.attn_sink[:, None]), dim=-1)
        probabilities = torch.softmax(logits_with_sink, dim=-1, dtype=torch.float32)
        out[batch_idx, 0] = probabilities[:, :-1] @ selected_kv
        # Preserve FlashMLA's established ABI: the sink participates in the
        # output normalization, while the returned LSE describes selected KV
        # logits only (the combine kernel merges the sink after storing LSE).
        lse[batch_idx, :, 0] = torch.logsumexp(logits, dim=-1)
    return out.to(torch.bfloat16), lse


def _run_native(
    native: Callable,
    case: _DecodeCase,
    sched_meta,
) -> tuple[torch.Tensor, torch.Tensor]:
    return native(
        q=case.q,
        k_cache=case.kv,
        kv_global_scale=case.kv_scale,
        indices=case.indices,
        topk_length=case.topk_length,
        attn_sink=case.attn_sink,
        tile_scheduler_metadata=sched_meta,
        head_dim_v=_HEAD_DIM,
        softmax_scale=case.sm_scale,
        extra_k_cache=case.extra_kv,
        extra_kv_global_scale=case.extra_scale,
        extra_indices_in_kvcache=case.extra_indices,
        extra_topk_length=case.extra_topk_length,
    )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@pytest.mark.parametrize(
    ("name", "batch_size", "h_q", "extra_page_size", "extra_topk"),
    [
        ("c0_flash_b1", 1, 64, None, 0),
        ("c0_flash_b2", 2, 64, None, 0),
        ("c0_pro", 2, 128, None, 0),
        ("c4_flash", 2, 64, 64, 512),
        ("c4_pro", 2, 128, 64, 1024),
        ("c128_flash", 2, 64, 2, 1024),
        ("c128_pro_b1", 1, 128, 2, 1024),
    ],
)
@torch.inference_mode()
def test_flashmla_dsv4_nvfp4_dual_source_correctness(
    name: str,
    batch_size: int,
    h_q: int,
    extra_page_size: Optional[int],
    extra_topk: int,
) -> None:
    del name
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _build_case(
        h_q=h_q,
        batch_size=batch_size,
        extra_page_size=extra_page_size,
        extra_topk=extra_topk,
    )
    out, lse = _run_native(native, case, FlashMLASchedMeta())
    out_ref, lse_ref = _reference(case)

    assert out.shape == out_ref.shape == (batch_size, 1, h_q, _HEAD_DIM)
    assert lse.shape == lse_ref.shape == (batch_size, h_q, 1)
    assert bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(lse).all())
    torch.testing.assert_close(out, out_ref, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL)
    torch.testing.assert_close(lse, lse_ref, atol=_LSE_ATOL, rtol=_LSE_RTOL)


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_nvfp4_rejects_invalid_contracts() -> None:
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _build_case(h_q=64, batch_size=1)

    with pytest.raises(RuntimeError, match="q must have dtype bfloat16"):
        native(
            case.q.float(),
            case.kv,
            case.kv_scale,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    with pytest.raises(RuntimeError, match="multiple of 64"):
        native(
            case.q,
            case.kv,
            case.kv_scale,
            case.indices[..., :-1].contiguous(),
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    bad_cache = torch.empty(
        (1, _SWA_PAGE_SIZE, 1, DSV4_NVFP4_BYTES_PER_TOKEN - 1),
        dtype=torch.uint8,
        device=case.q.device,
    )
    with pytest.raises((RuntimeError, ValueError), match="380"):
        native(
            case.q,
            bad_cache,
            case.kv_scale,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    cache_numel = case.kv.numel()
    misaligned_storage = torch.empty(
        cache_numel + 1, dtype=torch.uint8, device=case.q.device
    )
    misaligned_cache = misaligned_storage[1:].view_as(case.kv)
    assert misaligned_cache.data_ptr() % 4 != 0
    with pytest.raises(RuntimeError, match="4-byte aligned"):
        native(
            case.q,
            misaligned_cache,
            case.kv_scale,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        native(
            case.q,
            case.kv,
            case.kv_scale,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
            extra_k_cache=case.kv,
        )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_nvfp4_clamps_graph_replayed_lengths() -> None:
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _build_case(h_q=64, extra_page_size=64, extra_topk=512)
    assert case.extra_indices is not None
    assert case.extra_topk_length is not None

    # Warm lazy extension state outside capture. A fresh metadata object below
    # makes the private scheduler itself part of the captured graph.
    _run_native(native, case, FlashMLASchedMeta())
    torch.cuda.synchronize()

    capture_meta = FlashMLASchedMeta()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = _run_native(native, case, capture_meta)

    out_ptr = graph_out.data_ptr()
    lse_ptr = graph_lse.data_ptr()
    int32 = torch.iinfo(torch.int32)

    # Length tensors are graph-replayed device inputs. Exercise both signs on
    # both sources: request 0 attends the clamped-full primary and empty extra;
    # request 1 attends the empty primary and clamped-full extra. The scheduler
    # and producer must agree without ever indexing beyond either padded width.
    case.topk_length.copy_(
        torch.tensor(
            [int32.max, int32.min],
            dtype=torch.int32,
            device=case.q.device,
        )
    )
    case.extra_topk_length.copy_(
        torch.tensor(
            [int32.min, int32.max],
            dtype=torch.int32,
            device=case.q.device,
        )
    )

    graph.replay()
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(case)

    assert graph_out.data_ptr() == out_ptr
    assert graph_lse.data_ptr() == lse_ptr
    assert bool(torch.isfinite(graph_out).all())
    assert bool(torch.isfinite(graph_lse).all())
    torch.testing.assert_close(graph_out, out_ref, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL)
    torch.testing.assert_close(graph_lse, lse_ref, atol=_LSE_ATOL, rtol=_LSE_RTOL)


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@pytest.mark.parametrize(
    ("name", "h_q", "extra_page_size", "extra_topk"),
    [
        ("c0_flash", 64, None, 0),
        ("c0_pro", 128, None, 0),
        ("c4_flash", 64, 64, 512),
        ("c4_pro", 128, 64, 1024),
        ("c128_flash", 64, 2, 1024),
        ("c128_pro", 128, 2, 1024),
    ],
)
@torch.inference_mode()
def test_flashmla_dsv4_nvfp4_cuda_graph_replay(
    name: str, h_q: int, extra_page_size: Optional[int], extra_topk: int
) -> None:
    del name
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _build_case(
        h_q=h_q,
        extra_page_size=extra_page_size,
        extra_topk=extra_topk,
    )

    # Warm lazy extension state with a different metadata object.  The fresh
    # object used during capture intentionally records scheduler generation in
    # the graph, so replay can consume updated per-request top-k lengths.
    _run_native(native, case, FlashMLASchedMeta())
    torch.cuda.synchronize()

    static_case = _DecodeCase(
        q=torch.zeros_like(case.q),
        kv=case.kv,
        kv_scale=case.kv_scale,
        indices=case.indices.clone(),
        topk_length=case.topk_length.clone(),
        attn_sink=case.attn_sink,
        dequantized_kv=case.dequantized_kv,
        sm_scale=case.sm_scale,
        extra_kv=case.extra_kv,
        extra_scale=case.extra_scale,
        extra_indices=(
            case.extra_indices.clone() if case.extra_indices is not None else None
        ),
        extra_topk_length=(
            case.extra_topk_length.clone()
            if case.extra_topk_length is not None
            else None
        ),
        dequantized_extra_kv=case.dequantized_extra_kv,
    )
    capture_meta = FlashMLASchedMeta()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = _run_native(native, static_case, capture_meta)

    out_ptr = graph_out.data_ptr()
    lse_ptr = graph_lse.data_ptr()
    captured_out = graph_out.clone()
    captured_lse = graph_lse.clone()

    static_case.q.copy_(case.q)
    static_case.indices.copy_(torch.roll(case.indices, shifts=7, dims=-1))
    if case.extra_indices is not None:
        assert case.extra_topk_length is not None
        assert static_case.extra_indices is not None
        assert static_case.extra_topk_length is not None
        # Exercise clamping inside the scheduler kernel recorded by the graph,
        # not only in the eager safety test above. Request 1 still has valid
        # extra KV after its negative primary length is clamped to zero.
        static_case.topk_length.copy_(
            torch.tensor(
                [case.indices.shape[-1] + 17, -9],
                dtype=torch.int32,
                device=case.q.device,
            )
        )
        static_case.extra_indices.copy_(
            torch.roll(case.extra_indices, shifts=11, dims=-1)
        )
        static_case.extra_topk_length.copy_(
            torch.tensor(
                [extra_topk + 33, 37],
                dtype=torch.int32,
                device=case.q.device,
            )
        )
    else:
        static_case.topk_length.copy_(
            torch.tensor([65, 127], dtype=torch.int32, device=case.q.device)
        )

    graph.replay()
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(static_case)

    assert graph_out.data_ptr() == out_ptr
    assert graph_lse.data_ptr() == lse_ptr
    assert not torch.equal(graph_out, captured_out)
    assert not torch.equal(graph_lse, captured_lse)
    torch.testing.assert_close(graph_out, out_ref, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL)
    torch.testing.assert_close(graph_lse, lse_ref, atol=_LSE_ATOL, rtol=_LSE_RTOL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
