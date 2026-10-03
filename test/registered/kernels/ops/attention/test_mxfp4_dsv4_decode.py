"""Correctness coverage for the JIT-compiled MXFP4 DSV4 decode kernel.

The kernel (python/sglang/kernels/jit/csrc/mxfp4_dsv4_decode_sm90/) consumes
the production 368-byte MXFP4 DSV4 cache row and applies one online softmax
over the SWA cache, the optional compressed (C4/C128) cache, and the attention
sink. References start from the exactly dequantized physical cache rather than
the pre-quantization BF16 keys. Covers three layer kinds (C0/C4/C128) x flash
vs. profiling semantics, invalid-contract rejection, and CUDA-graph replay.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import msgspec
import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

try:
    from sglang.kernels.ops.attention.dsv4.mxfp4_dsv4_decode_sm90 import (
        FlashMLASchedMeta,
        flash_mla_with_kvcache_dsv4_mxfp4,
    )
except Exception as exc:
    FlashMLASchedMeta = None
    flash_mla_with_kvcache_dsv4_mxfp4 = None
    _flashmla_import_error = exc
else:
    _flashmla_import_error = None

try:
    from sglang.kernels.ops.attention.dsv4.mxfp4_k_cache import (
        MXFP4_BYTES_PER_TOKEN,
        dequantize_dsv4_mxfp4_k_cache_paged,
        quantize_dsv4_mxfp4_k_cache_into,
    )
except Exception as exc:
    MXFP4_BYTES_PER_TOKEN = 368
    dequantize_dsv4_mxfp4_k_cache_paged = None
    quantize_dsv4_mxfp4_k_cache_into = None
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
    if flash_mla_with_kvcache_dsv4_mxfp4 is None:
        pytest.fail(
            f"DSV4 MXFP4 FlashMLA wrapper is unavailable: {_flashmla_import_error}"
        )
    return flash_mla_with_kvcache_dsv4_mxfp4


def _require_codec() -> tuple[Callable, Callable]:
    if (
        quantize_dsv4_mxfp4_k_cache_into is None
        or dequantize_dsv4_mxfp4_k_cache_paged is None
    ):
        pytest.fail(f"DSV4 MXFP4 codec is unavailable: {_codec_import_error}")
    return (
        quantize_dsv4_mxfp4_k_cache_into,
        dequantize_dsv4_mxfp4_k_cache_paged,
    )


class _DecodeCase(msgspec.Struct):
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    topk_length: torch.Tensor
    attn_sink: torch.Tensor
    dequantized_kv: torch.Tensor
    sm_scale: float
    extra_kv: Optional[torch.Tensor] = None
    extra_indices: Optional[torch.Tensor] = None
    extra_topk_length: Optional[torch.Tensor] = None
    dequantized_extra_kv: Optional[torch.Tensor] = None


def _make_cache(
    *,
    page_size: int,
    minimum_tokens: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantize, dequantize = _require_codec()
    device = torch.device("cuda")
    num_pages = math.ceil(minimum_tokens / page_size)
    capacity = num_pages * page_size
    raw = torch.zeros(
        (num_pages, page_size * MXFP4_BYTES_PER_TOKEN),
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
    physical_rows = torch.arange(capacity, dtype=torch.int32, device=device)
    quantize(
        cache_k=source,
        kv_buffer=raw,
        loc=physical_rows,
        page_size=page_size,
    )
    dequantized = dequantize(
        raw,
        physical_rows,
        page_size=page_size,
    ).squeeze(1)
    cache_4d = raw.view(num_pages, page_size, 1, MXFP4_BYTES_PER_TOKEN)
    return cache_4d, dequantized


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
    kv, dequantized_kv = _make_cache(
        page_size=_SWA_PAGE_SIZE,
        minimum_tokens=batch_size * _SWA_TOPK,
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
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        dequantized_kv=dequantized_kv,
        sm_scale=1.0 / math.sqrt(_HEAD_DIM),
    )

    if extra_page_size is not None:
        extra_kv, dequantized_extra = _make_cache(
            page_size=extra_page_size,
            minimum_tokens=batch_size * extra_topk,
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
        indices=case.indices,
        topk_length=case.topk_length,
        attn_sink=case.attn_sink,
        tile_scheduler_metadata=sched_meta,
        head_dim_v=_HEAD_DIM,
        softmax_scale=case.sm_scale,
        extra_k_cache=case.extra_kv,
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
def test_flashmla_dsv4_mxfp4_dual_source_correctness(
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
def test_flashmla_dsv4_mxfp4_caller_accum_workspace_bounds_vram() -> None:
    """A caller-supplied split-K workspace must replace the module-global
    per-geometry scratch.

    The global cache pinned one (batch + sm parts)-row FP32 accumulator pair
    per geometry forever: a stock CUDA-graph capture sweep (one entry per
    captured batch size) left ~0.5 GiB resident at h_q=64 (~0.85 GiB at
    h_q=128) per runner, and two runners sharing a geometry would race on
    the same tensors.  A runner-sized arena prefix-sliced per batch pins one
    allocation instead; every batch size only touches accumulator rows
    [0, batch + parts), so all captured graphs share the arena's base
    address and replays stay correct.
    """
    native = _require_native()
    assert FlashMLASchedMeta is not None
    from sglang.kernels.ops.attention.dsv4.mxfp4_dsv4_decode_sm90 import (
        _SCRATCH,
        _num_sm_parts,
    )

    h_q = 64
    num_sm_parts = _num_sm_parts(1, 1, h_q, torch.device("cuda"))
    capture_bs = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    rows = capture_bs[-1] + num_sm_parts
    lse_arena = torch.empty((rows, 1, h_q), dtype=torch.float32, device="cuda")
    o_arena = torch.empty((rows, 1, h_q, _HEAD_DIM), dtype=torch.float32, device="cuda")

    # Lean sweep fixtures: one max-batch cache/query sliced per batch (no
    # dequantized reference — that FP32 copy would dwarf the scratch being
    # measured).
    quantize, _ = _require_codec()
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(20260818)
    max_capacity = capture_bs[-1] * _SWA_TOPK
    raw = torch.zeros(
        (max_capacity // _SWA_PAGE_SIZE, _SWA_PAGE_SIZE * MXFP4_BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device=device,
    )
    src = (
        torch.randn(
            (max_capacity, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        / 10
    )
    quantize(
        cache_k=src,
        kv_buffer=raw,
        loc=torch.arange(max_capacity, dtype=torch.int32, device=device),
        page_size=_SWA_PAGE_SIZE,
    )
    del src
    kv = raw.view(-1, _SWA_PAGE_SIZE, 1, MXFP4_BYTES_PER_TOKEN)
    q_max = (
        torch.randn(
            (capture_bs[-1], 1, h_q, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        / 10
    )
    indices_max = torch.arange(max_capacity, dtype=torch.int32, device=device).view(
        capture_bs[-1], 1, _SWA_TOPK
    )

    _SCRATCH.clear()
    for b in capture_bs:
        out, lse = native(
            q=q_max[:b],
            k_cache=kv,
            indices=indices_max[:b],
            topk_length=None,
            attn_sink=None,
            tile_scheduler_metadata=FlashMLASchedMeta(),
            head_dim_v=_HEAD_DIM,
            split_accum_buffers=(lse_arena, o_arena),
        )
        assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(lse).all())

    # The caller-owned arena fully replaces the module-global scratch: the
    # sweep over 15 captured batch sizes leaves it empty.  The default path
    # (no buffers — standalone callers) pins one accumulator pair per
    # geometry instead, which is exactly the unbounded accumulation the
    # arena exists to avoid (~440 MiB of FP32 across this sweep at h_q=64).
    assert not _SCRATCH
    for b in (2, 8, 32):
        native(
            q=q_max[:b],
            k_cache=kv,
            indices=indices_max[:b],
            topk_length=None,
            attn_sink=None,
            tile_scheduler_metadata=FlashMLASchedMeta(),
            head_dim_v=_HEAD_DIM,
        )
    assert len(_SCRATCH) == 3
    _SCRATCH.clear()

    # Same arena, second sweep: results identical to the global-scratch path
    # (the prefix slicing must not change numerics), and the arena tensors
    # are reused, not reallocated.
    out_ref, _ = _run_native(
        native, _build_case(h_q=h_q, batch_size=8), FlashMLASchedMeta()
    )
    case8 = _build_case(h_q=h_q, batch_size=8)
    out_arena, _ = native(
        q=case8.q,
        k_cache=case8.kv,
        indices=case8.indices,
        topk_length=case8.topk_length,
        attn_sink=case8.attn_sink,
        tile_scheduler_metadata=FlashMLASchedMeta(),
        head_dim_v=_HEAD_DIM,
        softmax_scale=case8.sm_scale,
        split_accum_buffers=(lse_arena, o_arena),
    )
    torch.testing.assert_close(out_arena, out_ref, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL)

    # An undersized or wrong-shaped arena must fail loudly, never truncate
    # silently.
    with pytest.raises(ValueError, match="rows"):
        native(
            q=case8.q,
            k_cache=case8.kv,
            indices=case8.indices,
            topk_length=case8.topk_length,
            attn_sink=case8.attn_sink,
            tile_scheduler_metadata=FlashMLASchedMeta(),
            split_accum_buffers=(lse_arena[:4], o_arena[:4]),
        )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_mxfp4_multirequest_parts_no_deadlock() -> None:
    """Batches larger than the SM-part count used to deadlock the GPU.

    The tile scheduler caps each SM part at a fixed block payload, so a part
    can serve more than one request. The producer's epilogue-drain barrier
    then waited on the wrong phase parity (one phase past the one the
    consumers had completed) and never released, spinning the persistent
    kernel forever. b = num_sm_parts + 1 guarantees every active part serves
    two requests; 2 * num_sm_parts exactly fills the per-part capacity.
    """
    native = _require_native()
    assert FlashMLASchedMeta is not None
    from sglang.kernels.ops.attention.dsv4.mxfp4_dsv4_decode_sm90 import _num_sm_parts

    num_sm_parts = _num_sm_parts(1, 1, 64, torch.device("cuda"))
    for batch_size in (num_sm_parts + 1, 2 * num_sm_parts):
        case = _build_case(h_q=64, batch_size=batch_size)
        out, lse = _run_native(native, case, FlashMLASchedMeta())
        out_ref, lse_ref = _reference(case)

        assert out.shape == out_ref.shape == (batch_size, 1, 64, _HEAD_DIM)
        assert bool(torch.isfinite(out).all())
        assert bool(torch.isfinite(lse).all())
        torch.testing.assert_close(out, out_ref, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL)
        torch.testing.assert_close(lse, lse_ref, atol=_LSE_ATOL, rtol=_LSE_RTOL)


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_mxfp4_rejects_invalid_contracts() -> None:
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _build_case(h_q=64, batch_size=1)

    with pytest.raises(RuntimeError, match="q must have dtype bfloat16"):
        native(
            case.q.float(),
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    with pytest.raises(RuntimeError, match="multiple of 64"):
        native(
            case.q,
            case.kv,
            case.indices[..., :-1].contiguous(),
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    bad_cache = torch.empty(
        (1, _SWA_PAGE_SIZE, 1, MXFP4_BYTES_PER_TOKEN - 1),
        dtype=torch.uint8,
        device=case.q.device,
    )
    with pytest.raises((RuntimeError, ValueError), match="368"):
        native(
            case.q,
            bad_cache,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    cache_numel = case.kv.numel()
    misaligned_storage = torch.empty(
        cache_numel + 16, dtype=torch.uint8, device=case.q.device
    )
    # Offsets 1 and 4 cover both a fully misaligned base and one that only
    # satisfies the old 4-byte contract (4 % 4 == 0 but 4 % 16 != 0 — the
    # kernel's 128-bit vector loads would fault).
    for misaligned_by in (1, 4):
        misaligned_cache = misaligned_storage[
            misaligned_by : misaligned_by + cache_numel
        ].view_as(case.kv)
        assert misaligned_cache.data_ptr() % 16 == misaligned_by
        with pytest.raises(RuntimeError, match="16-byte aligned"):
            native(
                case.q,
                misaligned_cache,
                case.indices,
                case.topk_length,
                case.attn_sink,
                FlashMLASchedMeta(),
            )

    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        native(
            case.q,
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
            extra_k_cache=case.kv,
        )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_mxfp4_dispatch_raises_instead_of_exiting() -> None:
    """The C++ entry point runs inside a serving process.

    Its contract violations used to fprintf + exit(1), killing the whole
    server on a bad input; they must raise a Python exception instead. The
    public wrapper rejects bad inputs first, so this drives the raw FFI
    dispatch with an invalid head_dim_v — the entry's first check.
    """
    from sglang.kernels.ops.attention.dsv4.mxfp4_dsv4_decode_sm90 import _get_dispatch

    device = torch.device("cuda")
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    q = torch.empty((1, 1, 64, 512), dtype=torch.bfloat16, device=device)
    with pytest.raises((ValueError, RuntimeError), match="512"):
        _get_dispatch()(
            q,
            q,  # k_cache placeholder; the head_dim_v check fires first
            empty_i32,  # indices
            empty_i32,  # topk_length
            empty_f32,  # attn_sink
            empty_i32,  # tile_scheduler_metadata
            empty_i32,  # num_splits
            empty_i32,  # extra_k_cache
            empty_i32,  # extra_indices
            empty_i32,  # extra_topk_length
            empty_f32,  # lse_accum
            empty_f32,  # o_accum
            empty_f32,  # out
            empty_f32,  # lse
            256,  # head_dim_v
            1.0,  # sm_scale
            0,  # generate_sched_meta
        )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_mxfp4_rejects_unsafe_kernel_contracts() -> None:
    """Guard the wrapper contract for values the SM90 kernel hardcodes.

    The kernel fixes head dim 512, 64/128 query heads, int32 index and length
    tensors, [B, S_Q] index geometry matching q, same-device placement, and
    contiguous layouts. Before these checks, a 256-wide bf16 q or an int64
    indices tensor passed validation and the kernel read memory with
    hardcoded 512-element int32 strides — an out-of-bounds or reinterpreted
    read instead of an exception.
    """
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _build_case(h_q=64, batch_size=1)

    # Head dims: a 256-wide q and a non-512 head_dim_v both sailed through
    # every earlier check while the kernel addresses q with 512 strides.
    narrow_q = case.q[..., :256].contiguous()
    with pytest.raises(ValueError, match="512"):
        native(
            narrow_q,
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )
    with pytest.raises(ValueError, match="512"):
        native(
            case.q,
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
            head_dim_v=256,
        )

    wide_q = case.q.repeat(1, 1, 3, 1)  # 192 heads
    with pytest.raises(ValueError, match="64 or 128"):
        native(
            wide_q,
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    # Index tensors: int64 reinterpreted as int32 by the kernel.
    with pytest.raises(ValueError, match="int32"):
        native(
            case.q,
            case.kv,
            case.indices.long(),
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )
    with pytest.raises(ValueError, match="int32"):
        native(
            case.q,
            case.kv,
            case.indices,
            case.topk_length.long(),
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    # Length-vector geometry: the scheduler kernel reads lengths[request] for
    # every request in the batch.
    with pytest.raises(ValueError, match="topk_length"):
        native(
            case.q,
            case.kv,
            case.indices,
            case.topk_length.repeat(2),
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    # Batch/sequence geometry mismatch between q and indices.
    with pytest.raises(ValueError, match="indices"):
        native(
            case.q,
            case.kv,
            case.indices.repeat(2, 1, 1),
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    # A strided q view: shape and dtype pass, but the kernel addresses q with
    # hardcoded contiguous strides.
    strided_q = torch.empty(
        (1, 1, 64, 1024), dtype=torch.bfloat16, device=case.q.device
    )[..., ::2]
    assert not strided_q.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        native(
            strided_q,
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    # Cross-device tensors: only q's device was ever checked before.
    cpu_cache = torch.empty_like(case.kv, device="cpu")
    with pytest.raises(ValueError, match="query device"):
        native(
            case.q,
            cpu_cache,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
        )

    # Extra source: an unaligned extra width and int64 extra indices.
    with pytest.raises(RuntimeError, match="multiple of 64"):
        native(
            case.q,
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
            extra_k_cache=case.kv,
            extra_indices_in_kvcache=case.indices[..., :32].contiguous(),
            extra_topk_length=case.topk_length,
        )
    with pytest.raises(ValueError, match="int32"):
        native(
            case.q,
            case.kv,
            case.indices,
            case.topk_length,
            case.attn_sink,
            FlashMLASchedMeta(),
            extra_k_cache=case.kv,
            extra_indices_in_kvcache=case.indices.long(),
            extra_topk_length=case.topk_length,
        )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_mxfp4_clamps_graph_replayed_lengths() -> None:
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


def _widen_primary_indices(case: _DecodeCase) -> _DecodeCase:
    """Double the primary top-k width (128 -> 256) so lengths can cross a
    second 64-token block boundary while reusing one FlashMLASchedMeta."""
    capacity = case.kv.shape[0] * case.kv.shape[1]
    wide = torch.cat(
        [case.indices, (case.indices + capacity) % capacity], dim=-1
    ).contiguous()
    case.indices = wide
    return case


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_mxfp4_eager_refreshes_stale_scheduler() -> None:
    """Eager (non-graph) calls with one FlashMLASchedMeta re-run the scheduler.

    The split assignment depends on the per-call top-k lengths; a growing
    sequence crossing a 64-token block boundary must not reuse the previous
    call's assignment (regression: the scheduler only ran on a fresh buffer,
    so the newly added KV block was silently dropped).
    """
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _widen_primary_indices(_build_case(h_q=64, batch_size=2))
    assert case.indices.shape[-1] == 256
    device = case.q.device

    meta = FlashMLASchedMeta()
    case.topk_length.copy_(torch.tensor([128, 96], dtype=torch.int32, device=device))
    out1, _ = _run_native(native, case, meta)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out1, _reference(case)[0], atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL
    )

    # Same scheduler instance, lengths now spanning three 64-token blocks.
    case.topk_length.copy_(torch.tensor([192, 160], dtype=torch.int32, device=device))
    out2, _ = _run_native(native, case, meta)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out2, _reference(case)[0], atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL
    )


@pytest.mark.skipif(not _is_sm90_supported(), reason="SM90 and CUDA >= 12.5 required")
@torch.inference_mode()
def test_flashmla_dsv4_mxfp4_fresh_meta_per_capture() -> None:
    """A second CUDA-graph capture with the same geometry gets a fresh
    scheduler (regression: the backend's capture dict reused the first
    graph's instance, so the second graph never recorded the scheduler
    kernel and replayed with a stale split assignment)."""
    native = _require_native()
    assert FlashMLASchedMeta is not None
    case = _widen_primary_indices(_build_case(h_q=64, batch_size=2))
    device = case.q.device

    # Warm lazy extension state (as the backend does before capture).
    _run_native(native, case, FlashMLASchedMeta())
    torch.cuda.synchronize()

    graph1 = torch.cuda.CUDAGraph()
    meta1 = FlashMLASchedMeta()
    case.topk_length.copy_(torch.tensor([128, 96], dtype=torch.int32, device=device))
    with torch.cuda.graph(graph1):
        _run_native(native, case, meta1)

    # The backend clears its capture dict at the start of a new capture
    # session, so the second graph records its own scheduler generation.
    graph2 = torch.cuda.CUDAGraph()
    meta2 = FlashMLASchedMeta()
    case.topk_length.copy_(torch.tensor([128, 96], dtype=torch.int32, device=device))
    with torch.cuda.graph(graph2):
        graph2_out, graph2_lse = _run_native(native, case, meta2)

    # Replay the second graph with lengths crossing the 64-block boundary.
    case.topk_length.copy_(torch.tensor([192, 160], dtype=torch.int32, device=device))
    graph2.replay()
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(case)
    torch.testing.assert_close(
        graph2_out, out_ref, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL
    )
    torch.testing.assert_close(graph2_lse, lse_ref, atol=_LSE_ATOL, rtol=_LSE_RTOL)


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
def test_flashmla_dsv4_mxfp4_cuda_graph_replay(
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
        indices=case.indices.clone(),
        topk_length=case.topk_length.clone(),
        attn_sink=case.attn_sink,
        dequantized_kv=case.dequantized_kv,
        sm_scale=case.sm_scale,
        extra_kv=case.extra_kv,
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
