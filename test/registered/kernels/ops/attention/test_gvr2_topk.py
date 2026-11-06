"""Correctness tests for the FlashInfer GVR V2 DSA decode top-k + page transform.

``gvr2_topk_transform_paged`` must match ``torch.topk`` over
``scores[r, :seq_lens[r]]`` mapped through the page table, up to interchangeable
ties, with ``-1`` padding for rows shorter than ``k``; also under CUDA graph
replay with changed lengths and scores.
``gvr2_topk`` (unfused) must return the same selection as row-local positions.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsa.gvr2_topk import (
    gvr2_fused_page_table_available,
    gvr2_supports,
    gvr2_topk,
    gvr2_topk_transform_paged,
)
from sglang.kernels.ops.attention.dsa.paged_mqa_logits import _restore_storage_extent
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

PAGE_SIZE = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="GVR V2 needs SM 100/103/107",
)


def _make_inputs(batch, width, seq_lens, row_stride=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    row_stride = row_stride or width
    arena = torch.randn(batch, row_stride, device="cuda", generator=g)
    scores = arena[:, :width]
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.stack(
        [
            torch.randperm(num_pages * 4, device="cuda", generator=g)[:num_pages]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return scores, lengths, page_table


def _slots_to_positions(slots, page_table):
    """Invert the page transform: physical slot -> row-local position."""
    positions = []
    pt = page_table.cpu().tolist()
    for r, row in enumerate(slots.cpu().tolist()):
        page_to_idx = {p: i for i, p in enumerate(pt[r])}
        positions.append(
            [
                page_to_idx[s // PAGE_SIZE] * PAGE_SIZE + s % PAGE_SIZE
                for s in row
                if s >= 0
            ]
        )
    return positions


def _assert_matches_reference(scores, lengths, page_table, k, out):
    assert out.shape == (scores.shape[0], k) and out.dtype == torch.int32
    scores_cpu = scores.float().cpu()
    for r, pos in enumerate(_slots_to_positions(out, page_table)):
        L = int(lengths[r])
        assert len(pos) == min(k, L), f"row {r}: {len(pos)} valid != {min(k, L)}"
        assert (out[r, len(pos) :] == -1).all() or len(pos) == k
        assert len(set(pos)) == len(pos) and all(0 <= p < L for p in pos)
        if L == 0:
            continue
        ref = torch.topk(scores_cpu[r, :L], min(k, L)).values
        got = scores_cpu[r, pos].sort(descending=True).values
        assert torch.equal(got, ref), f"row {r}: selected scores differ"


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "batch,width",
    [(1, 4096), (8, 8192), (32, 16384), (4, 65536), (64, 131072), (200, 32768)],
)
def test_gvr2_topk_paged(batch, width, k):
    g = torch.Generator().manual_seed(batch * 7 + k)
    seq_lens = torch.randint(1, width + 1, (batch,), generator=g).tolist()
    seq_lens[0] = width
    if batch > 2:
        seq_lens[1] = min(k, width)  # exactly k: identity
        seq_lens[2] = k // 3  # shorter than k: -1 tail
    scores, lengths, page_table = _make_inputs(batch, width, seq_lens)
    assert gvr2_supports(scores, k)
    out = gvr2_topk_transform_paged(scores, lengths, k, page_table, PAGE_SIZE)
    _assert_matches_reference(scores, lengths, page_table, k, out)


def test_gvr2_topk_paged_noncontiguous_page_table():
    batch, width, k = 8, 32768, 2048
    seq_lens = [width, 20000, k, 7, width, 9000, 0, 30000]
    scores, lengths, page_table = _make_inputs(batch, width, seq_lens)
    # A column slice of a wider table runs the separate Triton lookup.
    wide = torch.zeros(batch, page_table.shape[1] + 8, dtype=torch.int32, device="cuda")
    wide[:, : page_table.shape[1]] = page_table
    sliced = wide[:, : page_table.shape[1]]
    assert not sliced.is_contiguous()
    out = gvr2_topk_transform_paged(scores, lengths, k, sliced, PAGE_SIZE)
    _assert_matches_reference(scores, lengths, page_table, k, out)


@pytest.mark.skipif(
    not gvr2_fused_page_table_available(),
    reason="FlashInfer gvr_2 without page_table=",
)
def test_gvr2_topk_paged_fused_matches_lookup():
    batch, width, k = 12, 65536, 2048
    g = torch.Generator().manual_seed(5)
    seq_lens = torch.randint(0, width + 1, (batch,), generator=g).tolist()
    scores, lengths, page_table = _make_inputs(
        batch, width, seq_lens, row_stride=width + 64
    )
    fused = gvr2_topk_transform_paged(scores, lengths, k, page_table, PAGE_SIZE)
    wide = torch.zeros(batch, page_table.shape[1] + 8, dtype=torch.int32, device="cuda")
    wide[:, : page_table.shape[1]] = page_table
    lookup = gvr2_topk_transform_paged(
        scores, lengths, k, wide[:, : page_table.shape[1]], PAGE_SIZE
    )
    _assert_matches_reference(scores, lengths, page_table, k, fused)
    assert torch.equal(fused.sort(dim=1).values, lookup.sort(dim=1).values)


def _dlpack_padded_logits(rows, width, aligned):
    """DeepGEMM's tvm-ffi return: ``empty(rows, aligned)[:, :width]`` via DLPack."""
    base = torch.randn(rows, aligned, device="cuda")
    view = torch.from_dlpack(torch.utils.dlpack.to_dlpack(base[:, :width]))
    return base, view


@pytest.mark.parametrize("rows", [2, 9, 36])
def test_gvr2_topk_dlpack_padded_logits(rows):
    # DSPARK verify: a 16385-page table, not a multiple of the 256-column pitch.
    pages, k = 16385, 2048
    width = pages * PAGE_SIZE
    aligned = (width + 255) // 256 * 256
    base, view = _dlpack_padded_logits(rows, width, aligned)
    logits = _restore_storage_extent(view)
    assert logits.data_ptr() == base.data_ptr() and logits.stride() == (aligned, 1)
    assert logits.untyped_storage().size() // 4 >= rows * aligned
    del view
    g = torch.Generator(device="cuda").manual_seed(rows)
    lengths = torch.randint(
        k + 1, width + 1, (rows,), dtype=torch.int32, device="cuda", generator=g
    )
    lengths[-1] = width
    page_table = torch.stack(
        [
            torch.randperm(pages, device="cuda", dtype=torch.int32, generator=g)
            for _ in range(rows)
        ]
    )
    out = gvr2_topk_transform_paged(logits, lengths, k, page_table, PAGE_SIZE)
    _assert_matches_reference(logits, lengths, page_table, k, out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gvr2_topk_transform_paged(logits, lengths, k, page_table, PAGE_SIZE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = gvr2_topk_transform_paged(logits, lengths, k, page_table, PAGE_SIZE)
    torch.cuda.current_stream().wait_stream(stream)
    base.copy_(torch.randn_like(base))
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(logits, lengths, page_table, k, out)


def test_restore_storage_extent_passthrough():
    full = torch.randn(4, 8448, device="cuda")
    for t in (full, full[:, :8192], full[:1, :8000]):
        assert _restore_storage_extent(t) is t


def _identity_page_table(batch, width):
    num_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    row = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    return row.expand(batch, num_pages).contiguous()


@pytest.mark.parametrize("k", [512, 2048])
def test_gvr2_topk_unfused_positions(k):
    batch, width = 8, 65536
    seq_lens = [width, 0, 40000, k, k // 2, 1, 30000, width]
    scores, lengths, _ = _make_inputs(batch, width, seq_lens, row_stride=width + 64)
    out = gvr2_topk(scores, lengths, k)
    _assert_matches_reference(
        scores, lengths, _identity_page_table(batch, width), k, out
    )


def test_gvr2_topk_padded_rows_and_inf():
    batch, width, k = 6, 8192, 2048
    seq_lens = [width, 0, 5000, 3000, 1, width]
    scores, lengths, page_table = _make_inputs(batch, width, seq_lens, row_stride=8448)
    assert scores.stride(0) == 8448 and gvr2_supports(scores, k)
    # Init/local tokens forced in with +inf (DSAIndexer._mask_init_and_local_tokens).
    scores[:, :4] = float("inf")
    scores[4, 0] = float("inf")
    out = gvr2_topk_transform_paged(scores, lengths, k, page_table, PAGE_SIZE)
    assert (out[1] == -1).all()
    _assert_matches_reference(scores, lengths, page_table, k, out)
    for r in (0, 2, 3, 5):
        assert set(range(4)) <= set(
            _slots_to_positions(out[r : r + 1], page_table[r : r + 1])[0]
        )


def test_gvr2_topk_rejects_unsupported():
    scores = torch.randn(4, 8192, device="cuda")
    assert not gvr2_supports(scores, 256)
    assert not gvr2_supports(scores.bfloat16(), 2048)
    assert not gvr2_supports(scores[:, 1:], 2048)


@pytest.mark.parametrize("k", [2048])
def test_gvr2_topk_cuda_graph_replay(k):
    batch, width = 16, 32768
    seq_lens = [width] * batch
    scores, lengths, page_table = _make_inputs(batch, width, seq_lens)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            gvr2_topk_transform_paged(scores, lengths, k, page_table, PAGE_SIZE)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = gvr2_topk_transform_paged(scores, lengths, k, page_table, PAGE_SIZE)
    torch.cuda.current_stream().wait_stream(stream)

    g = torch.Generator().manual_seed(1)
    for step in range(3):
        new_lens = torch.randint(0, width + 1, (batch,), generator=g)
        new_lens[0] = k - 1
        lengths.copy_(new_lens.to(torch.int32))
        scores.copy_(torch.randn_like(scores))
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(scores, lengths, page_table, k, out)


def test_gvr2_backend_predicates():
    gvr2, sgl = DSATopKBackend.FLASHINFER_GVR2, DSATopKBackend.SGL_KERNEL
    assert gvr2.is_gvr2() and not gvr2.is_sgl_kernel()
    assert sgl.is_sgl_kernel() and not sgl.is_gvr2()
    assert gvr2.fallback() is sgl
    for backend in (sgl, DSATopKBackend.TORCH, DSATopKBackend.FLASHINFER):
        assert backend.fallback() is backend
    assert gvr2.should_use_topk_v2() == sgl.should_use_topk_v2()


def _spy_on_fallback(monkeypatch, name):
    original = getattr(DSATopKBackend, name)
    delegated = []

    def spy(self, *args, **kwargs):
        if self.is_sgl_kernel():
            delegated.append(kwargs)
            return "sgl-kernel"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DSATopKBackend, name, spy)
    return delegated


@pytest.mark.parametrize(
    "fused,case",
    [
        (True, "ragged"),
        (True, "row_starts"),
        (True, "batch_idx_list"),
        (True, "token_rows"),
        (False, "row_starts"),
    ],
)
def test_gvr2_backend_prefill_uses_fallback(monkeypatch, fused, case):
    delegated = _spy_on_fallback(
        monkeypatch, "topk_transform" if fused else "topk_func"
    )
    batch, width = 4, 8192
    scores, seq_lens, page_table = _make_inputs(batch, width, [width] * batch)
    method, kwargs = TopkTransformMethod.PAGED, {}
    if case == "ragged":
        method = TopkTransformMethod.RAGGED
        kwargs["topk_indices_offset"] = torch.zeros_like(seq_lens)
    elif case == "row_starts":
        kwargs["row_starts"] = torch.zeros_like(seq_lens)
    elif case == "batch_idx_list":
        kwargs["batch_idx_list"] = [0] * batch
    else:
        # One request's tokens: one logits row per token, one page-table row.
        page_table = page_table[:1]
    out = DSATopKBackend.FLASHINFER_GVR2.topk_transform(
        scores,
        seq_lens,
        2048,
        method,
        SimpleNamespace(real_page_table=page_table, page_size=PAGE_SIZE),
        force_unfused_topk=not fused,
        **kwargs,
    )
    assert out == "sgl-kernel"
    assert len(delegated) == 1
    for name, value in kwargs.items():
        assert delegated[0][name] is value


@pytest.mark.parametrize("fused", [True, False])
def test_gvr2_backend_decode_runs_gvr2(fused):
    batch, width, k = 4, 8192, 2048
    scores, seq_lens, page_table = _make_inputs(batch, width, [width, 5000, k, 7])
    # The unfused contract has no page table: GVR V2 must not read it.
    metadata = SimpleNamespace(
        real_page_table=page_table if fused else None, page_size=PAGE_SIZE
    )
    out = DSATopKBackend.FLASHINFER_GVR2.topk_transform(
        scores,
        seq_lens,
        k,
        TopkTransformMethod.PAGED,
        metadata,
        force_unfused_topk=not fused,
    )
    expected_table = page_table if fused else _identity_page_table(batch, width)
    _assert_matches_reference(scores, seq_lens, expected_table, k, out)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("case", ["unsupported_k", "bf16"])
def test_gvr2_backend_decode_rejects_unsupported(fused, case):
    batch, width = 4, 8192
    scores, seq_lens, page_table = _make_inputs(batch, width, [width] * batch)
    k = 2048
    if case == "unsupported_k":
        k = 256
    else:
        scores = scores.bfloat16()
    with pytest.raises(RuntimeError, match="flashinfer_gvr2"):
        DSATopKBackend.FLASHINFER_GVR2.topk_transform(
            scores,
            seq_lens,
            k,
            TopkTransformMethod.PAGED,
            SimpleNamespace(real_page_table=page_table, page_size=PAGE_SIZE),
            force_unfused_topk=not fused,
        )


def test_gvr2_backend_check_config():
    gvr2 = DSATopKBackend.FLASHINFER_GVR2
    gvr2.check_config("cuda", 2048)
    with pytest.raises(ValueError, match="top-k"):
        gvr2.check_config("cuda", 256)
    with pytest.raises(ValueError, match="CUDA"):
        gvr2.check_config("cpu", 2048)
    for backend in (DSATopKBackend.SGL_KERNEL, DSATopKBackend.FLASHINFER):
        backend.check_config("cuda", 256)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
