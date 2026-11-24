"""Correctness, determinism, and warmup tests for FlashTopK prefill top-k
(standard DSA `Indexer` and KPool DSA `IndexerKPool`) against DeepGEMM
`fp8_mqa_logits` + exact reference top-k / SGLang `topk_from_pooled_history_logits`.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest
import torch

from sglang.kernels.ops.attention.dsa.flashtopk_prefill import (
    _WARMUP_KEYS,
    _WARMUP_ROWS,
    FLASHTOPK_ADAPTIVE_MIN_KEYS,
    FLASHTOPK_HEAD_DIM,
    FLASHTOPK_HEADS,
    _get_selector,
    flashtopk_fp8_indexer_topk,
    flashtopk_kpool_indexer_topk,
    is_flashtopk_supported,
    warmup_flashtopk_prefill,
)
from sglang.kernels.ops.attention.topk_page_table import topk_page_table_transform_
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    topk_from_pooled_history_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

requires_flashtopk = pytest.mark.skipif(
    not is_flashtopk_supported(), reason="Requires CUDA SM90 or SM100"
)


def _make_ragged(
    requests: Sequence[Tuple[int, int, str]], device="cuda"
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build (ks, ke, total_keys) for packed requests: (ctx, rows, where)."""
    ks: List[int] = []
    ke: List[int] = []
    cum = 0
    for ctx, rows, where in requests:
        assert rows <= ctx
        first = ctx - rows if where == "last" else max(0, ctx // 2 - rows // 2)
        for p in range(first, first + rows):
            ks.append(cum)
            ke.append(cum + p + 1)
        cum += ctx
    return (
        torch.tensor(ks, dtype=torch.int32, device=device),
        torch.tensor(ke, dtype=torch.int32, device=device),
        cum,
    )


def _gen_inputs(rows: int, keys: int, seed: int = 0, device="cuda"):
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        rows,
        FLASHTOPK_HEADS,
        FLASHTOPK_HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=g,
    )
    kv = torch.randn(
        keys, FLASHTOPK_HEAD_DIM, device=device, dtype=torch.bfloat16, generator=g
    )
    weights = (
        torch.rand(
            rows, FLASHTOPK_HEADS, device=device, dtype=torch.float32, generator=g
        )
        + 0.1
    )
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scale = per_custom_dims_cast_to_fp8(kv, (0,), False)
    return q_fp8, weights, kv_fp8.contiguous(), kv_scale.contiguous()


def _ref_logits(q_fp8, weights, kv_fp8, kv_scale, ks, ke) -> torch.Tensor:
    import deep_gemm

    return deep_gemm.fp8_mqa_logits(
        q_fp8, (kv_fp8, kv_scale), weights, ks, ke, clean_logits=True
    )[:, : kv_fp8.shape[0]]


def _check_fp8_topk_against_ref(
    out: torch.Tensor,
    q: torch.Tensor,
    w: torch.Tensor,
    kv: torch.Tensor,
    sf: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    out_offsets: torch.Tensor | None = None,
    row_chunk: int = 256,
) -> None:
    rows = ks.numel()
    lens = (ke - ks).clamp(min=0).long()
    offsets = (
        torch.zeros_like(ks, dtype=torch.long)
        if out_offsets is None
        else out_offsets.long()
    )
    for r0 in range(0, rows, row_chunk):
        r1 = min(r0 + row_chunk, rows)
        logits = _ref_logits(q[r0:r1], w[r0:r1], kv, sf, ks[r0:r1], ke[r0:r1])
        o = out[r0:r1].long()
        n = lens[r0:r1].clamp(max=topk)
        valid = torch.arange(topk, device=out.device)[None, :] < n[:, None]
        assert (o[~valid] == -1).all(), "padding past min(topk, len) is not -1"

        # Recover absolute key index j = o - offsets + ks.
        abs_idx = o - offsets[r0:r1, None] + ks[r0:r1, None].long()
        inside = (abs_idx >= ks[r0:r1, None]) & (abs_idx < ke[r0:r1, None])
        assert (inside | ~valid).all(), "selected key outside [ks, ke)"

        srt = (
            torch.where(valid, abs_idx, torch.full_like(abs_idx, -1)).sort(dim=1).values
        )
        assert not ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any(), (
            "duplicate selection"
        )

        got = torch.where(
            valid,
            logits.gather(1, abs_idx.clamp(min=0)),
            torch.full_like(logits[:, :1], float("inf")),
        )
        k_eff = min(topk, logits.shape[1])
        if k_eff > 0:
            top_vals = logits.topk(k_eff, dim=1).values
            kth = top_vals.gather(1, (n - 1).clamp(min=0)[:, None])
            tol = 1e-4 * logits.masked_fill(~torch.isfinite(logits), 0).abs().amax(
                1, keepdim=True
            ).clamp(min=1.0)
            assert (got >= kth - tol).all(), "selected key below k-th logit"
            above = (logits > kth + tol).sum(1)
            assert torch.equal(((got > kth + tol) & valid).sum(1), above), (
                "key above k-th logit missed"
            )
        del logits


@requires_flashtopk
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_flashtopk_fp8_indexer_ragged(topk: int, with_offsets: bool):
    """Exactness and determinism on mixed ragged requests (short + long rows)."""
    ks, ke, keys = _make_ragged(
        [(256, 64, "last"), (4096, 128, "last"), (32768, 256, "mid")]
    )
    rows = ks.numel()
    q, w, kv, sf = _gen_inputs(rows, keys, seed=topk + int(with_offsets))
    # Use arbitrary positive offsets distinct from ks so -1 + (offset - ks) > 0 is exercised!
    out_offsets = (ks + 777) if with_offsets else None

    out1 = flashtopk_fp8_indexer_topk(
        q, w, kv, sf, ks, ke, topk, out_offsets=out_offsets
    )
    out2 = flashtopk_fp8_indexer_topk(
        q, w, kv, sf, ks, ke, topk, out_offsets=out_offsets
    )
    torch.cuda.synchronize()
    assert torch.equal(out1, out2), "FlashTopK must be bit-deterministic"
    _check_fp8_topk_against_ref(out1, q, w, kv, sf, ks, ke, topk, out_offsets)


@requires_flashtopk
def test_flashtopk_fp8_indexer_adaptive_pruning_128k():
    """Exercise >= 128K keys so FlashTopK adaptive key pruning executes end-to-end."""
    keys = FLASHTOPK_ADAPTIVE_MIN_KEYS + 2048
    rows = 512
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = torch.arange(keys - rows + 1, keys + 1, dtype=torch.int32, device="cuda")
    q, w, kv, sf = _gen_inputs(rows, keys, seed=17)
    out = flashtopk_fp8_indexer_topk(q, w, kv, sf, ks, ke, 2048, out_offsets=ks)
    torch.cuda.synchronize()
    _check_fp8_topk_against_ref(out, q, w, kv, sf, ks, ke, 2048, ks)


@requires_flashtopk
def test_flashtopk_fp8_indexer_paged_transform():
    """Verify FlashTopK + topk_page_table_transform_ matches dense PAGED slots."""
    requests = [(1024, 64, "last"), (8192, 128, "last")]
    ks, ke, keys = _make_ragged(requests)
    rows = ks.numel()
    topk = 512
    page_size = 64
    q, w, kv, sf = _gen_inputs(rows, keys, seed=42)

    # Synthetic physical page table (permuted page IDs).
    max_pages = 8192 // page_size
    perm = torch.randperm(max_pages * 3, dtype=torch.int32, device="cuda")
    page_table_64 = torch.stack(
        [perm[:max_pages], perm[max_pages : 2 * max_pages]], dim=0
    ).contiguous()
    row_to_batch = torch.cat(
        [
            torch.zeros(64, dtype=torch.int32, device="cuda"),
            torch.ones(128, dtype=torch.int32, device="cuda"),
        ]
    )

    out_rel = flashtopk_fp8_indexer_topk(q, w, kv, sf, ks, ke, topk, out_offsets=None)
    out_paged = out_rel.clone()
    topk_page_table_transform_(out_paged, page_table_64, row_to_batch, page_size)
    torch.cuda.synchronize()

    # Compare every valid slot against manual page_table lookup on out_rel.
    valid = out_rel >= 0
    expected = torch.where(
        valid,
        page_table_64[
            row_to_batch[:, None].long(), (out_rel.clamp(min=0) // page_size).long()
        ]
        * page_size
        + (out_rel.clamp(min=0) % page_size),
        torch.full_like(out_rel, -1),
    )
    assert torch.equal(out_paged, expected)
    _check_fp8_topk_against_ref(out_rel, q, w, kv, sf, ks, ke, topk, None)


@requires_flashtopk
def test_flashtopk_fp8_indexer_edge_cases():
    """Empty rows, empty keys, and strided output buffer."""
    q0, w0, kv0, sf0 = _gen_inputs(0, 1024, seed=1)
    ks0 = torch.empty(0, dtype=torch.int32, device="cuda")
    out0 = flashtopk_fp8_indexer_topk(q0, w0, kv0, sf0, ks0, ks0, 512)
    assert out0.shape == (0, 512)

    q, w, _, _ = _gen_inputs(16, 128, seed=2)
    kv_empty = torch.empty(
        0, FLASHTOPK_HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    sf_empty = torch.empty(0, dtype=torch.float32, device="cuda")
    ks = torch.zeros(16, dtype=torch.int32, device="cuda")
    out_empty = flashtopk_fp8_indexer_topk(q, w, kv_empty, sf_empty, ks, ks, 512)
    assert (out_empty == -1).all()

    # Non-contiguous (strided) output buffer slice.
    ks, ke, keys = _make_ragged([(2048, 32, "last")])
    q, w, kv, sf = _gen_inputs(32, keys, seed=3)
    wide = torch.full((32, 1024), -7, dtype=torch.int32, device="cuda")
    flashtopk_fp8_indexer_topk(q, w, kv, sf, ks, ke, 512, out=wide[:, :512])
    torch.cuda.synchronize()
    assert (wide[:, 512:] == -7).all()
    _check_fp8_topk_against_ref(wide[:, :512].contiguous(), q, w, kv, sf, ks, ke, 512)


@requires_flashtopk
@pytest.mark.parametrize(
    "mode", ["paged_indexed", "paged_per_row", "paged_broadcast", "ragged", "unfused"]
)
@pytest.mark.parametrize("padded_out_rows", [False, True])
def test_flashtopk_kpool_matches_dense(mode: str, padded_out_rows: bool):
    """Verify flashtopk_kpool_indexer_topk matches dense DeepGEMM +
    topk_from_pooled_history_logits across all KPool modes, tail sizes 0..3,
    short rows (pool_len < group_topk and pool_len == 0), and out_rows padding.
    """
    kpool = 4
    topk = 2048
    group_topk = topk // kpool  # 512
    # 4 packed requests:
    # req0: seq_len=3 (0 complete pools, 3 tail tokens -> pool_len=0!)
    # req1: seq_len=501 (125 pools < 512, tail=1)
    # req2: seq_len=4098 (1024 pools > 512, tail=2)
    # req3: seq_len=16384 (4096 pools > 512, tail=0)
    req_specs = [
        (3, 3, 2),  # (seq_len, q_len, req_pool_id)
        (501, 32, 5),
        (4098, 64, 1),
        (16384, 96, 7),
    ]
    ks_list: List[int] = []
    ke_list: List[int] = []
    pool_lens_list: List[int] = []
    seq_lens_list: List[int] = []
    pt_row_idx_list: List[int] = []
    topk_offsets_list: List[int] = []

    cum_pools = 0
    cum_tokens = 0
    for seq_len, q_len, req_id in req_specs:
        first_pos = seq_len - q_len
        for pos in range(first_pos, seq_len):
            tlen = pos + 1
            plen = tlen // kpool
            ks_list.append(cum_pools)
            ke_list.append(cum_pools + plen)
            pool_lens_list.append(plen)
            seq_lens_list.append(tlen)
            pt_row_idx_list.append(req_id)
            topk_offsets_list.append(cum_tokens)
        cum_pools += seq_len // kpool
        cum_tokens += seq_len

    n_real = len(ks_list)
    total_q = n_real + 13 if padded_out_rows else n_real
    device = "cuda"
    ks = torch.tensor(ks_list, dtype=torch.int32, device=device)
    ke = torch.tensor(ke_list, dtype=torch.int32, device=device)
    pool_lens = torch.tensor(pool_lens_list, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    q, w, kv, sf = _gen_inputs(total_q, cum_pools, seed=123)

    # Synthetic req_to_token table [10, max_seq_len] with unique physical slots.
    max_seq = max(s for s, _, _ in req_specs)
    req_to_token = (
        torch.arange(10 * max_seq, dtype=torch.int32, device=device).view(10, max_seq)
        + 100
    )

    page_table = None
    page_table_row_index = None
    topk_offsets = None
    if mode == "paged_indexed":
        page_table = req_to_token
        page_table_row_index = torch.tensor(
            pt_row_idx_list, dtype=torch.int32, device=device
        )
    elif mode == "paged_per_row":
        page_table = req_to_token[
            torch.tensor(pt_row_idx_list, dtype=torch.long, device=device)
        ].contiguous()
    elif mode == "paged_broadcast":
        # Non-contiguous broadcast page table (stride(0) == 0), as in _get_topk_ragged_kpool.
        # Restrict to single-request layout matching _get_topk_ragged_kpool.
        n_real = 64
        total_q = n_real + 7 if padded_out_rows else n_real
        seq_len = 8195  # 2048 pools, 3 tail tokens
        pool_total = seq_len // kpool
        ks = torch.zeros(n_real, dtype=torch.int32, device=device)
        seq_lens = torch.arange(
            seq_len - n_real + 1, seq_len + 1, dtype=torch.int32, device=device
        )
        pool_lens = torch.div(seq_lens, kpool, rounding_mode="floor").to(torch.int32)
        ke = pool_lens
        q, w, kv, sf = _gen_inputs(total_q, pool_total, seed=456)
        page_table = req_to_token[3, :seq_len].unsqueeze(0).expand(n_real, -1)
    elif mode == "ragged":
        topk_offsets = torch.tensor(topk_offsets_list, dtype=torch.int32, device=device)

    out_rows = total_q if padded_out_rows else None
    got = flashtopk_kpool_indexer_topk(
        q_fp8=q,
        weights=w,
        kv_fp8=kv,
        kv_scale=sf,
        starts=ks,
        ends=ke,
        pool_lens=pool_lens,
        kpool=kpool,
        topk=topk,
        seq_lens=seq_lens,
        page_table=page_table,
        page_table_row_index=page_table_row_index,
        topk_offsets=topk_offsets,
        out_rows=out_rows,
    )

    # Dense reference via DeepGEMM + SGLang's topk_from_pooled_history_logits.
    logits = _ref_logits(q[:n_real], w[:n_real], kv, sf, ks, ke)
    ref = topk_from_pooled_history_logits(
        logits=logits,
        group_lengths=pool_lens,
        kpool=kpool,
        topk=topk,
        page_table=page_table,
        topk_offsets=topk_offsets,
        seq_lens=seq_lens,
        row_starts=ks,
        out_rows=out_rows,
        page_table_row_index=page_table_row_index,
    )
    torch.cuda.synchronize()

    assert got.shape == ref.shape == (total_q, topk + kpool - 1)
    if padded_out_rows:
        assert (got[n_real:] == -1).all()
        assert (ref[n_real:] == -1).all()

    # Every row must select the exact same set of physical/ragged token slots
    # (order of pools may differ from unordered topk_v2, tail slots must match).
    got_sorted = got[:n_real].sort(dim=1).values
    ref_sorted = ref[:n_real].sort(dim=1).values
    assert torch.equal(got_sorted, ref_sorted), (
        f"KPool mismatch in mode={mode}: "
        f"{(got_sorted != ref_sorted).any(dim=1).sum().item()} rows differ"
    )


@requires_flashtopk
def test_flashtopk_kpool_adaptive_pruning_128k():
    """Exercise KPool FlashTopK past 128K pool keys (512K+ token context)."""
    kpool = 4
    topk = 2048
    pool_keys = FLASHTOPK_ADAPTIVE_MIN_KEYS + 1024
    rows = 256
    device = "cuda"
    ks = torch.zeros(rows, dtype=torch.int32, device=device)
    seq_lens = torch.arange(
        pool_keys * kpool - rows + 3,
        pool_keys * kpool + 3,
        dtype=torch.int32,
        device=device,
    )
    pool_lens = torch.div(seq_lens, kpool, rounding_mode="floor").to(torch.int32)
    ke = pool_lens
    q, w, kv, sf = _gen_inputs(rows, pool_keys, seed=99)
    offsets = torch.full((rows,), 1024, dtype=torch.int32, device=device)

    got = flashtopk_kpool_indexer_topk(
        q_fp8=q,
        weights=w,
        kv_fp8=kv,
        kv_scale=sf,
        starts=ks,
        ends=ke,
        pool_lens=pool_lens,
        kpool=kpool,
        topk=topk,
        seq_lens=seq_lens,
        topk_offsets=offsets,
    )
    logits = _ref_logits(q, w, kv, sf, ks, ke)
    ref = topk_from_pooled_history_logits(
        logits=logits,
        group_lengths=pool_lens,
        kpool=kpool,
        topk=topk,
        topk_offsets=offsets,
        seq_lens=seq_lens,
        row_starts=ks,
    )
    torch.cuda.synchronize()
    assert torch.equal(got.sort(dim=1).values, ref.sort(dim=1).values)


@requires_flashtopk
def test_flashtopk_warmup(monkeypatch):
    """warmup_flashtopk_prefill compiles both standard and KPool selectors cleanly."""
    monkeypatch.setenv("SGLANG_OPT_USE_PREFILL_FLASHTOPK", "1")

    class _DummyIndexer(torch.nn.Module):
        def __init__(self, topk: int, kpool: int = 1):
            super().__init__()
            self.use_prefill_flashtopk = True
            self.index_topk = topk
            self.index_kpool = kpool

    model = torch.nn.Sequential(_DummyIndexer(2048, 1), _DummyIndexer(2048, 4))
    warmup_flashtopk_prefill(model, torch.device("cuda", 0))
    # Verify selectors are cached and usable outside inference_mode even when
    # num_q > _CHUNK_SIZE (triggers multi-slice total.zero_() path).
    for eff_topk in (512, 2048):
        sel = _get_selector(0, eff_topk)
        assert sel.max_rows >= max(_WARMUP_ROWS)
        assert sel.max_kv >= max(_WARMUP_KEYS)
    rows_multi = max(_WARMUP_ROWS) + 16
    q, w, kv, sf = _gen_inputs(rows_multi, 2048, seed=7)
    ks = torch.zeros(rows_multi, dtype=torch.int32, device="cuda")
    ke = torch.full((rows_multi,), 2048, dtype=torch.int32, device="cuda")
    with torch.no_grad():
        out = flashtopk_fp8_indexer_topk(q, w, kv, sf, ks, ke, 512)
    torch.cuda.synchronize()
    assert out.shape == (rows_multi, 512)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
