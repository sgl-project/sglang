"""The chunked DSA kpool ragged prefill top-k must equal the single fused call.

`IndexerKPool._get_topk_ragged_kpool_plan` normally computes one dense
`[sum(q_len) x sum(pooled_seq_len)]` fp32 logits tensor for the whole batch. That
grows with the batch's total context even though each q row only ever reads its
own request's column window, which is how a 4-way long-context prefill asked for
73.65 GiB and killed every TP rank (sgl-project/sglang#37712).

`_topk_ragged_kpool_grouped` replaces that with one MQA call per request per
q-row chunk. Every index in it is a rebased slice — K column window, `ks`/`ke`,
`row_starts`, the fused top-k mapping tensors, the output rows — so the test that
matters is bit-equality against the unsplit path on the same inputs, across the
three top-k mapping modes and at chunk sizes that do not divide the q lengths.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

deep_gemm = pytest.importorskip("deep_gemm")

from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import (  # noqa: E402
    IndexerKPool,
)
from sglang.srt.layers.attention.dsa.kpool_fp8_index import (  # noqa: E402
    kpool_build_ragged_layout,
)
from sglang.srt.layers.attention.dsa.kpool_plan import (  # noqa: E402
    KPoolExtendPlan,
    PoolWriteRows,
    RaggedGroup,
    TailWriteRows,
    _append_local_rows,
    _KPoolCpuPlan,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")

POOL_SIZE = 4  # GLM-5.3-Flash index_kpool
SLOTS_PER_PAGE = 64  # real page_size
PAGE_SIZE = 64
INDEX_HEADS = 32  # index_n_heads
HEAD_DIM = 128  # index_head_dim
DEVICE = torch.device("cuda")


def _make_indexer(index_topk: int) -> IndexerKPool:
    # __new__ skips an __init__ that needs a model config, weights and a device;
    # the two shapes below are all the top-k path reads off self.
    indexer = IndexerKPool.__new__(IndexerKPool)
    indexer.index_topk = index_topk
    indexer.index_kpool = POOL_SIZE
    return indexer


def _build_plan(seq_lens: list[int], extend_lens: list[int]) -> KPoolExtendPlan:
    """A ragged extend plan built the way `_kpool_plan_to_gpu` builds one."""
    cpu = _KPoolCpuPlan()
    _append_local_rows(cpu, POOL_SIZE, SLOTS_PER_PAGE, extend_lens, seq_lens)

    # Per q row: the context length that row attends over.
    seqlens_expanded = torch.tensor(
        [
            seq_len - q_len + 1 + j
            for seq_len, q_len in zip(seq_lens, extend_lens, strict=True)
            for j in range(q_len)
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    pooled_seq_lens_expanded = torch.div(
        seqlens_expanded, POOL_SIZE, rounding_mode="floor"
    ).to(torch.int32)

    max_token_pages = max((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens)
    # Only the gather reads the page ids; ks/ke depend on the plan geometry.
    page_table = torch.arange(
        len(seq_lens) * max_token_pages, dtype=torch.int32, device=DEVICE
    ).view(len(seq_lens), max_token_pages)

    def _i32(values):
        return torch.tensor(values, dtype=torch.int32, device=DEVICE)

    total_pool_pages = cpu.total_pool_pages
    concat_page_table, q_ks, q_ke = kpool_build_ragged_layout(
        full_page_table=page_table,
        cu_pages_excl=_i32(cpu.cu_pages_excl),
        ragged_pool_pages=_i32(cpu.ragged_pool_pages),
        cu_q_len_excl=_i32(cpu.cu_q_len_excl),
        ragged_q_len=_i32(cpu.ragged_q_len),
        pooled_seq_lens_expanded=pooled_seq_lens_expanded,
        slots_per_page=SLOTS_PER_PAGE,
        total_pool_pages=total_pool_pages,
        total_q=pooled_seq_lens_expanded.shape[0],
        pool_size=POOL_SIZE,
    )

    ragged_groups = tuple(
        RaggedGroup(
            q_start=q_start,
            q_len=q_len,
            k_start=page_start * SLOTS_PER_PAGE,
            k_rows=pool_pages * SLOTS_PER_PAGE,
        )
        for q_start, q_len, page_start, pool_pages in zip(
            cpu.cu_q_len_excl,
            cpu.ragged_q_len,
            cpu.cu_pages_excl,
            cpu.ragged_pool_pages,
            strict=True,
        )
    )

    empty = torch.empty((0,), dtype=torch.int32, device=DEVICE)
    empty64 = torch.empty((0,), dtype=torch.int64, device=DEVICE)
    return KPoolExtendPlan(
        writes=PoolWriteRows(
            req=empty64,
            pool_id=empty64,
            n_from_tail=empty,
            chunk_src=empty64,
            tail_logical_base=empty,
            write_loc=empty64,
        ),
        tails=TailWriteRows(
            req=empty64, dst_logical_start=empty, chunk_src=empty64, n_write=empty
        ),
        pooled_seq_lens_expanded=pooled_seq_lens_expanded,
        seq_lens_expanded=seqlens_expanded,
        ragged_concat_page_table=concat_page_table,
        ragged_q_ks=q_ks,
        ragged_q_ke=q_ke,
        ragged_total_k_rows=total_pool_pages * SLOTS_PER_PAGE,
        ragged_k_u8=None,
        ragged_k_scale=None,
        ragged_paged_page_table=None,
        ragged_paged_page_table_row_index=None,
        ragged_groups=ragged_groups,
    )


def _make_inputs(
    total_q: int, total_k_rows: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def _randn(*shape):
        return torch.randn(*shape, generator=gen, device=DEVICE, dtype=torch.float32)

    q_fp8 = _randn(total_q, INDEX_HEADS, HEAD_DIM).to(torch.float8_e4m3fn)
    weights = _randn(total_q, INDEX_HEADS).abs()
    k_fp8 = _randn(max(total_k_rows, 1), HEAD_DIM)[:total_k_rows].to(
        torch.float8_e4m3fn
    )
    k_scale = _randn(max(total_k_rows, 1))[:total_k_rows].abs() + 0.25
    return q_fp8, weights, k_fp8, k_scale


def _fused_mapping(
    mode: str, plan: KPoolExtendPlan, seq_lens: list[int], extend_lens: list[int]
):
    """(page_table, page_table_row_index, topk_offsets) for one top-k mapping mode.

    Mirrors what `_kpool_ragged_topk_mapping` hands back for each of
    `TopkTransformMethod.PAGED` / `RAGGED` / fusion off. Production only ever
    pairs a paged table with a row index (`kpool_plan._kpool_plan_to_gpu`), so
    that is the only paged shape covered.
    """
    n_real = plan.seq_lens_expanded.shape[0]
    if mode == "none":
        return None, None, None
    if mode == "ragged":
        # Where each q row's request starts in the flat token index space.
        starts = torch.tensor(
            [
                start
                for start, q_len in zip(
                    [sum(seq_lens[:i]) for i in range(len(seq_lens))],
                    extend_lens,
                    strict=True,
                )
                for _ in range(q_len)
            ],
            dtype=torch.int32,
            device=DEVICE,
        )
        return None, None, starts
    assert mode == "paged"
    # Token-level table: raw_token < seq_len always, so max(seq_lens) columns.
    page_table = torch.randperm(
        len(seq_lens) * max(seq_lens), device=DEVICE, dtype=torch.int32
    ).view(len(seq_lens), max(seq_lens))
    row_index = torch.repeat_interleave(
        torch.arange(len(seq_lens), dtype=torch.int32, device=DEVICE),
        torch.tensor(extend_lens, dtype=torch.int32, device=DEVICE),
    )
    assert row_index.shape[0] == n_real
    return page_table, row_index, None


def _reference_unsplit(
    indexer: IndexerKPool,
    plan: KPoolExtendPlan,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    total_q: int,
    mapping,
) -> torch.Tensor:
    """The single fused call the chunked path has to reproduce."""
    page_table, page_table_row_index, topk_offsets = mapping
    n_real = plan.seq_lens_expanded.shape[0]
    logits = deep_gemm.fp8_mqa_logits(
        q_fp8[:n_real].contiguous(),
        (k_fp8.contiguous(), k_scale.contiguous()),
        weights[:n_real].contiguous(),
        plan.ragged_q_ks,
        plan.ragged_q_ke,
        clean_logits=True,
    )
    return indexer._topk_from_kpool_logits(
        logits,
        plan.pooled_seq_lens_expanded,
        seq_lens=plan.seq_lens_expanded,
        page_table=page_table,
        topk_offsets=topk_offsets,
        row_starts=plan.ragged_q_ks,
        out_rows=total_q,
        page_table_row_index=page_table_row_index,
    )


def _grouped(
    indexer: IndexerKPool,
    plan: KPoolExtendPlan,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    total_q: int,
    mapping,
    logits_budget_bytes: int,
) -> torch.Tensor:
    page_table, page_table_row_index, topk_offsets = mapping
    return indexer._topk_ragged_kpool_grouped(
        plan=plan,
        q_fp8=q_fp8,
        weights=weights,
        k_fp8=k_fp8,
        k_scale=k_scale,
        logits_budget_bytes=logits_budget_bytes,
        total_q=total_q,
        page_table=page_table,
        page_table_row_index=page_table_row_index,
        topk_offsets=topk_offsets,
    )


def _assert_same_selection(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Same selected tokens per row.

    Top-k store order inside a row is not pinned down by the fused kernel (it
    fills from `atomicAdd` positions), so rows are compared as sorted sets. The
    -1 padding sorts to the front and is compared too, which keeps the count of
    real selections exact.
    """
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.sort(dim=1).values, expected.sort(dim=1).values, rtol=0, atol=0
    )


# (seq_lens, extend_lens, q_padding)
CASES = {
    # Pure ragged prefill; 137 and 2049 are not page- or pool-aligned.
    "prefill_mixed": ([600, 137, 2049], [600, 137, 2049], 0),
    # Extend over existing history: q rows start mid-context.
    "extend_with_history": ([700, 300, 4100], [100, 64, 33], 0),
    # seq_len < index_kpool: no pooled history at all, tail-select only.
    "empty_pool_group": ([3, 512, 2], [3, 512, 2], 0),
    # CUDA-graph style padding: q_fp8 has rows the plan does not describe.
    "padded_q": ([521, 300], [521, 300], 37),
}

# Budgets small enough to force many chunks, chosen so the per-group row count
# does not divide the group's q_len.
BUDGETS = [3 * 64 * 4, 7 * 64 * 4, 5 * 4096, 1 << 40]


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("mode", ["none", "ragged", "paged"])
@pytest.mark.parametrize("budget", BUDGETS)
def test_chunked_ragged_topk_matches_the_single_fused_call(case, mode, budget):
    seq_lens, extend_lens, q_padding = CASES[case]
    indexer = _make_indexer(index_topk=512)
    plan = _build_plan(seq_lens, extend_lens)

    n_real = plan.seq_lens_expanded.shape[0]
    total_q = n_real + q_padding
    q_fp8, weights, k_fp8, k_scale = _make_inputs(
        total_q, plan.ragged_total_k_rows, seed=sorted(CASES).index(case)
    )
    mapping = _fused_mapping(mode, plan, seq_lens, extend_lens)

    expected = _reference_unsplit(
        indexer, plan, q_fp8, weights, k_fp8, k_scale, total_q, mapping
    )
    actual = _grouped(
        indexer,
        plan,
        q_fp8,
        weights,
        k_fp8,
        k_scale,
        total_q,
        mapping,
        logits_budget_bytes=budget,
    )
    _assert_same_selection(actual, expected)


def test_the_chunked_path_never_builds_the_whole_batch_logits():
    """The point of the split: peak logits bytes stop scaling with the batch."""
    seq_lens = [8192] * 6
    indexer = _make_indexer(index_topk=512)
    plan = _build_plan(seq_lens, seq_lens)
    q_fp8, weights, k_fp8, k_scale = _make_inputs(
        plan.seq_lens_expanded.shape[0], plan.ragged_total_k_rows, seed=7
    )

    dense_bytes = plan.seq_lens_expanded.shape[0] * plan.ragged_total_k_rows * 4
    budget = 64 << 20

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    before = torch.cuda.memory_allocated(DEVICE)
    _grouped(
        indexer,
        plan,
        q_fp8,
        weights,
        k_fp8,
        k_scale,
        plan.seq_lens_expanded.shape[0],
        (None, None, None),
        logits_budget_bytes=budget,
    )
    torch.cuda.synchronize()
    peak_growth = torch.cuda.max_memory_allocated(DEVICE) - before

    # One 8192-row group's own window is ~64 MiB of logits; the whole batch's
    # would be 6x that. Allow room for the int32 output and the ks/ke rebases.
    assert peak_growth < dense_bytes // 2, (peak_growth, dense_bytes)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
