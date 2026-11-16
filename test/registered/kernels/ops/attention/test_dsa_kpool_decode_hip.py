"""ROCm decode path of the DSA k-pool indexer (GLM-5.3-Flash).

Each part is compared bit for bit with the code it replaces:

- ``kpool_q_rotate_quant_gate``: the query Hadamard, the 128-wide fp8 quant of
  the TileLang ``act_quant`` and the head-gate scaling of
  ``IndexerKPool._get_logits_head_gate``, in one kernel.
- ``out_col_align`` of the fused k-pool top-k: the table comes out at the
  64-column width that TileLang sparse attention reads, and the extra columns
  are -1, as the padding of ``_forward_tilelang`` makes them.
- ``init/update_pooled_paged_mqa_metadata`` on ROCm decode: the pooled seqlens
  and page table are built once per step, and each layer gets the values it
  would otherwise build itself.

The radix top-k does not fix an output order, so the pooled top-k columns are
compared as a set; the tail and padding columns are positional and are compared
exactly.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils import is_gfx1250_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

if not (torch.cuda.is_available() and is_hip()):
    pytest.skip("ROCm GPU required", allow_module_level=True)

# GLM-5.3-Flash indexer.
N_HEADS = 32
HEAD_DIM = 128
KPOOL = 4
TOPK = 2048
PAGE = 64
TILELANG_COL_BLOCK = 64

requires_tilelang_act_quant = pytest.mark.skipif(
    is_gfx1250_supported(),
    reason="gfx1250 quantizes with the Triton act_quant, which the kernel does not reproduce",
)


# ---------------------------------------------------------------------------
# Query Hadamard + fp8 quant + head gate in one kernel
# ---------------------------------------------------------------------------


def _query_inputs(n: int, seed: int = 0):
    """bf16 query [n, heads, 128] and fp32 head weights [n, heads].

    Row 0 of a non-empty batch is all zero (amax hits its 1e-4 floor) and row 1
    is scaled up, so the scales cover a few binades."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        n, N_HEADS, HEAD_DIM, generator=g, device="cuda", dtype=torch.float32
    )
    if n > 0:
        query[0] = 0
    if n > 1:
        query[1] *= 300
    head_weights = torch.randn(n, N_HEADS, generator=g, device="cuda")
    return query.to(torch.bfloat16), head_weights


def _reference_query_prep(query, head_weights, *, rotate, round_scale):
    """The generic decode path: rotate_activation -> act_quant -> head gate."""
    from sglang.kernels.ops.quantization.act_quant_tilelang import act_quant
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

    if rotate:
        query = rotate_activation(query)
    q_fp8, q_scale = act_quant(
        query.contiguous(), HEAD_DIM, "ue8m0" if round_scale else None
    )
    weights = head_weights * N_HEADS**-0.5
    weights = weights.unsqueeze(-1) * q_scale * HEAD_DIM**-0.5
    return q_fp8, weights


def _fused_query_prep(query, head_weights, *, rotate, round_scale):
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        kpool_q_rotate_quant_gate,
    )

    return kpool_q_rotate_quant_gate(
        query,
        head_weights,
        rotate=rotate,
        round_scale=round_scale,
        weight_scale_1=N_HEADS**-0.5,
        weight_scale_2=HEAD_DIM**-0.5,
    )


def _assert_bit_identical(actual, expected):
    q_fp8, weights = actual
    ref_q_fp8, ref_weights = expected
    assert q_fp8.dtype == ref_q_fp8.dtype
    assert q_fp8.shape == ref_q_fp8.shape
    assert weights.shape == ref_weights.shape
    assert weights.dtype == torch.float32
    assert torch.equal(q_fp8.view(torch.uint8), ref_q_fp8.view(torch.uint8))
    assert torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))


@requires_tilelang_act_quant
@pytest.mark.parametrize("round_scale", [True, False])
@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("n", [1, 3, 16, 33, 96])
def test_query_prep_matches_generic_path(n, rotate, round_scale):
    query, head_weights = _query_inputs(n)
    _assert_bit_identical(
        _fused_query_prep(query, head_weights, rotate=rotate, round_scale=round_scale),
        _reference_query_prep(
            query, head_weights, rotate=rotate, round_scale=round_scale
        ),
    )


@requires_tilelang_act_quant
def test_query_prep_reads_strided_inputs():
    """Token and head strides of the inputs are free; only the inner dim is unit."""
    n = 16
    query, head_weights = _query_inputs(n, seed=1)
    wide_query = torch.zeros(
        n, N_HEADS, 2 * HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    wide_query[..., :HEAD_DIM] = query
    wide_weights = torch.zeros(n, 2 * N_HEADS, device="cuda")
    wide_weights[:, :N_HEADS] = head_weights
    strided_query = wide_query[..., :HEAD_DIM]
    strided_weights = wide_weights[:, :N_HEADS]
    assert not strided_query.is_contiguous()
    assert not strided_weights.is_contiguous()
    _assert_bit_identical(
        _fused_query_prep(
            strided_query, strided_weights, rotate=True, round_scale=True
        ),
        _reference_query_prep(query, head_weights, rotate=True, round_scale=True),
    )


def test_query_prep_empty_batch():
    query, head_weights = _query_inputs(0)
    q_fp8, weights = _fused_query_prep(
        query, head_weights, rotate=True, round_scale=True
    )
    assert q_fp8.shape == (0, N_HEADS, HEAD_DIM)
    assert weights.shape == (0, N_HEADS, 1)


@requires_tilelang_act_quant
def test_query_prep_graph_replay():
    """A graph replay with new buffer contents gives the eager result."""
    n = 16
    query, head_weights = _query_inputs(n, seed=2)

    def run():
        return _fused_query_prep(query, head_weights, rotate=True, round_scale=True)

    run()  # Compile before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for seed in (3, 4):
        new_query, new_weights = _query_inputs(n, seed=seed)
        query.copy_(new_query)
        head_weights.copy_(new_weights)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bit_identical(
            out,
            _reference_query_prep(query, head_weights, rotate=True, round_scale=True),
        )


# ---------------------------------------------------------------------------
# Top-k at the width TileLang reads
# ---------------------------------------------------------------------------


def _topk_inputs(groups_per_row, tails):
    """Distinct pooled scores, so the top-k set is unique, and seq_lens with
    ``tail`` live tokens after the last closed pool of each row."""
    torch.manual_seed(0)
    width = max(groups_per_row) + 8
    score = torch.stack(
        [torch.randperm(width, dtype=torch.float32) for _ in groups_per_row]
    ).cuda()
    lengths = torch.tensor(groups_per_row, dtype=torch.int32, device="cuda")
    seq_lens = lengths * KPOOL + torch.tensor(tails, dtype=torch.int32, device="cuda")
    return score, lengths, seq_lens


def _assert_aligned_table(aligned, unaligned):
    live = unaligned.shape[1]
    assert live == TOPK + KPOOL - 1
    assert aligned.shape[1] == -(-live // TILELANG_COL_BLOCK) * TILELANG_COL_BLOCK
    assert aligned.shape[1] == 2112
    aligned, unaligned = aligned.cpu(), unaligned.cpu()
    torch.testing.assert_close(
        aligned[:, :TOPK].sort(dim=1).values,
        unaligned[:, :TOPK].sort(dim=1).values,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        aligned[:, TOPK:live], unaligned[:, TOPK:], atol=0, rtol=0
    )
    assert bool((aligned[:, live:] == -1).all())


# Rows: shorter than the budget, at it, and past it; tails 0..3.
GROUPS = [3, 512, 600, 32768]
TAILS = [2, 0, 3, 1]


def test_topk_out_col_align_pads_with_minus_one():
    from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
        fast_kpool_topk_transform_fused,
    )

    score, lengths, seq_lens = _topk_inputs(GROUPS, TAILS)
    kwargs = dict(
        score=score, lengths=lengths, kpool=KPOOL, topk=TOPK, seq_lens=seq_lens
    )
    _assert_aligned_table(
        fast_kpool_topk_transform_fused(**kwargs, out_col_align=TILELANG_COL_BLOCK),
        fast_kpool_topk_transform_fused(**kwargs),
    )


def test_topk_out_col_align_with_page_table():
    """The decode call: token ids go through the page table, -1 stays -1."""
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        topk_from_pooled_history_logits,
    )

    score, lengths, seq_lens = _topk_inputs(GROUPS, TAILS)
    rows = len(GROUPS)
    page_width = score.shape[1] * KPOOL
    pages = torch.randperm(
        (rows + 1) * page_width, dtype=torch.int32, device="cuda"
    ).reshape(rows + 1, page_width)
    row_index = torch.tensor([2, 0, 4, 1], dtype=torch.int32, device="cuda")
    kwargs = dict(
        logits=score,
        group_lengths=lengths,
        kpool=KPOOL,
        topk=TOPK,
        page_table=pages,
        page_table_row_index=row_index,
        seq_lens=seq_lens,
    )
    _assert_aligned_table(
        topk_from_pooled_history_logits(**kwargs, out_col_align=TILELANG_COL_BLOCK),
        topk_from_pooled_history_logits(**kwargs),
    )


def test_topk_out_col_align_keeps_an_aligned_width():
    """Without tail columns the width is already 2048; the option changes nothing."""
    from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
        fast_kpool_topk_transform_fused,
    )

    score, lengths, _ = _topk_inputs(GROUPS, TAILS)
    kwargs = dict(score=score, lengths=lengths, kpool=KPOOL, topk=TOPK)
    aligned = fast_kpool_topk_transform_fused(
        **kwargs, out_col_align=TILELANG_COL_BLOCK
    )
    unaligned = fast_kpool_topk_transform_fused(**kwargs)
    assert aligned.shape == unaligned.shape == (len(GROUPS), TOPK)
    torch.testing.assert_close(
        aligned.sort(dim=1).values.cpu(),
        unaligned.sort(dim=1).values.cpu(),
        atol=0,
        rtol=0,
    )


def test_backend_asks_for_the_tilelang_width_on_decode_only():
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.model_executor.forward_mode import ForwardMode

    def align(mode=ForwardMode.DECODE, **overrides):
        backend = dict(
            dsa_index_kpool=KPOOL,
            dsa_decode_impl="tilelang",
            use_fused_topk=True,
            hisparse_coordinator=None,
        )
        backend.update(overrides)
        return DeepseekSparseAttnBackend._decode_topk_out_col_align(
            SimpleNamespace(**backend), SimpleNamespace(forward_mode=mode)
        )

    assert align() == TILELANG_COL_BLOCK
    assert align(ForwardMode.IDLE) == TILELANG_COL_BLOCK
    assert align(ForwardMode.EXTEND) == 1
    assert align(ForwardMode.TARGET_VERIFY) == 1
    assert align(dsa_index_kpool=1) == 1
    assert align(dsa_decode_impl="flashmla_kv") == 1
    assert align(use_fused_topk=False) == 1
    assert align(hisparse_coordinator=object()) == 1


# ---------------------------------------------------------------------------
# Pooled metadata once per step
# ---------------------------------------------------------------------------


def _dsa_metadata(seqlens, page_table):
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata

    bs = seqlens.shape[0]
    cu = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    return DSAMetadata(
        page_size=PAGE,
        cache_seqlens_int32=seqlens,
        max_seq_len_q=1,
        max_seq_len_k=int(seqlens.max()),
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        page_table_1=None,
        real_page_table=page_table,
        dsa_cache_seqlens_int32=seqlens,
        dsa_cu_seqlens_q=cu,
        dsa_cu_seqlens_k=cu,
        dsa_extend_seq_lens_list=[],
        dsa_seqlens_expanded=seqlens,
    )


def _per_layer_metadata(attn_metadata, page_table, seqlens):
    """What IndexerKPool._get_kpool_decode_metadata gives a layer."""
    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

    return IndexerKPool._get_kpool_decode_metadata(
        SimpleNamespace(index_kpool=KPOOL),
        SimpleNamespace(attn_metadata=attn_metadata),
        page_table,
        seqlens,
        PAGE,
        build_schedule_metadata=False,
    )


def _step_inputs(seqlens_host, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seqlens = torch.tensor(seqlens_host, dtype=torch.int32, device="cuda")
    pages = 4096 // PAGE
    page_table = torch.randint(
        0, 1 << 20, (len(seqlens_host), pages), generator=g, device="cuda"
    ).to(torch.int32)
    return seqlens, page_table


def test_pooled_metadata_is_built_for_rocm_decode_only():
    from sglang.srt.layers.attention.dsa.kpool_plan import (
        _pooled_paged_mqa_metadata_enabled,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode

    assert _pooled_paged_mqa_metadata_enabled(ForwardMode.DECODE)
    assert _pooled_paged_mqa_metadata_enabled(ForwardMode.IDLE)
    assert not _pooled_paged_mqa_metadata_enabled(ForwardMode.EXTEND)
    assert not _pooled_paged_mqa_metadata_enabled(ForwardMode.TARGET_VERIFY)


def test_pooled_metadata_matches_per_layer_build_and_updates_in_place():
    from sglang.srt.layers.attention.dsa.kpool_plan import (
        init_pooled_paged_mqa_metadata,
        update_pooled_paged_mqa_metadata,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode

    kwargs = dict(kpool=KPOOL, physical_page_size=PAGE, index_page_size=PAGE)
    seqlens, page_table = _step_inputs([1, 63, 64, 2051, 4096], seed=0)
    metadata = init_pooled_paged_mqa_metadata(
        _dsa_metadata(seqlens, page_table), seqlens, ForwardMode.DECODE, **kwargs
    )
    assert metadata.pooled_index_kpool == KPOOL
    # CUDA-only parts stay off on ROCm.
    assert metadata.pooled_paged_mqa_schedule_metadata is None
    assert metadata.pooled_topk_v2_plan is None
    pooled_seqlens = metadata.pooled_cache_seqlens_int32
    pooled_table = metadata.pooled_real_page_table

    # Graph replay: the step writes new values into the captured buffers.
    for seqlens_host, seed in (([1, 63, 64, 2051, 4096], 0), ([7, 4095, 5, 64, 2], 1)):
        new_seqlens, new_table = _step_inputs(seqlens_host, seed)
        seqlens.copy_(new_seqlens)
        page_table.copy_(new_table)
        update_pooled_paged_mqa_metadata(
            metadata, seqlens, ForwardMode.DECODE, **kwargs
        )
        assert metadata.pooled_cache_seqlens_int32 is pooled_seqlens
        assert metadata.pooled_real_page_table is pooled_table

        shared = _per_layer_metadata(metadata, page_table, seqlens)
        rebuilt = _per_layer_metadata(
            _dsa_metadata(seqlens, page_table), page_table, seqlens
        )
        # Every layer reads the step's buffers instead of building its own.
        assert shared[0].data_ptr() == pooled_seqlens.data_ptr()
        assert shared[2].data_ptr() == pooled_table.data_ptr()
        for name, got, want in zip(
            ("pool_seqlens", "pool_context_lens", "pool_block_tables"),
            shared[:3],
            rebuilt[:3],
        ):
            torch.testing.assert_close(got, want, atol=0, rtol=0, msg=name)
        assert shared[3] is None and rebuilt[3] is None
        assert shared[4] is None and rebuilt[4] is None


def test_pooled_metadata_leaves_extend_alone():
    from sglang.srt.layers.attention.dsa.kpool_plan import (
        init_pooled_paged_mqa_metadata,
        update_pooled_paged_mqa_metadata,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode

    kwargs = dict(kpool=KPOOL, physical_page_size=PAGE, index_page_size=PAGE)
    seqlens, page_table = _step_inputs([1, 63, 2051], seed=0)
    metadata = _dsa_metadata(seqlens, page_table)
    assert (
        init_pooled_paged_mqa_metadata(metadata, seqlens, ForwardMode.EXTEND, **kwargs)
        is metadata
    )
    update_pooled_paged_mqa_metadata(metadata, seqlens, ForwardMode.EXTEND, **kwargs)
    assert metadata.pooled_index_kpool == 1
    assert metadata.pooled_cache_seqlens_int32 is None
    assert metadata.pooled_real_page_table is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
