"""The grouped verify indexer must select each draft row's per-row decode top-k."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
from sglang.kernels.ops.attention.minimax_sparse.decode import (
    verify_topk_idx as verify_module,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.verify_topk_idx import (
    verify_topk_idx,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
)
from sglang.srt.utils import get_device, is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.quant_ref_utils import nvfp4_kv_quantize

# Portable Triton indexer; the MI35x suite also covers the gfx950 fused kernels.
register_amd_ci(est_time=120, stage="jit-kernel-unit", runner_config="amd")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

# MiniMax-M3 / M3.1 sparse layers: 128-token blocks, top-16, no init block.
BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16
DEVICE = get_device()
# M3 verifies 4 EAGLE3 draft rows per request, M3.1 verifies 8.
NUM_DRAFTS = pytest.mark.parametrize("num_draft", [4, 8])


def _build(
    prefixes,
    *,
    req_slots,
    num_draft,
    num_q_heads=1,
    num_kv_heads=1,
    k_dtype=torch.bfloat16,
    q_dtype=torch.bfloat16,
    draft_keys="away",
    padding_requests=0,
):
    gen = torch.Generator(device=DEVICE).manual_seed(len(prefixes) * 1000 + num_draft)
    max_len = max(prefixes) + num_draft
    num_table_rows = max(req_slots) + 1
    # Permuted token slots: every request reads a scattered slice of the pool.
    num_slots = num_table_rows * max_len + 1
    req_to_token = (
        torch.randperm(num_slots - 1, device=DEVICE, generator=gen)[
            : num_table_rows * max_len
        ]
        .add_(1)
        .to(torch.int32)
        .reshape(num_table_rows, max_len)
    )
    k_cache = torch.randn(
        num_slots, num_kv_heads, HEAD_DIM, device=DEVICE, generator=gen
    )
    rows = (len(prefixes) + padding_requests) * num_draft
    q = torch.randn(rows, num_q_heads, HEAD_DIM, device=DEVICE, generator=gen)
    # "away" draft keys score lowest for every row of the request, so a row's own
    # local block wins only when forced; "toward" keys win any block a row can see.
    sign = -10.0 if draft_keys == "away" else 10.0
    for r, (prefix, slot) in enumerate(zip(prefixes, req_slots)):
        key = q[r * num_draft : (r + 1) * num_draft].reshape(-1, HEAD_DIM).sum(0)
        for j in range(1, num_draft):
            k_cache[req_to_token[slot, prefix + j].long()] = key * sign
    seq_lens = [p + j + 1 for p in prefixes for j in range(num_draft)]
    slot_ids = [s for s in req_slots for _ in range(num_draft)]
    # CUDA-graph padding requests: seq_lens fill value 1 plus the draft offsets.
    seq_lens += [j + 2 for _ in range(padding_requests) for j in range(num_draft)]
    slot_ids += [0] * (padding_requests * num_draft)
    return dict(
        q=q.to(q_dtype),
        k_cache=k_cache.to(k_dtype),
        req_to_token=req_to_token,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=DEVICE),
        max_seqlen=max_len,
        slot_ids=torch.tensor(slot_ids, dtype=torch.int64, device=DEVICE),
    )


def _assert_matches_per_row(inputs, *, num_draft, local_blocks=1, radix_topk=True):
    with envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.override(radix_topk):
        _, ref_idx, _ = flash_decode_with_topk_idx(
            sink=None,
            v_cache=None,
            block_size=BLOCK_SIZE,
            topk=TOPK,
            init_blocks=0,
            local_blocks=local_blocks,
            disable_index_value=True,
            **inputs,
        )
        # A score column the grouped kernel skips would otherwise read the
        # reference's identical, just-freed score buffer.
        q, req_to_token = inputs["q"], inputs["req_to_token"]
        num_cols = -(-min(inputs["max_seqlen"], req_to_token.shape[1]) // BLOCK_SIZE)
        poison = torch.full((q.shape[1], q.shape[0], num_cols), 1e31, device=DEVICE)
        del poison
        grouped_idx = verify_topk_idx(
            num_draft=num_draft,
            block_size=BLOCK_SIZE,
            topk=TOPK,
            init_blocks=0,
            local_blocks=local_blocks,
            sm_scale=None,
            q_scale=None,
            k_scale=None,
            **inputs,
        )
    assert grouped_idx.shape == ref_idx.shape
    ref_sets = ref_idx.sort(dim=-1).values
    grouped_sets = grouped_idx.sort(dim=-1).values
    mismatched = (ref_sets != grouped_sets).any(-1).nonzero().tolist()
    assert mismatched == [], "(head, row) pairs with a different top-k set"


def _sparse_decode_args(inputs, *, seed, extra_rows=0):
    """`minimax_sparse_decode` arguments over `inputs`; `extra_rows` appends rows
    that repeat the first rows' lengths and slots."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=DEVICE, generator=gen).to(torch.bfloat16)

    rows = inputs["q"].shape[0] + extra_rows
    num_slots = inputs["k_cache"].shape[0]
    idx_q = torch.cat([inputs["q"], randn(extra_rows, 1, HEAD_DIM)])
    args = (
        randn(rows, 16, HEAD_DIM),
        None,
        randn(num_slots, 1, HEAD_DIM),
        randn(num_slots, 1, HEAD_DIM),
        idx_q,
        None,
        inputs["k_cache"],
        None,
    )
    kwargs = dict(
        req_to_token=inputs["req_to_token"],
        slot_ids=torch.cat([inputs["slot_ids"], inputs["slot_ids"][:extra_rows]]),
        seq_lens=torch.cat([inputs["seq_lens"], inputs["seq_lens"][:extra_rows]]),
        max_seqlen=inputs["max_seqlen"],
        block_size_q=1,
        block_size_k=BLOCK_SIZE,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        disable_index_value=True,
    )
    return args, kwargs


@NUM_DRAFTS
@pytest.mark.parametrize("radix_topk", [True, False])
def test_rows_straddling_block_boundaries(num_draft, radix_topk):
    """A shorter draft row keeps its own causal length and local block when the
    request's draft tail crosses into the next block."""
    prefixes = [BLOCK_SIZE * 40 - 2, BLOCK_SIZE * 23 - 1, 7777, BLOCK_SIZE * 60]
    inputs = _build(prefixes, req_slots=[5, 2, 7, 1], num_draft=num_draft)
    _assert_matches_per_row(inputs, num_draft=num_draft, radix_topk=radix_topk)


@NUM_DRAFTS
def test_score_chunks_spanning_several_blocks(num_draft):
    """At long context each score program walks several blocks of its chunk; 24
    requests get 64 chunks, so 150-block prefixes put 3 blocks in each."""
    prefixes = [BLOCK_SIZE * (150 - r % 3) - r for r in range(24)]
    inputs = _build(prefixes, req_slots=list(range(24)), num_draft=num_draft)
    _assert_matches_per_row(inputs, num_draft=num_draft)


@NUM_DRAFTS
@pytest.mark.parametrize(
    "q_dtype,k_dtype",
    [
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.bfloat16, torch.float8_e4m3fnuz),
    ],
)
def test_fp8_index_k(num_draft, q_dtype, k_dtype):
    """An fp8 index K, widened to a bf16 Q, selects the per-row blocks."""
    prefixes = [BLOCK_SIZE * 31 - 3, 4321, BLOCK_SIZE * 17 - 2]
    inputs = _build(
        prefixes,
        req_slots=[3, 6, 0],
        num_draft=num_draft,
        q_dtype=q_dtype,
        k_dtype=k_dtype,
    )
    _assert_matches_per_row(inputs, num_draft=num_draft)


@NUM_DRAFTS
def test_graph_padding_requests(num_draft):
    """Padding requests (short rows on slot 0) next to long ones disturb neither."""
    inputs = _build(
        [BLOCK_SIZE * 50 - 2, 3000],
        req_slots=[4, 9],
        num_draft=num_draft,
        padding_requests=2,
    )
    _assert_matches_per_row(inputs, num_draft=num_draft)


@NUM_DRAFTS
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(4, 1), (2, 2)])
def test_multiple_index_heads(num_draft, num_q_heads, num_kv_heads):
    """Index heads sharing an index-K head fill the M tile beside the draft rows
    (TP < 4), and each index-K head scores against its own keys."""
    inputs = _build(
        [BLOCK_SIZE * 25 - 1, 5000],
        req_slots=[1, 3],
        num_draft=num_draft,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )
    _assert_matches_per_row(inputs, num_draft=num_draft)


@NUM_DRAFTS
def test_no_local_blocks_keeps_rows_causal(num_draft):
    """Without a forced local block, a row must not see later draft rows' keys,
    which here outscore every prefix key."""
    inputs = _build(
        [BLOCK_SIZE * 25 - 2, BLOCK_SIZE * 12 - 3, 5000],
        req_slots=[1, 3, 0],
        num_draft=num_draft,
        draft_keys="toward",
    )
    _assert_matches_per_row(inputs, num_draft=num_draft, local_blocks=0)


@NUM_DRAFTS
@pytest.mark.parametrize(
    "extra_rows,score_type", [(1, "max"), (0, "lse")], ids=["ragged", "lse"]
)
def test_unsupported_verify_batches_fall_back_to_per_row(
    num_draft, extra_rows, score_type
):
    """A batch the grouped indexer cannot score (rows not a multiple of num_draft,
    or the lse score) must run the per-row decode indexer instead."""
    inputs = _build([BLOCK_SIZE * 20 - 2, 3000], req_slots=[2, 1], num_draft=num_draft)
    args, kwargs = _sparse_decode_args(inputs, seed=3, extra_rows=extra_rows)
    kwargs["score_type"] = score_type
    _, ref_o = minimax_sparse_decode(*args, **kwargs)
    _, verify_o = minimax_sparse_decode(*args, **kwargs, verify_num_draft=num_draft)
    torch.testing.assert_close(verify_o, ref_o, rtol=0, atol=0)


@NUM_DRAFTS
@pytest.mark.parametrize("grouped", [False, True], ids=["per_row", "grouped"])
def test_source_layer_publishes_topk_for_reuse_layers(num_draft, grouped):
    """A top-k sharing source layer writes its selection straight into the shared
    buffer; the reuse layers must read exactly that selection."""
    inputs = _build([BLOCK_SIZE * 30 - 2, 6000], req_slots=[2, 1], num_draft=num_draft)
    args, kwargs = _sparse_decode_args(inputs, seed=0)
    kwargs["verify_num_draft"] = num_draft if grouped else None
    _, ref_o = minimax_sparse_decode(*args, **kwargs)
    rows = inputs["q"].shape[0]
    shared = torch.full((1, rows, TOPK), -7, dtype=torch.int32, device=DEVICE)
    _, source_o = minimax_sparse_decode(*args, **kwargs, topk_out=shared)
    _, reuse_o = minimax_sparse_decode(*args, **kwargs, cached_topk_idx=shared)
    torch.testing.assert_close(source_o, ref_o, rtol=0, atol=0)
    torch.testing.assert_close(reuse_o, ref_o, rtol=0, atol=0)


def _build_paged(prefixes, *, num_draft, k_dtype, seed, tied_pages=False):
    """Like _build, but slots are whole 128-token pages, as the paged KV pool hands out.
    With ``tied_pages``, every page copies page 1 except every 13th, which copies page 2:
    most blocks tie on score and the few others rank above or below them."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    max_len = max(prefixes) + num_draft
    pages_per_req = -(-max_len // BLOCK_SIZE)
    num_pages = len(prefixes) * pages_per_req
    pages = torch.randperm(num_pages, device=DEVICE, generator=gen) + 1
    offs = torch.arange(BLOCK_SIZE, device=DEVICE)
    req_to_token = (
        (pages.view(len(prefixes), pages_per_req, 1) * BLOCK_SIZE + offs)
        .view(len(prefixes), -1)[:, :max_len]
        .to(torch.int32)
    )
    k_cache = torch.randn(
        (num_pages + 1) * BLOCK_SIZE, 1, HEAD_DIM, device=DEVICE, generator=gen
    )
    if tied_pages:
        k_pages = k_cache.view(num_pages + 1, BLOCK_SIZE * HEAD_DIM)
        rare = torch.arange(num_pages + 1, device=DEVICE) % 13 == 0
        k_pages.copy_(k_pages[1 + rare.long()])
    rows = len(prefixes) * num_draft
    q = torch.randn(rows, 1, HEAD_DIM, device=DEVICE, generator=gen)
    seq_lens = [p + j + 1 for p in prefixes for j in range(num_draft)]
    return dict(
        q=q.to(torch.bfloat16),
        k_cache=k_cache.to(k_dtype),
        req_to_token=req_to_token,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64, device=DEVICE),
        # graph capture sizes the score columns past the real lengths
        max_seqlen=max_len + 4 * BLOCK_SIZE,
        slot_ids=torch.arange(len(prefixes), device=DEVICE).repeat_interleave(
            num_draft
        ),
    )


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 fused verify indexer")
@NUM_DRAFTS
@pytest.mark.parametrize("k_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 2)])
# Above 32 requests the grid holds two programs per CU.
@pytest.mark.parametrize("num_reqs", [16, 40])
def test_fused_verify_indexer_matches_score_topk(
    num_draft, k_dtype, init_blocks, local_blocks, num_reqs
):
    """The gfx950 fused score + top-k kernels must emit exactly the ids and layout of
    the score kernel + radix top-k, short rows and request splits included."""
    from sglang.kernels.ops.attention.minimax_sparse.decode import (
        verify_topk_gfx950,
    )

    prefixes = [BLOCK_SIZE * (60 + 37 * (r % 16)) - r for r in range(num_reqs)]
    prefixes[3], prefixes[9] = 3 * BLOCK_SIZE - 2, 777  # rows with <= topk blocks
    inputs = _build_paged(prefixes, num_draft=num_draft, k_dtype=k_dtype, seed=7)
    kwargs = dict(
        num_draft=num_draft,
        block_size=BLOCK_SIZE,
        topk=TOPK,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=None,
        q_scale=None,
        k_scale=0.5,
        **inputs,
    )
    assert verify_topk_gfx950.can_fuse_verify_topk(
        inputs["q"], inputs["k_cache"], num_draft, TOPK, local_blocks, 128, 128
    )
    two_pass = verify_topk_idx(**kwargs)
    out = torch.full_like(two_pass, -7)
    fused = verify_topk_idx(**kwargs, page_size=BLOCK_SIZE, topk_idx_out=out)
    assert fused.data_ptr() == out.data_ptr()
    assert torch.equal(fused, two_pass)


def _nvfp4_pool(k_cache):
    """A [slots, 1, D] cache quantized into the NVFP4 index-K pool's page layout."""
    slots, _, dim = k_cache.shape
    payload, scales = nvfp4_kv_quantize(
        k_cache.view(slots, dim).to(torch.bfloat16), torch.ones(1, device=DEVICE)
    )
    pages = slots // BLOCK_SIZE
    return Nvfp4Cache(
        payload=payload.view(pages, BLOCK_SIZE, dim // 2),
        scales=scales.view(pages, BLOCK_SIZE, dim // 16),
    )


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 fused verify indexer")
@NUM_DRAFTS
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 2)])
# Below 8 requests a request takes up to 64 splits, from 8 up to 32.
@pytest.mark.parametrize("num_reqs", [1, 3, 8, 17, 40, 64])
def test_fused_verify_indexer_on_nvfp4_matches_score_topk(
    monkeypatch, num_draft, init_blocks, local_blocks, num_reqs
):
    """On an NVFP4 index-K pool the fused kernels must emit exactly the ids of the
    NVFP4 score kernel + radix top-k, scales folded in the same order."""
    from sglang.kernels.ops.attention.minimax_sparse.decode import (
        verify_topk_gfx950,
    )

    monkeypatch.setattr(verify_topk_gfx950, "VERIFY_FUSED_MIN_REQS_NVFP4", 1)
    prefixes = [BLOCK_SIZE * (60 + 37 * (r % 8)) - r for r in range(num_reqs)]
    if num_reqs > 5:
        prefixes[3], prefixes[5] = 3 * BLOCK_SIZE - 2, 777  # rows with <= topk blocks
    inputs = _build_paged(
        prefixes, num_draft=num_draft, k_dtype=torch.bfloat16, seed=11
    )
    inputs["k_cache"] = _nvfp4_pool(inputs["k_cache"])
    kwargs = dict(
        num_draft=num_draft,
        block_size=BLOCK_SIZE,
        topk=TOPK,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=0.07,
        q_scale=None,
        k_scale=0.5,
        page_size=BLOCK_SIZE,
        **inputs,
    )
    assert verify_topk_gfx950.can_fuse_verify_topk(
        inputs["q"], inputs["k_cache"], num_draft, TOPK, local_blocks, 128, 128
    )
    out = torch.full(
        (1, len(prefixes) * num_draft, TOPK), -7, dtype=torch.int32, device=DEVICE
    )
    fused = verify_topk_idx(**kwargs, topk_idx_out=out)
    monkeypatch.setattr(verify_module, "_use_fused_verify_topk", lambda *a, **kw: False)
    two_pass = verify_topk_idx(**kwargs)
    assert fused.data_ptr() == out.data_ptr()
    assert torch.equal(fused, two_pass)


@pytest.mark.skipif(not is_gfx95_supported(), reason="gfx950 fused verify indexer")
@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize(
    "split_config",
    [
        (32, 1, 4, False),  # one split per request
        (64, 3, 8, True),
        (256, 32, 1, False),
        (512, 32, 1, True),
        (256, 64, 1, True),
    ],
)
def test_fused_verify_ids_do_not_depend_on_the_split(monkeypatch, nvfp4, split_config):
    """However tied blocks split over programs and warps, the fused kernels keep a tie's
    lower block ids, as the score + radix top-k path does."""
    from sglang.kernels.ops.attention.minimax_sparse.decode import (
        verify_topk_gfx950,
    )

    prefixes = [BLOCK_SIZE * (20 + 31 * (r % 7)) - r for r in range(24)]
    inputs = _build_paged(
        prefixes, num_draft=8, k_dtype=torch.bfloat16, seed=5, tied_pages=True
    )
    if nvfp4:
        inputs["k_cache"] = _nvfp4_pool(inputs["k_cache"])
    else:
        inputs["k_cache"] = inputs["k_cache"].to(torch.float8_e4m3fn)
    kwargs = dict(
        num_draft=8,
        block_size=BLOCK_SIZE,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        sm_scale=None,
        q_scale=None,
        k_scale=None,
        page_size=BLOCK_SIZE,
        **inputs,
    )
    assert verify_topk_gfx950.can_fuse_verify_topk(
        inputs["q"], inputs["k_cache"], 8, TOPK, 1, 128, 128
    )
    monkeypatch.setattr(verify_topk_gfx950, "_split_config", lambda *a: split_config)
    fused = verify_topk_idx(**kwargs)
    monkeypatch.setattr(verify_module, "_use_fused_verify_topk", lambda *a, **kw: False)
    assert torch.equal(fused, verify_topk_idx(**kwargs))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
