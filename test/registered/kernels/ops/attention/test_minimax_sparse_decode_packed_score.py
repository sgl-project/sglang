"""Packed EAGLE-verify index scoring: block scores and top-k equal the per-row kernel's
for max scores and finite K, eagerly and under CUDA-graph replay."""

import sys

import pytest
import torch
import triton

import sglang.kernels.ops.attention.minimax_decode_topk as topk_ops
from sglang.kernels.ops.attention.minimax_sparse.decode import (
    flash_with_topk_idx as fwti,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# SM90 scores pages that hold whole blocks, as every case here, with its paged kernel.
pytestmark = pytest.mark.skipif(
    get_platform().is_sm90, reason="SM90 keeps the paged per-row kernel"
)

HD, TOPK, PAGE, NDT, BLK = 128, 16, 128, 4, 128
MAX_CONTEXT = 256 * 1024

PREFIXES = {
    # rows 1001-1004 stay at <= top-k blocks; rows 5120 | 5121 straddle a block edge
    "edge": [1000, 5118],
    # rows 2047-2050: the group has top-k + 1 blocks, its first two rows only top-k
    "topk_edge": [TOPK * BLK - 2],
    "long": [200_000, 77_777, 131_071],
    # the last two rows run past the context length and are clamped, as in the backend
    "clamped": [MAX_CONTEXT - 2, 33_000],
}
DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp8_k": (torch.float8_e4m3fn, torch.bfloat16),  # widening: bf16 q, fp8 index K
    "fp8_qk": (torch.float8_e4m3fn, torch.float8_e4m3fn),  # fp8 attn-GEMM mode
}
# Pinned to keep the file cheap, and unlike the packed kernel's fixed tile below;
# max scores must not depend on the tile, as both kernels lower to the same 16-row MMA.
REFERENCE_CONFIG = triton.Config({"BLOCK_SIZE_N": 256}, num_warps=8, num_stages=3)
PACKED_TILE_CONFIG = triton.Config({"BLOCK_SIZE_N": 128}, num_warps=4, num_stages=2)


@pytest.fixture(autouse=True)
def pinned_reference(monkeypatch):
    monkeypatch.setattr(fwti._decode_score_kernel.fn, "configs", [REFERENCE_CONFIG])


@pytest.fixture(autouse=True)
def nan_buffers(monkeypatch):
    # Unwritten scores then cannot match the per-row run's freed memory.
    orig = torch.empty

    def empty(*args, **kwargs):
        out = orig(*args, **kwargs)
        if out.dtype in (torch.float32, torch.bfloat16, torch.float16):
            out.fill_(float("nan"))
        return out

    monkeypatch.setattr(torch, "empty", empty)


class _Count:
    def __init__(self, kernel):
        self.kernel, self.launches = kernel, 0

    def __getitem__(self, grid):
        self.launches += 1
        return self.kernel[grid]


class _Case:
    def __init__(
        self,
        prefixes,
        gqa,
        dtypes,
        *,
        ndt=NDT,
        blk=BLK,
        init=0,
        local=1,
        max_context=MAX_CONTEXT,
        q_row_pad=0,
        seed=0,
    ):
        torch.manual_seed(seed)
        k_dtype, q_dtype = dtypes
        self.ndt, self.blk, self.init, self.local = ndt, blk, init, local
        self.num_reqs = len(prefixes)
        width = max_context + 4 + ndt  # ReqToTokenPool width with speculative decoding
        pages_per_req = (width + PAGE - 1) // PAGE
        pages = torch.randperm(self.num_reqs * pages_per_req, device="cuda")
        slots = pages.view(self.num_reqs, pages_per_req, 1) * PAGE + torch.arange(
            PAGE, device="cuda"
        )
        self.req_to_token = slots.reshape(self.num_reqs, -1)[:, :width].to(torch.int32)
        num_slots = self.num_reqs * pages_per_req * PAGE
        self.k_cache = torch.randn(num_slots, 1, HD, device="cuda").to(k_dtype)
        self.k_scale = 0.75 if k_dtype != torch.bfloat16 else None
        rows = self.num_reqs * ndt
        # q_row_pad > 0 slices q out of wider rows, as the fused index projection does.
        q = torch.randn(rows, gqa * HD + q_row_pad, dtype=torch.bfloat16, device="cuda")
        self.q = q[:, : gqa * HD].view(rows, gqa, HD).to(q_dtype)
        prefix = torch.tensor(prefixes, dtype=torch.int64, device="cuda")
        offsets = torch.arange(1, ndt + 1, dtype=torch.int64, device="cuda")
        self.seq_lens = (prefix[:, None] + offsets).reshape(-1).clamp(max=max_context)
        self.slot_ids = torch.arange(
            self.num_reqs, dtype=torch.int64, device="cuda"
        ).repeat_interleave(ndt)
        self.max_seqlen = int(self.seq_lens.max())


def _decode(case, packed, *, dense, score_type="max", max_seqlen=None):
    return fwti.flash_decode_with_topk_idx(
        q=case.q,
        sink=None,
        k_cache=case.k_cache,
        v_cache=None,
        req_to_token=case.req_to_token,
        seq_lens=case.seq_lens,
        max_seqlen=max_seqlen or case.max_seqlen,
        slot_ids=case.slot_ids,
        block_size=case.blk,
        topk=TOPK,
        init_blocks=case.init,
        local_blocks=case.local,
        score_type=score_type,
        disable_index_value=True,
        use_dense_main_attn=dense,
        page_size=PAGE,
        k_scale=case.k_scale,
        packed_queries=case.ndt if packed else 1,
    )


def _traced(case, packed, **kwargs):
    seen = {}
    counter = _Count(fwti._decode_score_packed_kernel)
    with pytest.MonkeyPatch.context() as mp:
        for name, orig in (
            ("minimax_decode_topk", topk_ops.minimax_decode_topk),
            ("minimax_decode_topk_page_table", topk_ops.minimax_decode_topk_page_table),
        ):

            def spy(score, *args, _orig=orig, **kw):
                seen["score"] = score.clone()
                return _orig(score, *args, **kw)

            mp.setattr(topk_ops, name, spy)
        mp.setattr(fwti, "_decode_score_packed_kernel", counter)
        out = _decode(case, packed, **kwargs)
    return out, seen.get("score"), counter.launches


def _assert_rows_equal(got, want, case, skip_trivial):
    blocks = (case.seq_lens + case.blk - 1) // case.blk
    for r, nb in enumerate(blocks.tolist()):
        if skip_trivial and nb <= TOPK:
            continue  # neither kernel needs to write a row the consumer never reads
        assert torch.equal(got[:, r, :nb], want[:, r, :nb]), f"row {r} ({nb} blocks)"


def _selected_blocks(idx, r, nb, case, dense):
    if not dense:  # block ids [1, rows, topk]: front-packed, -1 padded
        row = idx[0, r]
        return row[row >= 0]
    # page table [rows, pages]: each block's first page, mapped back to its block
    req = case.req_to_token[int(case.slot_ids[r])]
    page_to_block = {int(p) // PAGE: b for b, p in enumerate(req[:: case.blk].tolist())}
    ppb = case.blk // PAGE
    pages = idx[r, : min(nb, TOPK) * ppb : ppb].tolist()
    return torch.tensor([page_to_block[p] for p in pages], device=idx.device)


def _assert_selection_equal(got, want, score, case, dense):
    blocks = (case.seq_lens + case.blk - 1) // case.blk
    for r, nb in enumerate(blocks.tolist()):
        g = _selected_blocks(got, r, nb, case, dense)
        w = _selected_blocks(want, r, nb, case, dense)
        if nb <= TOPK:  # every block is selected from seq_lens, scores unread
            assert torch.equal(g, w), f"row {r}"
            continue
        # The radix top-k places final-round ties by atomic order, so compare the
        # selected scores: tied blocks may swap, the score multiset may not.
        sg = score[0, r, g.long()].sort().values
        sw = score[0, r, w.long()].sort().values
        assert g.numel() == w.numel() and torch.equal(sg, sw), f"row {r}"


def _check_pair(case, dense):
    with envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.override(True):
        (_, want_idx, want_len), want_score, per_row_launches = _traced(
            case, False, dense=dense
        )
        (_, got_idx, got_len), got_score, packed_launches = _traced(
            case, True, dense=dense
        )
    assert per_row_launches == 0 and packed_launches == 1
    _assert_rows_equal(got_score, want_score, case, skip_trivial=True)
    _assert_selection_equal(got_idx, want_idx, want_score, case, dense)
    assert (got_len is None) == (want_len is None)
    if got_len is not None:
        assert torch.equal(got_len, want_len)


@pytest.mark.parametrize(
    "prefixes,gqa,dtypes,dense,q_row_pad",
    [
        pytest.param("edge", 1, "bf16", False, 0, id="edge"),
        pytest.param("topk_edge", 1, "bf16", False, 0, id="topk_edge"),
        pytest.param("long", 1, "bf16", False, 0, id="long"),
        pytest.param("clamped", 1, "bf16", False, 0, id="clamped"),
        pytest.param("long", 4, "bf16", False, HD, id="long-gqa4-strided_q"),
        pytest.param("edge", 1, "fp8_k", False, 0, id="edge-fp8_k"),
        pytest.param("long", 1, "fp8_qk", False, 0, id="long-fp8_qk"),
        pytest.param("clamped", 1, "bf16", True, 0, id="clamped-page_table"),
    ],
)
def test_packed_matches_per_row(prefixes, gqa, dtypes, dense, q_row_pad):
    case = _Case(PREFIXES[prefixes], gqa, DTYPES[dtypes], q_row_pad=q_row_pad)
    _check_pair(case, dense)


@pytest.mark.parametrize("init,local", [(0, 0), (0, 1), (2, 0), (2, 1)])
@pytest.mark.parametrize("blk", [64, 128])
def test_masks_without_local_cover(init, local, blk):
    """With local_blocks 0 a row's partial last block rests on its own length mask."""
    # Every row has more than top-k blocks, so the consumer reads every score.
    prefixes = [2 * TOPK * blk + 5, 3 * TOPK * blk - 2, 5 * TOPK * blk + blk - 3]
    case = _Case(
        prefixes, 1, DTYPES["bf16"], blk=blk, init=init, local=local, max_context=16384
    )
    # q >= 0 and large K on the later drafts expose any row that reads past its length.
    case.q = case.q.abs()
    for g, prefix in enumerate(prefixes):
        case.k_cache[case.req_to_token[g, prefix + 1 : prefix + NDT].long()] = 4.0
    _check_pair(case, dense=False)


def test_twelve_row_tile():
    """ndt 3 with 4 index heads fills 12 of the 16 dot-tile rows."""
    _check_pair(_Case(PREFIXES["long"], 4, DTYPES["bf16"], ndt=3), dense=False)


@pytest.mark.parametrize("score_type", ["max", "lse"])
def test_packed_matches_per_row_triton_topk(monkeypatch, score_type):
    """The Triton two-stage top-k reads every row's scores, trivial rows included."""
    if score_type == "lse":
        # An lse block sum follows the MMA layout: give the reference the packed tile's.
        monkeypatch.setattr(
            fwti._decode_score_kernel.fn, "configs", [PACKED_TILE_CONFIG]
        )
    case = _Case(PREFIXES["edge"] + PREFIXES["topk_edge"], 1, DTYPES["bf16"])
    with envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.override(False):
        (_, want, _), _, _ = _traced(case, False, dense=False, score_type=score_type)
        (_, got, _), _, launches = _traced(
            case, True, dense=False, score_type=score_type
        )
    assert launches == 1
    assert torch.equal(got, want)


def test_unpackable_batch_falls_back():
    """A batch that is not whole groups of draft rows keeps the per-row kernel."""
    case = _Case(PREFIXES["long"], 1, DTYPES["bf16"])
    case.q, case.seq_lens, case.slot_ids = (
        t[:-1] for t in (case.q, case.seq_lens, case.slot_ids)
    )
    (_, want, _), want_score, _ = _traced(case, False, dense=False)
    (_, got, _), _, launches = _traced(case, True, dense=False)
    assert launches == 0
    _assert_selection_equal(got, want, want_score, case, dense=False)


@pytest.mark.parametrize("dense", [False, True], ids=["block_ids", "page_table"])
def test_graph_replay_at_longer_context(dense):
    """Captured on short rows, replayed at the real lengths, bounded by the context."""
    case = _Case(PREFIXES["long"] + PREFIXES["clamped"], 1, DTYPES["bf16"])
    real = case.seq_lens.clone()
    case.seq_lens.fill_(1)
    case.seq_lens += torch.arange(1, NDT + 1, device="cuda").repeat(case.num_reqs)
    with envs.SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX.override(True):
        _decode(case, packed=True, dense=dense, max_seqlen=MAX_CONTEXT)  # JIT warm-up
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # the score clone is captured too, so it holds the replayed scores
            out, got_score, launches = _traced(
                case, True, dense=dense, max_seqlen=MAX_CONTEXT
            )
        case.seq_lens.copy_(real)
        graph.replay()
        torch.cuda.synchronize()
        want, want_score, _ = _traced(case, False, dense=dense, max_seqlen=MAX_CONTEXT)
    assert launches == 1
    _assert_rows_equal(got_score, want_score, case, skip_trivial=True)
    _assert_selection_equal(out[1], want[1], want_score, case, dense)
    if dense:  # the forced local block holds the partial tail, so lengths never tie
        assert torch.equal(out[2], want[2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
