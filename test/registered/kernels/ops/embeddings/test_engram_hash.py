"""Compare decode and multimodal extend hashes with a per-token reference."""

import pytest
import torch

from sglang.kernels.ops.embeddings.engram_hash import (
    MODE_DECODE,
    MODE_EXTEND,
    engram_hash_ids,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="engram_hash_ids requires CUDA",
)

VOCAB = 1000
COMPRESSED = 300
N = 4
L = 2
H = 8
COLS = (N - 1) * H
IMAGE = 7
MM_PAD_SHIFT = 1_000_000
# (N - 1) * H primes per layer, all above COMPRESSED - 1, in ascending order.
PRIMES = [
    1009,
    1013,
    1019,
    1021,
    1031,
    1033,
    1039,
    1049,
    1051,
    1061,
    1063,
    1069,
    1087,
    1091,
    1093,
    1097,
    1103,
    1109,
    1117,
    1123,
    1129,
    1151,
    1153,
    1163,
    1171,
    1181,
    1187,
    1193,
    1201,
    1213,
    1217,
    1223,
    1229,
    1231,
    1237,
    1249,
    1259,
    1277,
    1279,
    1283,
    1289,
    1291,
    1297,
    1301,
    1303,
    1307,
    1319,
    1321,
]


def _tables(seed=0):
    g = torch.Generator().manual_seed(seed)
    token_map = torch.randint(0, COMPRESSED, (VOCAB,), generator=g, dtype=torch.int64)
    bound = (2**63 - 1) // COMPRESSED // 2
    multipliers = (
        torch.randint(0, bound, (L, N), generator=g, dtype=torch.int64) * 2 + 1
    )
    primes = torch.tensor(PRIMES, dtype=torch.int64).view(L, N - 1, H)
    flat = primes.view(L, COLS)
    offsets = torch.cumsum(torch.cat([flat.new_zeros(L, 1), flat[:, :-1]], 1), 1)
    return token_map, multipliers, primes, offsets


def naive(
    ids,
    pos,
    *,
    mode,
    history,
    token_map,
    multipliers,
    primes,
    offsets,
    pad_id,
    num_real,
    req_slots,
    block,
    row,
    starts,
    image_token_id,
):
    ids, pos, hist, tm = (
        ids.tolist(),
        pos.tolist(),
        history.tolist(),
        token_map.tolist(),
    )
    mult, pr, ofs = multipliers.tolist(), primes.tolist(), offsets.tolist()
    slots = req_slots.tolist() if req_slots is not None else None
    row = row.tolist() if row is not None else None
    starts = starts.tolist() if starts is not None else None
    T = len(ids)
    out = torch.zeros(T, L, COLS, dtype=torch.int64)
    toks = torch.zeros(T, N, dtype=torch.int32)
    for t in range(T):
        comp = [pad_id] * N
        if t < num_real:
            if mode == MODE_DECODE:
                r, off = t, 0
            else:
                r = row[t]
                off = t - starts[r]
            hrow = slots[r] if slots is not None else r
            blocked = False
            for s in range(N):
                tok = ids[t - s] if s <= off else hist[hrow][N - 2 - (s - off - 1)]
                if image_token_id is not None:
                    if tok >= MM_PAD_SHIFT:
                        tok = image_token_id
                    blocked = blocked or tok == image_token_id
                blocked = blocked or pos[t] < s
                toks[t, s] = tok
                comp[s] = pad_id if blocked else tm[tok]
        for l in range(L):
            rolling = comp[0] * mult[l][0]
            for i in range(1, N):
                rolling ^= comp[i] * mult[l][i]
                for h in range(H):
                    out[t, l, (i - 1) * H + h] = (
                        rolling % pr[l][i - 1][h] + ofs[l][(i - 1) * H + h]
                    )
    return out, toks


def _check(ids, pos, **kw):
    dev = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in kw.items()}
    got_ids, got_toks = engram_hash_ids(
        ids.cuda(), pos.cuda(), mm_pad_shift=MM_PAD_SHIFT, **dev
    )
    want_ids, want_toks = naive(ids, pos, **kw)
    assert got_toks.dtype == torch.int32 and got_ids.dtype == torch.int64
    assert torch.equal(got_toks.cpu(), want_toks), (got_toks.cpu(), want_toks)
    assert torch.equal(got_ids.cpu(), want_ids)
    return got_ids


def _common(seed=0, image=False):
    token_map, multipliers, primes, offsets = _tables(seed)
    return dict(
        token_map=token_map,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
        pad_id=int(token_map[2]),
        image_token_id=IMAGE if image else None,
    )


def test_decode_reads_history_through_req_slots():
    g = torch.Generator().manual_seed(1)
    bs, slots_total = 7, 12
    ids = torch.randint(0, VOCAB, (bs,), generator=g)
    pos = torch.randint(0, 50, (bs,), generator=g)
    pos[0] = 0  # sequence start: only the token itself survives
    history = torch.randint(
        0, VOCAB, (slots_total + 1, N - 1), generator=g, dtype=torch.int32
    )
    req_slots = torch.randperm(slots_total, generator=g)[:bs]
    _check(
        ids,
        pos,
        mode=MODE_DECODE,
        history=history,
        num_real=bs,
        req_slots=req_slots,
        block=1,
        row=None,
        starts=None,
        **_common(),
    )


def test_decode_commit_writes_live_rows_only():
    # With commit_out_loc the kernel writes each live request's new history row
    # (token plus its n - 2 newest predecessors, oldest first); a padded row
    # (out_cache_loc 0) writes nothing. Hash ids still read the old history.
    g = torch.Generator().manual_seed(6)
    bs, slots_total = 5, 9
    ids = torch.randint(0, VOCAB, (bs,), generator=g)
    pos = torch.randint(3, 50, (bs,), generator=g)
    history = torch.randint(
        0, VOCAB, (slots_total + 1, N - 1), generator=g, dtype=torch.int32
    )
    req_slots = torch.tensor([7, 2, 0, 5, 2])  # the padded row 4 aliases live row 1
    out_loc = torch.tensor([11, 12, 13, 14, 0])
    kw = _common()
    before = history.clone()
    want_ids, want_toks = naive(
        ids,
        pos,
        mode=MODE_DECODE,
        history=history,
        num_real=bs,
        req_slots=req_slots,
        block=1,
        row=None,
        starts=None,
        **kw,
    )
    dev = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in kw.items()}
    hist_dev = history.cuda()
    got_ids, got_toks = engram_hash_ids(
        ids.cuda(),
        pos.cuda(),
        mode=MODE_DECODE,
        history=hist_dev,
        num_real=bs,
        req_slots=req_slots.cuda(),
        block=1,
        row=None,
        starts=None,
        mm_pad_shift=MM_PAD_SHIFT,
        commit_out_loc=out_loc.cuda(),
        **dev,
    )
    assert torch.equal(got_ids.cpu(), want_ids) and torch.equal(
        got_toks.cpu(), want_toks
    )
    expect = before.clone()
    for t in range(bs - 1):  # row 4 is padding
        expect[req_slots[t]] = want_toks[t, : N - 1].flip(0)
    assert torch.equal(hist_dev.cpu(), expect)


def test_decode_commit_across_programs():
    # 100 rows span four 32-token programs; live rows hold distinct slots and the
    # padded rows alias live slots, so cross-program read/write order matters.
    g = torch.Generator().manual_seed(8)
    bs, slots_total, live = 100, 160, 90
    ids = torch.randint(0, VOCAB, (bs,), generator=g)
    pos = torch.randint(3, 50, (bs,), generator=g)
    history = torch.randint(
        0, VOCAB, (slots_total + 1, N - 1), generator=g, dtype=torch.int32
    )
    live_slots = torch.randperm(slots_total, generator=g)[:live]
    req_slots = torch.cat([live_slots, live_slots[: bs - live]])  # padded rows alias
    out_loc = torch.cat(
        [torch.arange(1, live + 1), torch.zeros(bs - live, dtype=torch.long)]
    )
    kw = _common()
    before = history.clone()
    dev = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in kw.items()}
    hist_dev = history.cuda()
    got_ids, got_toks = engram_hash_ids(
        ids.cuda(),
        pos.cuda(),
        mode=MODE_DECODE,
        history=hist_dev,
        num_real=bs,
        req_slots=req_slots.cuda(),
        block=1,
        row=None,
        starts=None,
        mm_pad_shift=MM_PAD_SHIFT,
        commit_out_loc=out_loc.cuda(),
        **dev,
    )
    want_ids, want_toks = naive(
        ids,
        pos,
        mode=MODE_DECODE,
        history=before,
        num_real=bs,
        req_slots=req_slots,
        block=1,
        row=None,
        starts=None,
        **kw,
    )
    # Live rows read the old history and hash identically to the oracle.
    assert torch.equal(got_ids[:live].cpu(), want_ids[:live])
    assert torch.equal(got_toks[:live].cpu(), want_toks[:live])
    expect = before.clone()
    for t in range(live):
        expect[req_slots[t]] = want_toks[t, : N - 1].flip(0)
    assert torch.equal(hist_dev.cpu(), expect)


def _extend_batch(g, lens, padded_to):
    starts = torch.cumsum(torch.tensor([0] + lens[:-1]), 0)
    row = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
    num_real = int(row.numel())
    ids = torch.randint(0, VOCAB, (padded_to,), generator=g)
    first_pos = torch.tensor([0, 17, 0, 5])[: len(lens)]
    pos = torch.zeros(padded_to, dtype=torch.int64)
    pos[:num_real] = first_pos[row] + (torch.arange(num_real) - starts[row])
    return ids, pos, row, starts, num_real


def test_extend_with_scheduler_history_and_image_spans():
    g = torch.Generator().manual_seed(4)
    lens = [6, 0, 3, 9]
    ids, pos, row, starts, num_real = _extend_batch(g, lens, padded_to=18)
    ids[2] = IMAGE  # inside request 0: shifts reaching it and beyond are PAD
    ids[12] = IMAGE
    history = torch.randint(0, VOCAB, (4, N - 1), generator=g, dtype=torch.int32)
    history[3, 1] = MM_PAD_SHIFT + 5  # multimodal pad id in the scheduler's history
    _check(
        ids,
        pos,
        mode=MODE_EXTEND,
        history=history,
        num_real=num_real,
        req_slots=None,
        block=1,
        row=row,
        starts=starts,
        **_common(image=True),
    )
