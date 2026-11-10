"""SM70 k-pool decode-update and q act_quant kernels must match their torch
reference bitwise (page bytes, scales, tails), and the target-verify write must
match the decode update run token by token.

Not a CI test:

  CUDA_VISIBLE_DEVICES=0 python test/manual/layers/test_kpool_sm70.py
"""

import torch

import sglang.kernels.ops.attention.kpool_sm70 as ks
from sglang.kernels.ops.attention.kpool_sm70 import (
    _compress,
    _two_pass_softmax,
    act_quant_sm70,
    kpool_decode_update_and_maybe_write_cache_sm70,
    kpool_write_tail_and_maybe_compress_sm70,
)
from sglang.srt.layers.attention.dsa.kpool_fp8_index import kpool_max_closed_pools


def act_quant_torch(x, round_scale):
    groups = x.size(-1) // 128
    xg = x.float().reshape(-1, groups, 128)
    amax = xg.abs().amax(dim=-1).clamp(min=1e-4)
    if round_scale:
        ratio = (amax / 448.0).clamp(min=torch.finfo(torch.float32).tiny)
        mant, exp = torch.frexp(ratio)
        scale = torch.ldexp(torch.ones_like(ratio), torch.where(mant == 0.5, exp - 1, exp))
    else:
        scale = amax / 448.0
    q = (xg / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return q.reshape(x.shape).to(torch.float8_e4m3fn), scale.reshape(*x.shape[:-1], groups)


def decode_update_torch(pool, buf, tail_k, tail_score, key, slot_score, ape, block_tables,
                        req_pool_indices, positions, seq_lens, out_cache_loc, round_scale):
    """The torch fallback the kernel replaced, verbatim."""
    n = key.shape[0]
    pool_size = pool.index_kpool
    tail_size = tail_k.shape[1]
    slots_per_page = pool.slots_per_page
    req_raw = req_pool_indices.to(torch.long)
    pos = positions.to(torch.long)
    seq_len = seq_lens.to(torch.long)
    cache_loc = out_cache_loc.to(torch.long)
    valid = (
        (req_raw >= 0)
        & (req_raw < tail_k.shape[0])
        & (cache_loc != 0)
        & (pos >= 0)
        & (pos < seq_len)
    )
    req = req_raw.clamp(0, tail_k.shape[0] - 1)
    safe_pos = pos.clamp(min=0)
    slot = torch.remainder(safe_pos, pool_size)
    closing = valid & (slot == pool_size - 1)
    start = safe_pos - slot
    keys = []
    scores = []
    for pool_slot in range(pool_size):
        phys = torch.remainder(start + pool_slot, tail_size)
        if pool_slot == pool_size - 1:
            k = key.float()
            s = slot_score.float()
        else:
            k = tail_k[req, phys].float()
            s = tail_score[req, phys].float()
        keys.append(k)
        scores.append(s + ape[pool_slot].float())
    pooled = _two_pass_softmax(torch.stack(keys, 0), torch.stack(scores, 0))
    quantized, scale = _compress(pooled, round_scale)
    pool_id = torch.div(safe_pos, pool_size, rounding_mode="floor")
    group = torch.div(pool_id, slots_per_page, rounding_mode="floor")
    table_col = (group * pool_size).clamp(0, block_tables.shape[1] - 1)
    page_id = block_tables[torch.arange(n, device=key.device), table_col].to(torch.long)
    offset = torch.remainder(pool_id, slots_per_page)
    loc = page_id * slots_per_page + offset
    page = torch.div(loc, slots_per_page, rounding_mode="floor")
    page_slot = loc - page * slots_per_page
    k_bytes = slots_per_page * key.shape[-1]
    n_pages = buf.shape[0]
    k_view = (
        buf[:, :k_bytes]
        .view(torch.float8_e4m3fn)
        .view(n_pages, slots_per_page, key.shape[-1])
    )
    old_k = k_view[page, page_slot].float()
    blended = torch.where(closing.unsqueeze(-1), quantized.float(), old_k)
    k_view[page, page_slot] = blended.to(torch.float8_e4m3fn)
    scale_view = buf.view(torch.float32)
    scale_idx = k_bytes // 4 + page_slot
    old_scale = scale_view[page, scale_idx]
    scale_view[page, scale_idx] = torch.where(closing, scale.float(), old_scale)

    phys = torch.remainder(safe_pos, tail_size)
    old_tail_k = tail_k[req, phys].float()
    new_tail_k = torch.where(valid.unsqueeze(-1), key.float(), old_tail_k)
    tail_k[req, phys] = new_tail_k.to(tail_k.dtype)
    old_tail_s = tail_score[req, phys].float()
    new_tail_s = torch.where(valid.unsqueeze(-1), slot_score.float(), old_tail_s)
    tail_score[req, phys] = new_tail_s.to(tail_score.dtype)


class Pool:
    index_kpool = 4
    slots_per_page = 64


def same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def decode_case(name, g, n, dtype, round_scale, key_scale=1.0, score_scale=1.0, zero_key=False):
    dev = "cuda"
    pool = Pool()
    tail = 4 + 4
    reqs = n + 3
    # Distinct pages and requests per row: the fallback's gather/scatter
    # write-back races when two rows alias a destination.
    pages = n * 16
    buf = torch.randint(0, 256, (pages, 64 * (128 + 4)), dtype=torch.uint8, device=dev, generator=g)
    tail_k = (torch.randn(reqs, tail, 128, device=dev, generator=g) * key_scale).bfloat16()
    tail_s = (torch.randn(reqs, tail, 128, device=dev, generator=g) * score_scale).bfloat16()
    key = (torch.randn(n, 128, device=dev, generator=g) * key_scale).to(dtype)
    if zero_key:
        key.zero_()
        tail_k.zero_()
    score = (torch.randn(n, 128, device=dev, generator=g) * score_scale).to(dtype)
    ape = torch.randn(4, 128, device=dev, generator=g)
    bt = torch.randperm(pages, device=dev, generator=g).reshape(n, 16).int()
    # Request 0 stays free for the invalid row below, which clamps onto it.
    req = torch.randperm(reqs - 1, device=dev, generator=g)[:n] + 1
    pos = torch.randint(0, 4096, (n,), device=dev, generator=g)
    pos[: n // 2] = pos[: n // 2] // 4 * 4 + 3  # half the rows close a pool
    seq = (pos + 1).int()
    loc = torch.randint(1, 1000, (n,), device=dev, generator=g)
    if n > 3:
        req[1] = -1  # invalid rows: bad req, pos past seq, cache_loc 0
        seq[2] = pos[2].int()
        loc[3] = 0
    args = [key, score, ape, bt, req, pos, seq, loc, round_scale]
    ref = [buf.clone(), tail_k.clone(), tail_s.clone()]
    decode_update_torch(pool, ref[0], ref[1], ref[2], *args)
    got = [buf.clone(), tail_k.clone(), tail_s.clone()]
    kpool_decode_update_and_maybe_write_cache_sm70(pool, got[0], got[1], got[2], *args)
    ok = all(same(a, b) for a, b in zip(ref, got))
    changed = int((ref[0] != buf).any(-1).sum())
    print(f"{name:40s} {'equal' if ok else 'DIFFER'}  (pages written {changed})")
    if not ok:
        for label, a, b in zip(("buf", "tail_k", "tail_s"), ref, got):
            diff = (a.view(torch.uint8) != b.view(torch.uint8)).nonzero()
            print(f"  {label}: {diff.shape[0]} bytes differ, first {diff[:4].tolist()}")
    return ok


def alias_case():
    """An invalid row clamps onto a valid row's request; the valid key must land."""
    dev = "cuda"
    tail_k = torch.zeros(2, 8, 128, dtype=torch.bfloat16, device=dev)
    tail_s = torch.zeros_like(tail_k)
    buf = torch.zeros(4, 64 * 132, dtype=torch.uint8, device=dev)
    key = torch.randn(2, 128, device=dev).half()
    score = torch.randn(2, 128, device=dev).half()
    req = torch.tensor([0, -1], device=dev)
    pos = torch.tensor([5, 5], device=dev)
    seq = torch.tensor([6, 6], dtype=torch.int32, device=dev)
    loc = torch.tensor([1, 1], device=dev)
    bt = torch.zeros(2, 4, dtype=torch.int32, device=dev)
    kpool_decode_update_and_maybe_write_cache_sm70(
        Pool(), buf, tail_k, tail_s, key, score, torch.zeros(4, 128, device=dev), bt, req, pos, seq, loc, True
    )
    ok = torch.equal(tail_k[0, 5], key[0].bfloat16()) and torch.equal(tail_s[0, 5], score[0].bfloat16())
    print(f"{'decode aliasing invalid row':40s} {'valid key kept' if ok else 'CLOBBERED'}")
    return ok


def verify_case(name, g, bs, n, dtype, round_scale, gate=None, pad_row=None):
    """One verify call over n drafts == n decode steps of one row each.

    gate[b] < n compresses only the pools closed by the first gate[b] drafts;
    the tail still takes all n. pad_row has out_cache_loc 0 and writes nothing.
    """
    dev = "cuda"
    pool = Pool()
    kp, spp, tail, cols = 4, 64, 4 + 4, 64
    reqs = bs + 2
    pages = bs * cols
    buf = torch.randint(0, 256, (pages, spp * (128 + 4)), dtype=torch.uint8, device=dev, generator=g)
    tail_k = torch.randn(reqs, tail, 128, device=dev, generator=g).bfloat16()
    tail_s = torch.randn(reqs, tail, 128, device=dev, generator=g).bfloat16()
    key = torch.randn(bs * n, 128, device=dev, generator=g).to(dtype)
    score = torch.randn(bs * n, 128, device=dev, generator=g).to(dtype)
    ape = torch.randn(kp, 128, device=dev, generator=g)
    bt = torch.randperm(pages, device=dev, generator=g).reshape(bs, cols).int()
    req = torch.randperm(reqs, device=dev, generator=g)[:bs]
    ws = torch.randint(0, 3000, (bs,), device=dev, generator=g)
    ws[: min(bs, 4)] = ws[: min(bs, 4)] // 4 * 4 + torch.arange(min(bs, 4), device=dev)
    loc = torch.randint(1, 1000, (bs * n,), device=dev, generator=g)
    if pad_row is not None:
        loc[pad_row * n] = 0
    closed = kpool_max_closed_pools(n, kp)
    write_loc = torch.zeros(bs, closed, dtype=torch.int64, device=dev)
    for b in range(bs):
        for c in range(closed):
            pool_id = int(ws[b]) // kp + c
            col = min((pool_id // spp) * kp, cols - 1)
            write_loc[b, c] = int(bt[b, col]) * spp + pool_id % spp

    ref = [buf.clone(), tail_k.clone(), tail_s.clone()]
    for b in range(bs):
        if pad_row == b:
            continue
        g_n = n if gate is None else int(gate[b])
        for i in range(n):
            row = b * n + i
            pos = ws[b : b + 1] + i
            if i < g_n:
                kpool_decode_update_and_maybe_write_cache_sm70(
                    pool, ref[0], ref[1], ref[2], key[row : row + 1], score[row : row + 1], ape,
                    bt[b : b + 1], req[b : b + 1], pos, (pos + 1).int(), loc[row : row + 1], round_scale,
                )
            else:
                phys = int(pos) % tail
                ref[1][req[b], phys] = key[row].bfloat16()
                ref[2][req[b], phys] = score[row].bfloat16()

    got = [buf.clone(), tail_k.clone(), tail_s.clone()]
    kpool_write_tail_and_maybe_compress_sm70(
        pool, got[0], key, score, got[1], got[2], ape, req, ws.int(), (ws // kp * kp).int(),
        write_loc, loc, n, round_scale, gate,
    )
    ok = all(same(a, b) for a, b in zip(ref, got))
    changed = int((ref[0] != buf).any(-1).sum())
    print(f"{name:40s} {'equal' if ok else 'DIFFER'}  (pages written {changed})")
    if not ok:
        for label, a, b in zip(("buf", "tail_k", "tail_s"), ref, got):
            diff = (a.view(torch.uint8) != b.view(torch.uint8)).nonzero()
            print(f"  {label}: {diff.shape[0]} bytes differ, first {diff[:4].tolist()}")
    return ok


def main() -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    ok = True
    for dtype in (torch.float16, torch.bfloat16):
        for round_scale in (True, False):
            for n in (1, 2, 7, 64, 512):
                ok &= decode_case(f"decode {str(dtype)[6:]} round={round_scale} n={n}", g, n, dtype, round_scale)
    ok &= decode_case("decode large scores", g, 64, torch.float16, True, score_scale=40.0)
    ok &= decode_case("decode tiny keys", g, 64, torch.float16, True, key_scale=1e-6)
    ok &= decode_case("decode zero keys", g, 64, torch.float16, True, zero_key=True)
    ok &= decode_case("decode large keys", g, 64, torch.float16, False, key_scale=3e4)

    ok &= alias_case()

    for dtype in (torch.float16, torch.bfloat16):
        for round_scale in (True, False):
            for n in (1, 2, 3, 4, 5):
                ok &= verify_case(f"verify {str(dtype)[6:]} round={round_scale} n={n}", g, 6, n, dtype, round_scale)
    gate = torch.tensor([1, 4, 2, 0, 3, 4], dtype=torch.int32, device="cuda")
    ok &= verify_case("verify gated effective_n", g, 6, 4, torch.float16, True, gate=gate)
    ok &= verify_case("verify padded row", g, 6, 4, torch.float16, True, pad_row=2)

    for dtype in (torch.float16, torch.bfloat16):
        for round_scale in (True, False):
            for shape in ((1, 64, 128), (3, 64, 128), (257, 32, 256)):
                x = (torch.randn(*shape, device="cuda", generator=g) * 3).to(dtype)
                x[0, 0] = 0
                x[-1, -1, :7] = 1e-7
                ry, rs = act_quant_torch(x, round_scale)
                y, s = act_quant_sm70(x, round_scale)
                good = same(ry, y) and same(rs, s)
                ok &= good
                print(f"act_quant {str(dtype)[6:]} round={round_scale} {shape}: {'equal' if good else 'DIFFER'}")
    assert ok, "SM70 k-pool kernels differ from the torch reference"


if __name__ == "__main__":
    main()
