"""Inkling 2 Hopper decode attention: split-KV partials against an fp64 reference."""

import sys
import types

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires a Hopper or newer GPU",
)

EXTENT, CLIP, HEAD_DIM = 1024, 22.627416997969522, 128
SCALE = HEAD_DIM**-0.5


def _reference_fp64(q, kc, vc, page_table, seqused, rel, scale, window):
    """Float64 reference: the last qlen positions of each sequence attend causally,
    with rel[..., dist] added below the extent and an optional left window.
    Returns (out, lse) per token and head."""
    batch = seqused.shape[0]
    tokens, heads, d = q.shape
    qlen = tokens // batch
    page, heads_k = kc.shape[1], kc.shape[2]
    group = heads // heads_k
    extent = rel.shape[-1]
    out = torch.empty(tokens, heads, d, dtype=torch.float64, device=q.device)
    lse = torch.empty(tokens, heads, dtype=torch.float64, device=q.device)
    for b in range(batch):
        n = int(seqused[b])
        pos = torch.arange(n, device=q.device)
        slot = page_table[b, pos // page].long() * page + pos % page
        keys = kc.view(-1, heads_k, d)[slot].double().repeat_interleave(group, 1)
        vals = vc.view(-1, heads_k, d)[slot].double().repeat_interleave(group, 1)
        rows = slice(b * qlen, (b + 1) * qlen)
        dist = (n - qlen + torch.arange(qlen, device=q.device))[:, None] - pos
        index = dist.clamp(0, extent - 1)[:, None, :].expand(qlen, heads, n)
        bias = torch.gather(rel[rows].double(), 2, index) * (dist < extent)[:, None]
        s = torch.einsum("thd,khd->thk", q[rows].double(), keys) * scale + bias
        hidden = dist < 0
        if window is not None:
            hidden |= dist > window
        s = s.masked_fill(hidden[:, None, :], float("-inf"))
        lse[rows] = torch.logsumexp(s, -1)
        out[rows] = torch.einsum("thk,khd->thd", torch.softmax(s, -1), vals)
    return out, lse


def _reduce_partials(o, m, l):
    """(out, lse) from split-KV partials, reduced as combine_gate_sink reduces them:
    o and l are read only where a split's max is above -inf."""
    active = m > float("-inf")
    top = m.max(0).values
    weight = torch.where(active, torch.exp(m - top), 0.0)
    total = (weight * torch.where(active, l, 0.0)).sum(0)
    o = torch.where(active[..., None], o, 0.0)
    out = (weight[..., None] * o).sum(0) / total[..., None]
    return out[:, 0].double(), (top + torch.log(total))[:, 0].double()


def _attention_inputs(heads_q, heads_k, qlen, window, page):
    """Paged inputs at the model's norms. The lengths put the end of the bias-free key
    range (seqlen - qlen - EXTENT + 1) on a 128- and 512-key boundary, one key before
    and one after, below zero and far above. Planted keys make a mistake decisive:
    the first head of each group has a +25 bias at distance EXTENT - 1, and the next
    verify token's key lines up with the second head's query. Every slot that is not
    a key is NaN."""
    torch.manual_seed(qlen * 100 + heads_k * 10 + page)
    group = heads_q // heads_k
    base = qlen + (EXTENT - 1 if window is None else window)
    lens = [qlen, qlen + 1, 700, 4999]
    lens += [base + x for x in (-1, 0, 1, 127, 128, 129, 511, 512, 513)]
    batch = len(lens)
    pages = [-(-n // page) for n in lens]
    order = torch.randperm(sum(pages)) + 1  # page 0 is never a sequence's
    page_table = torch.zeros(batch, max(pages) + 2, dtype=torch.int32)
    for b, own in enumerate(order.split(pages)):
        page_table[b, : pages[b]] = own
    page_table = page_table.cuda()
    shape = (sum(pages) + 1, page, heads_k, HEAD_DIM)
    kc = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    vc = torch.full_like(kc, float("nan"))
    q = torch.randn(batch * qlen, heads_q, HEAD_DIM, device="cuda")
    q = (CLIP * q / q.norm(dim=-1, keepdim=True)).bfloat16()
    rel = 2 * torch.randn(batch * qlen, heads_q, EXTENT, device="cuda")
    rel[:, ::group, EXTENT - 1] = 25.0
    rel = rel.bfloat16()
    for b, n in enumerate(lens):
        pos = torch.arange(n, device="cuda")
        slot = page_table[b, pos // page].long() * page + pos % page
        k = torch.randn(n, heads_k, HEAD_DIM, device="cuda")
        k = CLIP * k / k.norm(dim=-1, keepdim=True)
        for t in range(qlen - 1):
            key = q[b * qlen + t, 1::group].float()
            k[n - qlen + t + 1] = CLIP * key / key.norm(dim=-1, keepdim=True)
        kc.view(-1, heads_k, HEAD_DIM)[slot] = k.bfloat16()
        vc.view(-1, heads_k, HEAD_DIM)[slot] = torch.randn_like(k).bfloat16()
    seqused = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return q, kc, vc, page_table, seqused, rel


def _check_partials(partials, inputs, window):
    out, lse = _reduce_partials(*partials)
    ref_out, ref_lse = _reference_fp64(*inputs, SCALE, window)
    torch.testing.assert_close(lse, ref_lse, atol=2e-4, rtol=0)
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)


@hopper
@pytest.mark.parametrize(
    "heads_q,heads_k,qlen,window,page",
    [
        (6, 1, 8, None, 128),  # verify on a global layer, TP8 rank
        (12, 2, 8, None, 48),  # page size that is no multiple of any tile
        (6, 1, 5, None, 128),  # 32 rows: main's split with the bias gathered ahead
        (6, 1, 2, None, 128),  # short verify block, 16 rows: main's path
        (6, 1, 8, 1023, 128),  # verify on a sliding-window layer
        (12, 2, 1, None, 128),  # decode
    ],
)
def test_decode_partials_matches_reference(heads_q, heads_k, qlen, window, page):
    """decode_partials covers every visible key once, with the bias only below the
    extent, the causal mask inside the verify block and the window cut, and reads
    no slot past a sequence."""
    from sglang.kernels.ops.attention import inkling2_decode_attention_sm90 as attn

    inputs = _attention_inputs(heads_q, heads_k, qlen, window, page)
    partials = attn.decode_partials(*inputs, SCALE, window_size_left=window)
    _check_partials(partials, inputs, window)


@hopper
@pytest.mark.parametrize("splits", [1, 7, 64])
def test_interior_split_partition(monkeypatch, splits):
    """On global-layer verify, any split count partitions the interior tiles and
    the masked tail into disjoint ranges that cover every key once, including
    splits left without keys."""
    from sglang.kernels.ops.attention import inkling2_decode_attention_sm90 as attn

    # Enough SMs for any batch, so MAX_SPLITS alone sets the split count.
    arch = types.SimpleNamespace(num_sms=1 << 16, is_sm=attn.current_arch().is_sm)
    monkeypatch.setattr(attn, "current_arch", lambda: arch)
    monkeypatch.setattr(attn, "MAX_SPLITS", splits)
    inputs = _attention_inputs(6, 1, 8, None, 128)
    partials = attn.decode_partials(*inputs, SCALE)
    assert partials[0].shape[0] == splits
    _check_partials(partials, inputs, None)


@hopper
def test_sequences_without_interior_tiles_keep_main_split(monkeypatch):
    """In a batch that mixes sequences with and without interior tiles, the ones
    without get the same keys in each split as on the launch with no interior path,
    so short contexts run main's work layout: the same row maxima bit for bit, and
    outputs and row sums up to the order of their FP32 sums."""
    from sglang.kernels.ops.attention import inkling2_decode_attention_sm90 as attn

    qlen = 8
    inputs = _attention_inputs(6, 1, qlen, None, 128)
    seqused = inputs[4]
    tile, _ = attn._verify_path(qlen, 6, None, False)
    has_interior = (seqused - qlen - EXTENT + 1) // tile > 0
    assert has_interior.any() and not has_interior.all()
    mixed = attn.decode_partials(*inputs, SCALE)
    monkeypatch.setattr(attn, "_verify_path", lambda *args: (0, False))
    plain = attn.decode_partials(*inputs, SCALE)
    rows = (~has_interior).repeat_interleave(qlen)
    o, m, l = (x[:, rows] for x in mixed)
    o0, m0, l0 = (x[:, rows] for x in plain)
    assert torch.equal(m, m0)
    torch.testing.assert_close(o, o0, rtol=1e-6, atol=0)
    torch.testing.assert_close(l, l0, rtol=1e-6, atol=0)


def test_verify_paths_only_on_hopper_with_16bit_cache(monkeypatch):
    """The interior tiles and the bias prefetch stay off for an MXFP8 cache and on any
    GPU but Hopper (SM120 runs this kernel too): those launches are main's."""
    from sglang.kernels.ops.attention import inkling2_decode_attention_sm90 as attn

    def path(major, mxfp8):
        arch = types.SimpleNamespace(is_sm=lambda want, minor=None: want == major)
        monkeypatch.setattr(attn, "current_arch", lambda: arch)
        return attn._verify_path(8, 6, None, mxfp8)

    assert path(9, mxfp8=False) == (64, True)
    assert path(9, mxfp8=True) == (0, False)
    assert path(12, mxfp8=False) == (0, False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
