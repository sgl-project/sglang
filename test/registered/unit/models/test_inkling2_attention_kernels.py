"""Short-conv, attention prologue, rel-bias decode, and page-128 attention kernels."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.layout.paged_view import paged_view
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-gb300")

hopper_or_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires a Hopper or Blackwell GPU",
)


blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Requires a Blackwell GPU",
)


def _bits_equal(a, b):
    a, b = a.contiguous(), b.contiguous()
    return a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


W, T8 = 4, 8


def _triton_sconv(monkeypatch):
    from sglang.kernels.ops.inkling2 import sconv

    monkeypatch.setattr(sconv, "_cuda_sconv", lambda *args, **kwargs: False)


def _track_indices(seqs, slots):
    """Checkpoint slots, disjoint from the live slots _meta draws from [1, slots)."""
    return slots + torch.randperm(seqs, device="cuda")


def _meta(seqs, rows_per_seq, slots, index_dtype=torch.int64):
    idx = (torch.randperm(slots - 1, device="cuda") + 1)[:seqs].to(index_dtype)
    if seqs > 2:
        idx[1] = 0
    mask = torch.ones(seqs, device="cuda", dtype=torch.bool)
    if seqs > 3:
        mask[3] = False
    rows = seqs * rows_per_seq
    return {
        "safe_idx": idx,
        "cache_mask": mask.view(seqs, 1, 1),
        "cu": torch.arange(0, rows + 1, rows_per_seq, device="cuda", dtype=index_dtype),
        "si": torch.arange(seqs, device="cuda", dtype=torch.int32).repeat_interleave(
            rows_per_seq
        ),
    }


def _run_both(monkeypatch, fn, mutable):
    """fn() under CUDA and Triton on clones of the mutable tensors."""
    copies = [t.clone() for t in mutable]
    actual = fn(mutable)
    with monkeypatch.context() as m:
        _triton_sconv(m)
        expected = fn(copies)
    assert _bits_equal(actual, expected)
    for a, b in zip(mutable, copies):
        assert _bits_equal(a, b)


@hopper_or_blackwell
@pytest.mark.parametrize("seqs", [1, 64])
@pytest.mark.parametrize("mode", ["decode", "verify"])
@pytest.mark.parametrize("top_k", [0, 8])
@pytest.mark.parametrize("with_partial", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("rows_dtype", [torch.bfloat16, torch.float32])
def test_sconv_epilogue(
    monkeypatch, seqs, mode, top_k, with_partial, out_dtype, rows_dtype
):
    from sglang.kernels.ops.inkling2.sconv import routed_sconv_epilogue

    torch.manual_seed(seqs + top_k)
    hidden, slots = 4608, 2 * seqs + 5
    rows_per_seq = 1 if mode == "decode" else T8
    tokens = seqs * rows_per_seq
    meta = _meta(seqs, rows_per_seq, slots)
    rows = torch.randn(tokens * max(top_k, 1), hidden, device="cuda").to(rows_dtype)
    rows[::7] = -0.0
    src2dst = torch.randperm(tokens * top_k, device="cuda").int() if top_k else None
    weight = torch.randn(hidden, 1, W, device="cuda")[:, 0]
    cache = torch.randn(slots + seqs, W - 1, hidden, device="cuda").bfloat16()
    window = torch.randn(seqs, T8, W - 1, hidden, device="cuda").bfloat16()
    attention = torch.randn(tokens, hidden, device="cuda").to(out_dtype)
    partial = torch.randn(tokens, hidden, device="cuda").to(out_dtype)
    track_mask = torch.rand(seqs, device="cuda") < 0.5
    track_indices = _track_indices(seqs, slots)
    decode = mode == "decode"

    def run(t):
        return routed_sconv_epilogue(
            rows,
            src2dst,
            top_k,
            weight,
            t[0],
            meta,
            attention,
            partial if with_partial else None,
            track_mask=track_mask if decode else None,
            track_indices=track_indices if decode else None,
            window=None if decode else t[1],
            round_bf16=rows_dtype == torch.float32,
        )

    _run_both(monkeypatch, run, [cache, window])


@hopper_or_blackwell
@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("partials", [False, True])
@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["verify", "decode", "decode_track"])
@pytest.mark.parametrize("padded_group", [8, 6])
@pytest.mark.parametrize("H,KVH", [(12, 2), (6, 1)], ids=["tp4", "tp8"])
def test_cuda_prologue(
    monkeypatch, m, full, partials, param_dtype, mode, padded_group, H, KVH
):
    """CUDA compact-rel prologue (MTP verify, or decode with the conv cache shift
    and tracked rows) equals the Triton _attention_prologue, at TP4 and TP8 heads
    per rank."""
    from sglang.kernels.ops.inkling2 import prologue as P
    from sglang.kernels.ops.inkling2.linear import linear, linear_k_group_partials_pair

    if partials and not full:
        pytest.skip("dynamic weights are full-layer only")
    torch.manual_seed(m)
    verify = mode == "verify"
    D, DR, W, hidden = 128, 16, 4, 4608
    per_seq = T8 if verify else 1
    tokens = m * per_seq
    sizes = [H * D, KVH * D, KVH * D, H * DR]
    proj = (torch.randn(tokens, sum(sizes) + 8, device="cuda") * 3).bfloat16()
    q, k, v, r = proj[:, : sum(sizes)].split(sizes, dim=1)
    slots = 3 * m + 5
    idx = (torch.randperm(slots - 1, device="cuda") + 1)[:m]
    if m > 2:
        idx[1] = 0
    mask = torch.ones(m, 1, 1, device="cuda", dtype=torch.bool)
    if m > 3:
        mask[3] = False
    meta = {
        "safe_idx": idx,
        "cache_mask": mask,
        "cu": torch.arange(0, tokens + 1, per_seq, device="cuda"),
        "si": torch.arange(m, device="cuda", dtype=torch.int32).repeat_interleave(
            per_seq
        ),
    }
    track = None
    if mode == "decode_track":
        track_mask = torch.rand(m, device="cuda") < 0.5
        track = (track_mask, _track_indices(m, slots))
        slots += m
    if full:
        kw, vw = (
            torch.randn(W, KVH * D, device="cuda"),
            torch.randn(W, KVH * D, device="cuda"),
        )
        sources = [
            torch.randn(tokens, hidden, device="cuda").bfloat16() for _ in range(2)
        ]
        weights = [
            (torch.randn(KVH * W, hidden, device="cuda") * 0.02).bfloat16()
            for _ in range(2)
        ]
        if partials:
            kdyn, vdyn = linear_k_group_partials_pair(
                sources[0], weights[0], sources[1], weights[1]
            )
        else:
            kdyn, vdyn = linear(sources[0], weights[0]), linear(sources[1], weights[1])
    else:
        kw = torch.randn(KVH * D, 1, W, device="cuda")[:, 0]
        vw = torch.randn(KVH * D, 1, W, device="cuda")[:, 0]
        kdyn = vdyn = None
    kc0 = torch.randn(slots, W - 1, KVH * D, device="cuda").bfloat16()
    vc0 = torch.randn(slots, W - 1, KVH * D, device="cuda").bfloat16()
    kw, vw = kw.to(param_dtype), vw.to(param_dtype)
    q_scale = (torch.rand(H, D, device="cuda") * 2).to(param_dtype)
    r_scale = (torch.rand(H, DR, device="cuda") * 2).to(param_dtype)
    rel_proj = (torch.randn(DR, 1024, device="cuda") * 3).bfloat16()
    positions = torch.randint(0, 300000, (tokens,), device="cuda")

    def run():
        kc, vc = kc0.clone(), vc0.clone()
        windows = (
            [
                torch.zeros(m + 1, T8, W - 1, KVH * D, device="cuda").bfloat16()
                for _ in range(2)
            ]
            if verify
            else []
        )
        out = P.attention_prologue(
            q, k, v, r, H, KVH, DR, padded_group, kw, vw, kc, vc, meta, q_scale,
            r_scale, rel_proj, None, not full, 22.627416997969522, 8.0, 1e-6, 1e-6,
            k_dynamic=kdyn, v_dynamic=vdyn, positions=positions if full else None,
            tau_floor=128000, tau_alpha=0.1,
            verify_windows=tuple(windows) if verify else None,
            track_mask=track and track[0], track_indices=track and track[1],
        )  # fmt: skip
        return (*out, kc, vc, *windows)

    taken = []
    real = P._cuda_prologue
    monkeypatch.setattr(
        P, "_cuda_prologue", lambda *a: taken.append(real(*a)) or taken[-1]
    )
    actual = run()
    assert taken == [True]
    monkeypatch.setattr(P, "_cuda_prologue", lambda *a: False)
    names = ("q", "k", "v", "rel", "kcache", "vcache", "kwin", "vwin")
    for name, a, b in zip(names, actual, run()):
        assert _bits_equal(a, b), name


@hopper_or_blackwell
@blackwell
@pytest.mark.parametrize("m", [1, 61])
@pytest.mark.parametrize("window", [None, 1023])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize(
    "path,H,KVH", [("cuda", 12, 2), ("triton", 12, 2), ("triton", 6, 1)]
)
def test_prologue_sheared_verify_rel(monkeypatch, m, window, ragged, path, H, KVH):
    """The verify prologue's sheared rel rows (CUDA, or the Triton fallback, also
    at TP8 heads) equal what FA4's ShearingBias builds from the compact ones, and
    FA4 on them (presheared) equals FA4 on the compact rows, bitwise."""
    from sglang.kernels.ops.attention.flash_attention_v4 import (
        flash_attn_with_kvcache,
    )
    from sglang.kernels.ops.attention.flash_attn.cute import interface
    from sglang.kernels.ops.inkling2 import prologue as P

    torch.manual_seed(m * 4 + ragged * 2 + (window is None))
    D, DR, G, extent, page = 128, 16, 8, 1024, 128
    taken = []
    real = P._cuda_prologue
    monkeypatch.setattr(
        P,
        "_cuda_prologue",
        lambda *a: path == "cuda" and (taken.append(real(*a)) or taken[-1]),
    )
    lens = (torch.randint(1, T8 + 1, (m,)) if ragged else torch.full((m,), T8)).tolist()
    tokens = sum(lens)
    cu = torch.tensor([0] + lens, device="cuda").cumsum(0)
    sizes = [H * D, KVH * D, KVH * D, H * DR]
    proj = (torch.randn(tokens, sum(sizes) + 8, device="cuda") * 3).bfloat16()
    q, k, v, r = proj[:, : sum(sizes)].split(sizes, dim=1)
    slots = 3 * m + 5
    meta = {
        "safe_idx": (torch.randperm(slots - 1, device="cuda") + 1)[:m],
        "cache_mask": torch.ones(m, 1, 1, device="cuda", dtype=torch.bool),
        "cu": cu,
        "si": torch.repeat_interleave(
            torch.arange(m, device="cuda", dtype=torch.int32),
            torch.tensor(lens, device="cuda"),
        ),
    }
    kw = torch.randn(KVH * D, W, device="cuda").bfloat16()
    vw = torch.randn(KVH * D, W, device="cuda").bfloat16()
    kc0 = torch.randn(slots, W - 1, KVH * D, device="cuda").bfloat16()
    vc0 = torch.randn(slots, W - 1, KVH * D, device="cuda").bfloat16()
    q_scale = (torch.rand(H, D, device="cuda") * 2).bfloat16()
    r_scale = (torch.rand(H, DR, device="cuda") * 2).bfloat16()
    rel_proj = (torch.randn(DR, extent, device="cuda") * 3).bfloat16()
    seqlens_k = torch.tensor(lens, dtype=torch.int32) + torch.randint(0, 3000, (m,))
    for i, sk in enumerate([0, 120, 121, 1024 - 3, 1030, 128 * 7 + 2]):
        if i < m:
            seqlens_k[i] = lens[i] + sk
    seqlens_k = seqlens_k.to(torch.int32).cuda()

    def prologue(shear):
        windows = [
            torch.zeros(m + 1, T8, W - 1, KVH * D, device="cuda").bfloat16()
            for _ in range(2)
        ]
        return P.attention_prologue(
            q, k, v, r, H, KVH, DR, G, kw, vw, kc0.clone(), vc0.clone(), meta,
            q_scale, r_scale, rel_proj, None, window is not None,
            22.627416997969522, 8.0, 1e-6, 1e-6,
            verify_windows=tuple(windows), shear=shear,
        )  # fmt: skip

    q_out, _, _, rel_compact = prologue(None)
    _, _, _, rel_sheared = prologue((seqlens_k, window))
    assert taken == ([True, True] if path == "cuda" else [])
    assert rel_sheared.shape == ((tokens + 127) // 128 * 128, KVH * G, extent + 256)

    pages = (int(seqlens_k.max()) + page - 1) // page
    page_table = (
        torch.randperm(m * pages, device="cuda", dtype=torch.int32).view(m, pages) + 1
    )
    kcache = torch.randn(m * pages + 1, page, KVH, D, device="cuda").bfloat16()
    vcache = torch.randn_like(kcache)
    captured = []
    real_empty = interface._shear_bias_empty

    def capture(shape, dtype, device):
        captured.append(torch.full(shape, float("nan"), dtype=dtype, device=device))
        return captured[-1]

    def attend(rel, presheared, num_splits=1):
        return flash_attn_with_kvcache(
            q=q_out, k_cache=kcache, v_cache=vcache, cache_seqlens=seqlens_k,
            page_table=page_table, cu_seqlens_q=cu.to(torch.int32),
            max_seqlen_q=max(lens), softmax_scale=D**-0.5, causal=window is None,
            window_size=(-1, -1) if window is None else (window, 0), num_splits=num_splits,
            pack_gqa=True, rel_bias=rel, return_softmax_lse=True,
            rel_bias_scale_after_add=True, rel_bias_presheared=presheared,
        )  # fmt: skip

    monkeypatch.setattr(interface, "_shear_bias_empty", capture)
    expected = attend(rel_compact, False)
    monkeypatch.setattr(interface, "_shear_bias_empty", real_empty)
    assert len(captured) == 1
    assert _bits_equal(rel_sheared[:tokens], captured[0][:tokens])
    actual = attend(rel_sheared, True)
    for a, b in zip(actual, expected):
        assert _bits_equal(a, b)
    split_out, split_lse = attend(rel_sheared, True, num_splits=9)
    torch.testing.assert_close(
        split_out.float(), actual[0].float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(split_lse, actual[1], atol=1e-3, rtol=1e-3)


def _rel_bias_attention_ref(q, kc, vc, page_table, seqused, rel, scale, window):
    """FP32 reference: q (batch * qlen, heads, d) are the last qlen positions of each
    sequence, attending causally with the compact relative bias (0 past the extent)
    and an optional left window. Returns (out, lse) per token and head."""
    batch = seqused.shape[0]
    tokens, heads, d = q.shape
    qlen = tokens // batch
    page = kc.shape[1]
    group = heads // kc.shape[2]
    extent = rel.shape[-1]
    out = torch.empty(tokens, heads, d, device=q.device)
    lse = torch.empty(tokens, heads, device=q.device)
    for b in range(batch):
        n = int(seqused[b])
        loc = page_table[b, : (n + page - 1) // page]
        keys = kc[loc].reshape(-1, kc.shape[2], d)[:n].float()
        vals = vc[loc].reshape(-1, vc.shape[2], d)[:n].float()
        pos = torch.arange(n, device=q.device)
        for t in range(qlen):
            i = b * qlen + t
            qpos = n - qlen + t
            dist = qpos - pos
            for h in range(heads):
                s = (keys[:, h // group] @ q[i, h].float()) * scale
                bias = torch.zeros(n, device=q.device)
                inside = (dist >= 0) & (dist < extent)
                bias[inside] = rel[i, h, dist[inside]].float()
                s = s + bias
                s[dist < 0] = float("-inf")
                if window is not None:
                    s[dist > window] = float("-inf")
                lse[i, h] = torch.logsumexp(s, 0)
                out[i, h] = torch.softmax(s, 0) @ vals[:, h // group]
    return out, lse


@hopper_or_blackwell
@blackwell
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("qlen", [1])
@pytest.mark.parametrize("window", [None, 1023])
@pytest.mark.parametrize("num_sms", [None, 148])
def test_rel_bias_decode_multi_query(batch, qlen, window, num_sms):
    """paged_rel_bias_decode matches an FP32 reference, through the combine_gate_sink
    epilogue it feeds, with the 64-CTA and the one-wave split counts."""
    from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd_sm100_bias_decode import (
        paged_rel_bias_decode,
    )
    from sglang.kernels.ops.inkling2.attention import combine_gate_sink

    torch.manual_seed(batch * 10 + qlen)
    H, KVH, D, extent, page = 12, 2, 128, 1024, 128
    seqused = torch.randint(qlen, 5000, (batch,), device="cuda", dtype=torch.int32)
    seqused[0] = max(qlen, 3)
    pages = (int(seqused.max()) + page - 1) // page
    page_table = (
        torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(
            batch, pages
        )
        + 1
    )
    kc = torch.randn(batch * pages + 1, page, KVH, D, device="cuda").bfloat16()
    vc = torch.randn_like(kc)
    tokens = batch * qlen
    q = (torch.randn(tokens, H, D, device="cuda") * 0.3).bfloat16()
    rel = torch.randn(tokens, H, extent, device="cuda").bfloat16()
    scale = D**-0.5
    o, mx, sm = paged_rel_bias_decode(
        q, kc, vc, page_table, seqused, rel, scale, window_size_left=window,
        reduce=False, num_sms=num_sms,
    )  # fmt: skip
    ref_out, ref_lse = _rel_bias_attention_ref(
        q, kc, vc, page_table, seqused, rel, scale, window
    )
    gate = torch.randn(tokens, H * D, device="cuda").bfloat16()
    sinks = torch.randn(tokens, H, device="cuda").bfloat16()
    out = combine_gate_sink(o, mx, sm, gate=gate, sinks=sinks)
    weight = torch.sigmoid(gate.float().view(tokens, H, D)) * torch.sigmoid(
        ref_lse - sinks.float()
    ).unsqueeze(-1)
    expected = ref_out * weight
    torch.testing.assert_close(
        out.float().view(tokens, H, D), expected, atol=2e-2, rtol=2e-2
    )


H, KVH, D, DR, W, T8 = 12, 2, 128, 16, 4, 8


def _verify_meta(m, slots):
    tokens = m * T8
    idx = (torch.randperm(slots - 1, device="cuda").int() + 1)[:m].contiguous()
    if m > 2:
        idx[1] = 0  # CUDA-graph padding row on the reserved slot
    return idx, {
        "safe_idx": idx,
        "cache_mask": torch.ones(m, device="cuda", dtype=torch.int32),
        "cu": torch.arange(0, tokens + 1, T8, device="cuda", dtype=torch.int32),
        "si": torch.arange(m, device="cuda", dtype=torch.int32).repeat_interleave(T8),
    }


@blackwell
@pytest.mark.parametrize("m", [1, 3, 16])
def test_sconv_windows_and_routed_epilogue(m):
    from sglang.kernels.ops.inkling2.moe import unweighted_post_reorder
    from sglang.kernels.ops.inkling2.residual import residual_add
    from sglang.kernels.ops.inkling2.sconv import (
        routed_sconv_epilogue,
        short_convolution,
    )
    from sglang.srt.models.inkling_common.kernels.sconv import (
        save_intermediate_conv_windows,
    )

    torch.manual_seed(m)
    hidden, top_k, tokens = 4608, 8, m * T8
    slots = 3 * m + 5
    idx, meta = _verify_meta(m, slots)
    cache = torch.randn(slots, W - 1, hidden, device="cuda").bfloat16()
    weight = torch.randn(hidden, W, device="cuda")
    rows = torch.randn(tokens * top_k, hidden, device="cuda").bfloat16()
    src2dst = torch.randperm(tokens * top_k, device="cuda").int()
    attention = torch.randn(tokens, hidden, device="cuda").bfloat16()
    partial = torch.randn(tokens, hidden, device="cuda").bfloat16()
    windows = [
        torch.zeros(m + 2, T8, W - 1, hidden, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    real = idx != 0

    mlp = unweighted_post_reorder(rows, src2dst, top_k)
    conv = short_convolution(mlp, weight, cache, meta)
    save_intermediate_conv_windows(cache, mlp, idx, windows[0], m, T8)
    assert _bits_equal(
        short_convolution(mlp, weight, cache, meta, window=windows[1]), conv
    )
    assert _bits_equal(windows[1][:m][real], windows[0][:m][real])
    fused = routed_sconv_epilogue(
        rows, src2dst, top_k, weight, cache, meta, attention, partial, window=windows[2]
    )
    assert _bits_equal(fused, residual_add(partial, attention, conv))
    assert _bits_equal(windows[2][:m][real], windows[0][:m][real])


@blackwell
def test_markov_rnn_gates_and_softcap():
    from sglang.kernels.ops.activation.softcap import softcap_inplace_logits
    from sglang.kernels.ops.inkling2.mtp import rnn_gates
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    torch.manual_seed(0)
    for rows in (1, 16, 64):
        proj = (torch.randn(rows, 768, device="cuda") * 4).bfloat16()
        state = torch.randn(rows, 256, device="cuda").bfloat16()
        gate_raw, cand_raw, out_raw = proj.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state, latent = rnn_gates(proj, state)
        assert _bits_equal(
            new_state, gate * state + (1.0 - gate) * torch.tanh(cand_raw)
        )
        assert _bits_equal(latent, torch.tanh(out_raw))

    cap, limit, width = 30.0, 200058, 50256
    for rows in (1, 8):
        for start in (0, 150768):
            base = (torch.randn(rows, width, device="cuda") * 40).bfloat16()
            bias = (torch.randn(rows, width, device="cuda") * 5).bfloat16()
            temps = (torch.arange(rows, device="cuda") % 2).float()
            offset = torch.zeros(1, dtype=torch.int64, device="cuda")
            logits = base + bias
            softcap_inplace_logits(logits, cap)
            logits[..., max(limit - start, 0) :] = float("-inf")
            kept = [
                torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            common = dict(
                temperatures=temps,
                key=7,
                offset=offset,
                step=1,
                step_dtype=torch.bfloat16,
                vocab_start=start,
            )
            want = MarkovSampleStep.execute(
                base_logits=logits,
                bias=torch.zeros_like(bias),
                logits_out=kept[0],
                **common,
            )
            got = MarkovSampleStep.execute(
                base_logits=base,
                bias=bias,
                logits_out=kept[1],
                softcap=cap,
                vocab_limit=limit,
                **common,
            )
            assert torch.equal(got, want) and _bits_equal(kept[1], kept[0])


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires the compiled CUDA attention backend",
)


WINDOW = 1024


HEADS, KV_HEADS, HEAD_DIM = 8, 2, 64


def _setup(monkeypatch, page_size, committed, query_lens, mapping, device="cpu"):
    from sglang.kernels.ops.attention import flash_attention_v4 as fa4
    from sglang.srt.layers.attention import flashattention_backend as flash

    capacity = 1280
    pages = capacity // page_size
    batch_size = len(committed)
    table = torch.stack(
        [torch.roll(torch.arange(pages), index + 3) + 1 for index in range(batch_size)]
    ).to(torch.int32)
    translation = torch.arange(pages + 1).flip(0).to(torch.int32)
    selected = translation[table.long()] if mapping != "full" else table
    token_translation = (
        (translation[:, None] * page_size + torch.arange(page_size))
        .flatten()
        .to(device)
    )
    cu_q = torch.tensor([0, *torch.tensor(query_lens).cumsum(0)], dtype=torch.int32)
    ends = torch.tensor(committed, dtype=torch.int32) + torch.tensor(
        query_lens, dtype=torch.int32
    )
    metadata = flash.FlashAttentionMetadata(
        cache_seqlens_int32=ends.to(device),
        max_seq_len_q=max(query_lens),
        max_seq_len_k=capacity,
        cu_seqlens_q=cu_q.to(device),
        cu_seqlens_k=torch.cat([ends.new_zeros(1), ends.cumsum(0)]).int().to(device),
        page_table=table.to(device),
        swa_page_table=selected.to(device) if mapping == "precomputed" else None,
    )
    generator = torch.Generator().manual_seed(7)
    cache_shape = (pages + 1, page_size, 2, KV_HEADS, HEAD_DIM)
    key, value = [
        torch.randn(cache_shape, generator=generator).bfloat16().to(device)[:, :, 0]
        for _ in range(2)
    ]
    query = (
        torch.randn(sum(query_lens), HEADS, HEAD_DIM, generator=generator)
        .bfloat16()
        .to(device)
    )
    backend = object.__new__(flash.FlashAttentionBackend)
    backend.fa_impl_ver = 4
    backend.flash_attn_with_kvcache = fa4.flash_attn_with_kvcache
    backend._anchor_window_cache = {}
    backend._anchor_first_layer = {}
    backend.page_size = page_size
    backend.num_splits = 1
    backend.decode_num_splits = 1
    backend.topk = 1
    backend.use_mla = False
    backend.fa_skip_kv_cache = False
    backend.kv_cache_is_mxfp8 = False
    backend.local_attn_builder = None
    backend.use_sliding_window_kv_pool = mapping != "full"
    backend.forward_metadata = metadata
    backend.full_cg_prefill_metadata = None
    backend.is_prefill_aware_swa = False
    backend._decode_uses_static_max_seqlen_k = False
    backend.runner_ids = SimpleNamespace(
        ids_in=lambda _kind, ids: token_translation[ids.long()]
    )
    backend.prepare_paged_mha_query = lambda q, qr, kr, *_args, **_kwargs: (
        q,
        qr,
        kr,
        None,
        None,
    )
    backend._mxfp8_sf_kwargs = lambda *_args: {}
    backend.get_paged_mha_kv_cache = lambda _layer: (key, value)
    layer = SimpleNamespace(
        layer_id=0,
        window_anchor_at_start=True,
        sliding_window_size=WINDOW - 1,
        num_splits=1,
        pack_gqa=False,
        out_dtype=torch.float32,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER_BIDIRECTIONAL,
        tp_q_head_num=HEADS,
        tp_k_head_num=KV_HEADS,
        tp_v_head_num=KV_HEADS,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        scaling=HEAD_DIM**-0.5,
        logit_cap=0,
    )
    batch = SimpleNamespace(
        batch_size=batch_size,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=None,
    )
    monkeypatch.setattr(flash, "is_cp_active", lambda _batch: False)
    return SimpleNamespace(
        flash=flash,
        backend=backend,
        layer=layer,
        batch=batch,
        query=query,
        key=key,
        value=value,
        physical_pages=selected,
        committed=list(committed),
        query_lens=list(query_lens),
        page_size=page_size,
        capacity=capacity,
    )


def _token_table(case):
    width = min(case.capacity, WINDOW + max(case.query_lens))
    rows, lengths = [], []
    for pages, prefix, queries in zip(
        case.physical_pages.tolist(), case.committed, case.query_lens
    ):
        token_pool = [
            token
            for page in pages
            for token in range(page * case.page_size, (page + 1) * case.page_size)
        ]
        start = max(0, prefix - WINDOW)
        rows.append(
            [
                token_pool[min(start + offset, case.capacity - 1)]
                for offset in range(width)
            ]
        )
        lengths.append(prefix + queries - start)
    device = case.query.device
    return (
        torch.tensor(rows, dtype=torch.int32, device=device),
        torch.tensor(lengths, dtype=torch.int32, device=device),
    )


def _forward(case, *, return_lse=False):
    return case.backend.forward_extend(
        case.query,
        None,
        None,
        case.layer,
        case.batch,
        save_kv_cache=False,
        return_lse=return_lse,
    )


def _reference_attention(case):
    from sglang.kernels.ops.attention import flash_attention_v4 as fa4

    table, lengths = _token_table(case)
    output, lse = fa4.flash_attn_with_kvcache(
        q=case.query,
        k_cache=paged_view(case.key.view(-1, KV_HEADS, HEAD_DIM), 1),
        v_cache=paged_view(case.value.view(-1, KV_HEADS, HEAD_DIM), 1),
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=case.backend.forward_metadata.cu_seqlens_q,
        cu_seqlens_k_new=torch.cat([lengths.new_zeros(1), lengths.cumsum(0)]).int(),
        max_seqlen_q=max(case.query_lens),
        softmax_scale=case.layer.scaling,
        causal=False,
        window_size=(-1, -1),
        softcap=0,
        return_softmax_lse=True,
        num_splits=1,
        out=torch.empty_like(case.query, dtype=torch.float32),
        pack_gqa=False,
    )
    return output.reshape(-1, HEADS * HEAD_DIM), lse.T.contiguous()


def _assert_bitwise(actual, expected):
    for left, right in zip(actual, expected):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(
            left.contiguous().view(torch.uint8),
            right.contiguous().view(torch.uint8),
        )


@requires_cuda
@pytest.mark.parametrize("page_size", [1, 128])
def test_anchored_attention_matches_token_pool_reference(monkeypatch, page_size):
    case = _setup(
        monkeypatch,
        page_size,
        [0, 127, 1023, 1025, 1153],
        [8, 8, 8, 8, 8],
        "precomputed",
        "cuda",
    )
    _assert_bitwise(_forward(case, return_lse=True), _reference_attention(case))


@requires_cuda
def test_anchored_graph_replay_refreshes_lengths_and_physical_pages(
    monkeypatch,
):
    case = _setup(
        monkeypatch, 128, [1024, 1025, 1151], [8, 8, 8], "precomputed", "cuda"
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _forward(case, return_lse=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _forward(case, return_lse=True)
    for committed in ([1152, 127, 129], [1023, 1024, 1025]):
        case.committed = list(committed)
        case.physical_pages = case.physical_pages.roll(2, dims=1)
        metadata = case.backend.forward_metadata
        metadata.swa_page_table.copy_(case.physical_pages)
        metadata.cache_seqlens_int32.copy_(
            torch.tensor(
                [prefix + 8 for prefix in committed],
                device="cuda",
                dtype=torch.int32,
            )
        )
        graph.replay()
        _assert_bitwise(output, _reference_attention(case))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
