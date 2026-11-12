"""Inkling2's ROCm kernels against float64 / torch references, and their run-to-run
determinism (two ROCm Triton miscompilations showed up only as nondeterminism)."""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: the ROCm kernels and dispatch Inkling2 runs on MI35x
register_amd_ci(est_time=170, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires a ROCm GPU",
)

H, HKV, D, EXT = 6, 1, 128, 1024
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


def _rel(a, b):
    return (
        (a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)
    ).item()


def _attention_reference(q, k, v, bias, q_pos, window=None, causal=True):
    """q (Lq, H, D) at positions q_pos over keys 0..len(k)-1; bias (Lq, H, EXT) by
    query-key distance, added for distances 0..EXT-1."""
    kk = k.double().repeat_interleave(H // HKV, 1)
    vv = v.double().repeat_interleave(H // HKV, 1)
    s = torch.einsum("qhd,khd->hqk", q.double(), kk) * D**-0.5
    dist = q_pos[:, None] - torch.arange(k.shape[0], device=q.device)[None, :]
    rel = torch.where(
        (dist >= 0) & (dist < EXT),
        bias.double()
        .permute(1, 0, 2)
        .gather(2, dist.clamp(0, EXT - 1)[None].expand(H, -1, -1)),
        0.0,
    )
    keep = dist >= 0 if causal else torch.ones_like(dist, dtype=torch.bool)
    if window is not None:
        keep &= dist <= window
    s = torch.where(keep[None], s + rel, float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vv), torch.logsumexp(
        s, -1
    ).T


def _paged_problem(prefix, ext, window, page=128, seed=0):
    torch.manual_seed(seed)
    seq = [p + e for p, e in zip(prefix, ext)]
    pages = [-(-s // page) for s in seq]
    perm = torch.randperm(sum(pages), device="cuda")
    k_cache = torch.randn(
        sum(pages) * page, HKV, D, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)
    slots, base = [], 0
    for b, n in enumerate(pages):
        pg = perm[base : base + n]
        base += n
        slots.append(
            (pg[:, None] * page + torch.arange(page, device="cuda")).flatten()[: seq[b]]
        )
    q = torch.randn(sum(ext), H, D, device="cuda", dtype=torch.bfloat16) * 2
    bias = torch.randn(sum(ext), H, EXT, device="cuda", dtype=torch.bfloat16)
    qo_indptr = torch.tensor(
        [0, *torch.tensor(ext).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    k_ext = torch.cat([k_cache[slots[b][p:]] for b, p in enumerate(prefix)])
    v_ext = torch.cat([v_cache[slots[b][p:]] for b, p in enumerate(prefix)])
    kv_lens = [p if window is None else min(p, window) for p in prefix]
    kv_indptr = torch.tensor(
        [0, *torch.tensor(kv_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    kv_indices = torch.cat(
        [slots[b][p - n : p] for b, (p, n) in enumerate(zip(prefix, kv_lens))]
    ).long()
    return (
        q,
        k_ext,
        v_ext,
        k_cache,
        v_cache,
        bias,
        qo_indptr,
        kv_indptr,
        kv_indices,
        slots,
        seq,
    )


@pytest.mark.parametrize("window", [None, 1023])
@pytest.mark.parametrize("kernel", ["rel_extend", "rel_verify", "rel_verify_int64"])
def test_relative_bias_attention(kernel, window):
    from sglang.kernels.ops.inkling2 import attention_hip

    if kernel == "rel_extend":
        prefix, ext, fn = (
            (0, 300, 2000, 5000),
            (7, 129, 600, 1100),
            attention_hip.rel_extend_attention,
        )
    else:
        prefix, ext, fn = (
            (0, 37, 300, 1100, 70000),
            (1, 5, 8, 8, 2),
            attention_hip.rel_verify_attention,
        )
    q, k_ext, v_ext, kc, vc, bias, qo, kvp, kvi, slots, seq = _paged_problem(
        prefix, ext, window
    )
    if kernel == "rel_verify_int64":
        qo, kvp = qo.long(), kvp.long()
    o = torch.empty(q.shape, device="cuda", dtype=torch.float32)
    lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
    args = (q, k_ext, v_ext, kc, vc, bias, o, lse, qo, kvp, kvi, max(ext), D**-0.5)
    fn(*args, window=-1 if window is None else window, page_size=128)
    first = o.clone()
    for b in range(len(prefix)):
        rows = slice(qo[b].item(), qo[b + 1].item())
        q_pos = torch.arange(prefix[b], seq[b], device="cuda")
        ro, rl = _attention_reference(
            q[rows], kc[slots[b]], vc[slots[b]], bias[rows], q_pos, window
        )
        assert _rel(o[rows], ro) < 2e-3
        assert (lse[rows].double() - rl).abs().max().item() < 1e-4
    for _ in range(10):
        o.fill_(1e3)
        fn(*args, window=-1 if window is None else window, page_size=128)
        assert torch.equal(o, first)


@pytest.mark.parametrize("window", [None, 1023])
@pytest.mark.parametrize(
    "prefix, ext, pad", [((0, 300, 2000), (7, 129, 600), 37), ((300,), (100,), 12)]
)
def test_rel_extend_padded_request_slots(prefix, ext, pad, window):
    """Full-graph prefill replays rel_extend_attention on a bucket: padded token rows,
    empty request slots and the bucket's token count as max_extend_len leave the real
    rows bitwise unchanged while the block height stays (a single request and its
    bucket never straddle the 128-token switch)."""
    from sglang.kernels.ops.inkling2 import attention_hip

    slots = 16
    q, k_ext, v_ext, kc, vc, bias, qo, kvp, kvi, _, _ = _paged_problem(
        prefix, ext, window
    )
    w = -1 if window is None else window
    rows = q.shape[0]
    o = torch.empty(q.shape, device="cuda", dtype=torch.float32)
    lse = torch.empty(q.shape[:2], device="cuda", dtype=torch.float32)
    attention_hip.rel_extend_attention(
        q, k_ext, v_ext, kc, vc, bias, o, lse, qo, kvp, kvi, max(ext), D**-0.5,
        window=w, page_size=128,
    )  # fmt: skip
    padded = lambda t: torch.cat([t, torch.randn_like(t[:pad])])  # noqa: E731
    empty = lambda t: torch.cat([t, t[-1:].expand(slots - len(ext))])  # noqa: E731
    qp, kp, vp, bp = padded(q), padded(k_ext), padded(v_ext), padded(bias)
    op = torch.full(qp.shape, 1e3, device="cuda", dtype=torch.float32)
    lsep = torch.full(qp.shape[:2], 1e3, device="cuda", dtype=torch.float32)
    attention_hip.rel_extend_attention(
        qp, kp, vp, kc, vc, bp, op, lsep, empty(qo), empty(kvp), kvi, qp.shape[0],
        D**-0.5, window=w, page_size=128,
    )  # fmt: skip
    assert torch.equal(op[:rows], o)
    assert torch.equal(lsep[:rows], lse)


def test_decode_gate_sink_epilogue():
    """Decode with gate_sink applies the gate and sink in its reduction bitwise as
    gate_sink() does after the LSE-returning launch."""
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    from sglang.kernels.ops.attention.score_mod import relative_bias_score_mod
    from sglang.kernels.ops.inkling2.attention import gate_sink

    torch.manual_seed(3)
    seq_lens, splits = (5, 700, 3000), 8
    batch, total = len(seq_lens), sum(seq_lens)
    k_buf = torch.randn(total, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn_like(k_buf)
    kvp = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    kvi = torch.randperm(total, device="cuda").long()
    q = torch.randn(batch, H, D, device="cuda", dtype=torch.bfloat16) * 2
    bias = torch.randn(batch, H, EXT, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(batch, 3 * H * D, device="cuda", dtype=torch.bfloat16)[
        :, H * D : 2 * H * D
    ]
    sinks = torch.randn(batch, 2 * H, device="cuda", dtype=torch.bfloat16)[:, :H]
    nsplit = torch.tensor([1, 4, splits], device="cuda", dtype=torch.int32)
    logits, lse_buf = (
        torch.empty(batch, H, splits, D, device="cuda"),
        torch.empty(batch, H, splits, device="cuda"),
    )
    o32, lse = (
        torch.empty(batch, H, D, device="cuda"),
        torch.empty(batch, H, device="cuda"),
    )
    common = dict(score_mod=relative_bias_score_mod, aux_tensors=[bias])
    decode_attention_fwd(
        q,
        k_buf,
        v_buf,
        o32,
        kvp,
        kvi,
        logits,
        lse_buf,
        nsplit,
        splits,
        D**-0.5,
        1.0,
        1.0,
        lse_out=lse,
        **common,
    )
    ref = gate_sink(o32.view(batch, -1), gate, lse, sinks, output_dtype=torch.bfloat16)
    o16 = torch.empty(batch, H, D, device="cuda", dtype=torch.bfloat16)
    decode_attention_fwd(
        q,
        k_buf,
        v_buf,
        o16,
        kvp,
        kvi,
        logits,
        lse_buf,
        nsplit,
        splits,
        D**-0.5,
        1.0,
        1.0,
        gate_sink=(gate, sinks),
        **common,
    )
    assert torch.equal(o16.view(batch, -1), ref)


@pytest.mark.parametrize("bound,min_tokens", [(4, 64), (64, 64), (128, 32)])
def test_length_based_decode_splits_through_buffer_views(bound, min_tokens):
    """Decode with per-request length-based KV splits (the triton backend's
    SGLANG_TRITON_DECODE_SPLIT_MIN_TOKENS policy), launched over [:, :, :bound] views
    of a 128-split buffer as the backend does: bitwise equal to the same launch on
    contiguous bound-split buffers, and float64 attention to BF16 accuracy."""
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    from sglang.kernels.ops.attention.metadata import (
        get_num_kv_splits_by_length_triton,
    )

    torch.manual_seed(bound)
    seq_lens = (1, 63, 64, 65, 700, 3000, 9000)
    batch, total = len(seq_lens), sum(seq_lens)
    k_buf = torch.randn(total, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn_like(k_buf)
    lens = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    kvp = torch.cat([lens.new_zeros(1), lens.cumsum(0)]).int()
    kvi = torch.randperm(total, device="cuda").long()
    q = torch.randn(batch, H, D, device="cuda", dtype=torch.bfloat16)
    nsplit = torch.empty(batch, device="cuda", dtype=torch.int32)
    get_num_kv_splits_by_length_triton[(1,)](
        nsplit, lens, batch, 1, bound, MIN_TOKENS=min_tokens, MAX_NUM_SEQ=256
    )
    want = [max(1, min(-(-n // min_tokens), bound)) for n in seq_lens]
    assert nsplit.tolist() == want
    logits = torch.empty(batch, H, 128, D, device="cuda")
    lse_buf = torch.empty(batch, H, 128, device="cuda")
    o = torch.empty(batch, H, D, device="cuda")
    decode_attention_fwd(
        q, k_buf, v_buf, o, kvp, kvi, logits[:, :, :bound], lse_buf[:, :, :bound],
        nsplit, bound, D**-0.5, 1.0, 1.0,
    )  # fmt: skip
    contiguous = torch.empty_like(o)
    decode_attention_fwd(
        q, k_buf, v_buf, contiguous, kvp, kvi,
        torch.empty(batch, H, bound, D, device="cuda"),
        torch.empty(batch, H, bound, device="cuda"),
        nsplit, bound, D**-0.5, 1.0, 1.0,
    )  # fmt: skip
    assert torch.equal(o, contiguous)
    for b, n in enumerate(seq_lens):
        idx = kvi[kvp[b] : kvp[b] + n]
        k, v = k_buf[idx, 0].double(), v_buf[idx, 0].double()
        p = torch.softmax(q[b].double() @ k.T * D**-0.5, -1)
        assert _rel(o[b], p @ v) < 5e-3, (b, n)


def _attn_res_reference(sources, addend, queries, normalize):
    values = torch.stack(sources).double()
    if addend is not None:
        values[-1] += addend.double()
    inv = torch.rsqrt(values.square().mean(-1) + 1e-6)
    p = torch.softmax(torch.einsum("ntd,qd,nt->qtn", values, queries.double(), inv), -1)
    if normalize:
        values = values * inv.unsqueeze(-1)
    return torch.einsum("qtn,ntd->qtd", p, values)


@pytest.mark.parametrize("split", [False, 512, 1024])
@pytest.mark.parametrize("tokens", [1, 9, 64, 300])
@pytest.mark.parametrize("n,nq", [(1, 1), (2, 3), (5, 1), (9, 1), (12, 3)])
def test_block_attention_residual(tokens, n, nq, split):
    """One-CTA kernel and the split kernel at both chunk widths (300 rows take the
    one-CTA kernel either way)."""
    from functools import partial

    from sglang.kernels.ops.inkling2 import norm_hip

    attn_res = partial(norm_hip.attn_res, split=bool(split), split_chunk=split or None)

    torch.manual_seed(n * 7 + nq)
    srcs = [
        torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16) * (i + 1)
        for i in range(n)
    ]
    addend = (
        torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16)
        if n % 2
        else None
    )
    q = torch.randn(nq, 4608, device="cuda") * 4608**-0.5 * 3
    ref = _attn_res_reference(srcs, addend, q, True)
    first = [y.clone() for y in attn_res(srcs, addend, q, normalize_values=True)]
    for y, r in zip(first, ref):
        assert y.dtype == torch.float32 and _rel(y, r) < 1e-5
    norms = [
        (
            torch.randn(4608, device="cuda", dtype=torch.bfloat16),
            torch.rand(4608, device="cuda", dtype=torch.bfloat16) + 0.5,
            1e-6,
            torch.bfloat16,
        )
        for _ in range(nq)
    ]
    normed = attn_res(srcs, addend, q, normalize_values=True, norms=norms)
    for y, r, (w, pw, eps, _) in zip(normed, ref, norms):
        u = r * pw.double()
        want = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps) * w.double()
        assert y.dtype == torch.bfloat16 and _rel(y, want) < 4e-3
    for _ in range(10):
        assert all(
            torch.equal(a, b)
            for a, b in zip(attn_res(srcs, addend, q, normalize_values=True), first)
        )


def test_split_attention_residual_graph_replay_is_deterministic():
    """The split kernel's row counters persist across launches and graph replays. A
    graph mixing rows, sources, chunk widths and one or two barriers per launch
    (without and with the fused norm) replays bitwise equal to its eager run."""
    from sglang.kernels.ops.inkling2 import norm_hip

    torch.manual_seed(0)
    calls = []
    for i in range(24):
        tokens, n, nq = (1, 2, 8, 16, 64)[i % 5], 1 + i % 10, 3 if i % 4 == 0 else 1
        srcs = [
            torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16)
            for _ in range(n)
        ]
        addend = torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(nq, 4608, device="cuda") * 4608**-0.5
        norms = [
            (
                torch.randn(4608, device="cuda", dtype=torch.bfloat16),
                None,
                1e-6,
                torch.bfloat16,
            )
        ] * nq
        calls.append(
            (srcs, addend, q, norms if i % 2 else None, (512, 1024)[i % 3 == 0])
        )

    def run():
        return [
            torch.stack(
                norm_hip.attn_res(
                    s, a, q, normalize_values=True, norms=nm, split=True, split_chunk=c
                )
            )
            for s, a, q, nm, c in calls
        ]

    eager = [o.clone() for o in run()]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(outs, eager))


@pytest.mark.parametrize("seqs", [1, 4, 32])
@pytest.mark.parametrize("mode", ["decode", "verify"])
@pytest.mark.parametrize("with_partial", [False, True])
def test_conv_epilogue_takes_fp32_expert_sums(seqs, mode, with_partial):
    """ROCm decode and small verify pass the FP32 expert sums to the conv epilogue
    as Hopper does (identity src2dst, top_k 1, rounded to BF16 on load): equal to
    rounding them to BF16 values first and convolving those FP32 rows."""
    from sglang.kernels.ops.inkling2.sconv import routed_sconv_epilogue

    torch.manual_seed(seqs)
    hidden, width, rows_per_seq = 4608, 4, 1 if mode == "decode" else 8
    slots, tokens = 2 * seqs + 5, seqs * rows_per_seq
    idx = (torch.randperm(slots - 1, device="cuda") + 1)[:seqs]
    meta = {
        "safe_idx": idx,
        "cache_mask": torch.ones(seqs, 1, 1, device="cuda", dtype=torch.bool),
        "cu": torch.arange(0, tokens + 1, rows_per_seq, device="cuda"),
        "si": torch.arange(seqs, device="cuda", dtype=torch.int32).repeat_interleave(
            rows_per_seq
        ),
    }
    sums = torch.randn(tokens, hidden, device="cuda") * 3
    sums[::5] = -0.0
    identity = torch.arange(tokens, device="cuda", dtype=torch.int32)
    weight = torch.randn(hidden, width, device="cuda")
    attention = torch.randn(tokens, hidden, device="cuda").bfloat16()
    partial = torch.randn(tokens, hidden, device="cuda").bfloat16()
    cache = torch.randn(slots, width - 1, hidden, device="cuda").bfloat16()
    window = torch.randn(seqs, 8, width - 1, hidden, device="cuda").bfloat16()
    decode = mode == "decode"

    def run(rows, src2dst, top_k, state):
        return routed_sconv_epilogue(
            rows,
            src2dst,
            top_k,
            weight,
            state[0],
            meta,
            attention,
            partial if with_partial else None,
            window=None if decode else state[1],
            round_bf16=top_k > 0,
        )

    routed, rounded = [cache.clone(), window.clone()], [cache, window]
    actual = run(sums, identity, 1, routed)
    expected = run(sums.bfloat16().float(), None, 0, rounded)
    # Values, not bits: the top-k gather sums from +0, so -0 rows come out +0.
    assert torch.equal(actual, expected)
    for a, b in zip(routed, rounded):
        assert torch.equal(a, b)


def test_rms_norm_and_l2_clip():
    """norm_hip's RMSNorm with a pre-weight and its scaled L2 clip match float64."""
    from sglang.kernels.ops.inkling2 import norm_hip

    torch.manual_seed(0)
    x = torch.randn(37, 4608, device="cuda", dtype=torch.bfloat16) * 3
    w, pw = (
        torch.randn(4608, device="cuda", dtype=torch.bfloat16),
        torch.rand(4608, device="cuda", dtype=torch.bfloat16) + 0.5,
    )
    u = x.double() * pw.double()
    want = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + 1e-6) * w.double()
    assert (
        _rel(
            norm_hip.rms_norm(x, w, 1e-6, pre_weight=pw, output_dtype=torch.float32),
            want,
        )
        < 1e-6
    )
    xs = (torch.randn(37, 3 * 6 * 128, device="cuda", dtype=torch.bfloat16) * 4)[
        :, : 6 * 128
    ].view(37, 6, 128)
    scale = torch.rand(6, 128, device="cuda") + 0.5
    v = xs.double() * scale.double()
    clip = v * (
        22.627 / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-6)
    ).clamp(max=1.0)
    assert _rel(norm_hip.l2_clip(xs, 22.627, 1e-6, scale=scale), clip) < 4e-3


def _dequant(w, s):
    if w.dtype == torch.uint8:
        lut = torch.tensor(
            E2M1 + [-v for v in E2M1], device=w.device, dtype=torch.float64
        )
        v = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], -1).flatten(-2)
        return v * s.double().repeat_interleave(16, -1)
    return w.double() * torch.exp2(s.double() - 127).repeat_interleave(32, -1)


def _sum_splits(partials, dtype):
    """expert_gemm's split-K slabs added in split order, as its consumers sum them."""
    total = partials[0].float()
    for part in partials[1:]:
        total = total + part
    return total.to(dtype)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [1, 5, 8, 13, 300])
def test_expert_gemm(fmt, tokens):
    """NVFP4: products of e2m1, e4m3 scales and NVFP4-fake-quantized activations are
    exact in FP16, so the FP32 accumulator equals float64 up to summation order."""
    from sglang.kernels.ops.inkling2 import experts_hip, experts_sm90

    torch.manual_seed(tokens)
    nexp, n, k, top_k, start = 16, 512, 1024, 8, 8
    if fmt == "nvfp4":
        w = torch.randint(0, 256, (nexp, n, k // 2), device="cuda", dtype=torch.uint8)
        s = (torch.rand(nexp, n, k // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    else:
        w = (torch.randn(nexp, n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
        s = torch.randint(
            120, 130, (nexp, n, k // 32), device="cuda", dtype=torch.uint8
        )
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 2
    xq = (
        experts_sm90.fake_quantize(x, experts_hip.activation_dtype(w.dtype))[0]
        if fmt == "nvfp4"
        else experts_sm90.fake_quantize_mxfp8(x)
    )
    ids = (
        torch.stack(
            [torch.randperm(nexp, device="cuda")[:top_k] for _ in range(tokens)]
        ).int()
        + start
    )
    route = experts_hip.route(ids.flatten(), start, nexp, 64)
    out = experts_hip.expert_gemm(
        xq, w, s, route, n, top_k, torch.float32, ids.numel(), num_experts=64
    )
    out = _sum_splits(out, torch.float32)
    wd, flat = _dequant(w, s), ids.flatten().long()
    xr = xq.double()[torch.arange(ids.numel(), device="cuda") // top_k]
    ref = torch.stack([xr[r] @ wd[flat[r] - start].T for r in range(ids.numel())])
    assert _rel(out, ref) < (1e-7 if fmt == "nvfp4" else 1e-6)
    assert all(
        torch.equal(
            _sum_splits(
                experts_hip.expert_gemm(
                    xq,
                    w,
                    s,
                    route,
                    n,
                    top_k,
                    torch.float32,
                    ids.numel(),
                    num_experts=64,
                ),
                torch.float32,
            ),
            out,
        )  # fmt: skip
        for _ in range(5)
    )


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("gather", [8, 1])
def test_expert_gemm_software_nvfp4_tile(monkeypatch, warps, gather):
    """The NVFP4 GEMM off gfx950 (software dequant, BF16 MMA) at its 128 x 128 x 128
    prefill tile, rebuilt here without NATIVE_CVT: exact against float64 and run to run.
    Its FP32 dequant once became v_pk_mul_f32 reading the scales through op_sel, which on
    gfx950 zeroed some weights in lanes 48-63, differently every run."""
    import types

    import triton
    import triton.language as tl

    from sglang.kernels.ops.inkling2 import experts_hip, experts_sm90
    from sglang.kernels.ops.inkling2.experts_hip import GemmConfig

    fn = experts_hip._grouped_dequant_gemm.fn
    software = dict(fn.__globals__, NATIVE_CVT=tl.constexpr(False))
    kernel = types.FunctionType(fn.__code__, software)
    kernel.__annotations__ = fn.__annotations__
    monkeypatch.setattr(experts_hip, "_grouped_dequant_gemm", triton.jit(kernel))
    monkeypatch.setattr(experts_hip, "NATIVE_CVT", tl.constexpr(False))
    torch.manual_seed(warps + gather)
    nexp, n, k, top_k, start, tokens = 16, 512, 1024, 8, 16, 256
    w = torch.randint(0, 256, (nexp, n, k // 2), device="cuda", dtype=torch.uint8)
    s = (torch.rand(nexp, n, k // 16, device="cuda") * 2 + 0.25).to(torch.float8_e4m3fn)
    rows = tokens * top_k
    x = torch.randn(rows // gather, k, device="cuda", dtype=torch.bfloat16) * 2
    xq = experts_sm90.fake_quantize(x, torch.bfloat16)[0]
    ids = torch.stack(
        [torch.randperm(4 * nexp, device="cuda")[:top_k] for _ in range(tokens)]
    ).int()
    cfg = GemmConfig(128, 128, 128, 1, warps, 1)
    route = experts_hip.route(ids.flatten(), start, nexp, 4 * nexp, block_m=cfg.bm)
    args = (xq, w, s, route, n, gather, torch.float32, rows)
    out = experts_hip.expert_gemm(*args, num_experts=4 * nexp, config=cfg)[0]
    flat = ids.flatten().long()
    wd, xr = _dequant(w, s), xq.double()[torch.arange(rows, device="cuda") // gather]
    for e in range(nexp):
        sel = (flat == start + e).nonzero().flatten()
        assert _rel(out[sel], xr[sel] @ wd[e].T) < 1e-7
    local = ((flat >= start) & (flat < start + nexp)).nonzero().flatten()
    assert all(
        torch.equal(
            experts_hip.expert_gemm(*args, num_experts=4 * nexp, config=cfg)[0][local],
            out[local],
        )
        for _ in range(5)
    )


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [1, 4, 40, 128])
def test_consumers_sum_split_k_partials(fmt, tokens):
    """The SwiGLU and combine kernels sum expert_gemm's split-K partials in split order:
    bitwise equal to summing them first, with several splits and with one."""
    from sglang.kernels.ops.inkling2 import experts_hip, experts_sm90, marlin_sm90

    torch.manual_seed(tokens)
    nexp, n2, k, top_k, start, caps = 16, 512, 1024, 8, 8, (7.0, 7.0)
    if fmt == "nvfp4":
        w13 = torch.randint(
            0, 256, (nexp, n2, k // 2), device="cuda", dtype=torch.uint8
        )
        s13 = (torch.rand(nexp, n2, k // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
        w2 = torch.randint(0, 256, (nexp, k, n2 // 4), device="cuda", dtype=torch.uint8)
        s2 = (torch.rand(nexp, k, n2 // 32, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    else:
        w13 = (torch.randn(nexp, n2, k, device="cuda") * 8).to(torch.float8_e4m3fn)
        s13 = torch.randint(
            120, 130, (nexp, n2, k // 32), device="cuda", dtype=torch.uint8
        )
        w2 = (torch.randn(nexp, k, n2 // 2, device="cuda") * 8).to(torch.float8_e4m3fn)
        s2 = torch.randint(
            120, 130, (nexp, k, n2 // 64), device="cuda", dtype=torch.uint8
        )
    nvfp4 = fmt == "nvfp4"
    act = experts_hip.activation_dtype(w13.dtype)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 2
    if nvfp4:
        xq, x_amax = experts_sm90.fake_quantize(x, act)
    else:
        xq = experts_sm90.fake_quantize_mxfp8(x)
    ids = (
        torch.stack(
            [torch.randperm(2 * nexp, device="cuda")[:top_k] for _ in range(tokens)]
        ).int()
        + start // 2
    )  # some rows routed to other ranks
    flat, rows = ids.flatten(), ids.numel()
    gammas = torch.rand(rows, device="cuda")
    route = experts_hip.route(flat, start, nexp, 64)
    out_dtype = torch.float32 if nvfp4 else torch.bfloat16
    w_amax = torch.rand(nexp, device="cuda") + 0.5
    denominator = torch.tensor([3.0], device="cuda")

    def run(reduce):
        def gemm(*args):
            partials = experts_hip.expert_gemm(*args, num_experts=64)
            return _sum_splits(partials, out_dtype) if reduce else partials

        acc = gemm(xq, w13, s13, route, n2, top_k, out_dtype, rows)
        if nvfp4:
            hidden, h_amax = marlin_sm90.swiglu_fake_quantize(
                acc, flat, gammas, (x_amax, w_amax, denominator), start, nexp, top_k,
                caps, act,
            )  # fmt: skip
            scales = (h_amax, w_amax, denominator)
        else:
            hidden = marlin_sm90.swiglu_fake_quantize_mxfp8(
                acc, flat, gammas, start, nexp, caps
            )
            h_amax, scales = None, (None, None, None)
        down = gemm(hidden, w2, s2, route, k, 1, out_dtype, rows)
        sums = marlin_sm90.source_partial(
            down, flat, scales, start, nexp, top_k, round_bf16=not nvfp4
        )
        return hidden, h_amax, sums

    ref, fused = run(True), run(False)
    # Rows routed to other ranks are left unset; compare this rank's rows.
    local = (flat >= start) & (flat < start + nexp)
    bits = torch.int16 if ref[0].element_size() == 2 else torch.int8
    assert torch.equal(ref[0][local].view(bits), fused[0][local].view(bits))
    if nvfp4:
        assert torch.equal(ref[1][local], fused[1][local])
    assert torch.equal(ref[2].view(torch.int32), fused[2].view(torch.int32))


@pytest.mark.parametrize("m", [1, 7, 64])
@pytest.mark.parametrize("local", [False, True])
def test_prologue_stores_kv_pool(m, local):
    """The ROCm decode prologue with kv_pool writes each row's K and V into the KV
    cache pool at its loc: bitwise what the attention backend's save stores (the
    same rows, padding rows included), with its other outputs unchanged."""
    from sglang.kernels.ops.inkling2.prologue import attention_prologue

    torch.manual_seed(m + local)
    H, KVH, D, DR, W = 6, 1, 128, 16, 4
    sizes = [H * D, KVH * D, KVH * D, H * DR]
    proj = (torch.randn(m, sum(sizes) + 8, device="cuda") * 3).bfloat16()
    q, k, v, r = proj[:, : sum(sizes)].split(sizes, dim=1)
    slots = 3 * m + 5
    idx = (torch.randperm(slots - 1, device="cuda") + 1)[:m]
    meta = {
        "safe_idx": idx,
        "cache_mask": torch.ones(m, 1, 1, device="cuda", dtype=torch.bool),
        "cu": torch.arange(0, m + 1, device="cuda"),
        "si": torch.arange(m, device="cuda", dtype=torch.int32),
    }
    if local:
        kw, vw = (
            torch.randn(KVH * D, W, device="cuda"),
            torch.randn(KVH * D, W, device="cuda"),
        )
        dyn = {}
    else:
        kw, vw = (
            torch.randn(W, KVH * D, device="cuda"),
            torch.randn(W, KVH * D, device="cuda"),
        )
        dyn = dict(
            k_dynamic=torch.randn(m, KVH * W, device="cuda").bfloat16(),
            v_dynamic=torch.randn(m, KVH * W, device="cuda").bfloat16(),
        )
    q_scale = torch.rand(H, D, device="cuda") * 2
    r_scale = torch.rand(H, DR, device="cuda") * 2
    rel_proj = (torch.randn(DR, 1024, device="cuda") * 3).bfloat16()
    seq_lens = torch.full((m,), 5000, device="cuda", dtype=torch.int32)
    pool_slots = 4 * m + 16
    loc = (torch.randperm(pool_slots - 1, device="cuda") + 1)[:m].long()
    if m > 1:
        loc[-1] = 0  # a CUDA-graph padding row writes the reserved slot like the save

    def run(fused):
        kc = torch.ones(slots, W - 1, KVH * D, device="cuda").bfloat16()
        vc = kc.clone() * 2
        k_pool = torch.zeros(pool_slots, KVH, D, device="cuda", dtype=torch.bfloat16)
        v_pool = torch.zeros_like(k_pool)
        out = attention_prologue(
            q, k, v, r, H, KVH, DR, H // KVH, kw, vw, kc, vc, meta, q_scale, r_scale,
            rel_proj, seq_lens, local, 22.627416997969522, 8.0, 1e-6, 1e-6,
            kv_pool=(k_pool, v_pool, loc) if fused else None, **dyn,
        )  # fmt: skip
        if not fused:
            k_pool[loc], v_pool[loc] = out[1], out[2]
        return (*out, kc, vc, k_pool, v_pool)

    for a, b in zip(run(True), run(False)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("local_heavy", [False, True])
@pytest.mark.parametrize("tokens", [1, 2, 3, 4])
def test_fused_route_matches_separate_kernels(rank, with_bias, local_heavy, tokens):
    """route_topk_metadata (one token, ROCm decode) is bitwise route_topk followed
    by route_metadata: weights, ids and every routing metadata tensor (local_heavy
    routes all eight ids to this rank's experts, filling the schedule)."""
    from sglang.kernels.ops.inkling2.route import route_metadata, route_topk
    from sglang.kernels.ops.inkling2.route_hip import route_topk_metadata

    experts, num_local, block_m = 384, 48, 16
    one = torch.ones((), device="cuda")
    start = rank * num_local
    for seed in range(20 if tokens == 1 else 6):
        torch.manual_seed(
            1000 * rank + 100 * with_bias + 10 * local_heavy + seed + 7 * tokens
        )
        partials = torch.randn(5, tokens, experts, device="cuda") * 3
        if local_heavy:
            partials[0, :, start : start + num_local] += 30
        bias = torch.randn(experts, device="cuda") * 0.1 if with_bias else None
        _, ids, weights = route_topk(partials, bias, 8, 8.0, one)
        meta = route_metadata(
            ids.flatten(), weights.flatten(), start, num_local, experts, block_m
        )
        f_weights, f_ids, f_meta, quantized = route_topk_metadata(
            partials, bias, 8.0, one, start, num_local, block_m
        )
        assert quantized is None
        assert torch.equal(f_ids, ids)
        assert torch.equal(f_weights.view(torch.int32), weights.view(torch.int32))
        for a, b in zip(f_meta, meta):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize(
    "quant,out_dtype",
    [("nvfp4", torch.bfloat16), ("nvfp4", torch.float16), ("mxfp8", torch.bfloat16)],
)
@pytest.mark.parametrize("width", [4608, 1000])
@pytest.mark.parametrize("tokens", [1, 3, 4])
def test_fused_route_quantizes_activations(quant, out_dtype, width, tokens):
    """route_topk_metadata's extra programs fake-quantize the token's activations
    bitwise as fake_quantize / fake_quantize_mxfp8, with the routing unchanged."""
    from sglang.kernels.ops.inkling2 import experts_sm90
    from sglang.kernels.ops.inkling2.route_hip import route_topk_metadata

    one = torch.ones((), device="cuda")
    for seed in range(10):
        torch.manual_seed(seed)
        partials = torch.randn(5, tokens, 384, device="cuda") * 3
        bias = torch.randn(384, device="cuda") * 0.1
        x = (torch.randn(tokens, width, device="cuda") * (seed + 1)).to(torch.bfloat16)
        x[0, seed * 7] = 0.0
        plain = route_topk_metadata(partials, bias, 8.0, one, 48, 48, 16)
        fused = route_topk_metadata(
            partials, bias, 8.0, one, 48, 48, 16, x, quant, out_dtype
        )
        for a, b in zip(
            (plain[0], plain[1], *plain[2]), (fused[0], fused[1], *fused[2])
        ):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        xq, x_amax = fused[3]
        if quant == "nvfp4":
            ref, ref_amax = experts_sm90.fake_quantize(x, out_dtype)
            assert torch.equal(x_amax.view(torch.int32), ref_amax.view(torch.int32))
        else:
            ref = experts_sm90.fake_quantize_mxfp8(x)
            assert x_amax is None
        assert xq.dtype == ref.dtype
        assert torch.equal(xq.view(torch.int16), ref.view(torch.int16))


def test_router_matches_exact_sigmoid_selection():
    """route_topk (the gate's top-k at every row count) selects the experts of torch's
    sigmoid plus bias, with renorm_topk_logits_scaled's weights for them."""
    from sglang.kernels.ops.inkling2.route import route_topk
    from sglang.srt.models.inkling_common.moe import renorm_topk_logits_scaled

    torch.manual_seed(0)
    one = torch.ones((), device="cuda")
    for m in (1, 7, 64, 512):
        logits = torch.randn(m, 384, device="cuda") * 3
        bias = torch.randn(384, device="cuda") * 0.1
        ids_ref = (torch.sigmoid(logits) + bias).topk(8, dim=-1).indices
        _, ids, w = route_topk(logits[None], bias, 8, 8.0, one)
        ids = ids.long()
        assert torch.equal(ids.sort(-1).values, ids_ref.sort(-1).values)
        w_ref, _ = renorm_topk_logits_scaled(logits, ids, 0, 8.0, one)
        assert (w - w_ref).abs().max().item() < 1e-6


def test_dynamic_conv_projection_partials():
    """On ROCm the global layers' dynamic conv projections (4 outputs per rank) take
    the paired split-K partials path, which uses_k_groups otherwise leaves to
    hipBLASLt; the summed partials match a float64 reference."""
    from sglang.kernels.ops.inkling2.linear import (
        linear_k_group_partials_pair,
        uses_k_groups,
    )

    for m in (1, 7, 64, 128):
        torch.manual_seed(m)
        x, x2 = (torch.randn(2, m, 4608, device="cuda") * 2).bfloat16()
        w, w2 = (torch.randn(2, 4, 4608, device="cuda") * 0.02).bfloat16()
        assert not uses_k_groups(x, w)
        assert uses_k_groups(x, w, partials=True)
        partials = linear_k_group_partials_pair(x, w, x2, w2)
        assert partials.shape == (2, 5, m, 4)
        for p, a, b in zip(partials, (x, x2), (w, w2)):
            ref = a.double() @ b.double().T
            assert (p.double().sum(0) - ref).abs().max().item() < 1e-3
    assert not uses_k_groups(
        torch.randn(129, 4608, device="cuda").bfloat16(), w, partials=True
    )


@pytest.mark.parametrize("inkling", [True, False])
@pytest.mark.parametrize(
    "batch,seq,splits,bound",
    [
        (1, 2560, 80, 128),
        (1, 2560, 40, 64),
        (4, 999, 16, 16),
        (3, 100, 2, 64),
        (2, 9000, 64, 64),
    ],
)
def test_decode_stage2_tree_merge_matches_loop(
    monkeypatch, inkling, batch, seq, splits, bound
):
    """ROCm's stage 2 tree merge of the splits matches the online-softmax loop to
    within one BF16 ulp (it reassociates the FP32 merge), and almost all outputs are
    bitwise equal: Inkling2's relative bias with the gate / sink epilogue, and plain
    GQA with sinks and the LSE output, including splits past the sequence end."""
    from sglang.kernels.ops.attention import decode_attention as da
    from sglang.kernels.ops.attention.score_mod import relative_bias_score_mod

    torch.manual_seed(batch * 1000 + seq)
    H, HKV, D = (6, 1, 128) if inkling else (32, 8, 128)
    total = batch * seq
    k = torch.randn(total, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kvp = torch.arange(0, total + 1, seq, device="cuda", dtype=torch.int32)
    kvi = torch.randperm(total, device="cuda").long()
    q = torch.randn(batch, H, D, device="cuda", dtype=torch.bfloat16)
    extra = (
        dict(
            score_mod=relative_bias_score_mod,
            aux_tensors=[torch.randn(batch, H, 1024, device="cuda").bfloat16()],
            gate_sink=(
                torch.randn(batch, H * D, device="cuda").bfloat16(),
                torch.randn(batch, H, device="cuda").bfloat16(),
            ),
        )
        if inkling
        else dict(
            sinks=torch.randn(H, device="cuda"),
            lse_out=torch.empty(batch, H, device="cuda"),
        )
    )
    nsplit = torch.full((batch,), splits, device="cuda", dtype=torch.int32)

    def run():
        o = torch.empty(batch, H, D, device="cuda", dtype=torch.bfloat16)
        logits = torch.empty(batch, H, bound, D, device="cuda")
        lse = torch.empty(batch, H, bound, device="cuda")
        da.decode_attention_fwd(
            q, k, v, o, kvp, kvi, logits, lse, nsplit, bound, D**-0.5, 1.0, 1.0,
            **extra,
        )  # fmt: skip
        return o

    tree = run()
    monkeypatch.setattr(da, "STAGE2_TREE_MAX_ELEMENTS", 0)
    loop = run()
    torch.testing.assert_close(tree.float(), loop.float(), rtol=2**-7, atol=1e-6)
    assert (tree != loop).sum().item() <= max(1, tree.numel() // 500)


@pytest.mark.parametrize("shape", [(1, 4608), (40, 4608), (8, 6, 128)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
def test_hip_rmsnorm_triton_matches_native(shape, weight_dtype, with_residual):
    """ROCm without vLLM's kernels: an opted-in RMSNorm's (the DSpark draft's)
    forward_hip Triton kernel matches forward_native, the residual sum bitwise and
    the output to within one BF16 ulp."""
    from sglang.srt.layers.layernorm import RMSNorm

    torch.manual_seed(len(shape) + with_residual)
    norm = RMSNorm(shape[-1], eps=1e-6).cuda()
    # only opted-in norms take the Triton kernel; other models keep forward_native
    assert not norm.uses_hip_triton()
    norm.hip_triton = True
    norm.weight.data = (torch.rand(shape[-1], device="cuda") + 0.5).to(weight_dtype)
    x = (torch.randn(*shape, device="cuda") * 3).bfloat16()
    residual = torch.randn_like(x) if with_residual else None
    fast = norm.forward_hip(x.clone(), None if residual is None else residual.clone())
    ref = norm.forward_native(x.clone(), None if residual is None else residual.clone())
    if with_residual:
        assert torch.equal(fast[1], ref[1])
        fast, ref = fast[0], ref[0]
    assert fast.shape == ref.shape and fast.dtype == ref.dtype
    torch.testing.assert_close(fast.float(), ref.float(), rtol=2**-7, atol=1e-6)


@pytest.mark.parametrize("causal", [False, True])
def test_rel_verify_attention_without_bias_matches_attention(causal):
    """The DSpark draft's short blocks on ROCm: rel_verify_attention with a
    zero-extent bias is plain attention over the cached prefix plus the block's own
    keys (bidirectional, or causal), against a float64 reference."""
    from sglang.kernels.ops.inkling2.attention_hip import rel_verify_attention

    torch.manual_seed(causal)
    H, HKV, D, ext = 6, 1, 128, 8
    prefixes = (0, 5, 700, 3000)
    batch = len(prefixes)
    pool = sum(prefixes) + 7
    k_buf = torch.randn(pool, HKV, D, device="cuda").bfloat16()
    v_buf = torch.randn_like(k_buf)
    kv_indices = torch.randperm(pool, device="cuda")[: sum(prefixes)].long()
    kv_indptr = torch.tensor(
        [0, *torch.tensor(prefixes).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    qo_indptr = torch.arange(0, batch * ext + 1, ext, device="cuda", dtype=torch.int32)
    q = torch.randn(batch * ext, H, D, device="cuda").bfloat16()
    k_ext = torch.randn(batch * ext, HKV, D, device="cuda").bfloat16()
    v_ext = torch.randn_like(k_ext)
    o = torch.empty_like(q)
    lse = torch.empty(batch * ext, H, device="cuda")
    rel_verify_attention(
        q, k_ext, v_ext, k_buf, v_buf, q.new_zeros((1, 1, 1))[:, :, :0], o, lse,
        qo_indptr, kv_indptr, kv_indices, ext, D**-0.5, causal=causal,
    )  # fmt: skip
    for b, p in enumerate(prefixes):
        idx = kv_indices[kv_indptr[b] : kv_indptr[b] + p]
        keys = torch.cat([k_buf[idx, 0], k_ext[b * ext : (b + 1) * ext, 0]]).double()
        vals = torch.cat([v_buf[idx, 0], v_ext[b * ext : (b + 1) * ext, 0]]).double()
        logits = q[b * ext : (b + 1) * ext].double() @ keys.T * D**-0.5
        if causal:
            pos = torch.arange(ext, device="cuda")
            hide = torch.cat(
                [
                    torch.zeros(ext, p, dtype=torch.bool, device="cuda"),
                    pos[None, :] > pos[:, None],
                ],
                1,
            )
            logits = logits.masked_fill(hide[:, None, :], float("-inf"))
        ref = torch.softmax(logits, -1) @ vals
        assert _rel(o[b * ext : (b + 1) * ext], ref) < 5e-3, (b, p)


@pytest.mark.parametrize("seqs", [1, 3])
@pytest.mark.parametrize("local", [False, True])
def test_verify_prologue_stores_kv_pool(seqs, local):
    """The MTP verify prologue (verify_windows, 4 rows per sequence) with kv_pool
    writes each row's K and V into the pool at its loc: bitwise the backend's save,
    with the prologue's other outputs and windows unchanged."""
    from sglang.kernels.ops.inkling2.prologue import attention_prologue

    torch.manual_seed(seqs + 10 * local)
    H, KVH, D, DR, W, per = 6, 1, 128, 16, 4, 4
    m = seqs * per
    sizes = [H * D, KVH * D, KVH * D, H * DR]
    proj = (torch.randn(m, sum(sizes) + 8, device="cuda") * 3).bfloat16()
    q, k, v, r = proj[:, : sum(sizes)].split(sizes, dim=1)
    slots = 3 * seqs + 5
    meta = {
        "safe_idx": (torch.randperm(slots - 1, device="cuda") + 1)[:seqs],
        "cache_mask": torch.ones(seqs, 1, 1, device="cuda", dtype=torch.bool),
        "cu": torch.arange(0, m + 1, per, device="cuda"),
        "si": torch.arange(seqs, device="cuda", dtype=torch.int32).repeat_interleave(
            per
        ),
    }
    if local:
        kw, vw = (torch.randn(KVH * D, W, device="cuda") for _ in range(2))
        dyn = {}
    else:
        kw, vw = (torch.randn(W, KVH * D, device="cuda") for _ in range(2))
        dyn = dict(
            k_dynamic=torch.randn(m, KVH * W, device="cuda").bfloat16(),
            v_dynamic=torch.randn(m, KVH * W, device="cuda").bfloat16(),
        )
    q_scale = torch.rand(H, D, device="cuda") * 2
    r_scale = torch.rand(H, DR, device="cuda") * 2
    rel_proj = (torch.randn(DR, 1024, device="cuda") * 3).bfloat16()
    seq_lens = torch.full((seqs,), 5000, device="cuda", dtype=torch.int32)
    pool_slots = 4 * m + 16
    loc = (torch.randperm(pool_slots - 1, device="cuda") + 1)[:m].long()

    def run(fused):
        kc = torch.ones(slots, W - 1, KVH * D, device="cuda").bfloat16()
        vc = kc.clone() * 2
        windows = tuple(
            torch.zeros(seqs, per, W - 1, KVH * D, device="cuda").bfloat16()
            for _ in range(2)
        )
        k_pool = torch.zeros(pool_slots, KVH, D, device="cuda", dtype=torch.bfloat16)
        v_pool = torch.zeros_like(k_pool)
        out = attention_prologue(
            q, k, v, r, H, KVH, DR, H // KVH, kw, vw, kc, vc, meta, q_scale, r_scale,
            rel_proj, seq_lens, local, 22.627416997969522, 8.0, 1e-6, 1e-6,
            verify_windows=windows,
            kv_pool=(k_pool, v_pool, loc) if fused else None, **dyn,
        )  # fmt: skip
        if not fused:
            k_pool[loc], v_pool[loc] = out[1], out[2]
        return (*out, kc, vc, *windows, k_pool, v_pool)

    for a, b in zip(run(True), run(False)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("batch,ext", [(1, 8), (3, 8), (2, 5)])
def test_rel_verify_gate_sink_in_combine(batch, ext):
    """rel_verify_attention(gate_sink=...) writes, from its own reduction, bitwise
    what gate_sink(o, gate, lse, sinks) gives after it (the ROCm MTP verify path),
    with o and lse unchanged."""
    from sglang.kernels.ops.inkling2.attention import gate_sink
    from sglang.kernels.ops.inkling2.attention_hip import rel_verify_attention

    torch.manual_seed(batch * 10 + ext)
    H, HKV, D = 6, 1, 128
    prefixes = [700 + 1000 * i for i in range(batch)]
    pool = sum(prefixes) + 7
    k_buf = torch.randn(pool, HKV, D, device="cuda").bfloat16()
    v_buf = torch.randn_like(k_buf)
    kv_indices = torch.randperm(pool, device="cuda")[: sum(prefixes)].long()
    kv_indptr = torch.tensor(
        [0, *torch.tensor(prefixes).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    qo_indptr = torch.arange(0, batch * ext + 1, ext, device="cuda", dtype=torch.int32)
    tokens = batch * ext
    q = torch.randn(tokens, H, D, device="cuda").bfloat16()
    k_ext = torch.randn(tokens, HKV, D, device="cuda").bfloat16()
    v_ext = torch.randn_like(k_ext)
    bias = torch.randn(tokens, H, 1024, device="cuda").bfloat16()
    proj = torch.randn(tokens, H * D + H + 4, device="cuda").bfloat16()
    gate, sinks = proj[:, : H * D], proj[:, H * D : H * D + H]

    def run(fused):
        o = torch.empty_like(q)
        lse = torch.empty(tokens, H, device="cuda")
        y = torch.empty(tokens, H * D, device="cuda", dtype=q.dtype)
        rel_verify_attention(
            q, k_ext, v_ext, k_buf, v_buf, bias, o, lse, qo_indptr, kv_indptr,
            kv_indices, ext, D**-0.5, gate_sink=(gate, sinks, y) if fused else None,
        )  # fmt: skip
        if not fused:
            y = gate_sink(o.view(tokens, -1), gate, lse, sinks, output_dtype=q.dtype)
        return o, lse, y

    for a, b in zip(run(True), run(False)):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    "quant,out_dtype",
    [("nvfp4", torch.bfloat16), ("nvfp4", torch.float16), ("mxfp8", torch.bfloat16)],
)
@pytest.mark.parametrize("tokens", [2, 8, 16, 64, 128])
def test_route_metadata_quantize_matches_separate(quant, out_dtype, tokens):
    """route_metadata_quantize is bitwise route_metadata followed by the rows' fake
    quantization (fake_quantize / fake_quantize_mxfp8), for 16 to 1024 ids."""
    from sglang.kernels.ops.inkling2 import experts_sm90
    from sglang.kernels.ops.inkling2.route import route_metadata
    from sglang.kernels.ops.inkling2.route_hip import route_metadata_quantize

    torch.manual_seed(tokens + len(quant))
    ids = torch.stack(
        [torch.randperm(384, device="cuda")[:8] for _ in range(tokens)]
    ).int()
    weights = torch.rand(tokens, 8, device="cuda")
    x = (torch.randn(tokens, 4608, device="cuda") * 2).bfloat16()
    for start in (0, 144):
        meta = route_metadata(ids.flatten(), weights.flatten(), start, 48, 384, 16)
        f_meta, (xq, x_amax) = route_metadata_quantize(
            ids.flatten(), weights.flatten(), start, 48, 384, 16, x, quant, out_dtype
        )
        for a, b in zip(f_meta, meta):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        if quant == "nvfp4":
            ref, ref_amax = experts_sm90.fake_quantize(x, out_dtype)
            assert torch.equal(x_amax.view(torch.int32), ref_amax.view(torch.int32))
        else:
            ref = experts_sm90.fake_quantize_mxfp8(x)
            assert x_amax is None
        assert torch.equal(xq.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("reqs,width", [(1, 8), (3, 8), (4, 16)])
def test_ctx_kv_store_matches_separate(heads, reqs, width):
    """ctx_kv_store writes the DSpark draft's context K / V rows bitwise as
    write_target_hidden_kv's separate kernels do: k_norm_ctx (Triton RMSNorm), the
    rotary kernel through apply_k_rope, and set_kv_buffer_prefix_valid's store."""
    from sglang.kernels.ops.inkling2.mtp_hip import ctx_kv_store
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_prefix_valid_impl

    torch.manual_seed(heads * 100 + reqs * 10 + width)
    norm = RMSNorm(D, eps=1e-6).cuda()
    norm.hip_triton = True  # as use_triton_norms sets it on the draft
    assert norm.uses_hip_triton()
    norm.weight.data = (torch.rand(D, device="cuda") + 0.5).bfloat16()
    rope = get_rope(D, rotary_dim=D, max_position=262144, base=1e6).cuda()
    rows, size = reqs * width, 4096
    kv = (torch.randn(rows, 2 * heads * D, device="cuda") * 4).bfloat16()
    positions = torch.randint(0, 262144, (rows,), device="cuda")
    loc_2d = torch.randperm(size, device="cuda")[:rows].view(reqs, width)
    commit_lens = torch.tensor([width, 0, 3, 5][:reqs], device="cuda").int()
    cache = torch.randn(2, size, heads, D, device="cuda").bfloat16()
    k_ref, v_ref = cache.clone().unbind(0)
    k_out, v_out = cache.clone().unbind(0)

    k, v = kv.split([heads * D, heads * D], dim=-1)
    k = norm(k.reshape(-1, D)).view_as(k)
    _, k = rope(positions, k.new_empty(k.shape), k)
    _set_kv_buffer_prefix_valid_impl(
        k.view(-1, heads, D),
        v.view(-1, heads, D),
        k_ref,
        v_ref,
        loc_2d,
        commit_lens,
        heads * D,
        torch.bfloat16,
    )
    ctx_kv_store(
        kv,
        positions,
        norm.weight,
        1e-6,
        rope.cos_sin_cache,
        k_out,
        v_out,
        loc_2d,
        commit_lens,
    )
    assert torch.equal(k_out.view(torch.int16), k_ref.view(torch.int16))
    assert torch.equal(v_out.view(torch.int16), v_ref.view(torch.int16))


@pytest.mark.parametrize("reqs", [1, 3, 16])
def test_mix_rmsnorm_last_matches_gather_and_copy(reqs):
    """mix_rmsnorm_last equals gathering each request's last committed row,
    mix_rmsnorm and index_copy_ into the f_rnn buffer (Inkling2 MTP's
    store_f_rnn after verify), bitwise."""
    from sglang.kernels.ops.inkling2.mtp import mix_rmsnorm
    from sglang.kernels.ops.inkling2.mtp_hip import mix_rmsnorm_last

    torch.manual_seed(reqs)
    width, taps, hidden, slots = 8, 8, 4608, 64
    targets = torch.randn(reqs * width, taps, hidden, device="cuda").bfloat16()
    commit_lens = torch.randint(1, width + 1, (reqs,), device="cuda").int()
    req_slots = torch.randperm(slots, device="cuda")[:reqs]
    alpha = torch.softmax(torch.randn(taps, device="cuda"), 0).bfloat16()
    weight = (torch.rand(hidden, device="cuda") + 0.5).bfloat16()
    buf = torch.randn(slots, hidden, device="cuda").bfloat16()
    expected = buf.clone()
    rows = torch.arange(reqs, device="cuda")
    last = targets.view(reqs, width, taps, hidden)[rows, commit_lens.long() - 1]
    expected.index_copy_(0, req_slots, mix_rmsnorm(last, alpha, weight, 1e-6))
    mix_rmsnorm_last(targets, width, commit_lens, req_slots, alpha, weight, 1e-6, buf)
    assert torch.equal(buf.view(torch.int16), expected.view(torch.int16))


def test_fused_kv_write_declines_translated_locs(monkeypatch):
    """A unified (translating) KV pool hands the batch virtual write locs, which the
    prologue must not scatter into: decode graph capture raised on them."""
    from types import SimpleNamespace

    from sglang.srt.layers.attention import triton_backend

    monkeypatch.setattr(triton_backend, "_is_hip", True)
    k_buffer = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    v_buffer = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    kv_layers = SimpleNamespace(
        read=lambda layer_id: SimpleNamespace(kv=lambda: (k_buffer, v_buffer))
    )
    layer = SimpleNamespace(
        layer_id=0, is_cross_attention=False, k_scale=None,
        tp_k_head_num=1, tp_v_head_num=1, qk_head_dim=8, v_head_dim=8,
    )  # fmt: skip
    batch = SimpleNamespace(
        out_cache_loc=torch.zeros(2, dtype=torch.int64), out_cache_loc_is_physical=True
    )
    for translating in (True, False):
        backend = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(dtype=torch.bfloat16),
            kv_layers=kv_layers,
            use_mla=False,
            use_sliding_window_kv_pool=False,
            dcp_size=1,
            runner_ids=SimpleNamespace(is_translating=translating),
            forward_metadata=SimpleNamespace(swa_out_cache_loc=None),
        )
        buffers = triton_backend.TritonAttnBackend.get_fused_kv_write_buffers(
            backend, layer, batch
        )
        if translating:
            assert buffers is None
        else:
            assert buffers[0] is k_buffer and buffers[1] is v_buffer
            assert torch.equal(buffers[2], batch.out_cache_loc)


@pytest.mark.parametrize("rows", [1, 8, 16, 64, 128])
def test_apply_wo_ud_matches_hipblaslt(rows):
    """Attention's wo_ud on ROCm at TP8 (K 768): aiter's Triton GEMM where it routes, up to
    64 rows, is bitwise F.linear (hipBLASLt); larger batches keep the layer's own path."""
    from sglang.srt.models.inkling_common import inkling2_attn_hip

    if inkling2_attn_hip._aiter_a16w16 is None:
        pytest.skip("aiter's Triton GEMM needs gfx95")

    class OutProjection(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight, self.bias, self.calls = weight, None, 0

        def forward(self, x):
            self.calls += 1
            return torch.nn.functional.linear(x, self.weight), None

    torch.manual_seed(rows)
    wo_ud = OutProjection((torch.randn(4608, 768, device="cuda") * 0.03).bfloat16())
    for scale in (1, 3):
        x = (torch.randn(rows, 768, device="cuda") * scale).bfloat16()
        out = inkling2_attn_hip.apply_wo_ud(wo_ud, x)
        assert torch.equal(out, torch.nn.functional.linear(x, wo_ud.weight))
    assert wo_ud.calls == (0 if rows <= 64 else 2)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [160, 256, 512])
def test_expert_gemm_large_batch_tiles(fmt, tokens):
    """1,280-4,096 routed rows over 384 experts (MTP verify above c16) take the
    large-batch tile (16, 64, 256), which equals float64 up to summation order."""
    from sglang.kernels.ops.inkling2 import experts_hip, experts_sm90
    from sglang.kernels.ops.inkling2.experts_hip import GemmConfig

    torch.manual_seed(tokens)
    nexp, n, k, top_k, start, num_experts = 48, 512, 1024, 8, 48, 384
    if fmt == "nvfp4":
        w = torch.randint(0, 256, (nexp, n, k // 2), device="cuda", dtype=torch.uint8)
        s = (torch.rand(nexp, n, k // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    else:
        w = (torch.randn(nexp, n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
        s = torch.randint(
            120, 130, (nexp, n, k // 32), device="cuda", dtype=torch.uint8
        )
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 2
    xq = (
        experts_sm90.fake_quantize(x, experts_hip.activation_dtype(w.dtype))[0]
        if fmt == "nvfp4"
        else experts_sm90.fake_quantize_mxfp8(x)
    )
    ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(tokens)]
    ).int()
    rows = ids.numel()
    cfg = experts_hip.default_config(rows, num_experts, fmt == "nvfp4")
    assert cfg == GemmConfig(16, 64, 256, 1, 4, 1)
    route = experts_hip.route(ids.flatten(), start, nexp, num_experts, block_m=cfg.bm)
    out = experts_hip.expert_gemm(
        xq, w, s, route, n, top_k, torch.float32, rows, num_experts=num_experts
    )[0]
    flat = ids.flatten().long()
    local = ((flat >= start) & (flat < start + nexp)).nonzero().flatten()
    assert local.numel() > 0
    wd = _dequant(w, s)
    xr = xq.double()[local // top_k]
    ref = torch.stack([xr[i] @ wd[flat[r] - start].T for i, r in enumerate(local)])
    assert _rel(out[local], ref) < (1e-7 if fmt == "nvfp4" else 1e-6)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("gather", [8, 1])
@pytest.mark.parametrize("tokens", [1, 8, 32, 128, 256, 384, 1024])
def test_expert_gemm_mfma_order(monkeypatch, fmt, tokens, gather):
    """Weights in MFMA order: 16-row tiles run the Gluon kernel and the others the
    Triton tile on that order, both bitwise the checkpoint layout's partials in
    every default_config band (8-8,192 rows over 64 experts), and run to run."""
    from sglang.kernels.ops.inkling2 import experts_hip, experts_sm90

    if not experts_hip.MFMA_ORDER:
        pytest.skip("MFMA-order experts need gfx950 and Gluon")
    gluon = experts_hip.experts_gluon
    torch.manual_seed(tokens + gather)
    # w2's 2,304-wide K splits into 3-tile loops (the unrolled NVFP4 path) at <= 64 rows.
    k = 1024 if gather == 8 else 2304
    nexp, n, top_k, start, num_experts = 16, 512, 8, 16, 64
    nvfp4 = fmt == "nvfp4"
    if nvfp4:
        w = torch.randint(0, 256, (nexp, n, k // 2), device="cuda", dtype=torch.uint8)
        s = (torch.rand(nexp, n, k // 16, device="cuda") * 2 + 0.25).to(
            torch.float8_e4m3fn
        )
    else:
        w = (torch.randn(nexp, n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
        s = torch.randint(
            120, 130, (nexp, n, k // 32), device="cuda", dtype=torch.uint8
        )
    rows = tokens * top_k
    x = torch.randn(rows // gather, k, device="cuda", dtype=torch.bfloat16) * 2
    xq = (
        experts_sm90.fake_quantize(x, experts_hip.activation_dtype(w.dtype))[0]
        if nvfp4
        else experts_sm90.fake_quantize_mxfp8(x)
    )
    ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:top_k] for _ in range(tokens)]
    ).int()
    cfg = experts_hip.default_config(rows, num_experts, nvfp4, mfma_order=True)
    route = experts_hip.route(ids.flatten(), start, nexp, num_experts, block_m=cfg.bm)
    calls = []
    kernel = gluon.grouped_gemm
    monkeypatch.setattr(gluon, "grouped_gemm", lambda *a: calls.append(a) or kernel(*a))
    args = (route, n, gather, torch.float32, rows)
    plain = experts_hip.expert_gemm(xq, w, s, *args, num_experts=num_experts)
    ordered = gluon.mfma_order_weights(w), gluon.mfma_order_scales(s)
    order = [
        experts_hip.expert_gemm(
            xq,
            *ordered,
            *args,
            num_experts=num_experts,
            mfma_order=True,
        )  # fmt: skip
        for _ in range(4)
    ]
    assert len(calls) == 4 * (cfg.bm == 16)
    flat = ids.flatten()
    local = ((flat >= start) & (flat < start + nexp)).nonzero().flatten()
    assert local.numel() > 0
    assert all(torch.equal(plain[:, local], o[:, local]) for o in order)


@pytest.mark.parametrize("precision", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [3, 40, 600])
def test_experts_load_in_mfma_order(monkeypatch, precision, tokens):
    """ROCm quantized experts load in MFMA order: their forward (Gluon at 6 and 80
    routed rows, the Triton MFMA-order tile at 1,200) is bitwise the same experts
    written in the checkpoint layout, and a reload rewrites the buffers in place."""
    from types import SimpleNamespace

    from sglang.kernels.ops.inkling2 import experts_hip
    from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.inkling_common import inkling2_moe as moe
    from sglang.test.inkling2_test_utils import precision_group

    if not experts_hip.MFMA_ORDER:
        pytest.skip("MFMA-order experts need gfx950 and Gluon")
    parallel = SimpleNamespace(
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(all_reduce=lambda x: x),
    )
    monkeypatch.setattr(moe, "get_parallel", lambda: parallel)
    prefix = "model.llm.layers.0.mlp.experts."
    names = [prefix + name + "_weight" for name in ("w13", "w2")]
    quant = Inkling2QuantConfig(
        {"model": {"parameter_groups": [precision_group(precision, names)]}}
    )
    config = SimpleNamespace(
        hidden_size=256,
        intermediate_size=256,
        mlp_gate_softcap=4,
        mlp_up_softcap=25,
        num_experts_per_tok=2,
        route_scale=8.0,
    )
    torch.manual_seed(tokens)
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        plain = moe.Inkling2Experts(config, 8, prefix, quant)
        loaded = moe.Inkling2Experts(config, 8, prefix, quant)
        x = torch.randn(tokens, 256)
        ids = torch.randint(0, 8, (tokens, 2), dtype=torch.int32)
        gammas = torch.rand(tokens, 2, dtype=torch.float32)
        buffers = [p.data_ptr() for p in loaded.parameters()]
        for _ in range(2):
            checkpoint = {}
            for name, value in plain.named_parameters():
                if name.endswith("outer_scale_denominator"):
                    checkpoint[name] = torch.full_like(value, 2688)
                elif value.dtype == torch.uint8:
                    # NVFP4 weights; MXFP8 ue8m0 scales near 1
                    checkpoint[name] = (
                        torch.randint_like(value, 0, 256)
                        if precision == "nvfp4"
                        else torch.randint_like(value, 120, 130)
                    )
                else:
                    checkpoint[name] = (torch.rand(value.shape) + 0.125).to(value.dtype)
                value.copy_(checkpoint[name])
            for name, value in loaded.named_parameters():
                value.weight_loader(value, checkpoint[name])
            plain.swizzle_nvfp4_scales()
            loaded.swizzle_nvfp4_scales()
            assert [p.data_ptr() for p in loaded.parameters()] == buffers
            assert loaded.in_mfma_order("w13") and loaded.in_mfma_order("w2")
            assert not plain.in_mfma_order("w13") and not plain.in_mfma_order("w2")
            assert not torch.equal(loaded.w13_weight, plain.w13_weight)
            expected = plain(x, gammas, ids)
            actual = loaded(x, gammas, ids)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("bs", [1, 3, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("window", [None, 4096])
def test_verify_indptrs_matches_torch(bs, dtype, window):
    """verify_indptrs (target-verify metadata in one launch) writes the integers the
    triton backend's separate torch ops computed, into the same buffers."""
    from sglang.kernels.ops.attention.metadata import verify_indptrs

    torch.manual_seed(bs)
    tokens = 8
    seq_lens = torch.randint(1, 3 * 4096, (bs,), device="cuda", dtype=dtype)
    bufs = [
        torch.zeros(bs + 9, device="cuda", dtype=d)
        for d in (torch.int64, torch.int32, torch.int64, torch.int32, torch.int32)
    ]
    qo, kv, mask, win, offsets = (b.clone() for b in bufs)
    ref_qo, ref_kv, ref_mask, ref_win, ref_offsets = bufs
    lens, start = verify_indptrs(
        seq_lens, tokens, qo[: bs + 1], kv[: bs + 1], mask[: bs + 1], window, win,
        offsets,
    )  # fmt: skip
    ref_qo[: bs + 1] = torch.arange(
        0, (1 + bs) * tokens, step=tokens, dtype=torch.int32, device="cuda"
    )
    ref_kv[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
    ref_mask[1 : bs + 1] = torch.cumsum(tokens * (seq_lens + tokens), dim=0)
    for out, ref in ((qo, ref_qo), (kv, ref_kv), (mask, ref_mask)):
        assert torch.equal(out, ref)
    if window is None:
        assert lens is None and start is None
        return
    ref_lens = torch.minimum(seq_lens, torch.tensor(window))
    ref_win[1 : bs + 1] = torch.cumsum(ref_lens, dim=0)
    ref_start = seq_lens - ref_lens
    ref_offsets[:bs] = ref_start
    assert lens.dtype == ref_lens.dtype and torch.equal(lens, ref_lens)
    assert start.dtype == ref_start.dtype and torch.equal(start, ref_start)
    assert torch.equal(win, ref_win) and torch.equal(offsets, ref_offsets)


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_rnn_step_matches_gemm_path(rows):
    """rnn_step (Inkling2 MTP's Markov RNN step for a few rows, in two launches) equals
    rnn_inputs, the rnn_proj GEMM and rnn_gates up to one BF16 step where the split-K
    sum rounds differently, and repeats bitwise."""
    import torch.nn.functional as F

    from sglang.kernels.ops.inkling2.mtp import rnn_gates, rnn_inputs
    from sglang.kernels.ops.inkling2.mtp_hip import rnn_step, rnn_step_fits

    torch.manual_seed(rows)
    rank, hidden, vocab = 256, 4608, 4096
    embedding = (torch.randn(vocab, rank, device="cuda") * 0.05).bfloat16()
    weight = (
        torch.randn(3 * rank, 2 * rank + 2 * hidden, device="cuda") * 0.01
    ).bfloat16()
    bias = (torch.randn(3 * rank, device="cuda") * 0.1).bfloat16()
    state = (torch.randn(rows, rank, device="cuda") * 0.5).bfloat16()
    tokens = torch.randint(0, vocab, (rows,), device="cuda")
    hidden_states = torch.randn(rows, 2 * hidden, device="cuda").bfloat16()[:, :hidden]
    f_rnn = torch.randn(rows, hidden, device="cuda").bfloat16()
    assert rnn_step_fits(state, hidden_states, weight, bias)
    inputs = rnn_inputs(state, tokens, embedding, hidden_states, f_rnn)
    ref_state, ref_latent = rnn_gates(F.linear(inputs, weight, bias), state)
    new_state, latent = rnn_step(
        state, tokens, embedding, hidden_states, f_rnn, weight, bias
    )
    for out, ref in ((new_state, ref_state), (latent, ref_latent)):
        assert out.dtype == ref.dtype and out.shape == ref.shape
        # A projection one BF16 step off moves the gates by a few BF16 steps.
        assert (out.float() - ref.float()).abs().max() <= 0.02
        assert (out != ref).float().mean() < 0.01
    again = rnn_step(state, tokens, embedding, hidden_states, f_rnn, weight, bias)
    assert torch.equal(again[0], new_state) and torch.equal(again[1], latent)


@pytest.mark.parametrize("rows", [1, 4])
def test_markov_step_fold_matches_addmm_bias(rows):
    """A sharded DSpark Markov step with the folded bias computed in its sampling kernel
    (fold) picks the tokens of the step given torch.addmm's bias, and keeps the same
    step logits up to rare one-step BF16 roundings of the bias; it repeats bitwise."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    torch.manual_seed(rows)
    padded, rank = 25128, 256
    base = torch.randn(rows, padded, device="cuda").bfloat16() * 4
    latents = torch.tanh(torch.randn(rows, rank, device="cuda")).bfloat16()
    weight = (torch.randn(padded, rank, device="cuda") * 0.1).bfloat16()
    fold_bias = torch.randn(padded, device="cuda").bfloat16()
    temperatures = torch.tensor([0.0, 0.7, 1.3, 0.0][:rows], device="cuda")
    common = dict(
        base_logits=base, temperatures=temperatures, key=99,
        offset=torch.tensor([3], device="cuda", dtype=torch.int64), step=1,
        step_dtype=torch.bfloat16, vocab_start=2 * padded, gather=lambda p: p,
        softcap=30.0, vocab_limit=3 * padded - 7,
    )  # fmt: skip
    expected_logits = torch.empty_like(base)
    expected = MarkovSampleStep.triton(
        bias=torch.addmm(fold_bias, latents, weight.T), logits_out=expected_logits,
        **common,
    )  # fmt: skip
    outs = []
    for _ in range(2):
        logits = torch.empty_like(base)
        tokens = MarkovSampleStep.triton(
            bias=latents, fold=(weight, fold_bias), logits_out=logits, **common
        )
        outs.append((tokens, logits))
    tokens, logits = outs[0]
    assert torch.equal(tokens, expected)
    assert torch.equal(outs[1][0], tokens) and torch.equal(outs[1][1], logits)
    finite = expected_logits.isfinite()
    assert torch.equal(finite, logits.isfinite())
    assert (logits != expected_logits).float().mean() < 0.01
    diff = (logits.float() - expected_logits.float())[finite].abs().max()
    assert diff <= 0.25  # one BF16 step at |logit| < 30
