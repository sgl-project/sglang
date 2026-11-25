"""Inkling2's ROCm kernels against float64 / torch references, and their run-to-run
determinism (two ROCm Triton miscompilations showed up only as nondeterminism)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: the ROCm kernels and dispatch Inkling2 runs on MI35x
register_amd_ci(est_time=40, suite="base-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires a ROCm GPU",
)

H, HKV, D, EXT = 6, 1, 128, 1024


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
    from sglang.kernels.ops.attention import inkling2_attn_hip as attention_hip

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
    from sglang.kernels.ops.attention import inkling2_attn_hip as attention_hip

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
    from sglang.kernels.ops.attention.inkling2_attn import gate_sink
    from sglang.kernels.ops.attention.score_mod import relative_bias_score_mod

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


@pytest.mark.parametrize("bound,min_tokens", [(4, 64), (128, 32)])
def test_length_based_decode_splits_through_buffer_views(bound, min_tokens):
    """Decode with per-request length-based KV splits (the triton backend's
    SGLANG_TRITON_DECODE_SPLIT_MIN_TOKENS policy), launched over [:, :, :bound] views
    of a 128-split buffer as the backend does: bitwise equal to the same launch on
    contiguous bound-split buffers, and float64 attention to BF16 accuracy."""
    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    from sglang.kernels.ops.attention.metadata_hip import (
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


@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("local", [False, True])
def test_prologue_stores_kv_pool(m, local):
    """The ROCm decode prologue with kv_pool writes each row's K and V into the KV
    cache pool at its loc: bitwise what the attention backend's save stores (the
    same rows, padding rows included), with its other outputs unchanged."""
    from sglang.kernels.ops.attention.inkling2_attn_prologue import attention_prologue

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


@pytest.mark.parametrize("inkling", [True, False])
@pytest.mark.parametrize(
    "batch,seq,splits,bound",
    [
        (1, 2560, 80, 128),
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


def test_rel_verify_attention_bidirectional_without_bias():
    """The DSpark draft's short blocks on ROCm: rel_verify_attention with a
    zero-extent bias and causal=False is plain attention over the cached prefix plus
    all of the block's own keys, against a float64 reference."""
    from sglang.kernels.ops.attention.inkling2_attn_hip import rel_verify_attention

    torch.manual_seed(0)
    ext = 8
    prefixes = (0, 63, 700, 3000)
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
        qo_indptr, kv_indptr, kv_indices, ext, D**-0.5, causal=False,
    )  # fmt: skip
    for b, p in enumerate(prefixes):
        idx = kv_indices[kv_indptr[b] : kv_indptr[b] + p]
        keys = torch.cat([k_buf[idx, 0], k_ext[b * ext : (b + 1) * ext, 0]]).double()
        vals = torch.cat([v_buf[idx, 0], v_ext[b * ext : (b + 1) * ext, 0]]).double()
        logits = q[b * ext : (b + 1) * ext].double() @ keys.T * D**-0.5
        ref = torch.softmax(logits, -1) @ vals
        assert _rel(o[b * ext : (b + 1) * ext], ref) < 5e-3, (b, p)


@pytest.mark.parametrize("batch,ext", [(3, 8), (2, 5)])
def test_rel_verify_gate_sink_in_combine(batch, ext):
    """rel_verify_attention(gate_sink=...) writes, from its own reduction, bitwise
    what gate_sink(o, gate, lse, sinks) gives after it (the ROCm MTP verify path),
    with o and lse unchanged."""
    from sglang.kernels.ops.attention.inkling2_attn import gate_sink
    from sglang.kernels.ops.attention.inkling2_attn_hip import rel_verify_attention

    torch.manual_seed(batch * 10 + ext)
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


@pytest.mark.parametrize("bs", [1, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("window", [None, 4096])
def test_verify_indptrs_matches_torch(bs, dtype, window):
    """verify_indptrs (target-verify metadata in one launch) writes the integers the
    triton backend's separate torch ops computed, into the same buffers."""
    from sglang.kernels.ops.attention.metadata_hip import verify_indptrs

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
