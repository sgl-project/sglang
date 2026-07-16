from __future__ import annotations

import math
import sys

import pytest
import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")


DTYPE_FP8 = torch.float8_e4m3fn
D_V = 512
H_Q = 64
H_KV = 1
TOPK = 128
S_KV = 256

# DeepSeek NSA E2E currently does not plumb a per-head attention sink into
# sparse MLA. No-sink cases are the E2E proxy; sink cases below exercise the
# optional kernel full path and partial topk_length handling.


def _sm90_available() -> bool:
    return is_sm90_supported()


def _make_fp8_tensor(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    tensor = torch.randn(shape, device="cuda", generator=generator, dtype=torch.float32)
    return (tensor * 0.05).to(DTYPE_FP8)


def _make_case(
    d_qk: int,
    with_sink: bool,
    s_q: int = 2,
    topk: int = TOPK,
    s_kv: int = S_KV,
):
    q = _make_fp8_tensor((s_q, H_Q, d_qk), seed=1000 + d_qk + s_q * 13 + topk)
    kv = torch.zeros((s_kv + 1, H_KV, d_qk), dtype=DTYPE_FP8, device="cuda")
    kv[:s_kv] = _make_fp8_tensor((s_kv, H_KV, d_qk), seed=2000 + d_qk + s_kv)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(3000 + d_qk + s_q * 17 + topk)
    indices = torch.randint(
        0,
        s_kv,
        (s_q, H_KV, topk),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )

    q_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    kv_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(d_qk)

    if not with_sink:
        return q, kv, indices, sm_scale, q_scale, kv_scale, None, None

    attn_sink = torch.linspace(-0.05, 0.05, H_Q, dtype=torch.float32, device="cuda")
    # Vary topk_length per query row to exercise the partial-topk path.
    lengths = [topk if i % 2 == 0 else max(topk - 32, topk // 2) for i in range(s_q)]
    topk_length = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    for q_idx, valid_topk in enumerate(lengths):
        if valid_topk < topk:
            indices[q_idx, 0, valid_topk:] = -1
    return q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length


def _torch_sparse_attention_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    attn_sink: torch.Tensor | None,
    topk_length: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk = indices.shape[-1]
    q_f32 = q.float() * q_scale.item()
    kv_f32 = kv.float() * kv_scale.item()
    out = torch.empty(
        (q.shape[0], q.shape[1], D_V), dtype=torch.float32, device=q.device
    )
    max_logits = torch.empty(
        (q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device
    )
    lse = torch.empty_like(max_logits)

    for q_idx in range(q.shape[0]):
        valid_topk = topk if topk_length is None else int(topk_length[q_idx].item())
        token_ids = indices[q_idx, 0, :valid_topk].to(torch.long)
        keys = kv_f32[token_ids, 0, :]
        values = kv_f32[token_ids, 0, :D_V]
        scores = torch.matmul(q_f32[q_idx], keys.transpose(0, 1)) * sm_scale
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        denom = exp_scores.sum(dim=-1, keepdim=True)
        max_logits[q_idx] = score_max.squeeze(-1)
        lse[q_idx] = torch.log(denom.squeeze(-1)) + score_max.squeeze(-1)
        if attn_sink is not None:
            denom = denom + torch.exp(attn_sink[:, None] - score_max)
        out[q_idx] = torch.matmul(exp_scores, values) / denom

    return out, max_logits, lse


def _run_and_check(d_qk, with_sink, s_q=2, topk=TOPK, s_kv=S_KV):
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length = _make_case(
        d_qk, with_sink, s_q=s_q, topk=topk, s_kv=s_kv
    )

    out, max_logits, lse = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    ref, ref_max_logits, ref_lse = _torch_sparse_attention_ref(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

    assert out.shape == (q.shape[0], H_Q, D_V)
    assert out.dtype == torch.bfloat16
    assert max_logits.shape == (q.shape[0], H_Q)
    assert lse.shape == (q.shape[0], H_Q)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(max_logits.float()).all()
    assert torch.isfinite(lse.float()).all()
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=8e-2)
    if attn_sink is None:
        torch.testing.assert_close(
            max_logits.float(), ref_max_logits, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(lse.float(), ref_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize("d_qk,with_sink", [(512, False), (576, False)])
def test_sparse_mla_q8kv8_prefill_matches_reference(d_qk: int, with_sink: bool):
    _run_and_check(d_qk, with_sink)


# Corner cases: minimal s_q, larger s_q, larger topk/s_kv, crossed d_qk
# configurations, and optional sink+topk_length feature coverage. The kernel
# requires topk to be a multiple of 128, so 128 is the minimum supported.
@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize(
    "d_qk,with_sink,s_q,topk,s_kv",
    [
        (576, True, 1, TOPK, S_KV),
        (576, True, 8, TOPK, S_KV),
        (576, True, 2, 256, 512),
        (512, False, 65, 256, 592),
        (512, True, 2, TOPK, S_KV),
        (576, False, 65, 256, 592),
    ],
)
def test_sparse_mla_q8kv8_prefill_corner_cases(
    d_qk: int, with_sink: bool, s_q: int, topk: int, s_kv: int
):
    _run_and_check(d_qk, with_sink, s_q=s_q, topk=topk, s_kv=s_kv)


# topk_length WITHOUT attn_sink (the production early-exit path for
# SGLANG_ENABLE_DSA_Q8KV8_TOPK_LENGTH): rows with a trailing -1 pad run must
# be BITWISE identical to the full-topk dispatch that masks those pads, and
# must match the fp32 reference on the truncated index range.
@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize(
    "d_qk,s_q,topk,s_kv",
    [
        (576, 8, TOPK, S_KV),
        (576, 65, 256, 592),
        (512, 8, TOPK, S_KV),
    ],
)
def test_sparse_mla_q8kv8_prefill_topk_length_only(
    d_qk: int, s_q: int, topk: int, s_kv: int
):
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    q, kv, indices, sm_scale, q_scale, kv_scale, _, _ = _make_case(
        d_qk, False, s_q=s_q, topk=topk, s_kv=s_kv
    )
    # Trailing pad runs of varying size, including a 1-valid-entry row (the
    # production clamp(min=1) floor) and full rows.
    lengths = [
        topk if i % 3 == 0 else (1 if i % 3 == 1 else max(topk - 32, topk // 2))
        for i in range(s_q)
    ]
    topk_length = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    for q_idx, valid_topk in enumerate(lengths):
        if valid_topk < topk:
            indices[q_idx, 0, valid_topk:] = -1

    out, max_logits, lse = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=None,
        topk_length=topk_length,
    )
    out_full, max_logits_full, lse_full = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=None,
        topk_length=None,
    )
    torch.cuda.synchronize()

    assert torch.equal(out, out_full)
    assert torch.equal(max_logits, max_logits_full)
    assert torch.equal(lse, lse_full)

    ref, ref_max_logits, ref_lse = _torch_sparse_attention_ref(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        attn_sink=None,
        topk_length=topk_length,
    )
    torch.testing.assert_close(out.float(), ref, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(max_logits.float(), ref_max_logits, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse.float(), ref_lse, atol=2e-3, rtol=2e-3)


# Precision / accuracy: no-sink only because these metrics are intended to
# approximate the current DeepSeek NSA E2E path. Sink behavior is still covered
# above as kernel feature coverage, but sink-enabled precision numbers should
# not be used as E2E proxy results until the E2E pipeline wires attn_sink.
@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize(
    "d_qk,s_q,topk,s_kv",
    [
        (512, 4, 256, 512),
        (576, 4, 256, 512),
        (512, 64, 256, 1024),
        (576, 64, 256, 1024),
    ],
)
def test_sparse_mla_q8kv8_prefill_precision(d_qk: int, s_q: int, topk: int, s_kv: int):
    """Demonstrate that Q8KV8 kernel precision is near-lossless versus the
    fp32 reference: max/mean/p99 absolute error are small and the fraction
    of elements exceeding 0.1 absolute error is under 1%."""
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    with_sink = False
    q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length = _make_case(
        d_qk, with_sink, s_q=s_q, topk=topk, s_kv=s_kv
    )

    out, max_logits, lse = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    ref, ref_max_logits, ref_lse = _torch_sparse_attention_ref(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

    out_f32 = out.float()
    abs_diff = (out_f32 - ref).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    p99_diff = torch.quantile(abs_diff.flatten(), 0.99).item()
    fail_rate = (abs_diff > 0.1).float().mean().item() * 100
    has_bad = bool(torch.isnan(out_f32).any() or torch.isinf(out_f32).any())
    ref_abs_mean = ref.abs().mean().clamp_min(1e-12).item()
    rel_mean = mean_diff / ref_abs_mean
    cos_diff = 1 - 2 * (out_f32.double() * ref.double()).sum().item() / max(
        (out_f32.double().square() + ref.double().square()).sum().item(), 1e-12
    )
    max_logits_diff = (max_logits.float() - ref_max_logits).abs().max().item()
    lse_diff = (lse.float() - ref_lse).abs().max().item()

    print(
        f"\n  d_qk={d_qk} with_sink={with_sink} s_q={s_q} topk={topk} s_kv={s_kv}: "
        f"max_diff={max_diff:.2e}, p99_diff={p99_diff:.2e}, "
        f"mean_diff={mean_diff:.2e}, rel_mean={rel_mean:.2e}, "
        f"cos_diff={cos_diff:.2e}, fail_rate(>0.1)={fail_rate:.3f}%, "
        f"max_logits_diff={max_logits_diff:.2e}, lse_diff={lse_diff:.2e}"
    )

    assert not has_bad, "Q8KV8 output contains NaN/Inf"
    assert fail_rate < 1.0, f"fail_rate {fail_rate:.3f}% exceeds 1% threshold"
    # Tight bounds on aggregate error to lock in near-lossless behavior.
    assert max_diff < 1e-3, f"max_diff {max_diff:.2e} exceeds 1e-3"
    assert mean_diff < 5e-3, f"mean_diff {mean_diff:.2e} exceeds 5e-3"
    assert p99_diff < 5e-2, f"p99_diff {p99_diff:.2e} exceeds 5e-2"
    assert cos_diff < 1e-4, f"cos_diff {cos_diff:.2e} exceeds 1e-4"
    assert max_logits_diff < 1e-2, f"max_logits_diff {max_logits_diff:.2e} exceeds 1e-2"
    assert lse_diff < 2e-3, f"lse_diff {lse_diff:.2e} exceeds 2e-3"


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
def test_sparse_mla_q8kv8_prefill_no_alias_between_calls():
    """Two default-allocation calls with the same shape must return independent
    storage. This guards against regressing to a module-scope output cache."""
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length = _make_case(
        d_qk=576, with_sink=False
    )

    out1, ml1, lse1 = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    snapshot = out1.clone()

    out2, ml2, lse2 = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    assert out1.data_ptr() != out2.data_ptr()
    assert ml1.data_ptr() != ml2.data_ptr()
    assert lse1.data_ptr() != lse2.data_ptr()
    # The first call's output must not be overwritten by the second call.
    torch.testing.assert_close(out1, snapshot)


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
def test_sparse_mla_q8kv8_prefill_caller_owned_buffers():
    """Caller-provided ``out`` / ``max_logits`` / ``lse`` tensors must be
    written into in-place and returned as-is."""
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length = _make_case(
        d_qk=576, with_sink=False
    )
    s_q = q.shape[0]
    out_buf = torch.empty((s_q, H_Q, D_V), dtype=torch.bfloat16, device="cuda")
    ml_buf = torch.empty((s_q, H_Q), dtype=torch.float32, device="cuda")
    lse_buf = torch.empty((s_q, H_Q), dtype=torch.float32, device="cuda")

    out, ml, lse = sparse_mla_q8kv8_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        d_v=D_V,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out_buf,
        max_logits=ml_buf,
        lse=lse_buf,
    )
    torch.cuda.synchronize()

    assert out.data_ptr() == out_buf.data_ptr()
    assert ml.data_ptr() == ml_buf.data_ptr()
    assert lse.data_ptr() == lse_buf.data_ptr()
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(ml.float()).all()
    assert torch.isfinite(lse.float()).all()


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
def test_sparse_mla_q8kv8_prefill_rejects_bad_buffers():
    """Validation: wrong shape/dtype and aliasing must raise ValueError."""
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    q, kv, indices, sm_scale, q_scale, kv_scale, attn_sink, topk_length = _make_case(
        d_qk=576, with_sink=False
    )
    s_q = q.shape[0]

    def _call(**overrides):
        kwargs = dict(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            d_v=D_V,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        kwargs.update(overrides)
        return sparse_mla_q8kv8_prefill_fwd(**kwargs)

    # Wrong dtype.
    bad_out = torch.empty((s_q, H_Q, D_V), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError):
        _call(out=bad_out)

    # Wrong shape.
    bad_ml = torch.empty((s_q + 1, H_Q), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        _call(max_logits=bad_ml)

    # Aliased max_logits / lse.
    shared = torch.empty((s_q, H_Q), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        _call(max_logits=shared, lse=shared)

    # d_v != 512.
    with pytest.raises(ValueError):
        _call(d_v=256)


# ---------------------------------------------------------------------------
# End-to-end-discovered corner-case gates.
#
# Everything ABOVE (matches_reference / corner_cases / precision /
# no_alias_between_calls / caller_owned_buffers / rejects_bad_buffers) is the
# original unit suite: small, all-valid (or topk_length-bounded) shapes checked
# against a reference that REPRODUCES the kernel's own clamp semantics.  That
# suite is blind to three bug classes that only surface under real
# DeepSeek-V3.2 serving; the gates below reproduce them as standalone kernel
# tests:
#
#   1. masked -1-sentinel SEMANTICS on few-valid rows (ctx << topk): pad slots
#      must contribute NOTHING to the softmax denominator.  This needs a MASKED
#      (-inf) fp32 reference -- a reference that mimics the kernel's own clamp
#      is blind to the bug.
#   2. s_q ENVELOPE to 6144: first-band NaNs from an is_kv_valid data race that
#      only appears past s_q=2048 (never exercised above).
#   3. LARGE S_KV (65536) / large index values: gathered multi-request buffers
#      reach tens of thousands of rows in e2e; the suite above used s_kv<=1024.
#
# These use h=128 (the real DeepSeek head count) and large s_q/s_kv, so they are
# heavier than the suite above; same SM90 skipif.  They also DOCUMENT that the
# kernel is run-to-run nondeterministic at the fp8 noise floor, so they compare
# against an fp32 reference (never bitwise / self-consistency).
# ---------------------------------------------------------------------------

_D_FULL = 576  # nope(512) + rope(64): the real DeepSeek MLA absorbed q/kv width


def _ref_masked_blocked(q, kv, indices, sm_scale, d_v, row_start, row_end):
    """fp32 reference with PROPER -1 masking (pad slots -> -inf), computed over a
    block of query rows [row_start, row_end) to bound peak memory.  Unlike
    ``_torch_sparse_attention_ref`` (which bounds validity via topk_length and so
    reproduces the kernel's clamp), this masks every -1 index out of the softmax,
    making it sensitive to the denominator-pollution bug."""
    q_f = q.float()
    kv_f = kv.float()[:, 0, :]
    idx_block = indices[row_start:row_end, 0, :].long()
    gathered = kv_f[idx_block.clamp(min=0)]
    scores = torch.einsum("qhd,qkd->qhk", q_f[row_start:row_end], gathered) * sm_scale
    scores = scores.masked_fill((idx_block < 0)[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,qkd->qhd", probs, gathered[:, :, :d_v])


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize("s_q", [2048, 4096])
def test_sparse_mla_q8kv8_prefill_masked_sentinels(s_q: int):
    """NEW gate (bug class 1): causal -1 structure (row i has min(1+i, topk)
    valid slots, the rest -1).  The kernel must mask pad slots out of the softmax
    denominator.  Checked against a MASKED (-inf) fp32 reference; a reference
    that reproduced the kernel's clamp would be blind to this.
    Gate: per-band cos > 0.97 AND magnitude ratio > 0.9
    (the denominator-pollution bug crushes magnitude 50-2000x, unmistakable even
    under fp8 noise)."""
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    h, topk, band, n = 128, 2048, 512, 4608
    s_kv = n + topk
    g = torch.Generator(device="cuda").manual_seed(11)
    q = torch.randn((s_q, h, _D_FULL), device="cuda", generator=g).to(DTYPE_FP8)
    kv = torch.zeros((s_kv, H_KV, _D_FULL), dtype=DTYPE_FP8, device="cuda")
    kv[:n] = torch.randn((n, H_KV, _D_FULL), device="cuda", generator=g).to(DTYPE_FP8)
    idx = torch.full((s_q, H_KV, topk), -1, dtype=torch.int32, device="cuda")
    slot = torch.arange(topk, device="cuda")
    valid = torch.clamp(1 + torch.arange(s_q, device="cuda"), max=topk)
    rnd = torch.randint(
        0, n, (s_q, topk), dtype=torch.int32, device="cuda", generator=g
    )
    idx[:, 0, :] = torch.where(
        slot[None, :] < valid[:, None], rnd, torch.full_like(rnd, -1)
    )
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(_D_FULL)

    out, _, _ = sparse_mla_q8kv8_prefill_fwd(
        q=q, kv=kv, indices=idx, sm_scale=sm_scale, q_scale=one, kv_scale=one, d_v=D_V
    )
    torch.cuda.synchronize()

    worst_cos, worst_mag = 1.0, 1.0
    for s in range(0, s_q, band):
        e = min(s + band, s_q)
        ref = _ref_masked_blocked(q, kv, idx, sm_scale, D_V, s, e)
        ob = out[s:e].float()
        cos = torch.nn.functional.cosine_similarity(
            ob.reshape(-1), ref.reshape(-1), dim=0
        ).item()
        mag = (ob.norm() / ref.norm().clamp(min=1e-9)).item()
        worst_cos = min(worst_cos, cos)
        if mag < 1.0:
            worst_mag = min(worst_mag, mag)
        del ref, ob
        torch.cuda.empty_cache()

    print(
        f"\n  masked-sentinels s_q={s_q}: worst cos={worst_cos:.4f} "
        f"worst |out|/|ref|={worst_mag:.3f}"
    )
    assert worst_cos > 0.97, f"cos {worst_cos:.4f} <= 0.97 (denominator pollution?)"
    assert worst_mag > 0.9, f"mag {worst_mag:.3f} <= 0.9 (denominator pollution?)"


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize("s_q", [2048, 4096, 6144])
def test_sparse_mla_q8kv8_prefill_sq_envelope(s_q: int):
    """NEW gate (bug class 2): all-valid correctness across the s_q envelope.
    s_q=6144 previously produced first-band NaNs (an is_kv_valid data race that
    only appears past s_q=2048)."""
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    h, topk, s_kv, band = 128, 2048, 8192, 1024
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn((s_q, h, _D_FULL), device="cuda", generator=g).to(DTYPE_FP8)
    kv = torch.randn((s_kv, H_KV, _D_FULL), device="cuda", generator=g).to(DTYPE_FP8)
    idx = torch.randint(
        0, s_kv, (s_q, H_KV, topk), dtype=torch.int32, device="cuda", generator=g
    )
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(_D_FULL)

    out, _, _ = sparse_mla_q8kv8_prefill_fwd(
        q=q, kv=kv, indices=idx, sm_scale=sm_scale, q_scale=one, kv_scale=one, d_v=D_V
    )
    torch.cuda.synchronize()

    has_nan = torch.isnan(out.float()).any().item()
    worst_cos = 1.0
    for s in range(0, s_q, band):
        e = min(s + band, s_q)
        ref = _ref_masked_blocked(q, kv, idx, sm_scale, D_V, s, e)
        cos = torch.nn.functional.cosine_similarity(
            out[s:e].float().reshape(-1), ref.reshape(-1), dim=0
        ).item()
        worst_cos = min(worst_cos, cos)
        del ref
        torch.cuda.empty_cache()

    print(f"\n  s_q-envelope s_q={s_q}: nan={has_nan} worst cos={worst_cos:.4f}")
    assert not has_nan, f"NaN in output at s_q={s_q} (is_kv_valid race)"
    assert worst_cos > 0.99, f"cos {worst_cos:.4f} <= 0.99"


@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
def test_sparse_mla_q8kv8_prefill_large_skv():
    """NEW gate (bug class 3): large gathered buffers / large index values
    (s_kv=65536, indices in [33000, 65536)).  E2E multi-request gather buffers
    reach tens of thousands of rows; the suite above used s_kv<=1024."""
    from sglang.jit_kernel.sparse_mla_q8kv8_prefill_sm90 import (
        sparse_mla_q8kv8_prefill_fwd,
    )

    h, topk, s_kv, s_q = 128, 2048, 65536, 2048
    g = torch.Generator(device="cuda").manual_seed(13)
    q = torch.randn((s_q, h, _D_FULL), device="cuda", generator=g).to(DTYPE_FP8)
    kv = torch.randn((s_kv, H_KV, _D_FULL), device="cuda", generator=g).to(DTYPE_FP8)
    idx = torch.randint(
        33000, s_kv, (s_q, H_KV, topk), dtype=torch.int32, device="cuda", generator=g
    )
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    sm_scale = 1.0 / math.sqrt(_D_FULL)

    out, _, _ = sparse_mla_q8kv8_prefill_fwd(
        q=q, kv=kv, indices=idx, sm_scale=sm_scale, q_scale=one, kv_scale=one, d_v=D_V
    )
    torch.cuda.synchronize()

    ref = _ref_masked_blocked(q, kv, idx, sm_scale, D_V, 0, 1024)
    cos = torch.nn.functional.cosine_similarity(
        out[:1024].float().reshape(-1), ref.reshape(-1), dim=0
    ).item()
    print(f"\n  large-S_KV={s_kv}: band-0 cos={cos:.4f}")
    assert cos > 0.99, f"cos {cos:.4f} <= 0.99"


# Backend-side topk_length derivation (backscan Triton kernel): must equal the
# reference "last non-negative position + 1 (min 1)" on every pad pattern the
# production topk output can produce (trailing runs), plus adversarial ones
# (interleaved -1s, all-pad, full rows) where the trailing-run semantics still
# define the correct consumed range.
@pytest.mark.skipif(
    not _sm90_available(), reason="Q8KV8 sparse prefill requires SM90 CUDA"
)
@pytest.mark.parametrize("s_q,topk", [(437, 2048), (7, 128), (65, 256), (4096, 2048)])
def test_q8kv8_topk_length_backscan(s_q: int, topk: int):
    from sglang.kernels.ops.kvcache.cache_ops import (
        q8kv8_topk_length_from_indices,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(4000 + s_q + topk)
    indices = torch.randint(
        0, 1 << 20, (s_q, topk), dtype=torch.int32, device="cuda", generator=generator
    )
    # Row patterns: full, trailing pad runs of every length, all-pad,
    # interleaved -1s inside the valid range.
    for i in range(s_q):
        mode = i % 5
        if mode == 1:
            indices[i, max(1, i % topk) :] = -1
        elif mode == 2:
            indices[i, :] = -1
        elif mode == 3:
            indices[i, i % topk :: 7] = -1  # interleaved + trailing mix
        elif mode == 4:
            indices[i, topk - 1 :] = -1

    got = q8kv8_topk_length_from_indices(indices)

    ramp = torch.arange(1, topk + 1, dtype=torch.int32, device="cuda")
    ref = ((indices >= 0).int() * ramp).amax(dim=-1).clamp_(min=1)
    assert torch.equal(got, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
