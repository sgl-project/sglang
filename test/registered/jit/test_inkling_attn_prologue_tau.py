"""The fused attn prologue's conditional log-scaling-tau fold on the q path:
q_out must equal {per-head RMSNorm -> bf16 -> * tau -> bf16} (the unfused
{prologue -> apply_log_scaling_tau} rounding, exactly), and the k/v outputs
must be untouched by tau.
"""

import pytest
import torch

from sglang.jit_kernel.inkling_attn_prologue import inkling_attn_prologue_decode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD = 128
W = 4
EPS = 1e-6


def _run(t, dq, dkv, tau):
    torch.manual_seed(1234)
    dev = "cuda"
    row = dq + 2 * dkv + 64  # packed qkvr row with an r tail
    qkvr = torch.randn(t, row, device=dev, dtype=torch.bfloat16)
    pool = 64
    k_cache = torch.randn(pool, W - 1, dkv, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(pool, W - 1, dkv, device=dev, dtype=torch.bfloat16)
    ci = torch.arange(t, device=dev, dtype=torch.int32) + 3
    cm = torch.ones(t, device=dev, dtype=torch.bool)
    kw = torch.randn(dkv, W, device=dev, dtype=torch.bfloat16) * 0.3
    vw = torch.randn(dkv, W, device=dev, dtype=torch.bfloat16) * 0.3
    qg = torch.randn(HEAD, device=dev, dtype=torch.bfloat16)
    kg = torch.randn(HEAD, device=dev, dtype=torch.bfloat16)
    slots = 256
    loc = (torch.arange(t, device=dev, dtype=torch.int64) * 7 + 5) % slots
    k_buf = torch.zeros(slots, dkv // HEAD, HEAD, device=dev, dtype=torch.bfloat16)
    v_buf = torch.zeros_like(k_buf)
    return inkling_attn_prologue_decode(
        qkvr,
        k_cache,
        v_cache,
        ci,
        cm,
        kw,
        vw,
        qg,
        kg,
        EPS,
        loc,
        k_buf,
        v_buf,
        0,
        dq,
        dq + dkv,
        dq,
        dkv,
        activation=None,
        use_residual=True,
        do_store=True,
        log_scaling_tau=tau,
    )


def _q_ref(t, dq, dkv, tau):
    torch.manual_seed(1234)
    row = dq + 2 * dkv + 64
    qkvr = torch.randn(t, row, device="cuda", dtype=torch.bfloat16)
    # (regenerate the SAME rng stream as _run for the remaining tensors)
    _ = torch.randn(64, W - 1, dkv, device="cuda", dtype=torch.bfloat16)
    _ = torch.randn(64, W - 1, dkv, device="cuda", dtype=torch.bfloat16)
    kw = torch.randn(dkv, W, device="cuda", dtype=torch.bfloat16) * 0.3
    vw = torch.randn(dkv, W, device="cuda", dtype=torch.bfloat16) * 0.3
    del kw, vw
    qg = torch.randn(HEAD, device="cuda", dtype=torch.bfloat16)
    q = qkvr[:, :dq].float().view(t, dq // HEAD, HEAD)
    inv = torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS)
    out = (q * inv * qg.float()).bfloat16()
    if tau is not None:
        out = (out.float() * tau.view(-1, 1, 1)).bfloat16()
    return out.view(t, dq)


@pytest.mark.parametrize("t", [1, 3, 8, 32])
def test_prologue_decode_tau_fold(t):
    dq, dkv = 2048, 256
    tau = 1.0 + 0.1 * torch.rand(t, device="cuda", dtype=torch.float32)

    q_tau, k_tau, v_tau, _ = _run(t, dq, dkv, tau)
    q_ref = _q_ref(t, dq, dkv, tau)
    torch.testing.assert_close(q_tau.float(), q_ref.float(), rtol=2e-2, atol=2e-2)

    # tau must not touch the k/v legs.
    q_off, k_off, v_off, _ = _run(t, dq, dkv, None)
    assert torch.equal(k_tau, k_off)
    assert torch.equal(v_tau, v_off)
    # And with tau=None the q path matches the tau-free reference bit-wise
    # modulo the norm's fp32 reduction (tolerance).
    torch.testing.assert_close(
        q_off.float(), _q_ref(t, dq, dkv, None).float(), rtol=2e-2, atol=2e-2
    )
    # The fold itself must be exactly {round -> fp32 mul -> round}: applying
    # tau to the tau-free kernel output reproduces the fused output bit-wise.
    refold = (q_off.float() * tau.view(-1, 1)).bfloat16()
    assert torch.equal(q_tau, refold)


@pytest.mark.parametrize("t", [1, 7, 64, 2048])
def test_rel_logits_proj_prescale_tau(t):
    """RelLogitsProj's operand-side tau fold (r*tau before the einsum) must
    match the legacy output-side scale within bf16 rounding."""
    from sglang.kernels.ops.attention.log_scaling_tau import apply_log_scaling_tau
    from sglang.srt.models.inkling_common.attn import RelLogitsProj

    torch.manual_seed(t)
    h, d_rel, e = 16, 16, 1024
    m = RelLogitsProj(d_rel, e).cuda()
    m.proj.data = torch.randn(d_rel, e, device="cuda", dtype=torch.bfloat16) * 0.1
    r = torch.randn(t, h, d_rel, device="cuda", dtype=torch.bfloat16)
    tau = 1.0 + 0.1 * torch.rand(t, device="cuda", dtype=torch.float32)

    assert m._prescale_tau  # default-on flag
    out = m(r, tau)
    ref = apply_log_scaling_tau(
        torch.einsum("thd,de->the", r, m.proj), tau.view(-1, 1, 1)
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    # And without tau it is the plain einsum, bit-exact.
    assert torch.equal(m(r), torch.einsum("thd,de->the", r, m.proj))


_QKVR_ROW = 2816  # dq 2048 + 2*dkv 512 + h*d_rel 256 (the TP4 packed row)


def _strided_r(t, h=16, d_rel=16, elem_offset=0):
    """r exactly as production builds it: the trailing slice of the packed
    qkvr projection output, viewed [t, h, d_rel] (row stride = full row)."""
    torch.manual_seed(t + elem_offset)
    qkvr = torch.randn(t, _QKVR_ROW + elem_offset, device="cuda", dtype=torch.bfloat16)
    off = _QKVR_ROW + elem_offset - h * d_rel
    return qkvr[:, off:].view(t, h, d_rel)


@pytest.mark.parametrize("t", [1, 2, 48, 49, 64, 200, 1024])
def test_rel_logits_proj_strided_dispatch(t):
    """_project on the production strided layout must be BIT-identical to the
    plain einsum in both dispatch bands -- the zero-copy batched matmul
    (t <= _REL_PROJ_MATMUL_MAX_T) and {JIT row-compact -> einsum} above it.
    Guards the band boundary, the as_strided compaction math, and the
    batched-GEMM == flat-GEMM reduction-order claim the dispatch relies on."""
    from sglang.srt.models.inkling_common.attn import RelLogitsProj

    h, d_rel, e = 16, 16, 1024
    m = RelLogitsProj(d_rel, e).cuda()
    m.proj.data = torch.randn(d_rel, e, device="cuda", dtype=torch.bfloat16) * 0.1
    assert m._proj_dispatch  # default-on flag

    r = _strided_r(t, h, d_rel)
    ref = torch.einsum("thd,de->the", r.contiguous(), m.proj)
    out = m(r)
    assert out.is_contiguous()
    assert torch.equal(out, ref)

    # The tau path on the same strided layout (prescale compacts first).
    from sglang.kernels.ops.attention.log_scaling_tau import apply_log_scaling_tau

    tau = 1.0 + 0.1 * torch.rand(t, device="cuda", dtype=torch.float32)
    ref_tau = apply_log_scaling_tau(ref, tau.view(-1, 1, 1))
    torch.testing.assert_close(m(r, tau).float(), ref_tau.float(), rtol=2e-2, atol=2e-2)


def test_rel_logits_proj_dispatch_fallbacks():
    """The compact band must fall back to the plain einsum (and stay exact)
    when the JIT copy is ineligible -- e.g. a 2-byte-aligned r slice -- and
    flag-off must restore the undispatched einsum on every input."""
    from sglang.srt.environ import envs
    from sglang.srt.models.inkling_common.attn import (
        _REL_PROJ_MATMUL_MAX_T,
        RelLogitsProj,
    )

    h, d_rel, e = 16, 16, 1024
    m = RelLogitsProj(d_rel, e).cuda()
    m.proj.data = torch.randn(d_rel, e, device="cuda", dtype=torch.bfloat16) * 0.1

    t = _REL_PROJ_MATMUL_MAX_T + 16  # inside the compact band
    r_misaligned = _strided_r(t, h, d_rel, elem_offset=1)
    assert r_misaligned.data_ptr() % 16 != 0
    ref = torch.einsum("thd,de->the", r_misaligned.contiguous(), m.proj)
    assert torch.equal(m(r_misaligned), ref)

    with envs.SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH.override(False):
        m_off = RelLogitsProj(d_rel, e).cuda()
        m_off.proj.data = m.proj.data
        assert not m_off._proj_dispatch
        r = _strided_r(t, h, d_rel)
        assert torch.equal(m_off(r), torch.einsum("thd,de->the", r, m_off.proj))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
