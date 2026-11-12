# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Fold (commit-then-verify) vs the stock verify + ReplaySSM commit, over multi-step chains.

Stock arm, per step: ``fused_kda_decode_mtp_dspark`` (CACHE_RING) reads the committed state, then
``commit_kda_replayssm_spec`` replays the accepted prefix into it. Fold arm, per step: the same verify with
``fold_pending`` replays the *previous* step's accepted records in-kernel, writes the committed state back, verifies
into the other ring half, and the real fold post-verify (``commit_kda_replayssm_fold_after_verify``) only records the
accept length; a final ``kda_fold_flush`` brings the fold state level with the stock one. The "mixed" pattern also
runs one step the way a non-fold forward does (flush, then the stock verify writing ring rows from 0), which checks
the ring-half bookkeeping across both kinds of verify.

Shapes are K3.1 at TP8: 12 heads/rank, K = V = 128, FP32 state (or BF16 state, the opt-in
SGLANG_OPT_KDA_CUTE_VERIFY_BF16_STATE path), BF16 raw k / v rings and FP32 g / beta rings (L=16), 8 verify tokens
(bonus + 7), the fused gated RMSNorm on. N=3 (with a CUDA-graph padding row) runs the 512-thread block, N=20 the
256-thread block.

Tolerances, fixed before the first run:
- fold vs stock, FP32 state: max|dS| / max|S| <= 1e-5 (reduction order and approximate exp/sqrt differ).
- fold vs stock, BF16 state: max|dS| / max|S| <= 2e-2 (about two BF16 ulps at the top binade) and at most 5% of the
  live state elements differ (both arms round the same FP32 replay once per commit; only values within the two
  replays' FP32 difference of a rounding boundary can flip, and a flip feeds later steps).
- fold vs stock, outputs (bf16): max|do| / max|o| <= 1e-2 and cosine >= 0.99999 (FP32 state; a few bf16 rounding
  flips) or >= 0.9999 (BF16 state).
- both arms vs an independent float64 torch chain written from the KDA definition: outputs max-norm error <= 2e-2 and
  cosine >= 0.9995; state <= 1e-3 relative (FP32 state) or <= 2e-2 (BF16 state). The chain commits from the raw k / v
  records the stock verify stored in its BF16 rings, each first checked against the definition's value within one
  BF16 ulp + 1e-5 absolute, and rounds a BF16 state once per commit.
- the padding row's outputs are zero and its slot is untouched; slots no request uses stay bit-identical.
- test_fold_verify_reads_stored_state: the fold verify's outputs equal, bit for bit, a plain verify's outputs on the
  state the fold kernel stored (same verify code on the same values).
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("KDA CuTe MTP fold needs SM10x (CuTe DSL kernel).", allow_module_level=True)

from types import SimpleNamespace  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from sglang.kernels.ops.attention.kda_decode_mtp import (  # noqa: E402
    fused_kda_decode_mtp_dspark,
)
from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (  # noqa: E402
    commit_kda_replayssm_fold_after_verify,
    commit_kda_replayssm_spec,
    kda_fold_flush,
)

DEV = "cuda"
H, K, W, L = 12, 128, 4, 16
LB = -5.0
EPS = 1e-6
SCALE = K**-0.5
STATE_TOL = {torch.float32: 1e-5, torch.bfloat16: 2e-2}
STATE_DIFF_FRAC_BF16 = 0.05
OUT_TOL = 1e-2
OUT_COS = {torch.float32: 0.99999, torch.bfloat16: 0.9999}
REF_OUT_TOL, REF_OUT_COS = 2e-2, 0.9995
REF_STATE_TOL = {torch.float32: 1e-3, torch.bfloat16: 2e-2}
STATE_DTYPES = pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30)).item()


def _cos(a, b):
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def _check_state(got, want, what):
    """Fold vs stock state; returns the max-norm relative error."""
    e = _rel(got, want)
    assert e <= STATE_TOL[got.dtype], f"{what}: rel {e:.3e}"
    if got.dtype == torch.bfloat16:
        frac = (got != want).double().mean().item()
        assert frac <= STATE_DIFF_FRAC_BF16, f"{what}: {frac:.3%} of BF16 state elements differ"
    return e


class Chain:
    """Inputs for `steps` verify steps over N requests (slots 1..N; slot N+1 never used)."""

    def __init__(self, N, num_spec, steps, seed, pad_row, state_dtype=torch.float32):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.N, self.T_req, self.steps = N, 1 + num_spec, steps
        T = N * self.T_req
        self.num_slots = N + 2
        rnd = lambda *s, dt=torch.float32, sc=1.0: (torch.randn(*s, device=DEV, generator=g) * sc).to(dt)  # noqa
        self.w = [rnd(H * K, W, sc=0.3) for _ in range(3)]
        self.A_log = rnd(H, sc=0.5)
        self.dt_bias = rnd(H * K, sc=0.5)
        self.onw = 1 + rnd(K, sc=0.1)
        self.h0 = rnd(self.num_slots, H, K, K, dt=state_dtype, sc=0.05)
        self.cs0 = [rnd(self.num_slots, H * K, W - 1, dt=torch.bfloat16) for _ in range(3)]
        self.x = [[rnd(1, T, H, K, dt=torch.bfloat16) for _ in range(3)] for _ in range(steps)]
        self.a = [rnd(1, T, H, K, dt=torch.bfloat16) for _ in range(steps)]
        self.b = [rnd(1, T, H, dt=torch.bfloat16) for _ in range(steps)]
        self.gate = [rnd(1, T, H, K, dt=torch.bfloat16) for _ in range(steps)]
        self.slots = torch.arange(1, N + 1, device=DEV, dtype=torch.int32)
        if pad_row:
            self.slots[-1] = -1
        self.scratch = torch.arange(N, device=DEV, dtype=torch.int32)
        self.cu = torch.arange(0, T + 1, self.T_req, device=DEV, dtype=torch.int32)

    def accepts(self, pattern, step):
        T = self.T_req
        if pattern == "all":
            v = [T] * self.N
        elif pattern == "bonus_only":
            v = [1] * self.N
        else:  # "mixed": per request and per step, covering 1 and T
            g = torch.Generator().manual_seed(1000 + step)
            v = torch.randint(1, T + 1, (self.N,), generator=g).tolist()
            v[0], v[-1] = (1, T) if step % 2 == 0 else (T, 1)
        return torch.tensor(v, device=DEV, dtype=torch.int32)


def _verify(c, step, h0, cs, rings, fold_pending=None):
    ic = [torch.zeros(c.N, c.T_req, H * K, W - 1, device=DEV, dtype=torch.bfloat16) for _ in range(3)]
    out = fused_kda_decode_mtp_dspark(
        x_q=c.x[step][0], x_k=c.x[step][1], x_v=c.x[step][2],
        w_q=c.w[0], w_k=c.w[1], w_v=c.w[2],
        cs_q=cs[0], cs_k=cs[1], cs_v=cs[2],
        g=c.a[step], beta=c.b[step], A_log=c.A_log, dt_bias=c.dt_bias,
        recurrent_state=h0, intermediate_ssm=None, intermediate_state_indices=c.scratch,
        intermediate_conv_q=ic[0], intermediate_conv_k=ic[1], intermediate_conv_v=ic[2],
        ssm_state_indices=c.slots, cu_seqlens=c.cu, lower_bound=LB, scale=SCALE,
        replayssm_rawv=rings[0], replayssm_rawk=rings[1], replayssm_g=rings[2], replayssm_beta=rings[3],
        onorm_gate=c.gate[step], onorm_weight=c.onw, onorm_eps=EPS,
        fold_pending=fold_pending,
    )  # fmt: skip
    return out, ic


def _roll_conv(c, cs, ic, acc):
    """Conv rollback (the stock fused_conv_window_scatter's effect): window after the last accepted token."""
    for n in range(c.N):
        s = int(c.slots[n])
        if s < 0:
            continue
        for j in range(3):
            cs[j][s] = ic[j][n, int(acc[n]) - 1]


def _rings(c):
    """Served ReplaySSM ring dtypes: raw v / k in the activation dtype (BF16), g / beta FP32."""
    z = lambda *s, dt=torch.float32: torch.zeros(*s, device=DEV, dtype=dt)  # noqa
    return [
        z(c.num_slots, H, L, K, dt=torch.bfloat16), z(c.num_slots, H, L, K, dt=torch.bfloat16),
        z(c.num_slots, H, L, K), z(c.num_slots, H, L),
    ]  # fmt: skip


def _spec(h, rings):
    return SimpleNamespace(
        temporal=h.unsqueeze(0), replayssm_rawv=rings[0].unsqueeze(0),
        replayssm_rawk=rings[1].unsqueeze(0), replayssm_g=rings[2].unsqueeze(0),
        replayssm_beta=rings[3].unsqueeze(0), conv=[], intermediate_conv_window=[],
    )  # fmt: skip


def _commit(h0, rings, slots, lens):
    commit_kda_replayssm_spec(
        h0, rings[0], rings[1], rings[2], rings[3], slots, lens,
        max_cache_len=L, num_k_heads=H, use_qk_l2norm_in_kernel=True, null_block_id=-1,
    )  # fmt: skip


def _ulp_bf16(x):
    return torch.ldexp(torch.ones_like(x), torch.frexp(x).exponent - 8)


def _ref_chain(c, accepts, ring_snaps):
    """Float64 chain from the KDA definition: outputs per step, committed state and conv per slot.

    The verify uses k and v as computed. The commit replays the raw k / v records the stock verify stored in its BF16
    rings (``ring_snaps[step]`` = (rawk, rawv) right after that step's verify), after checking each record is the
    definition's value within one BF16 ulp (+1e-5 absolute for FP32 cancellation in the conv sum); g and beta are
    recomputed in float64. A BF16 state is rounded once per commit.
    """
    S = c.h0.double().clone()
    cs = [t.double().clone() for t in c.cs0]
    outs = []
    w = [t.double() for t in c.w]
    eA = torch.exp(c.A_log.double()).view(H, 1)
    l2n = lambda x: x / torch.sqrt((x**2).sum(-1, keepdim=True) + 1e-6)  # noqa
    for step in range(c.steps):
        o_step = torch.zeros(c.N * c.T_req, H, K, dtype=torch.float64, device=DEV)
        for n in range(c.N):
            s = int(c.slots[n])
            if s < 0:
                continue
            Sn = S[s].clone()
            Sc = S[s].clone()
            win = [cs[j][s].clone() for j in range(3)]  # [H*K, 3], oldest first
            for t in range(c.T_req):
                r = n * c.T_req + t
                y = []
                for j in range(3):
                    xt = c.x[step][j][0, r].double().reshape(H * K, 1)
                    full = torch.cat([win[j], xt], 1)  # [H*K, 4]
                    y.append(F.silu((full * w[j]).sum(1)).view(H, K))
                    win[j] = full[:, 1:]
                q = l2n(y[0]) * SCALE
                k = l2n(y[1])
                v = y[2]
                gk = LB * torch.sigmoid(eA * (c.a[step][0, r].double() + c.dt_bias.double().view(H, K)))
                beta = torch.sigmoid(c.b[step][0, r].double()).view(H, 1)
                Sn = Sn * torch.exp(gk).unsqueeze(1)
                d = (v - torch.einsum("hvk,hk->hv", Sn, k)) * beta
                Sn = Sn + d.unsqueeze(2) * k.unsqueeze(1)
                o = torch.einsum("hvk,hk->hv", Sn, q)
                o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS) * c.onw.double()
                o_step[r] = o * torch.sigmoid(c.gate[step][0, r].double().view(H, K))
                # The commit's replay of this record from the rings.
                rk = ring_snaps[step][0][s, :, t].double()
                rv = ring_snaps[step][1][s, :, t].double()
                for name, rec, want in (("rawk", rk, y[1]), ("rawv", rv, y[2])):
                    off = (rec - want).abs() - _ulp_bf16(want) - 1e-5
                    assert off.max().item() <= 0, f"{name} ring record step {step} slot {s} token {t}"
                kc, vc = l2n(rk), rv
                Sc = Sc * torch.exp(gk).unsqueeze(1)
                dc = (vc - torch.einsum("hvk,hk->hv", Sc, kc)) * beta
                Sc = Sc + dc.unsqueeze(2) * kc.unsqueeze(1)
                if t + 1 == int(accepts[step][n]):
                    S_commit, cs_commit = Sc.clone(), [x.clone() for x in win]
            S[s] = S_commit.to(c.h0.dtype).double()
            for j in range(3):
                cs[j][s] = cs_commit[j]
        outs.append(o_step)
    return outs, S


@pytest.mark.parametrize("N,pad_row", [(3, True), (20, False)], ids=["n3_wide_pad", "n20_narrow"])
@pytest.mark.parametrize("pattern", ["all", "bonus_only", "mixed"])
@STATE_DTYPES
def test_fold_matches_stock_two_kernel_path(N, pad_row, pattern, state_dtype):
    steps = 4
    c = Chain(N, num_spec=7, steps=steps, seed=7 + N, pad_row=pad_row, state_dtype=state_dtype)
    accepts = [c.accepts(pattern, s) for s in range(steps)]

    # Stock: verify, commit, conv rollback.
    h_s, cs_s, rings_s = c.h0.clone(), [t.clone() for t in c.cs0], _rings(c)
    outs_s, committed_s, ring_snaps = [], [], []
    for step in range(steps):
        out, ic = _verify(c, step, h_s, cs_s, rings_s)
        outs_s.append(out)
        ring_snaps.append((rings_s[1].clone(), rings_s[0].clone()))
        _commit(h_s, rings_s, c.slots, accepts[step])
        _roll_conv(c, cs_s, ic, accepts[step])
        committed_s.append(h_s.clone())

    # Fold: verify(fold_pending), fold post-verify (pending), conv rollback; flush at the end.
    h_f, cs_f, rings_f = c.h0.clone(), [t.clone() for t in c.cs0], _rings(c)
    spec = _spec(h_f, rings_f)
    pending = torch.zeros(c.num_slots, device=DEV, dtype=torch.int32)
    live = c.slots[c.slots >= 0].long()
    stock_step = 2 if pattern == "mixed" else -1
    outs_f = []
    m = dict(N=N, pattern=pattern, dtype=str(state_dtype), inkernel_state=0.0, out_rel=0.0, out_cos=1.0)
    for step in range(steps):
        if step == stock_step:
            # A non-fold forward: flush first, then the stock verify (ring rows from 0).
            kda_fold_flush(spec_state=spec, pending=pending, state_batch_indices=c.slots)
            _check_state(h_f[live], committed_s[step - 1][live], f"flush before step {step}")
            out, ic = _verify(c, step, h_f, cs_f, rings_f)
        else:
            out, ic = _verify(c, step, h_f, cs_f, rings_f, fold_pending=pending)
        outs_f.append(out)
        if step > 0 and step != stock_step:
            # The kernel wrote the state committed through step-1: stock's state after step-1's commit.
            e = _check_state(h_f[live], committed_s[step - 1][live], f"in-kernel commit, step {step}")
            m["inkernel_state"] = max(m["inkernel_state"], e)
        acc = accepts[step]
        commit_kda_replayssm_fold_after_verify(
            spec_state=spec, pending=pending, state_batch_indices=c.slots, accept_lens=acc,
            last_correct_step_indices=(acc - 1).to(torch.int64),
        )  # fmt: skip
        _roll_conv(c, cs_f, ic, acc)
    kda_fold_flush(spec_state=spec, pending=pending, state_batch_indices=c.slots)
    torch.cuda.synchronize()

    for step in range(steps):
        e, cs_ = _rel(outs_f[step], outs_s[step]), _cos(outs_f[step], outs_s[step])
        m["out_rel"], m["out_cos"] = max(m["out_rel"], e), min(m["out_cos"], cs_)
        assert e <= OUT_TOL and cs_ >= OUT_COS[state_dtype], (
            f"step {step} outputs fold vs stock: rel {e:.3e} cos {cs_:.7f}"
        )
    m["final_state"] = _check_state(h_f[live], h_s[live], "final state fold vs stock")
    unused = torch.ones(c.num_slots, dtype=torch.bool, device=DEV)
    unused[live] = False
    assert torch.equal(h_f[unused], c.h0[unused]), "fold touched a slot no request owns"
    if pad_row:
        rows = slice((N - 1) * c.T_req, N * c.T_req)
        for out in outs_f:
            assert torch.count_nonzero(out[0, rows]) == 0, "padding row output not zero"

    # Independent reference (catches bugs both arms share).
    ref_outs, ref_S = _ref_chain(c, accepts, ring_snaps)
    for arm, outs, h in (("stock", outs_s, h_s), ("fold", outs_f, h_f)):
        for step in range(steps):
            o = outs[step][0].double()
            e, cs_ = _rel(o, ref_outs[step]), _cos(o, ref_outs[step])
            assert e <= REF_OUT_TOL and cs_ >= REF_OUT_COS, (
                f"{arm} step {step} vs float64 reference: rel {e:.3e} cos {cs_:.6f}"
            )
        e = _rel(h[live], ref_S[live])
        m[f"{arm}_state_vs_f64"] = e
        assert e <= REF_STATE_TOL[state_dtype], f"{arm} final state vs float64 reference: rel {e:.3e}"
    print("KF_METRIC", m)


@pytest.mark.parametrize("N,pad_row", [(3, True), (20, False)], ids=["n3_wide_pad", "n20_narrow"])
@STATE_DTYPES
def test_fold_verify_reads_stored_state(N, pad_row, state_dtype):
    """A fold verify's outputs equal a plain verify's on the state the fold kernel stored, bit for bit.

    Catches a fold kernel that verifies from a state other than the one it stores (for BF16 state: from its unrounded
    FP32 replay), which would make a window's outputs depend on whether its predecessor was committed in-kernel or by
    a flush. FP32 state is the control: there the stored state is the register state.
    """
    c = Chain(N, num_spec=7, steps=2, seed=31 + N, pad_row=pad_row, state_dtype=state_dtype)
    acc0 = c.accepts("mixed", 0)
    h, cs, rings = c.h0.clone(), [t.clone() for t in c.cs0], _rings(c)
    pending = torch.zeros(c.num_slots, device=DEV, dtype=torch.int32)
    live = c.slots[c.slots >= 0].long()
    _, ic0 = _verify(c, 0, h, cs, rings, fold_pending=pending)  # nothing pending: verify only
    commit_kda_replayssm_fold_after_verify(
        spec_state=_spec(h, rings), pending=pending, state_batch_indices=c.slots, accept_lens=acc0,
        last_correct_step_indices=(acc0 - 1).to(torch.int64),
    )  # fmt: skip
    _roll_conv(c, cs, ic0, acc0)
    cs_before, h_before = [t.clone() for t in cs], h.clone()
    out_f, _ = _verify(c, 1, h, cs, rings, fold_pending=pending)  # replays step 0's window, stores, verifies
    h_stored = h.clone()
    out_c, _ = _verify(c, 1, h_stored.clone(), cs_before, [r.clone() for r in rings])
    torch.cuda.synchronize()
    assert not torch.equal(h_stored[live], h_before[live]), "the fold verify must have committed a window"
    assert torch.equal(out_f, out_c), (
        f"fold verify outputs differ from a verify on the stored state: rel {_rel(out_f, out_c):.3e}, "
        f"{(out_f != out_c).double().mean().item():.4%} of elements"
    )


def test_fold_rejects_bad_pending():
    c = Chain(2, num_spec=7, steps=1, seed=0, pad_row=False)
    bad = torch.zeros(c.num_slots, device=DEV, dtype=torch.int64)
    with pytest.raises(ValueError):
        _verify(c, 0, c.h0.clone(), [t.clone() for t in c.cs0], _rings(c), fold_pending=bad)
