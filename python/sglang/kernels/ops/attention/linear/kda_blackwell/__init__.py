# SPDX-License-Identifier: Apache-2.0
# KDA (Kimi Delta Attention) SM100/Blackwell CuteDSL prefill pipeline.
#
# Mirrors gdn_blackwell but for KDA's PER-CHANNEL decay gate. A fused Triton
# prologue computes the per-chunk cumsum g_cu and five pre-scaled key/query
# tensors; three cutedsl kernels then run the chunked gated delta rule:
#   prologue -> kkt_inv_uw (U,W) -> h (V_new, per-chunk state, final state) -> o
import torch

from .kernel_h import kda_h_cutedsl
from .kernel_kkt_inv_uw import kkt_inv_uw_cutedsl
from .kernel_o import kda_o_cutedsl
from .prologue import kda_prologue

__all__ = ["chunk_kda_cutedsl", "prepare_metadata"]


def prepare_metadata(cu_seqlens: torch.Tensor, chunk_size: int = 64):
    """Build (chunk_indices [NT,2], chunk_offsets [N+1], total_chunks [1]).

    chunk_indices[g] = (seq_id, local_chunk_id) for global chunk g.
    chunk_offsets[s] = number of chunks before sequence s.
    """
    dev = cu_seqlens.device
    cs = cu_seqlens.to(torch.int64)
    seqlens = cs[1:] - cs[:-1]
    nchunks = (seqlens + chunk_size - 1) // chunk_size  # [N]
    n = seqlens.numel()
    chunk_offsets = torch.zeros(n + 1, dtype=torch.int32, device=dev)
    chunk_offsets[1:] = nchunks.cumsum(0).to(torch.int32)
    total = int(chunk_offsets[-1].item())
    seq_id = torch.repeat_interleave(torch.arange(n, device=dev), nchunks)
    local = torch.arange(total, device=dev) - chunk_offsets[seq_id].to(torch.int64)
    chunk_indices = torch.stack(
        [seq_id.to(torch.int32), local.to(torch.int32)], dim=1
    ).contiguous()
    total_chunks = torch.tensor([total], dtype=torch.int32, device=dev)
    return chunk_indices, chunk_offsets, total_chunks, total


# Per-(Hv,K,V,device) grow-only scratch workspace. The cutedsl KKT/h/o kernels
# are fast; the per-call PyTorch overhead (re-allocating + re-zeroing the eye and
# the two pack buffers ~200MB/call, metadata recompute, a `.item()` sync) was what
# dragged the full function below Triton. Reusing scratch across calls removes it.
# Safe because KDA layers run sequentially on one CUDA stream (the next call's
# kernels are ordered after this call's), and only the returned o/ht are fresh.
_KDA_WS: dict = {}


def _kda_workspace(q, T, Hv, K, V, cu_seqlens):
    import torch as _t

    dev = q.device
    # Key by the current CUDA stream too: the scratch is process-global and
    # mutable, so two KDA forwards running concurrently on different streams
    # (e.g. two-batch overlap) must not share buffers. Within one forward all
    # KDA layers run on the same stream -> same key -> the reuse benefit holds.
    stream = _t.cuda.current_stream(device=dev).cuda_stream
    key = (Hv, K, V, dev, q.dtype, stream)
    ws = _KDA_WS.get(key)

    # metadata: recompute only when cu_seqlens changes (object identity -> no
    # sync; within one forward all KDA layers share the same cu_seqlens object).
    if ws is None or ws["cu"] is not cu_seqlens:
        ci, co, tcs, total = prepare_metadata(cu_seqlens)
    else:
        ci, co, tcs, total = ws["ci"], ws["co"], ws["tcs"], ws["total"]
    pad_t = total * 64

    if ws is None or ws["Tcap"] < T or ws["padcap"] < pad_t or ws["totalcap"] < total:
        Tcap = T if ws is None else max(T, ws["Tcap"])
        padcap = pad_t if ws is None else max(pad_t, ws["padcap"])
        totalcap = total if ws is None else max(total, ws["totalcap"])
        ws = {
            "kL": q.new_zeros(Tcap, Hv, K, dtype=_t.bfloat16),
            "qg2": q.new_zeros(Tcap, Hv, K, dtype=_t.bfloat16),
            "eye": q.new_zeros(Tcap, Hv, K, dtype=_t.bfloat16),
            "U": q.new_empty(padcap, Hv, V, dtype=_t.bfloat16),
            "W": q.new_empty(padcap, Hv, K, dtype=_t.bfloat16),
            "Vn": q.new_empty(padcap, Hv, V, dtype=_t.bfloat16),
            "hc": q.new_empty(totalcap, Hv, V, K, dtype=_t.bfloat16),
            "Tcap": Tcap,
            "padcap": padcap,
            "totalcap": totalcap,
            "cu": None,
            "eye_hw": 0,
        }
        _KDA_WS[key] = ws

    ws["ci"], ws["co"], ws["tcs"], ws["total"] = ci, co, tcs, total

    # eye is the one-hot(chunk-position) identity injection: recompute only on a
    # cu_seqlens change. Clear the prior high-water region then scatter the new 1s.
    if ws["cu"] is not cu_seqlens:
        eye = ws["eye"]
        hw = max(ws["eye_hw"], T)
        eye[:hw].zero_()
        # Match cu_seqlens' dtype (typically int32) so searchsorted/indexing avoid
        # the int64 casts, while staying correct if cu_seqlens is passed as int64.
        tok = _t.arange(T, device=dev, dtype=cu_seqlens.dtype)
        seq_of = _t.searchsorted(cu_seqlens, tok, right=True) - 1
        pos = (tok - cu_seqlens[seq_of]) % 64
        eye[tok, :, pos] = 1.0
        ws["eye_hw"] = T
        ws["cu"] = cu_seqlens
    return ws, ci, co, tcs, total, pad_t


def chunk_kda_cutedsl(
    q: torch.Tensor,  # [T, Hv, K] bf16, L2-normed
    k: torch.Tensor,  # [T, Hv, K] bf16, L2-normed
    v: torch.Tensor,  # [T, Hv, V] bf16
    g: torch.Tensor,  # [T, Hv, K] log-decay. RAW if A_log given, else pre-activated
    beta: torch.Tensor,  # [T, Hv] fp32, post-sigmoid
    h0: torch.Tensor,  # [N, Hv, V, K] state, or the state POOL with h0_indices
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    num_sms: int | None = None,
    A_log: torch.Tensor | None = None,  # [Hv]; if set, activate g internally
    dt_bias: torch.Tensor | None = None,  # [Hv, K] or [Hv*K]
    lower_bound: float | None = None,
    h0_indices: torch.Tensor | None = None,  # [N] int32 pool slots
):
    """Run the KDA chunk gated-delta-rule prefill. Returns (o [T,Hv,V], ht).

    Dense mode (``h0_indices is None``): ``h0`` is [N, Hv, V, K]; the final state
    is returned in a fresh ``ht`` and ``h0`` is left untouched.

    Pool mode: ``h0`` is the state pool [num_slots, Hv, V, K] and ``h0_indices``
    maps each sequence to its slot; the h kernel reads AND writes the pool rows
    in place (fused state gather/scatter — no [N, Hv, V, K] intermediates), and
    the returned ``ht`` is the pool tensor itself.
    """
    import torch.nn.functional as F

    T, Hv, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count

    # Gate activation (standard KDA gate). Fused into the prologue is a B2 TODO;
    # for now a small PyTorch pass, matching chunk_kda's kda_gate_chunk_cumsum.
    if A_log is not None:
        if lower_bound is not None:
            raise NotImplementedError(
                "KDA cutedsl: safe_gate (lower_bound) not yet supported"
            )
        x = g.float()
        if dt_bias is not None:
            x = x + dt_bias.float().view(1, Hv, K)
        g_act = -torch.exp(A_log.float()).view(1, Hv, 1) * F.softplus(x)
    else:
        g_act = g.float()

    # Reusable scratch (eye/pack/U/W/V_new/h_chunks) + cached metadata; only the
    # returned o/ht are freshly allocated. This removes the ~0.2-0.6ms/call host
    # overhead (re-alloc + re-zero of ~200MB + metadata sync) that otherwise drags
    # the (fast) cutedsl kernels below Triton.
    ws, chunk_indices, chunk_offsets, total_chunks, total, pad_t = _kda_workspace(
        q, T, Hv, K, V, cu_seqlens
    )

    # KL/qg2 from the prologue fold the decay with a chunk-global g_last reference
    # (exp(g_cu - g_last)), which overflows fp32 for real per-channel gates. They
    # are recomputed below; the prologue still gives the bounded KR/KG/qg/g_cu.
    _, KR, KG, qg, _, g_cu = kda_prologue(
        q, k, g_act, float(scale), cu_seqlens, chunk_indices, total
    )

    # Sub-chunk-normalized intra-chunk gated KKT / QK from the FLA kernel (stable),
    # injected through the cutedsl KKT/Aqk MMAs as an identity-right-operand pass:
    # with kL'=M (M in the first 64 K-slots) and kR'=onehot(chunk-pos), the MMA
    # kL'@kR'.T == M, so kkt_inv_uw/kernel_o see the correct matrix without overflow.
    from sglang.kernels.ops.attention.fla.kda import (
        RCP_LN2,
        chunk_kda_scaled_dot_kkt_fwd,
    )

    ones_beta = q.new_ones(1, T, Hv, dtype=torch.float32)
    # The FLA kkt kernels consume log2-space gate cumsums (exp2-based); g_cu must
    # stay natural-log for the cutedsl kernels below, so convert on a copy.
    M_kk, M_qk = chunk_kda_scaled_dot_kkt_fwd(
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        gk=(g_cu * RCP_LN2).unsqueeze(0),
        beta=ones_beta,
        scale=float(scale),
        cu_seqlens=cu_seqlens,
        chunk_size=64,
    )

    # Pack M into the first 64 K-slots of the reused buffers; cols [64:128] stay 0
    # (never written since the one-time zeroed alloc), so the MxI injection is exact.
    kL_inj = ws["kL"][:T]
    qg2_inj = ws["qg2"][:T]
    kL_inj[:, :, :64] = M_kk[0].to(torch.bfloat16)
    qg2_inj[:, :, :64] = M_qk[0].to(torch.bfloat16)
    eye = ws["eye"][:T]

    U = ws["U"][:pad_t]
    W = ws["W"][:pad_t]
    kkt_inv_uw_cutedsl(
        kL_inj,
        eye,
        KG,
        v,
        U,
        W,
        beta,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        num_sms=num_sms,
    )

    V_new = ws["Vn"][:pad_t]
    h_chunks = ws["hc"][:total]
    if h0_indices is None:
        # Dense mode: preserve the return-fresh-ht contract.
        ht = torch.empty_like(h0)
        state_indices = torch.arange(
            cu_seqlens.numel() - 1, device=q.device, dtype=torch.int32
        )
    else:
        # Pool mode: read and write the pool rows in place.
        ht = h0
        state_indices = h0_indices
    kda_h_cutedsl(
        KR,
        U,
        W,
        V_new,
        g_cu,
        h_chunks,
        h0,
        ht,
        cu_seqlens,
        chunk_offsets,
        state_indices,
    )

    o = q.new_empty(T, Hv, V, dtype=torch.bfloat16)
    kda_o_cutedsl(
        qg,
        qg2_inj,
        eye,
        V_new,
        h_chunks,
        o,
        cu_seqlens,
        chunk_indices,
        total_chunks,
        num_sms=num_sms,
    )
    return o, ht
