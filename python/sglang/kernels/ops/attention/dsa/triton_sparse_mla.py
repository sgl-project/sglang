"""Triton sparse-MLA forward for the DSA fp8 prefill path — gfx942 (MI308X) port.

A per-(merged)query flash-attention kernel over the indexer-selected topk KV.
On gfx950 this is ~1.6x faster than the TileLang partial+combine kernel for the
prefill regime (n_groups=1): the attention tile is tiny (one 16x16 MFMA), so a
small-warp per-program kernel avoids the intra-block coordination overhead of
the 256-thread TileLang block.

gfx942 port note (v2 — 2-query merged program)
------------------------------------------------
The per-query kernel computes tl.dot([H, D_V], [D_V, BLOCK_N]), placing the
head count H on the MFMA **M** dimension. CDNA3 matrix cores require M >= 16,
but with 8 heads/card on MI308X (TP8) H=8 < 16 — pad-to-16 head duplication
(v1) satisfied the constraint but wasted 2x FLOPs.

v2 instead merges **2 query tokens** into one program. 2 tokens x 8 heads = 16
rows, so M = 16 is legal with *zero* wasted compute (each row is a distinct
real (token, head) and the two tokens' KV sets are kept disjoint via a
structured (row_tok == tok2) block-diagonal mask). grid = ceil(seq/2).
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0

# Number of query tokens fused into a single program. 2 tokens * 8 heads = 16
# rows -> tl.dot M = 16 (legal on CDNA3), with no duplicate work.
_NTOK = 2


def _prune_configs(configs, named_args, **kwargs):
    """Drop configs whose KV tile exceeds topk (pure waste)."""
    topk = named_args["topk"]
    keep = [c for c in configs if c.kwargs["BLOCK_N"] <= topk]
    return keep or [configs[0]]


# The best (BLOCK_N, num_warps, num_stages) is shape- and arch-sensitive, so
# autotune over a grid keyed on the attention shape. Benchmarked once per key
# (a one-time stall on the first prefill of each new shape), then cached.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=ns)
    for bn in (32, 64, 128)
    for w in (1, 2, 4)
    for ns in (1, 2)
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["topk", "H", "DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sparse_mla_fwd_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    sm_scale,
    fp8_max,
    topk,
    seq,
    H: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NTOK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)

    ROWS: tl.constexpr = NTOK * H
    row_tok = tl.arange(0, ROWS) // H   # [rows] which token within this program
    row_h = tl.arange(0, ROWS) % H      # [rows] which head within the token
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)

    # ---- load q for all NTOK tokens of this program ----------------------
    # Row layout: [tok0 h0..h_{H-1}, tok1 h0..h_{H-1}, ...]; M = rows = 16.
    tok_idx = s_i * NTOK + row_tok                 # [rows] global token index
    q_off_nope = tok_idx[:, None] * H * D_V + row_h[:, None] * D_V + dv[None, :]
    q_off_rope = tok_idx[:, None] * H * D_TAIL + row_h[:, None] * D_TAIL + dt[None, :]
    q_mask = tok_idx[:, None] < seq                # mask odd-boundary token out

    q_main = tl.load(
        q_nope_ptr + q_off_nope, mask=q_mask, other=0.0
    ).to(q_nope_ptr.dtype.element_ty)              # [rows, D_V]
    q_tail = tl.load(
        q_rope_ptr + q_off_rope, mask=q_mask, other=0.0
    ).to(q_nope_ptr.dtype.element_ty)              # [rows, D_TAIL]

    m_i = tl.full([ROWS], -float("inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D_V], tl.float32)

    # ---- gather KV for all NTOK tokens, laid out as NTOK * BLOCK_N cols ---
    col = tl.arange(0, NTOK * BLOCK_N)             # [NTOK*BLOCK_N]
    tok2 = col // BLOCK_N                          # which token each column feeds
    within2 = col % BLOCK_N                         # index within that token's topk

    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + within2) < topk
        base = s_i * NTOK + tok2                   # global token for this column
        idx_off = base * topk + k0 + within2
        idx_mask = kmask & (base < seq)
        idx_cat = tl.load(
            idx_ptr + idx_off, mask=idx_mask, other=-1
        )                                          # [NTOK*BLOCK_N]
        valid_page = idx_cat >= 0                  # [NTOK*BLOCK_N]
        kbase = kv_ptr + idx_cat[:, None] * DIM
        kv_main_cat = tl.load(
            kbase + dv[None, :], mask=valid_page[:, None], other=0.0
        ).to(q_nope_ptr.dtype.element_ty)          # [NTOK*BLOCK_N, D_V]
        kv_tail_cat = tl.load(
            kbase + (D_V + dt)[None, :], mask=valid_page[:, None], other=0.0
        ).to(q_nope_ptr.dtype.element_ty)          # [NTOK*BLOCK_N, D_TAIL]

        # qk: [rows, NTOK*BLOCK_N]. dot M = rows = NTOK*H = 16 (legal).
        qk = tl.dot(q_main, tl.trans(kv_main_cat)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail_cat)).to(tl.float32)

        # Block-diagonal mask: token t's rows only attend its own BLOCK_N cols.
        valid_2d = (
            (row_tok[:, None] == tok2[None, :])
            & valid_page[None, :]
            & kmask[None, :]
        )                                          # [rows, NTOK*BLOCK_N]
        qk = tl.where(valid_2d, qk * sm_scale, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # Guard an all-masked row (m_new == -inf): shift by 0 rather than
        # exp(-inf + inf) = NaN. Identical to m_new whenever the row has a key.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])           # [rows, NTOK*BLOCK_N]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(q_nope_ptr.dtype.element_ty)
        pv = tl.dot(p_fp8, kv_main_cat).to(tl.float32) * (1.0 / fp8_max)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + q_off_nope,
        acc.to(o_ptr.dtype.element_ty),
        mask=q_mask,
    )


def triton_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """q_nope: [seq, H, d_v] fp8, q_rope: [seq, H, dim-d_v] fp8,
    kv: [num_pages, 1, dim] fp8, indices: [seq, 1, topk].

    Fuses NTOK=2 query tokens per program so the MFMA M dimension is
    NTOK*H = 16 (>= 16, legal on CDNA3) with no duplicate compute (v2).
    Returns [1, seq, H, d_v] bf16 to match tilelang_sparse_fwd.
    """
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    # v2 needs NTOK*H >= 16 for the tl.dot M dimension; with NTOK=2 that means
    # H >= 8. The MI308X TP8 target has H=8 (8 heads/card) -> M = 16 exactly.
    assert 2 * H >= 16, "gfx942 v2 kernel needs NTOK*H >= 16 (H >= 8 for NTOK=2)"
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]

    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    # grid: one program per NTOK tokens; BLOCK_N/warps/stages via autotune.
    _sparse_mla_fwd_kernel[((seq + _NTOK - 1) // _NTOK,)](
        q_nope,
        q_rope,
        kv,
        indices,
        out,
        sm_scale,
        _FP8_MAX,
        topk,
        seq,
        H=H,
        DIM=dim,
        D_V=d_v,
        D_TAIL=d_tail,
        NTOK=_NTOK,
    )
    return out.unsqueeze(0)
