"""Triton sparse-MLA forward for the DSA fp8 prefill path.

A per-query flash-attention kernel over the indexer-selected topk KV. On
gfx950 this is ~1.6x faster than the TileLang partial+combine kernel for the
prefill regime (n_groups=1): the attention tile is tiny (M=16 heads = one
16x16 MFMA), so a small-warp per-program kernel avoids the intra-block
coordination overhead of the 256-thread TileLang block.
"""

import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0


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

# SM120: the 18-config autotune sweep (nvcc compile + benchmark of every config,
# on every DP rank at once) overruns the 300s scheduler watchdog on the first
# prefill. A *single* config means triton skips benchmarking and compiles once.
# Env-tunable if you want to try other tiles. Off-SM120 keeps the full sweep.
try:
    from sglang.srt.utils import is_sm120_supported as _is_sm120_supported

    if _is_sm120_supported():
        # SM120 smem cap is 101376 B. Calibrated from two measured points at
        # H=64, BLOCK_N=64: stages=2 -> 127232 B, stages=1 -> 122880 B (so a
        # pipeline stage is only ~4352 B; the q/kv TILES dominate, not stages).
        # Decompose: BLOCK_N-scaling = (D_V + D_TAIL + H)*BLOCK_N = 640*BLOCK_N;
        # fixed = 122880 - 640*64 = 81920 (q tile [H,D_V] + scratch).
        # Predicted (stages=1):  BLOCK_N=64->122880(OOM), 32->102400(OOM, ~1KB
        # over), 16->92160(FITS), 8->87040(FITS). So 32+ do NOT fit at H=64.
        # Use {8, 16}: both provably fit; triton picks 16 (faster), 8 is a floor.
        _w = int(os.environ.get("SGLANG_SM120_DSA_ATTN_WARPS", "4"))
        _ns = int(os.environ.get("SGLANG_SM120_DSA_ATTN_STAGES", "1"))
        _bn_env = os.environ.get("SGLANG_SM120_DSA_ATTN_BLOCK_N")
        _bns = [int(_bn_env)] if _bn_env else [8, 16]
        _AUTOTUNE_CONFIGS = [
            triton.Config({"BLOCK_N": bn}, num_warps=_w, num_stages=_ns)
            for bn in _bns
        ]
except Exception:  # noqa: BLE001
    pass


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
    H: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)

    h = tl.arange(0, H)
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    # q is read as two separate tensors (q_nope width D_V, q_rope width D_TAIL):
    # the upstream concat into a single [.., DIM] tensor is skipped since this
    # kernel splits q into main/tail anyway.
    q_main = tl.load(q_nope_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :]).to(
        q_nope_ptr.dtype.element_ty
    )  # [H, D_V]
    q_tail = tl.load(
        q_rope_ptr + s_i * H * D_TAIL + h[:, None] * D_TAIL + dt[None, :]
    ).to(
        q_nope_ptr.dtype.element_ty
    )  # [H, D_TAIL]

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0)
        kbase = kv_ptr + page[:, None] * DIM
        kv_main = tl.load(kbase + dv[None, :], mask=valid[:, None], other=0.0).to(
            q_nope_ptr.dtype.element_ty
        )  # [BLOCK_N, D_V] -- reused as V
        kv_tail = tl.load(
            kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(
            q_nope_ptr.dtype.element_ty
        )  # [BLOCK_N, D_TAIL]

        qk = tl.dot(q_main, tl.trans(kv_main)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail)).to(tl.float32)
        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # Guard an all-masked row (m_new == -inf): shift by 0 instead so that
        # exp(-inf - 0) = 0 rather than exp(-inf + inf) = NaN. Identical to
        # m_new whenever the row has >=1 valid key.
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])  # [H, BLOCK_N]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(q_nope_ptr.dtype.element_ty)
        pv = tl.dot(p_fp8, kv_main).to(tl.float32) * (1.0 / fp8_max)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
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

    Reads q from the two un-concatenated tensors directly (no q_nope/q_rope
    concat). Returns [1, seq, H, d_v] bf16 to match tilelang_sparse_fwd.
    """
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    # BLOCK_N / num_warps / num_stages are chosen by @triton.autotune.
    _sparse_mla_fwd_kernel[(seq,)](
        q_nope,
        q_rope,
        kv,
        indices,
        out,
        sm_scale,
        _FP8_MAX,
        topk,
        H=H,
        DIM=dim,
        D_V=d_v,
        D_TAIL=d_tail,
    )
    return out.unsqueeze(0)


_sm120_route_logged = False


def sm120_triton_sparse_mla(layer, q_nope, q_rope, kv_cache, page_table_1):
    """SM120 has no working tilelang sparse-attention kernel (it is Hopper-only:
    166 KB smem / block_I=64 hardwired, exceeds SM120's ~100 KB). Use this triton
    sparse-MLA kernel instead when the absorbed-MLA shapes/dtype match (fp8 KV,
    d_v=512, tail=64, topk=2048 -- the GLM-5.2 DSA path). The head count is NOT
    fixed: under DP-attention each rank carries the full 64 heads; under plain TP
    it is fewer. The triton kernel handles any (power-of-2) head count. Returns
    the attn output, or None to fall back to the (tilelang) path.

    Lives here (not in dsa_backend) so it can be imported without pulling in the
    heavy dsa_backend import chain (flashinfer.comm etc.).
    """
    global _sm120_route_logged
    if q_rope is None:
        return None

    h = layer.tp_q_head_num
    ok = (
        kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and h in (16, 32, 64, 128)
        and layer.v_head_dim == 512
        and (layer.head_dim - layer.v_head_dim) == 64
        and page_table_1.shape[-1] == 2048
    )
    if not _sm120_route_logged:
        _sm120_route_logged = True
        import logging

        logging.getLogger(__name__).warning(
            "SM120 DSA attention: %s triton sparse-MLA "
            "(heads=%s kv_dtype=%s v_head_dim=%s head_dim=%s topk=%s)",
            "USING" if ok else "FALLING BACK (tilelang) -- gate not matched, from",
            h, kv_cache.dtype, layer.v_head_dim, layer.head_dim,
            page_table_1.shape[-1],
        )
    if not ok:
        return None
    return triton_sparse_mla_fwd(
        q_nope=q_nope,
        q_rope=q_rope,
        kv=kv_cache,
        indices=page_table_1.unsqueeze(1),
        sm_scale=layer.scaling,
        d_v=layer.v_head_dim,
    )
