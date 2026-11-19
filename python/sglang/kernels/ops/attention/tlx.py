"""gfx950 attention input preparation from fbtriton ``tlx.ops.qk_norm_rope_kv_cache``, vendored in ``sglang.kernels.tlx_ops``.

``qk_norm_rope_kv_cache`` computes what ``qk_norm_rope_cache`` (``kernels/ops/attention/qk_norm_rope_cache.py``)
does, in one launch for any number of tokens, with the per-head sums of squares in another fixed order: weightless
per-head RMSNorm of q and k with the BF16 round trips of the unfused sequence, the query scale, interleaved RoPE on
BF16-rounded cos/sin and products, and the FP8 K/V store into ``[num_blocks, page, HK, D]`` caches through the slot
mapping (optionally remapped full -> SWA; negative slots write nothing). It returns q only; k is consumed by the
cache store. Rejected (see ``qk_norm_rope_kv_cache_supported``): other dtypes, head dims or head counts that are not
powers of two, rows wider than 4096. The caller gates the architecture. The kernel is plain Triton (no TLX primitives), so stock ROCm Triton runs it.
"""

import torch


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def qk_norm_rope_kv_cache_supported(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor
) -> bool:
    """Whether ``qk_norm_rope_kv_cache`` handles these operands; host-only, never reads device data."""
    if k_cache.dim() != 4 or q.dim() != 2 or k.dim() != 2 or v.shape != k.shape:
        return False
    _, _, num_kv_heads, head_dim = k_cache.shape
    return (
        q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.stride(1) == k.stride(1) == v.stride(1) == 1
        and _pow2(head_dim)
        and 16 <= head_dim <= 256
        and q.shape[1] % head_dim == 0
        and _pow2(q.shape[1] // head_dim)
        and _pow2(num_kv_heads)
        and q.shape[1] <= 4096
        and k.shape == (q.shape[0], num_kv_heads * head_dim)
    )


def qk_norm_rope_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    slot_remap: torch.Tensor | None,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    eps: float,
    q_scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """Normalized, scaled and rotated BF16 q ``(T, HQ * D)``; K/V written to the FP8 caches."""
    from sglang.kernels.tlx_ops.kernels.fused.qk_norm_rope_kv_cache.gfx950 import (
        qk_norm_rope_kv_cache as _qk_norm_rope_kv_cache,
    )

    return _qk_norm_rope_kv_cache(
        q,
        k,
        v,
        k_cache,
        v_cache,
        slot_mapping,
        eps,
        q_scale,
        positions,
        cos_sin_cache,
        True,
        slot_remap,
        k_scale,
        v_scale,
        None,
        None,
    )


_PAGED_DECODE_WORKSPACES: dict = {}
_PAGED_DECODE_SCALES: dict = {}


class PagedDecodeNormGate:
    """Hand-off between the model and the attention backend: while ``gate`` is set, a decode that runs
    ``paged_attention_decode`` applies the attention output's weightless per-head RMSNorm times sigmoid(``gate``) in
    its split-K reduce and sets ``done``; otherwise the model applies the norm-gate itself."""

    def __init__(self):
        self.gate = None
        self.eps = 0.0
        self.done = False


NORM_GATE = PagedDecodeNormGate()


def _scale_1(s, device) -> torch.Tensor:
    """A ``[1]`` FP32 dequant scale tensor for a Python float or a 0-/1-element tensor."""
    if isinstance(s, torch.Tensor):
        return (
            s
            if s.dtype == torch.float32 and s.dim() == 1 and s.numel() == 1
            else s.reshape(1).float()
        )
    key = (float(s), device)
    t = _PAGED_DECODE_SCALES.get(key)
    if t is None:
        t = _PAGED_DECODE_SCALES[key] = torch.full(
            (1,), float(s), dtype=torch.float32, device=device
        )
    return t


def paged_attention_decode_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> bool:
    """Whether ``paged_attention_decode`` handles one-token-per-sequence decode on these operands; host-only.

    q is ``[T, HQ, D]`` BF16, the caches are SGLang's page-major ``[num_blocks, page, HK, D]`` float8_e4m3fn views.
    """
    return (
        q.dim() == 3
        and k_cache.dim() == 4
        and v_cache.shape == k_cache.shape
        and q.dtype == torch.bfloat16
        and k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
        and q.stride(2) == 1
        and q.shape[2] == k_cache.shape[3]
        and _pow2(q.shape[2])
        and q.shape[1] % k_cache.shape[2] == 0
        and block_table.dtype == torch.int32
        and block_table.dim() == 2
        # CUDA-graph decode passes the whole page-table buffer; rows past the batch are never read.
        and block_table.shape[0] >= q.shape[0]
    )


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    sm_scale: float,
    sliding_window: int,
    max_context_len: int,
    k_descale,
    v_descale,
    out_gate: torch.Tensor | None = None,
    out_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Decode attention into ``out`` ``[T, HQ, D]`` (BF16); ``sliding_window`` 0 attends to the whole context.
    With ``out_gate`` (``[T, HQ * 64]`` BF16, unit inner stride), ``out`` is ``rmsnorm_gate(attention, out_gate)``.

    ``seq_lens`` may be int32 or int64. The split plan depends on the batch size, the window and
    ``max_context_len`` (the page table width times the page size, fixed per CUDA graph), and its workspace is
    allocated once per plan key, so a captured graph replays with the same buffers.
    """
    from sglang.kernels.tlx_ops.kernels.paged_attn import gfx950_decode as m

    norm_gate = out_gate is not None
    key = (
        q.shape[0],
        q.shape[1],
        k_cache.shape[2],
        sliding_window,
        max_context_len,
        block_table.shape[1],
        norm_gate,
        q.device,
    )
    ws = _PAGED_DECODE_WORKSPACES.get(key)
    if ws is None:
        ws = _PAGED_DECODE_WORKSPACES[key] = m.paged_attention_decode_workspace(
            q,
            k_cache,
            v_cache,
            block_table,
            1,
            sliding_window,
            max_context_len,
            "NHD",
            norm_gate=norm_gate,
        )
    return m.paged_attention_decode(
        q,
        k_cache,
        v_cache,
        seq_lens,
        block_table,
        sm_scale,
        1,
        sliding_window,
        max_context_len,
        out,
        ws,
        "NHD",
        k_descale=_scale_1(k_descale, q.device),
        v_descale=_scale_1(v_descale, q.device),
        out_gate=out_gate,
        out_norm_eps=out_norm_eps,
    )
