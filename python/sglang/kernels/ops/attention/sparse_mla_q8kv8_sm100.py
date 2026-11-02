"""JIT-compiled Q8KV8 sparse MLA attention for SM100 (B200).

Q and KV are e4m3 on all 576 dims (512 NoPE + 64 RoPE) with per-tensor scales, and QK^T and PV both
run as FP8 tcgen05 MMAs. The inputs are those of FlashInfer's TRT-LLM-gen sparse MLA as the DSA
``trtllm`` backend builds them: merged e4m3 Q, the 576 B e4m3 paged KV cache, token-level top-k slots,
and the fused scales ``bmm1_scale = q_scale * k_scale * sm_scale`` and ``bmm2_scale``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.attention.utils import check_out_buffer
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

D_QK = 576
D_V = 512
_B_H = 64
# Split only long top-k lists, into chunks of at least 256 entries. Measured on B200: a split never
# paid off at topk=512, and at topk=2048 the best was the largest of 8 / 4 / 2 that keeps every work
# item on its own SM (decode batches of 1-64 query tokens).
_MIN_TOPK_TO_SPLIT = 1024
_MIN_SPLIT_LEN = 256


@cache_once
def _jit_sparse_mla_q8kv8_sm100_module() -> Module:
    # The kernel is built on tcgen05 and TMA gather4 and has only been validated on SM100.
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("sparse_mla_q8kv8_sm100 requires an SM100 (B200) GPU")
    return load_jit(
        "sparse_mla_q8kv8_sm100",
        cuda_files=["sparse_mla_q8kv8_sm100/entry.cuh"],
        cuda_wrappers=[("dispatch", "sparse_mla_q8kv8_sm100_dispatch")],
        # --use_fast_math maps the softmax exp2f to MUFU.EX2.
        extra_cuda_cflags=[
            "-O3",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_dependencies=["cutlass"],
    )


@register_custom_op(
    op_name="sparse_mla_q8kv8_sm100",
    mutates_args=[
        "out",
        "max_logits",
        "lse",
        "partial_out",
        "partial_max_logits",
        "partial_lse",
    ],
)
def _sparse_mla_q8kv8_sm100_op(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_length: torch.Tensor,
    out: torch.Tensor,
    max_logits: torch.Tensor,
    lse: torch.Tensor,
    partial_out: torch.Tensor,
    partial_max_logits: torch.Tensor,
    partial_lse: torch.Tensor,
    s_q: int,
    s_kv: int,
    h_q: int,
    topk: int,
    num_splits: int,
    stride_q_s_q: int,
    stride_q_h_q: int,
    stride_kv_s_kv: int,
    stride_indices_s_q: int,
    bmm1_scale: float,
    bmm2_scale: float,
    have_attn_sink: int,
    have_topk_length: int,
) -> None:
    _jit_sparse_mla_q8kv8_sm100_module().dispatch(
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        out,
        max_logits,
        lse,
        partial_out,
        partial_max_logits,
        partial_lse,
        s_q,
        s_kv,
        h_q,
        topk,
        num_splits,
        stride_q_s_q,
        stride_q_h_q,
        stride_kv_s_kv,
        stride_indices_s_q,
        bmm1_scale,
        bmm2_scale,
        have_attn_sink,
        have_topk_length,
    )


@cache_once
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@cache_once
def _placeholder(device: torch.device, dtype: torch.dtype, role: str) -> torch.Tensor:
    # One zero-size tensor per role, so the mutated arguments of the custom op never alias.
    return torch.empty(0, dtype=dtype, device=device)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _choose_num_splits(s_q: int, h_q: int, topk: int, num_sms: int) -> int:
    """Largest of 8 / 4 / 2 that keeps every work item on its own SM, else 1 (no split)."""
    if topk < _MIN_TOPK_TO_SPLIT:
        return 1
    items = s_q * (h_q // _B_H)
    for splits in (8, 4, 2):
        if (
            topk % (64 * splits) == 0
            and topk // splits >= _MIN_SPLIT_LEN
            and items * splits <= num_sms
        ):
            return splits
    return 1


@debug_kernel_api
def sparse_mla_q8kv8_sm100_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    num_splits: Optional[int] = None,
    *,
    out: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP8 sparse MLA on SM100: logits = bmm1_scale * q.k, out = bmm2_scale * softmax(logits) @ v.

    V is the first 512 dims of each KV row. Indices < 0 or >= s_kv and positions >= topk_length are
    masked. Masked positions with an in-range index still load their row, which must not hold NaN.

    Args:
        q:           [s_q, h_q, 576], float8_e4m3fn, NoPE then RoPE dims
        kv:          [s_kv, 576] or paged [pages, (1,) page_size, 576], float8_e4m3fn
        indices:     [s_q, topk] or [s_q, 1, topk], int32 token rows of kv
        bmm1_scale:  q_scale * kv_scale * sm_scale
        bmm2_scale:  kv_scale of V
        attn_sink:   optional [h_q], float32
        topk_length: optional [s_q], int32 number of leading valid positions per row
        num_splits:  1, 2, 4 or 8 top-k splits; None chooses from the shapes
        out:         optional [s_q, h_q, 512] bfloat16 buffer for the output
        max_logits:  optional [s_q, h_q] float32 buffer
        lse:         optional [s_q, h_q] float32 buffer

    Returns:
        out:        [s_q, h_q, 512], bfloat16
        max_logits: [s_q, h_q], float32, -inf for rows without valid indices
        lse:        [s_q, h_q], float32, natural log without the sink, +inf for rows without
                    valid indices
    """
    _check(
        q.dim() == 3 and q.shape[2] == D_QK,
        f"q must be [s_q, h_q, {D_QK}], got {tuple(q.shape)}",
    )
    _check(
        q.dtype == torch.float8_e4m3fn and kv.dtype == torch.float8_e4m3fn,
        "q and kv must be float8_e4m3fn",
    )
    _check(kv.shape[-1] == D_QK, f"kv must end in {D_QK} dims, got {tuple(kv.shape)}")
    if kv.dim() != 2:
        _check(
            kv.is_contiguous(),
            "a paged or 3D kv must be contiguous so its token rows flatten without a copy",
        )
        kv = kv.view(-1, D_QK)
    if indices.dim() == 3:
        _check(indices.shape[1] == 1, "only h_kv == 1 is supported")
        indices = indices.squeeze(1)
    _check(indices.dtype == torch.int32, "indices must be int32")
    s_q, h_q_real, _ = q.shape
    s_kv, topk = kv.shape[0], indices.shape[1]
    _check(indices.shape[0] == s_q, "indices must have s_q rows")
    _check(topk % 64 == 0, f"topk must be a multiple of 64, got {topk}")
    device = q.device
    for name, t in (("q", q), ("kv", kv), ("indices", indices)):
        _check(t.device == device, f"{name} must be on {device}, got {t.device}")
        _check(t.stride(-1) == 1, f"{name} must be contiguous in its last dimension")

    h_q = h_q_real
    out_user = max_logits_user = lse_user = None
    if h_q % _B_H != 0:
        _check(_B_H % h_q == 0, f"h_q must be a multiple of 64 or divide 64, got {h_q}")
        q_pad = q.new_zeros(s_q, _B_H, D_QK)
        q_pad[:, :h_q] = q
        q, h_q = q_pad, _B_H
        if attn_sink is not None:
            pad = attn_sink.new_full((_B_H - h_q_real,), float("-inf"))
            attn_sink = torch.cat([attn_sink, pad])
        out_user, max_logits_user, lse_user = out, max_logits, lse
        out = max_logits = lse = None
    if attn_sink is not None:
        _check(
            attn_sink.dtype == torch.float32
            and attn_sink.numel() == h_q
            and attn_sink.is_contiguous(),
            "attn_sink must be a contiguous [h_q] float32 tensor",
        )
    if topk_length is not None:
        _check(
            topk_length.dtype == torch.int32
            and topk_length.numel() == s_q
            and topk_length.is_contiguous(),
            "topk_length must be a contiguous [s_q] int32 tensor",
        )
    if num_splits is None:
        num_splits = _choose_num_splits(s_q, h_q, topk, _num_sms(device.index))
    _check(
        num_splits in (1, 2, 4, 8) and topk % (64 * num_splits) == 0,
        f"bad num_splits {num_splits} for topk {topk}",
    )

    if out is None:
        out = torch.empty(s_q, h_q, D_V, dtype=torch.bfloat16, device=device)
    else:
        check_out_buffer(out, "out", (s_q, h_q, D_V), torch.bfloat16, device)
    if max_logits is None:
        max_logits = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        check_out_buffer(max_logits, "max_logits", (s_q, h_q), torch.float32, device)
    if lse is None:
        lse = torch.empty(s_q, h_q, dtype=torch.float32, device=device)
    else:
        check_out_buffer(lse, "lse", (s_q, h_q), torch.float32, device)
    _check(
        len({out.data_ptr(), max_logits.data_ptr(), lse.data_ptr()}) == 3,
        "out, max_logits and lse must not alias",
    )
    if num_splits > 1:
        partial_out = torch.empty(
            num_splits, s_q, h_q, D_V, dtype=torch.bfloat16, device=device
        )
        partial_max_logits = torch.empty(
            num_splits, s_q, h_q, dtype=torch.float32, device=device
        )
        partial_lse = torch.empty(
            num_splits, s_q, h_q, dtype=torch.float32, device=device
        )
    else:
        partial_out = _placeholder(device, torch.bfloat16, "partial_out")
        partial_max_logits = _placeholder(device, torch.float32, "partial_max_logits")
        partial_lse = _placeholder(device, torch.float32, "partial_lse")

    _sparse_mla_q8kv8_sm100_op(
        q,
        kv,
        indices,
        (
            attn_sink
            if attn_sink is not None
            else _placeholder(device, torch.float32, "attn_sink")
        ),
        (
            topk_length
            if topk_length is not None
            else _placeholder(device, torch.int32, "topk_length")
        ),
        out,
        max_logits,
        lse,
        partial_out,
        partial_max_logits,
        partial_lse,
        s_q,
        s_kv,
        h_q,
        topk,
        num_splits,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        indices.stride(0),
        float(bmm1_scale),
        float(bmm2_scale),
        int(attn_sink is not None),
        int(topk_length is not None),
    )
    if h_q != h_q_real:
        out = out[:, :h_q_real]
        max_logits = max_logits[:, :h_q_real]
        lse = lse[:, :h_q_real]
        if out_user is not None:
            out = out_user.copy_(out)
        if max_logits_user is not None:
            max_logits = max_logits_user.copy_(max_logits)
        if lse_user is not None:
            lse = lse_user.copy_(lse)
    return out, max_logits, lse


def sparse_mla_q8kv8_sm100_trtllm(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Same arguments and result as FlashInfer's TRT-LLM-gen sparse MLA call.

    That is ``trtllm_batch_decode_with_kv_cache_mla(..., sparse_mla_top_k=topk)``. trtllm-gen treats
    the first ``min(seq_len, topk)`` slots of a row as valid; ``seq_lens`` applies the same rule here,
    on top of masking -1 and out-of-range slots, and lets the kernel skip the blocks past it.

    Args:
        query:        [batch, q_len, h_q, 576], float8_e4m3fn
        kv_cache:     [pages, (1,) page_size, 576], float8_e4m3fn
        block_tables: [batch, q_len, topk], int32 token slots
        bmm1_scale:   q_scale * kv_scale * sm_scale
        bmm2_scale:   kv_scale of V
        seq_lens:     optional [batch], int32; needs q_len == 1

    Returns:
        [batch, q_len, h_q, 512], bfloat16
    """
    batch, q_len, h_q, _ = query.shape
    topk_length = None
    if seq_lens is not None:
        _check(q_len == 1, f"seq_lens needs q_len == 1, got q_len={q_len}")
        _check(
            seq_lens.dtype == torch.int32 and seq_lens.numel() == batch,
            f"seq_lens must be [{batch}] int32, got {tuple(seq_lens.shape)} {seq_lens.dtype}",
        )
        topk_length = seq_lens.reshape(batch)
    out, _, _ = sparse_mla_q8kv8_sm100_fwd(
        query.view(batch * q_len, h_q, D_QK),
        kv_cache,
        block_tables.view(batch * q_len, block_tables.shape[-1]),
        bmm1_scale,
        bmm2_scale,
        topk_length=topk_length,
    )
    return out.view(batch, q_len, h_q, D_V)
