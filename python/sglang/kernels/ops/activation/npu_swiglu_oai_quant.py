"""swiglu_oai + optional quantization for Ascend NPU.

Adapted from: sgl_kernel_npu package (swiglu_quant.py)

Implements swiglu_oai (MiniMax-M3 variant) fused with optional per-row int8 quant:

    gate = x1.clamp(-inf, limit)
    up   = x2.clamp(-limit, limit)
    out  = gate * sigmoid(gate * alpha) * (up + 1)

where x = [gate | up] is concatenated along the last dimension.
When quantization is enabled, each row is scaled to int8 with per-row scale.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _swiglu_oai_quant_kernel_moe(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    alpha,
    limit,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
):
    # calc real total_rows
    if GROUP_LIST_TYPE == 0:  # cusum (group_list length = NUM_EXPERTS, last element = total)
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS - 1).to(tl.int32)
    else:
        gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
        gl_mask = gl_offsets < NUM_EXPERTS
        group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
        total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # swiglu_oai: gate * sigmoid(gate * alpha) * (up + 1)  with clamping
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        gate = al.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        up = al.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )
        gate = tl.minimum(gate, limit)                    # clamp(-inf, limit]
        up = tl.minimum(tl.maximum(up, -limit), limit)    # clamp[-limit, limit]
        out = gate * tl.sigmoid(gate * alpha) * (up + 1.0)

        # quant
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            # store scale
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp_out = al.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
                tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")

                o_offsets = (
                    row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                )
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(
                    out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=mask
                )
        else:
            # store out
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))


@triton.jit
def _swiglu_oai_quant_kernel_dense(
    x_ptr,
    out_ptr,
    scale_ptr,
    alpha,
    limit,
    total_rows,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Simplified kernel for dense MLP: total_rows is passed directly, no group_list."""
    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # swiglu_oai: gate * sigmoid(gate * alpha) * (up + 1)  with clamping
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        gate = al.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        up = al.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )
        gate = tl.minimum(gate, limit)                    # clamp(-inf, limit]
        up = tl.minimum(tl.maximum(up, -limit), limit)    # clamp[-limit, limit]
        out = gate * tl.sigmoid(gate * alpha) * (up + 1.0)

        # quant
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp_out = al.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
                tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")

                o_offsets = (
                    row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                )
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(
                    out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=mask
                )
        else:
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))


def swiglu_oai_quant(x, alpha, limit, need_quant=True, group_list=None, group_list_type=None):
    """swiglu_oai activation with optional int8 quantization.

    Supports two dispatch modes:

    **Dense MLP mode** (``group_list`` is ``None``, default):
        All rows are processed as a single group. No CPU-device sync for group_list.

    **MoE grouped mode** (``group_list`` is provided):
        Tokens are routed per-expert. ``group_list`` holds per-expert token counts
        (or cumsum), and ``group_list_type`` specifies the encoding (0=cusum, 1=count).

    Args:
        x: Input tensor [..., 2d], concatenated [gate | up] layout.
        alpha: Sigmoid scaling parameter.
        limit: Clamp limit.
        need_quant: If True, quantize output to int8 with per-row scale.
        group_list: Optional tensor of expert token counts/cumsum.
        group_list_type: Required if group_list is given (0=cusum, 1=count).

    Returns:
        (out, scale) — out shape [..., d], scale shape [num_rows].
    """
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    s, h = x.shape
    out_dtype = torch.int8 if need_quant else x.dtype
    out = torch.empty((s, h // 2), dtype=out_dtype, device=x.device)
    scale = torch.empty((s,), dtype=torch.float32, device=x.device)
    _, num_vectorcore = get_device_properties()

    if group_list is not None:
        # ── MoE grouped path ──────────────────────────────────────────
        if group_list_type not in (0, 1):
            raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
        num_experts = group_list.shape[0]
        # ub must be 32-byte aligned on npu
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, "
                f"got {group_list.dtype}"
            )

        _swiglu_oai_quant_kernel_moe[(num_vectorcore,)](
            x,
            group_list,
            out,
            scale,
            alpha,
            limit,
            TOTAL_COLS=h,
            HALF_COLS=h // 2,
            COL_BLOCK_SIZE=1536,
            NUM_EXPERTS=num_experts,
            NUM_EXPERTS_ALGIN=num_experts_algin,
            GROUP_LIST_TYPE=group_list_type,
            NUM_CORES=num_vectorcore,
            DTYPE_MAX=127,
            SCALE=need_quant,
            multibuffer=True,
        )
    else:
        # ── Dense MLP path ────────────────────────────────────────────
        _swiglu_oai_quant_kernel_dense[(num_vectorcore,)](
            x,
            out,
            scale,
            alpha,
            limit,
            s,
            TOTAL_COLS=h,
            HALF_COLS=h // 2,
            COL_BLOCK_SIZE=1536,
            NUM_CORES=num_vectorcore,
            DTYPE_MAX=127,
            SCALE=need_quant,
            multibuffer=True,
        )

    return out.reshape(*orig_shape[:-1], -1), scale
