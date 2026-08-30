import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _swiglu_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    SCALE: tl.constexpr,
    DO_LIMIT: tl.constexpr,
    LIMIT: tl.constexpr,
):
    # calc real total_rows
    if GROUP_LIST_TYPE == 0:  # cusum
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
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
        # swiglu; int64 row offset: triton-ascend does not auto-promote
        # row_idx * TOTAL_COLS, which overflows int32 at large M * N.
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        x_offsets = row_off + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets).to(tl.float32)
        x1 = al.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        x2 = al.extract_slice(
            cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,)
        )

        # DeepSeek-V4-Flash Expert.forward, in fp32: clamp gate (max only) and
        # up (symmetric) BEFORE silu, then out = silu(gate) * up.
        if DO_LIMIT:
            gate = tl.minimum(x1, LIMIT)
            up = tl.maximum(tl.minimum(x2, LIMIT), -LIMIT)
            out = gate * tl.sigmoid(gate) * up
        else:
            out = x1 * tl.sigmoid(x1) * x2

        # mxfp8 quant: e4m3 payload + one e8m0 scale per 32-element block.
        # CANN floor convention (cf. kv_compress_epilog): e is the biased fp32
        # exponent of amax * (1/448), so scale = 2^(e-127) and the block amax
        # lands in [448, 896) -> clamps to the e4m3 max.
        if SCALE:
            NUM_SUB_BLK: tl.constexpr = COL_BLOCK_SIZE // 32
            row_scale_off = row_idx.to(tl.int64) * (HALF_COLS // 32)
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                blk = al.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                blk2d = tl.reshape(blk, (NUM_SUB_BLK, 32))
                amax = tl.max(tl.abs(blk2d), axis=1)
                m = tl.maximum(amax, 1e-4)
                m2 = m * (1.0 / 448.0)
                e = (m2.to(tl.int32, bitcast=True) >> 23) & 0xFF
                descale = tl.reshape(
                    tl.exp2(e.to(tl.float32) - 127.0), (NUM_SUB_BLK, 1)
                )
                q = tl.clamp(blk2d / descale, -448.0, 448.0)
                q = tl.reshape(q, (COL_BLOCK_SIZE,)).to(out_ptr.dtype.element_ty)

                o_offsets = (
                    row_idx.to(tl.int64) * HALF_COLS
                    + col_blk_idx
                    + tl.arange(0, COL_BLOCK_SIZE)
                )
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(out_ptr + o_offsets, q, mask=mask)

                s_idx = col_blk_idx // 32
                s_offsets = row_scale_off + s_idx + tl.arange(0, NUM_SUB_BLK)
                s_mask = (s_idx + tl.arange(0, NUM_SUB_BLK)) < (HALF_COLS // 32)
                tl.store(scale_ptr + s_offsets, e.to(tl.uint8), mask=s_mask)
        else:
            # store out
            o_offsets = row_idx.to(tl.int64) * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))


def swiglu_quant(
    x, group_list, group_list_type, need_quant=True, do_limit=False, limit=7.0
):
    """SwiGLU with DeepSeek-V4-Flash Expert.forward semantics + MXFP8 quant.

    Activation, in fp32: up = clamp(up, -limit, limit); gate = clamp(gate,
    max=limit) applied BEFORE silu; out = silu(gate) * up. ``do_limit=False``
    matches swiglu_limit == 0 (plain silu(gate) * up).

    ``need_quant=True`` quantizes to MXFP8 per 32-element block with the CANN
    floor convention (e8m0 exponent = biased fp32 exponent of amax / 448):
    returns ``out`` as ``float8_e4m3fn [s, h/2]`` and ``scale`` as
    ``float8_e8m0fnu [s, h/2/32]``. The scale buffer holds uint8 biased
    exponents viewed as e8m0 (the ``npu_dynamic_mx_quant`` format); reshape it
    to ``[s, h/2/64, 2]`` (a free view) for ``npu_grouped_matmul``
    ``per_token_scale``. With ``need_quant=False`` the unquantized activation
    is returned and ``scale`` is uninitialised (caller must ignore it).

    Supports both prefill and decode: total rows are derived on device from
    ``group_list`` (0 = cusum, e.g. prefill TP / DeepEP-normal; 1 = per-expert
    counts, e.g. decode routing_v2 / DeepEP-LL), no host sync.
    """
    # group_list_type must be 0 cusum or 1 count
    if group_list_type not in [0, 1]:
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    s, h = x.shape
    half_cols = h // 2
    if need_quant and half_cols % 32 != 0:
        raise ValueError(
            f"MXFP8 quant requires h // 2 divisible by 32, but got {half_cols}"
        )
    out_dtype = torch.float8_e4m3fn if need_quant else x.dtype
    out = torch.empty((s, half_cols), dtype=out_dtype, device=x.device)
    scale = torch.empty((s, half_cols // 32), dtype=torch.uint8, device=x.device)
    num_experts = group_list.shape[0]
    # ub must be 32-byte aligned on npu
    if group_list.dtype == torch.int64:
        num_experts_algin = (num_experts + 7) // 8 * 8
    elif group_list.dtype == torch.int32:
        num_experts_algin = (num_experts + 15) // 16 * 16
    else:
        raise ValueError(
            f"group_list dtype must be torch.int32 or torch.int64, but got {group_list.dtype}"
        )

    _, num_vectorcore = get_device_properties()
    _swiglu_quant_kernel[(num_vectorcore,)](
        x,
        group_list,
        out,
        scale,
        TOTAL_COLS=h,
        HALF_COLS=half_cols,
        COL_BLOCK_SIZE=1536,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_ALGIN=num_experts_algin,
        GROUP_LIST_TYPE=group_list_type,
        NUM_CORES=num_vectorcore,
        SCALE=need_quant,
        multibuffer=True,
        DO_LIMIT=do_limit,
        LIMIT=limit,
    )
    if need_quant:
        return out, scale.view(torch.float8_e8m0fnu)
    return out, scale
