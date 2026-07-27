import functools
import logging
import sys
from enum import IntEnum
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_npu_memory_capacity, is_npu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)
_is_npu = is_npu()
indexer_weight_stream = None
gva_is_inited = False

from typing import Optional

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice

from sgl_kernel_npu.utils.triton_utils import get_device_properties


@triton.jit
def _situ_and_mul_quant_kernel(
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
    N_ROWS,
    NUM_CORES: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    SCALE: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    # total_rows: from group_list (routed MoE) or N_ROWS (dense / shared).
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:  # cusum
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
        else:  # count
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
            total_rows = tl.sum(group_list)
    else:
        total_rows = N_ROWS

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    # full-row load (d<=6144 fits UB): situ computed once, single tl.max over the row.
    cols = tl.arange(0, HALF_COLS)
    for row_idx in range(row_begin, row_end):
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        gate = tl.load(x_ptr + row_off + cols).to(tl.float32)
        up = tl.load(x_ptr + row_off + HALF_COLS + cols).to(tl.float32)
        situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
        if DO_LINEAR_BETA:
            up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
        out = situ_a * up

        if SCALE:
            scale = tl.maximum(tl.max(tl.abs(out)) / DTYPE_MAX, 1e-30)
            tl.store(scale_ptr + row_idx.to(tl.int64), scale.to(scale_ptr.dtype.element_ty))
            # quantize in COL_BLOCK_SIZE slices (a full-row rint overflows UB, cf. swiglu_quant).
            for cb in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp = al.extract_slice(out, offsets=(cb,), sizes=(COL_BLOCK_SIZE,), strides=(1,))
                tmp = tmp.to(tl.float32) / scale
                tmp = tl.floor(tmp + 0.5)
                tmp = tl.clamp(tmp, -128, 127).to(tl.int8)
                c_idx = cb + tl.arange(0, COL_BLOCK_SIZE)
                mask = c_idx < HALF_COLS
                tl.store(out_ptr + row_idx.to(tl.int64) * HALF_COLS + c_idx,
                         tmp.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            tl.store(out_ptr + row_idx.to(tl.int64) * HALF_COLS + cols,
                     out.to(out_ptr.dtype.element_ty))


@triton.autotune(
    configs=[triton.Config({"BLOCK_H": b, "multibuffer": True}) for b in (1024, 2048, 4096, 8192)],
    key=["HALF_COLS", "HAS_GROUP_LIST"],
)
@triton.jit
def _situ_and_mul_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    N_ROWS,
    NUM_CORES: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # total_rows: from group_list (routed MoE) or N_ROWS (dense / shared, no group_list).
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:  # cusum
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
        else:  # count
            gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
            gl_mask = gl_offsets < NUM_EXPERTS
            group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
            total_rows = tl.sum(group_list)
    else:
        total_rows = N_ROWS

    # rows distributed across vector cores (manual split + early return for over-provision).
    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    # H-tile over the OUTPUT dim (HALF_COLS): out[i] only needs gate[i]=x[i] and
    # up[i]=x[d+i], so every [h:h+BLOCK_H] tile is self-contained -- no full-row resident,
    # which is what keeps large d (e.g. 33792) within UB. gate = first half, up = second half.
    h_offs = tl.arange(0, BLOCK_H)
    for row_idx in range(row_begin, row_end):
        # int64 row offset: row_idx * stride stays int32 by default on triton-ascend
        # (no auto-promote), which overflows when N*d is large (e.g. N=32768, d=33792).
        row_off = row_idx.to(tl.int64) * TOTAL_COLS
        gate_base = x_ptr + row_off
        up_base = x_ptr + row_off + HALF_COLS
        out_base = out_ptr + row_idx.to(tl.int64) * HALF_COLS
        for h_start in range(0, HALF_COLS, BLOCK_H):
            h_idx = h_start + h_offs
            mask = h_idx < HALF_COLS
            gate = tl.load(gate_base + h_idx, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(up_base + h_idx, mask=mask, other=0.0).to(tl.float32)
            situ_a = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
            if DO_LINEAR_BETA:
                up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
            out = situ_a * up
            tl.store(out_base + h_idx, out.to(out_ptr.dtype.element_ty), mask=mask)


def situ_and_mul_quant(
    x,
    group_list=None,
    group_list_type=None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
    need_quant: bool = True,
    quant_type: int = 0,
):
    """SituAndMul activation + fused dynamic int8 quant (d<=6144); unquant fallback (d=33792).

    Args:
        x: ``[..., 2d]`` tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
            ``None`` = dense / shared path (all rows). Required for routed MoE.
        group_list_type: 0 = cusum, 1 = count. Ignored when ``group_list is None``.
        beta / linear_beta: SituAndMul soft-saturation bounds (``linear_beta=None`` leaves up).
        need_quant: True -> int8 out + per-token fp32 scale; False -> activation out (scale is
            uninitialised, caller must ignore).
        quant_type: 0 = int8 (default), 1 = fp8 (deferred -> NotImplementedError).

    Returns:
        ``(out, scale)``. For d<=6144 + quant: ``out`` int8, ``scale`` fp32. For d>6144 (e.g.
        33792) or need_quant=False: ``out`` is the BF16/FP32 activation (no quant), ``scale``
        uninitialised -- quant only supports d in {3072, 6144}.
    """
    if quant_type not in (0, 1):
        raise ValueError(f"quant_type must be 0 (int8) or 1 (fp8), but got {quant_type}")
    if need_quant and quant_type == 1:
        raise NotImplementedError(
            "fp8 (quant_type=1) is deferred: A5-only, uses npu_dynamic_mx_quant (not fusible "
            "into Triton); MoE MXFP8 downstream still WIP in sglang. Use quant_type=0 (int8)."
        )

    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    x_2d = x.reshape(-1, x.shape[-1])
    s, h = x_2d.shape
    half_cols = h // 2
    # quant only for small d (3072/6144); large d (33792) -> unquant fallback.
    do_quant = need_quant and (half_cols <= 6144)
    out_dtype = torch.int8 if do_quant else x.dtype
    out = torch.empty((s, half_cols), dtype=out_dtype, device=x.device)
    scale = torch.empty((s,), dtype=torch.float32, device=x.device)

    if has_group_list:
        num_experts = group_list.shape[0]
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, but got {group_list.dtype}"
            )
        group_list_arg = group_list
        num_experts_arg = num_experts
        num_experts_algin_arg = num_experts_algin
        gl_type_arg = group_list_type
    else:
        group_list_arg = x_2d
        num_experts_arg = 1
        num_experts_algin_arg = 1
        gl_type_arg = 0

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    if do_quant:
        _situ_and_mul_quant_kernel[(num_vectorcore,)](
            x_2d, group_list_arg, out, scale,
            TOTAL_COLS=h, HALF_COLS=half_cols, COL_BLOCK_SIZE=half_cols,
            NUM_EXPERTS=num_experts_arg, NUM_EXPERTS_ALGIN=num_experts_algin_arg,
            GROUP_LIST_TYPE=gl_type_arg, N_ROWS=s, NUM_CORES=num_vectorcore,
            HAS_GROUP_LIST=has_group_list, BETA=beta, INV_BETA=1.0 / beta,
            DO_LINEAR_BETA=do_linear_beta, LINEAR_BETA=linear_beta_v,
            INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
            SCALE=need_quant, DTYPE_MAX=127, multibuffer=True,
        )
    else:
        raise NotImplementedError(
            "SituAndMul quantization is only implemented for d<=6144 (int8). "
        )
        # _situ_and_mul_kernel[(num_vectorcore,)](
        #     x_2d, group_list_arg, out,
        #     TOTAL_COLS=h, HALF_COLS=half_cols,
        #     NUM_EXPERTS=num_experts_arg, NUM_EXPERTS_ALGIN=num_experts_algin_arg,
        #     GROUP_LIST_TYPE=gl_type_arg, N_ROWS=s, NUM_CORES=num_vectorcore,
        #     HAS_GROUP_LIST=has_group_list, BETA=beta, INV_BETA=1.0 / beta,
        #     DO_LINEAR_BETA=do_linear_beta, LINEAR_BETA=linear_beta_v,
        #     INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
        # )
    return out.reshape(*x.shape[:-1], half_cols), scale


def situ_and_mul(
    x,
    group_list=None,
    group_list_type=None,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
):
    """SituAndMul activation with optional MoE group_list.

    Args:
        x: ``[..., 2d]`` tensor (gate | up halves along the last dim).
        group_list: per-expert token counts (count) or cumulative sum (cusum).
            ``None`` = dense / shared-expert path: process ALL rows of ``x`` (no dispatch
            padding). Required only for the routed-MoE path.
        group_list_type: 0 = cusum, 1 = count. Ignored when ``group_list is None``.
        beta: SituAndMul beta (soft-saturation bound on the gate path).
        linear_beta: optional soft-saturation bound on the up path; ``None`` leaves ``up``.

    Returns:
        ``[..., d]`` tensor. With ``group_list``: only the first ``sum(group_list)`` rows
        are written (rest is padding). Without: all rows written.
    """
    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"x last dim must be even, but got {x.shape[-1]}")

    x_2d = x.reshape(-1, x.shape[-1])
    s, h = x_2d.shape
    out = torch.empty((s, h // 2), dtype=x.dtype, device=x.device)

    if has_group_list:
        num_experts = group_list.shape[0]
        if group_list.dtype == torch.int64:
            num_experts_algin = (num_experts + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_algin = (num_experts + 15) // 16 * 16
        else:
            raise ValueError(
                f"group_list dtype must be torch.int32 or torch.int64, "
                f"but got {group_list.dtype}"
            )
        group_list_arg = group_list
        num_experts_arg = num_experts
        num_experts_algin_arg = num_experts_algin
        gl_type_arg = group_list_type
    else:
        # dense / shared: kernel skips the group_list block (HAS_GROUP_LIST=False),
        # so these are never read -- pass harmless dummies.
        group_list_arg = x_2d
        num_experts_arg = 1
        num_experts_algin_arg = 1
        gl_type_arg = 0

    do_linear_beta = linear_beta is not None
    linear_beta_v = linear_beta if do_linear_beta else 1.0

    _, num_vectorcore = get_device_properties()
    _situ_and_mul_kernel[(num_vectorcore,)](
        x_2d,
        group_list_arg,
        out,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        NUM_EXPERTS=num_experts_arg,
        NUM_EXPERTS_ALGIN=num_experts_algin_arg,
        GROUP_LIST_TYPE=gl_type_arg,
        N_ROWS=s,
        NUM_CORES=num_vectorcore,
        HAS_GROUP_LIST=has_group_list,
        BETA=beta,
        INV_BETA=1.0 / beta,
        DO_LINEAR_BETA=do_linear_beta,
        LINEAR_BETA=linear_beta_v,
        INV_LINEAR_BETA=(1.0 / linear_beta_v) if do_linear_beta else 1.0,
    )
    return out.reshape(*x.shape[:-1], h // 2)


@triton.jit(
    do_not_specialize=[
        "N",
        "B",
    ]
)
def _apply_attn_res_kernel(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    out_ptr,
    N,
    H: tl.constexpr,
    B,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    tok0 = pid * block_size
    if tok0 >= N:
        return
    tok1 = tl.minimum(tok0 + block_size, N)

    cols = tl.arange(0, H)                                   # full-row (non-pow2 OK on Ascend)
    s_idx = tl.arange(0, NB)                                 # padded stream-index block

    # Fused mul: score_weight = norm_w * proj_w (computed once, resident)
    norm_w = tl.load(norm_w_ptr + cols).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols).to(tl.float32)
    w = norm_w * proj_w

    br_stride = B * H  # block_residual row stride

    for tok in range(tok0, tok1):
        # ---- pass 1: per stream, one full-row load; MS + vw; score = rstd * vw ----
        scores = tl.full([NB], -float("inf"), dtype=tl.float32)
        for s in range(B + 1):
            if s < B:
                v = tl.load(block_residual_ptr + tok * br_stride + s * H + cols).to(tl.float32)
            else:
                v = tl.load(prefix_sum_ptr + tok * H + cols).to(tl.float32)
            ms = tl.sum(v * v) / H
            rstd = tl.rsqrt(ms + EPS)
            k = v * rstd  # normalize first (matches reference FP32 path exactly)
            scores = tl.where(s_idx == s, tl.sum(k * w), scores)

        # ---- softmax (manual: tl.max + tl.exp + tl.sum; tl.softmax unusable) ----
        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        weights = exp_scores / tl.sum(exp_scores)

        # ---- pass 2: weighted sum of raw streams (full-row reload) ----
        out = tl.zeros([H], dtype=tl.float32)
        for s in range(B + 1):
            if s < B:
                v = tl.load(block_residual_ptr + tok * br_stride + s * H + cols).to(tl.float32)
            else:
                v = tl.load(prefix_sum_ptr + tok * H + cols).to(tl.float32)
            w_s = tl.sum(tl.where(s_idx == s, weights, 0.0))
            out += w_s * v

        tl.store(out_ptr + tok * H + cols, out.to(out_ptr.dtype.element_ty))


def apply_attn_res(prefix_sum, block_residual, proj, norm):
    """K3 learned attn-residual: softmax-mix B+1 residual streams per token.

    Fused operations (no host-side temporaries):
      - score_weight = norm_w * proj_w  (mul inside kernel)
      - v = cat(block_residual, prefix_sum.unsqueeze(1))  (two-pointer read inside kernel)

    Args:
        prefix_sum: [N, H] BF16 (current running sum).
        block_residual: [N, B, H] BF16 (B past block snapshots).
        proj: nn.Linear(H, 1) — learned per-channel scoring projection.
        norm: KimiRMSNorm-like — has .weight [H] and .variance_epsilon (float).

    Returns:
        [N, H] BF16 — the softmax-weighted mix of the B+1 streams.
    """
    N, H = prefix_sum.shape
    B = block_residual.shape[1]
    proj_w = proj.weight.squeeze(0)
    norm_w = norm.weight
    eps = norm.variance_epsilon

    out = torch.empty((N, H), dtype=prefix_sum.dtype, device=prefix_sum.device)
    NB = triton.next_power_of_2(B + 1)

    _, num_vectorcore = get_device_properties()
    _apply_attn_res_kernel[(num_vectorcore,)](
        block_residual,
        prefix_sum,
        norm_w,
        proj_w,
        out,
        N=N,
        H=H,
        B=B,
        EPS=eps,
        NUM_CORES=num_vectorcore,
        NB=NB,
        multibuffer=True,
    )
    return out


class NPUACLFormat(IntEnum):
    ACL_FORMAT_UNDEFINED = -1
    ACL_FORMAT_ND = 2
    ACL_FORMAT_FRACTAL_NZ = 29


class FusedMoEMode(IntEnum):
    FUSED_DEEP_MOE = 1
    DISPATCH_FFN_COMBINE = 2


def _call_once(fn: Callable):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if getattr(fn, "_has_been_called", False):
            logger.debug("Function {} has already been called.", fn.__name__)
            return

        fn._has_been_called = True
        return fn(*args, **kwargs)

    return wrapper


def set_default_server_args(args: "ServerArgs"):
    """
    Set default server arguments for NPU backend.
    """

    # NPU only works with "ascend" attention backend for now
    args.attention_backend = "ascend"
    args.prefill_attention_backend = "ascend"
    args.decode_attention_backend = "ascend"
    if args.page_size is None:
        args.page_size = 128

    # NPU memory settings
    decode = args.cuda_graph_config.decode
    npu_mem = get_npu_memory_capacity()
    if npu_mem <= 32 * 1024:
        # Ascend 910B4,910B4_1
        # (chunked_prefill_size 4k, max_bs 16 if tp < 4 else 64)
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = 4 * 1024
        if decode.max_bs is None:
            if args.tp_size < 4:
                decode.max_bs = 16
            else:
                decode.max_bs = 64
    elif npu_mem <= 64 * 1024:
        # Ascend 910B1,910B2,910B2C,910B3,910_9391,910_9392,910_9381,910_9382,910_9372,910_9362
        # (chunked_prefill_size 8k, max_bs 64 if tp < 4 else 256)
        if args.chunked_prefill_size is None:
            args.chunked_prefill_size = 8 * 1024
        if decode.max_bs is None:
            if args.tp_size < 4:
                decode.max_bs = 64
            else:
                decode.max_bs = 256

    # NPU does not support CustomAllReduce
    args.disable_custom_all_reduce = True

    # handles hierarchical cache configs
    if args.enable_hierarchical_cache:
        args.hicache_io_backend = "kernel_ascend"
        if args.use_mla_backend():
            args.hicache_mem_layout = "page_first_kv_split"
        else:
            args.hicache_mem_layout = "page_first_direct"


@_call_once
def init_npu_backend():
    """
    Initialize NPU backend. This function should be called only once.
    """

    assert _is_npu, "NPU backend initialization called on non-NPU device."

    try:
        import custom_ops  # noqa: F401
        import sgl_kernel_npu  # noqa: F401
    except ImportError as e:
        logger.warning("NPU custom kernel packages unavailable: %s", e)

    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    # Re-mock torch.cuda.is_available cuz transfer_to_npu mocks it True
    torch.cuda.is_available = lambda: False

    torch_npu.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)


def _is_nz_aligned(tensor: torch.Tensor) -> bool:
    """Check whether the last two dims satisfy FRACTAL_NZ alignment rules.

    Ascend FRACTAL_NZ requires:
      BF16 / FP16 : both dims divisible by 16
      INT8         : k % 16 == 0  and  n % 32 == 0
      INT4         : k % 16 == 0  and  n % 64 == 0
      FP4          : both dims divisible by 64
    """
    if tensor.dim() < 2:
        return False
    k, n = tensor.shape[-2], tensor.shape[-1]
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return k % 16 == 0 and n % 16 == 0
    if tensor.dtype == torch.int8:
        return k % 16 == 0 and n % 32 == 0
    if tensor.dtype in (torch.uint8, torch.int32):
        # INT4 is typically packed into uint8/int32; be conservative
        return k % 16 == 0 and n % 64 == 0
    return True


def npu_format_cast(
    tensor: torch.Tensor,
    acl_format: NPUACLFormat = NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
) -> torch.Tensor:
    """
    Cast a tensor to a specific NPU ACL format.

    Args:
        tensor (torch.Tensor): The input tensor.
        acl_format (NPUACLFormat): The target NPU ACL format.

    Returns:
        torch.Tensor: The tensor cast to the specified NPU ACL format.
    """

    if not _is_npu:
        return tensor

    if envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get():
        return tensor

    if tensor.device == torch.device("cpu"):
        logger.warning_once(
            "Warning: The conversion from 'ND' to 'NZ' does not work on the CPU. "
            "Please disable offloading, otherwise the performance will be "
            "significantly reduced. --dit-cpu-offload false"
        )
        return tensor

    if acl_format == NPUACLFormat.ACL_FORMAT_FRACTAL_NZ and not _is_nz_aligned(tensor):
        k, n = tensor.shape[-2], tensor.shape[-1]
        logger.warning_once(
            "Skipping FRACTAL_NZ format cast: tensor shape (%d, %d) dtype %s "
            "is not aligned to NZ requirements. Falling back to 'ND' format, "
            "which may reduce NPU performance.",
            k,
            n,
            tensor.dtype,
        )
        return tensor

    # Skip format cast for meta tensors (used in offloader)
    if tensor.device.type == "meta":
        return tensor

    return torch.ops.npu.npu_format_cast(tensor, acl_format.value)


def get_indexer_weight_stream():
    global indexer_weight_stream
    if indexer_weight_stream is None:
        indexer_weight_stream = torch.npu.Stream()
    return indexer_weight_stream


def init_zbal(world_size, gpu_id, world_rank, do_check=True):
    """
    init zbal, if is mix alloc mode, only register for sma & comm
    """
    zbal_mem_size = envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get()
    if not zbal_mem_size > 0:
        return 1

    global gva_is_inited
    from zbal import is_mix_alloc, switch_to_allocator, zbal_init

    if is_mix_alloc():
        switch_to_allocator()
        # use lazy init for mix alloc
        return 1
    else:
        if envs.SGLANG_ZBAL_BOOTSTRAP_URL.get():
            ret = zbal_init(
                world_size,
                gpu_id,
                world_rank,
                zbal_mem_size * (1024**2),
                ip_port=envs.SGLANG_ZBAL_BOOTSTRAP_URL.get(),
            )
        else:
            ret = zbal_init(world_size, gpu_id, world_rank, zbal_mem_size * (1024**2))

        gva_is_inited = True

        if do_check and not ret:
            logger.error("[ZBAL] zbal init failed!")
            sys.exit(-1)

        return ret


def lazy_init_zbal_gva_mem(
    device, gpu_id, world_rank, world_size, cpu_group=None, do_check=True
):
    """
    lazy init zbal gva mem, keep weights and kv remains alloc by dma vmm to avoid memory fragment
    """
    from zbal import is_mix_alloc, zbal_init

    if not is_mix_alloc():
        logger.info(
            "lazy init is supported only in mix alloc mode, this action will be passed"
        )
        return 1

    global gva_is_inited
    from sglang.srt.utils.common import get_available_gpu_memory

    # TODO need to use allgather if you want use total_memory stats from mem_get_info as unbalance os
    total_memory = 61.2  # 2.5GB for other (workspace & os) outside torch
    free_gpu_memory = get_available_gpu_memory(
        device,
        gpu_id,
        distributed=world_size > 1,
        cpu_group=cpu_group,
        empty_cache=True,
    )

    used_memory = total_memory - free_gpu_memory

    used_memory_in_mb = int(used_memory * 1024)
    gva_in_mb = envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() - used_memory_in_mb
    gva_in_mb = gva_in_mb - gva_in_mb % 128  # align to 128MB
    print(f"[ZBAL] rank {world_rank} allocated {gva_in_mb} MB gva space.")

    assert not gva_is_inited, "zbal gva should be inited only once"
    # zbal_set_logger_level(0)
    if envs.SGLANG_ZBAL_BOOTSTRAP_URL.get():
        res = zbal_init(
            world_size,
            gpu_id,
            world_rank,
            gva_in_mb * (1024**2),
            ip_port=envs.SGLANG_ZBAL_BOOTSTRAP_URL.get(),
        )
    else:
        res = zbal_init(world_size, gpu_id, world_rank, gva_in_mb * (1024**2))

    gva_is_inited = True
    if do_check and not res:
        logger.error("[ZBAL] zbal lazy init failed!")
        sys.exit(-1)
    return res


share_stream = None
routed_stream = None


def get_share_stream():
    global share_stream
    return share_stream


def set_share_stream(stream):
    global share_stream
    share_stream = stream
    # TODO LKL: set stream limit has impact on precision
    # torch.npu.set_stream_limit(share_stream, 8, 16)


def get_routed_stream():
    global routed_stream
    return routed_stream


def set_routed_stream(stream):
    global routed_stream
    routed_stream = stream
    # TODO LKL: set stream limit has impact on precision
    # torch.npu.set_stream_limit(routed_stream, 16, 32)


def wait_share_stream():
    stream = get_share_stream()
    if stream is not None:
        cur_stream = torch.get_device_module().current_stream()
        cur_stream.wait_stream(stream)


def wait_routed_stream():
    stream = get_routed_stream()
    if stream is not None:
        cur_stream = torch.get_device_module().current_stream()
        cur_stream.wait_stream(stream)


def process_shared_expert(hidden_states, forward_func):
    stream = get_share_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_share_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states)
    return shared_output


def process_routed_expert(hidden_states, topk_output, forward_func):
    stream = get_routed_stream()
    if stream is None:
        stream = torch.get_device_module().Stream()
        set_routed_stream(stream)
    stream.wait_stream(torch.get_device_module().current_stream())
    with torch.get_device_module().stream(stream):
        shared_output = forward_func(hidden_states, topk_output)
    return shared_output
