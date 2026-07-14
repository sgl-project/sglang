from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from sglang.third_party.triton_kernels.numerics_details.mxfp import quantize_mxfp8_fn
from sglang.third_party.triton_kernels.numerics_details.flexpoint import float_to_flex, load_scale
from sglang.third_party.triton_kernels.numerics import InFlexData, OutFlexData, MAX_FINITE_FLOAT8E4B8, MAX_FINITE_FLOAT8E4NV, MAX_FINITE_FLOAT8E5
from typing import Optional
from .specialize import SpecializationModule, ClosureArg, FnSpecs


@dataclass(frozen=True)
class PostprocessFn:
    specs: FnSpecs = FnSpecs.default()
    fn_args: tuple[object] = tuple()


@triton.jit
def _reduce(X, stride_xr: tl.int64, stride_x0: tl.int64, stride_x1,  # x tensor (input)
            XMx, stride_xmxr, stride_xmx0, stride_xmx1,  # x mx scale
            Y, stride_y0: tl.int64, stride_y1,  # y tensor (output)
            YMx, stride_ymx0, stride_ymx1,  # y mx scale
            Mask, stride_mr, stride_m0, stride_m1,  # mask tensor
            Scale, stride_sr, stride_s0, stride_s1,  # scale tensor
            K, S0, X_S1, Y_S1,  # shape (K = reduction dim; S0, IN_S1 = input dims, OUT_S1 = output dims)
            POSTPROCESS_FN1: tl.constexpr, postprocess_fn1_args,  #
            POSTPROCESS_FN2: tl.constexpr, postprocess_fn2_args,  #
            XFlex,  # x flex (global) scale
            YFlexExpected, YFlexActual, YFlexChecksum, Y_FLEX_SATURATE_INF: tl.constexpr,  # y flex (global) scale
            IS_MASK_NONE: tl.constexpr,  #
            BROADCAST_R: tl.constexpr,  #
            BROADCAST_S0: tl.constexpr,  #
            BROADCAST_S1: tl.constexpr,  #
            IS_SCALE_NONE: tl.constexpr,  #
            SCALE_BROADCAST_R: tl.constexpr,  #
            SCALE_BROADCAST_S0: tl.constexpr,  #
            SCALE_BROADCAST_S1: tl.constexpr,  #
            BLOCK_S0: tl.constexpr,  #
            BLOCK_X_S1: tl.constexpr,  #
            BLOCK_Y_S1: tl.constexpr,  #
            ):
    pid_s0 = tl.program_id(0)
    pid_s1 = tl.program_id(1)
    tl.static_assert(BLOCK_X_S1 % 32 == 0)
    BLOCK_X_SMX1: tl.constexpr = BLOCK_X_S1 // 32
    BLOCK_Y_SMX1: tl.constexpr = BLOCK_Y_S1 // 32
    offs_s0 = pid_s0 * BLOCK_S0 + tl.arange(0, BLOCK_S0)
    offs_x_s1 = pid_s1 * BLOCK_X_S1 + tl.arange(0, BLOCK_X_S1)
    offs_x_smx1 = pid_s1 * BLOCK_X_SMX1 + tl.arange(0, BLOCK_X_SMX1)
    valid_s0 = offs_s0 < S0
    valid_x_s1 = offs_x_s1 < X_S1
    valid_in_smx1 = offs_x_smx1 < tl.cdiv(X_S1, 32)
    y = tl.zeros((BLOCK_S0, BLOCK_X_S1), dtype=tl.float32)
    x_flex_scale = load_scale(XFlex)
    for k in tl.range(0, K, num_stages=2):
        x_ptrs = X + k * stride_xr + offs_s0[:, None] * stride_x0 + offs_x_s1[None, :] * stride_x1
        x = tl.load(x_ptrs, mask=valid_s0[:, None] & valid_x_s1[None, :], other=0.0)
        x = x.to(tl.float32)
        if XMx is not None:
            xmx_ptrs = XMx + k * stride_xmxr + offs_s0[:, None] * stride_xmx0 + offs_x_smx1[None, :] * stride_xmx1
            xmx = tl.load(xmx_ptrs, mask=valid_s0[:, None] & valid_in_smx1[None, :], other=0.0)
            xmx = (xmx.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            x = (xmx[:, :, None] * x.reshape([BLOCK_S0, BLOCK_X_S1 // 32, 32])).reshape([BLOCK_S0, BLOCK_X_S1])
        x = x * x_flex_scale
        if not IS_SCALE_NONE:
            k_term_s = 0 if SCALE_BROADCAST_R else (k * stride_sr)
            s0_term_s = 0 if SCALE_BROADCAST_S0 else (offs_s0[:, None] * stride_s0)
            s1_term_s = 0 if SCALE_BROADCAST_S1 else (offs_x_s1[None, :] * stride_s1)
            s_ptrs = Scale + k_term_s + s0_term_s + s1_term_s
            s = tl.load(s_ptrs, mask=valid_s0[:, None] & valid_x_s1[None, :], other=1)
            x = x * s
        if not IS_MASK_NONE:
            k_term = 0 if BROADCAST_R else (k * stride_mr)
            s0_term = 0 if BROADCAST_S0 else (offs_s0[:, None] * stride_m0)
            s1_term = 0 if BROADCAST_S1 else (offs_x_s1[None, :] * stride_m1)
            m_ptrs = Mask + k_term + s0_term + s1_term
            m = tl.load(m_ptrs, mask=valid_s0[:, None] & valid_x_s1[None, :], other=1)
            x = tl.where(m != 0, x, 0.0)
        y += x
    if POSTPROCESS_FN1 is not None:
        y = POSTPROCESS_FN1(y, *postprocess_fn1_args)
    offs_y_s1 = pid_s1 * BLOCK_Y_S1 + tl.arange(0, BLOCK_Y_S1)
    offs_y_smx1 = pid_s1 * BLOCK_Y_SMX1 + tl.arange(0, BLOCK_Y_SMX1)
    valid_y_s1 = offs_y_s1 < Y_S1
    valid_y_smx1 = offs_y_smx1 < tl.cdiv(Y_S1, 32)
    y = float_to_flex(y, YFlexExpected, YFlexActual, YFlexChecksum, None, Y, Y_FLEX_SATURATE_INF)
    # TODO (phil): keeping for backward compatibility, but will remove !
    if YMx is None and POSTPROCESS_FN2 is not None:
        y = POSTPROCESS_FN2(y, *postprocess_fn2_args, target_dtype=Y.dtype.element_ty)
    y_ptrs = Y + offs_s0[:, None] * stride_y0 + offs_y_s1[None, :] * stride_y1
    if YMx is not None:
        y, y_scale = quantize_mxfp8_fn(y, valid_y_s1[None, :])
        y_mx_ptrs = YMx + offs_s0[:, None] * stride_ymx0 + offs_y_smx1[None, :] * stride_ymx1
        tl.store(y_mx_ptrs, y_scale, mask=valid_s0[:, None] & valid_y_smx1[None, :])
    tl.store(y_ptrs, y, mask=valid_s0[:, None] & valid_y_s1[None, :])


specializations = SpecializationModule(
    "reduce",
    kernels=[("_reduce", _reduce)],
    closure_args={
        "postprocess_fn1": ClosureArg("POSTPROCESS_FN1", "postprocess_fn1_args"),
        "postprocess_fn2": ClosureArg("POSTPROCESS_FN2", "postprocess_fn2_args"),
    },
)


def reduce(
    x: torch.Tensor,
    dim: int,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    x_mxscale: Optional[torch.Tensor] = None,
    x_flex: Optional[InFlexData] = InFlexData(),
    y_dtype: Optional[torch.dtype] = None,
    y_flex: Optional[OutFlexData] = OutFlexData(),
    y_flex_saturate_inf: bool = False,
    y_has_mx: Optional[bool] = None,
    y: Optional[torch.Tensor] = None,
    postprocess_fn1: Optional[PostprocessFn] = None,
    # TODO: keeping for backward compatibility, but will remove !
    postprocess_fn2: Optional[PostprocessFn] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Performs a reduction over the specified dimension of the input tensor,
    optionally multiplied by `scale` and ignoring masked elements.

    Arguments:
        - x: Tensor
          input tensor to reduce.
        - dim: int
          dimension along which `x` should be reduce.
        - mask: Optional[torch.Tensor]
          integer mask of the same shape as `x` (or broadcastable to it).
          entries that are `0` are ignored in the reduction.
          if `mask is None`, all elements are included.
        - scale: Optional[torch.Tensor]
          scale factors of the same shape as `x` (or broadcastable to it).
          the reduction is performed over `x * scale`. If `scale is None`,
          a value of 1 is used everywhere.

    Returns:
        - output: torch.Tensor
          The reduced tensor with `dim` removed.
        - output_mxscale: Optional[torch.Tensor]
          The output mx scale if input is micro-scaled, else None.
    """
    if x.ndim != 3:
        raise NotImplementedError("reduce only supports 3D inputs in this implementation")
    if dim < 0:
        dim += x.ndim
    if dim not in (0, 1, 2):
        raise ValueError("dim must be in {0,1,2}")
    if x_mxscale is not None:
        if dim == 2:
            raise ValueError("reduction over the micro-scaled dimension not supported")
        assert x.shape[:-2] == x_mxscale.shape[:-2]
        assert triton.cdiv(x.shape[-1], 32) * 32 == x_mxscale.shape[-1] * 32
        assert dim != -1
    # assert not y_flex.is_per_batch
    if postprocess_fn1 is None:
        postprocess_fn1 = PostprocessFn()
    if postprocess_fn2 is None:
        postprocess_fn2 = PostprocessFn()
    if y_dtype is None:
        y_dtype = x.dtype
    if y_flex is None:
        y_flex = OutFlexData()
    if x_flex is None:
        x_flex = InFlexData()
    if y_has_mx is None:
        y_has_mx = x_mxscale is not None
    # input shapes
    dims = (0, 1, 2)
    nonred = tuple(d for d in dims if d != dim)
    S0, X_S1 = x.shape[nonred[0]], x.shape[nonred[1]]
    Y_S1 = X_S1 // postprocess_fn1.specs.reduction_n
    if y is None:
        y = torch.empty((S0, Y_S1), device=x.device, dtype=y_dtype)
    assert y.shape == (S0, Y_S1), f"y.shape: {y.shape} != ({S0}, {Y_S1})"
    y_mxscale = None
    if y_has_mx:
        y_mxscale = torch.empty((S0, triton.cdiv(Y_S1, 32)), device=x.device, dtype=torch.uint8)
    # Strides for X along reduced and non-reduced dims
    stride_xr = x.stride(dim)
    stride_x0 = x.stride(nonred[0])
    stride_x1 = x.stride(nonred[1])
    # Strides for X mx scales
    stride_xmxr = None if x_mxscale is None else x_mxscale.stride(dim)
    stride_xmx0 = None if x_mxscale is None else x_mxscale.stride(nonred[0])
    stride_xmx1 = None if x_mxscale is None else x_mxscale.stride(nonred[1])
    # Strides for Y mx scales
    stride_ymx0 = None if y_mxscale is None else y_mxscale.stride(0)
    stride_ymx1 = None if y_mxscale is None else y_mxscale.stride(1)
    # Mask strides (broadcast allowed via stride 0)
    if mask is not None:
        mstr0, mstr1, mstr2 = mask.stride()
        stride_mr = (mstr0 if dim == 0 else (mstr1 if dim == 1 else mstr2))
        stride_m0 = (mstr0 if nonred[0] == 0 else (mstr1 if nonred[0] == 1 else mstr2))
        stride_m1 = (mstr0 if nonred[1] == 0 else (mstr1 if nonred[1] == 1 else mstr2))
    else:
        stride_mr = stride_m0 = stride_m1 = 0
    # Scale strides (broadcast allowed via stride 0)
    if scale is not None:
        sstr0, sstr1, sstr2 = scale.stride()
        stride_sr = (sstr0 if dim == 0 else (sstr1 if dim == 1 else sstr2))
        stride_s0 = (sstr0 if nonred[0] == 0 else (sstr1 if nonred[0] == 1 else sstr2))
        stride_s1 = (sstr0 if nonred[1] == 0 else (sstr1 if nonred[1] == 1 else sstr2))
    else:
        stride_sr = stride_s0 = stride_s1 = 0
    K = x.shape[dim]
    # Always use the 2D tiled kernel with constexpr metaprogramming for mask broadcasting
    BLOCK_S0 = 64
    BLOCK_X_S1 = 128
    BLOCK_Y_S1 = 128 // postprocess_fn1.specs.reduction_n
    grid = (triton.cdiv(S0, BLOCK_S0), triton.cdiv(Y_S1, BLOCK_Y_S1))
    mask_arg = mask if mask is not None else None
    scale_arg = scale if scale is not None else None
    reduce_kernel = specializations.get(postprocess_fn1=postprocess_fn1.specs,
                                        postprocess_fn2=postprocess_fn2.specs)._reduce
    reduce_kernel[grid](
        x_flex.reinterpret(x), stride_xr, stride_x0, stride_x1,  #
        x_mxscale, stride_xmxr, stride_xmx0, stride_xmx1,  #
        y_flex.reinterpret(y), y.stride(0), y.stride(1),  #
        y_mxscale, stride_ymx0, stride_ymx1,  #
        mask_arg, stride_mr, stride_m0, stride_m1,  #
        scale_arg, stride_sr, stride_s0, stride_s1,  #
        K, S0, X_S1, Y_S1,  #
        *postprocess_fn1.fn_args, *postprocess_fn2.fn_args,  #
        x_flex.scale, y_flex.expected_scale, y_flex.actual_scale, y_flex.checksum_scale,  #
        y_flex_saturate_inf,  #
        IS_MASK_NONE=(mask is None),  #
        BROADCAST_R=(stride_mr == 0),  #
        BROADCAST_S0=(stride_m0 == 0),  #
        BROADCAST_S1=(stride_m1 == 0),  #
        IS_SCALE_NONE=(scale is None),  #
        SCALE_BROADCAST_R=(stride_sr == 0),  #
        SCALE_BROADCAST_S0=(stride_s0 == 0),  #
        SCALE_BROADCAST_S1=(stride_s1 == 0),  #
        BLOCK_S0=BLOCK_S0,  #
        BLOCK_X_S1=BLOCK_X_S1,  #
        BLOCK_Y_S1=BLOCK_Y_S1,  #
        num_warps=4  #
    )
    return y, y_mxscale


def compute_actual_scale(x, dtype, per_batch_scale=False):
    max_finite = {
        torch.float8_e5m2: MAX_FINITE_FLOAT8E5,
        torch.float8_e4m3fn: MAX_FINITE_FLOAT8E4NV,
        torch.float8_e4m3fnuz: MAX_FINITE_FLOAT8E4B8,
    }[dtype]
    maxvals = x.abs().amax(dim=tuple(range(1, x.ndim))) if per_batch_scale else x.abs().max()
    return maxvals / max_finite


def reduce_torch(x: torch.Tensor, dim: int, mask: Optional[torch.Tensor] = None,  #
                 scale: Optional[torch.Tensor] = None,  #
                 x_mxscale: Optional[torch.Tensor] = None,  #
                 x_flex: Optional[InFlexData] = InFlexData(), y_flex: Optional[OutFlexData] = OutFlexData(),
                 y_flex_saturate_inf: bool = False, postprocess_fn1: Optional[callable] = None):
    from sglang.third_party.triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch, upcast_from_mxfp_torch
    x_dtype = x.dtype
    # upcast input
    if x_mxscale is not None:
        x = upcast_from_mxfp_torch(x, x_mxscale, torch.float32, axis=-1)
    x = x.to(torch.float32)
    if x_flex is not None:
        x *= x_flex.scale
    # upcast scale
    if scale is None:
        scale = torch.ones(1, dtype=torch.float32, device=x.device)
    scale = scale.to(torch.float32)
    # initialize mask
    if mask is None:
        mask = torch.ones(1, dtype=torch.bool, device=x.device)
    mask = mask.to(torch.bool)
    ret = torch.where(mask, x * scale, 0).sum(dim=dim)
    if postprocess_fn1 is not None:
        ret = postprocess_fn1(ret)
    if y_flex is not None:
        y_flex.actual_scale.copy_(compute_actual_scale(ret, x_dtype, y_flex.is_per_batch))
        ret = (ret / y_flex.expected_scale).to(x_dtype)
    # downcast output
    ret_mxscale = None
    if x_mxscale is not None:
        assert y_flex is None
        ret, ret_mxscale = downcast_to_mxfp_torch(ret, torch.float8_e4m3fn, axis=-1)
    return ret.to(x_dtype), ret_mxscale
