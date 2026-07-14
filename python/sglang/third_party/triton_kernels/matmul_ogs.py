# isort: off
# fmt: off
from dataclasses import dataclass, field
import itertools
import torch
import triton
from enum import Enum, auto
import math
# utilities
from sglang.third_party.triton_kernels import target_info
from sglang.third_party.triton_kernels.numerics import InFlexData, OutFlexData
from sglang.third_party.triton_kernels.target_info import is_cuda
# details
from .matmul_ogs_details._matmul_ogs import _matmul_ogs
from .matmul_ogs_details._p_matmul_ogs import _p_matmul_ogs, get_per_device_per_stream_alloc_fn
from .numerics_details.mxfp import MXFP_BLOCK_SIZE
from .tensor_details.layout_details.strided import StridedLayout
from .matmul_ogs_details.opt_flags import make_opt_flags, update_opt_flags_constraints
from .specialize import FnSpecs, SpecializationModule, ClosureArg
from .tensor import Storage, Tensor, FP4, bitwidth, wrap_torch_tensor, RaggedTensorMetadata
from .reduce import reduce
from .reduce import PostprocessFn as ReducePostprocessFn


@dataclass
class GatherIndx:
    """
    Indices for an operation that performs:
    Y = X[src_idx, :]
    """
    # array such that `dst_idx[src_idx] = arange(0, N)`
    src_indx: torch.Tensor
    dst_indx: torch.Tensor


@dataclass
class ScatterIndx:
    """
    Indices for an operation that performs:
    Y[dst_idx, :] = X
    """
    # array such that `dst_idx[src_idx] = arange(0, N)`
    src_indx: torch.Tensor
    dst_indx: torch.Tensor

@dataclass
class RoutingData:
    gate_scal: torch.Tensor = field()
    expt_hist: torch.Tensor = field()
    n_expts_tot: int = field()
    n_expts_act: int = field()
    expt_data: RaggedTensorMetadata = None

    # Used to make perf annotation cleaner: when we use expert sharding, we can
    # use this to tell the "expected" number of local tokens per expert, because
    # the actual number can vary per each input.
    expected_tokens_per_expt: int = field(default=None)

    def n_blocks(self, n_rows, block_m):
        if n_rows <= self.n_expts_tot:
            return n_rows
        else:
            return triton.cdiv(max(n_rows - self.n_expts_tot + 1, 0), block_m) + self.n_expts_tot - 1

@dataclass(frozen=True)
class FusedActivation:
    specs: FnSpecs = FnSpecs.default()
    fn_args: tuple[object] = tuple()


@dataclass(frozen=True)
class Epilogue:
    specs: FnSpecs = FnSpecs.default()
    fn_arg_values_matmul: tuple[object] = tuple()
    fn_arg_values_finalize: tuple[object] = tuple()
    effective_itemsize: float = None

class FnName(Enum):
    QUANTIZE_MXFP8 = auto()


@dataclass(frozen=True)
class FusedComm:
    out_handles: torch.Tensor
    scatter_shard_indx: torch.Tensor | None = None
    reduce_rank: int = 0
    n_reduce_shards: int = 1

specializations = SpecializationModule("matmul_ogs",
    kernels=[("_matmul_ogs", _matmul_ogs), ("_p_matmul_ogs", _p_matmul_ogs)],
    closure_args={
        "epilogue": ClosureArg("EPILOGUE_FN", "epilogue_fn_args"), #
        "activation": ClosureArg("ACTIVATION_FN", "activation_fn_args"), #
    },
)
# -----------------------------------------------------------------------------
#                    Matrix Multiplication + Outer Gather/Scatter
# -----------------------------------------------------------------------------


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


# This supports computing dw for ragged matmul.  Note the correspondence:
#   fwd pass:      y = matmul_ogs(x, w, ...)
#   bwd pass (dw): dw = matmul_ogs(x.T, dy, ...)
#
# Thus, "our" x, w, and y (as seen by matmul_ogs) correspond to x.T, dy, dw, respectively.
# To avoid confusion, now we'll stick to "x, w, y" terminology.
#
# Assume that y.shape == (N_EXPTS, M, N), x.shape == (M, K), w.shape = (K_W, N).
#
# To make things feasible, we require that x and w satisfy the following condition:
#   (1) We don't support gather/scatter indices: in x, all columns for expt #0 are grouped at
#       the leftmost part, followed by expt #1, and so on.  Ditto for w (top to bottom).
#   (2) At least one of x and w are padded: each expert uses a multiple of block_k columns
#       (or rows), and unused values are filled with zero.
#   (3) No inf or nan are allowed in x or w (except for the final padding - see below).
#       This is because we use "multiplying by padded zero region" in lieu of masking.
#   (4) The number of actually used columns/rows equals self.base.expt_hist.sum() and may be
#       less than K or K_W.  In this case, the final "unused" values can be left uninitialized.
#       However, if x or w is unpadded, the first block_k columns/rows of the unused part must
#       not contain nan or inf.
#
# For example, assume N_EXPTS == 5, block_k == 32, and expt_hist == [60, 33, 0, 32, 25].
#
#               if unpadded     if padded
#               -----------     ---------
#   x: expt #0: x[:, :60]       x[:, :60]
#                               x[:, 60:64] - zero padded
#      expt #1: x[:, 60:93]     x[:, 64:97]
#                               x[:, 97:128] - zero padded
#      expt #3: x[:, 93:125]    x[:, 128:160]
#      expt #4: x[:, 125:150]   x[:, 160:185]
#                               x[:, 185:192] - zero padded
#               x[:, 150:min(182, K)] - must not contain inf/nan
#
#               x[:, 182:]      x[:, 192:] - unused (may contain garbage, including inf/nan)
#
#   w is the same, except that rows columns are flipped.
@dataclass
class InnerRoutingData:
    base: RoutingData | None = None
    block_k: int | None = None
    x_is_padded: bool = False
    w_is_padded: bool = False

    # Return value contains: ExptHist, ExptOffs, ExptTileOffs, ExptData,
    #                        EXPT_IS_INNER, X_IS_PADDED, W_IS_PADDED, ExptHistMax
    @staticmethod
    def make_kernel_args(data, block_m):
        if isinstance(data, RoutingData):
            expt_data, block = data.expt_data, block_m
            args = (False, False, False, None)
        elif isinstance(data, InnerRoutingData):
            expt_data, block = data.base.expt_data, data.block_k
            args = (
                True, data.x_is_padded, data.w_is_padded, expt_data.slice_sizes.max()
            )
        elif data is None:
            expt_data = None
        else:
            assert None

        if expt_data is None:
            return (None, None, None, None, False, False, False, None)

        return (
            expt_data.slice_sizes,
            expt_data.slice_offs,
            expt_data.block_offs(block),
            expt_data.block_schedule(block),
        ) + args


# ---------------------
# Numerics
# ---------------------

# fmt: off

@dataclass(frozen=True)
class FlexCtx:
    lhs_data: InFlexData = InFlexData()
    rhs_data: InFlexData = InFlexData()
    out_data: OutFlexData = OutFlexData()
    acc_data: InFlexData = InFlexData()

@dataclass
class PrecisionConfig:
    max_num_imprecise_acc: int = None
    allow_tf32: bool = True
    flex_ctx: FlexCtx = FlexCtx()
    acc_scale: int = 1.0
    flexpoint_saturate_inf: bool = False
    report_quantization_err_fn: callable = None
    act_scale: Tensor | None = None
    weight_scale: Tensor| None = None
    out_scale: Tensor | None = None
    out_dtype: torch.dtype = None
    enforce_bitwise_invariance: bool = False


# TODO: merge in opt_flags
def get_swap_xw(precision_config, opt_flags):
    if target_info.cuda_capability_geq(10, 0):
        return precision_config.weight_scale is not None and opt_flags.block_m <= 64 and opt_flags.is_persistent
    return False

# ---------------------
# Allocation
# ---------------------

@dataclass
class MatmulAllocation:
    device: str
    output: tuple[tuple[int], torch.dtype]
    scratchpads: dict[str, tuple]

def init_allocation(x, w, precision_config, fused_activation,
                    routing_data, gather_indx, scatter_indx, inner_routing_data,
                    n_reduce_shards, opt_flags):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.src_indx.shape[0]
    if scatter_indx is not None:
        M = scatter_indx.src_indx.shape[0]
    if scatter_indx is None:
        y_rows = M
    else:
        y_rows = M // routing_data.n_expts_act
    y_rows *= n_reduce_shards
    if inner_routing_data is not None:
        batch_dim = inner_routing_data.base.n_expts_tot
    else:
        batch_dim = x.shape[0] if x.ndim == 3 else 1
    out_shape = (batch_dim, y_rows, N // fused_activation.specs.reduction_n)
    out_dtype = precision_config.out_dtype or x.dtype
    output = (out_shape, out_dtype)
    # ---- scratchpad -----#
    scratchpad = dict()
    N_scratch = N // fused_activation.specs.reduction_n if opt_flags.split_k == 1 else N
    if opt_flags.split_k > 1 or (scatter_indx is not None and (not is_cuda() or routing_data.n_expts_act > 1)):
        scratch_out_dtype = torch.float32 if opt_flags.split_k > 1 else out_dtype
        scratchpad["matmul"] = ((opt_flags.split_k, batch_dim, M, N_scratch), scratch_out_dtype)
    if "matmul" in scratchpad and precision_config.out_scale is not None:
        assert batch_dim == 1, "batch_dim > 1 not supported yet"
        scratchpad["mx_out_scale"] = ((opt_flags.split_k, 1, M, triton.cdiv(N_scratch, MXFP_BLOCK_SIZE)), torch.uint8)
    return MatmulAllocation(x.device, output, scratchpad)

def apply_allocation(allocation: MatmulAllocation, output):
    ret = dict()
    if output is None:
        output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
    else:
        if output.ndim == 2:
            output = output[None, :, :]
        assert output.shape == allocation.output[0]
    ret["output"] = output[None, :, :]
    ret["scratchpad"] = {
        k: torch.empty(v[0], device=allocation.device, dtype=v[1])
            for k, v in allocation.scratchpads.items()
    }
    return ret

# -----------------------------------------------------------------------------
# Canonicalize
# -----------------------------------------------------------------------------
# the `matmul_ogs` kernel can operate on 2D or 3D inputs depending on the mode being used
# we can canonicalize storages to make the implementation more uniform

def _canonicalize_storage(storage, out_ndim, flex_data):
    assert out_ndim >= storage.data.ndim
    # Need to use as_strided instead of view because for a tensor with
    # shape[-2] == 1 can have ambuiguity related to col-wise. Fo example,
    # > t = torch.randn(2, 5, 1).mT
    # > t_view = t.view(t.shape)
    # > t.stride(), t_view.stride()
    # ((5, 1, 1), (5, 5, 1))
    # Our check t_view is col-wise fails since t_view.stride(-2) != 1
    # This case is covered by (m, n, k) == (1000, 700, 2) in test_matmul.py
    new_storage_shape = [1] * (out_ndim - storage.data.ndim) + list(storage.data.shape)
    new_storage_stride = [0] * (out_ndim - storage.data.ndim) + list(storage.data.stride())
    new_storage_data = storage.data.as_strided(new_storage_shape, new_storage_stride)
    if flex_data is not None:
        new_storage_data = flex_data.reinterpret(new_storage_data)
    return Storage(new_storage_data, storage.layout)


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------

def matmul_ogs_set_idle_sms(num_idle_sms):
    """
    persistent kernels will leave `num_idle_sms` idle
    """
    update_opt_flags_constraints({"idle_sms": num_idle_sms})

def matmul_ogs(x, w, bias,
    routing_data: RoutingData | None = None,
    gather_indx: GatherIndx | None = None,
    scatter_indx: ScatterIndx | None = None,
    precision_config: PrecisionConfig | None = None,
    betas: torch.Tensor | None = None,
    gammas: torch.Tensor | None = None,
    out_alpha: float | None = None,
    y: torch.Tensor | None = None,
    fused_comm: FusedComm | None = None,
    fused_activation: FusedActivation | None = None,
    epilogue: Epilogue | None = None,
    y_acc_in: torch.Tensor | None = None,
    inner_routing_data: InnerRoutingData | None = None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])

    matmul can be optionally fused with all gather or scatter at the end for the output. When fused_comm is specified, the m-th row of the output will be stored to (m * n_reduce_shards + reduce_rank) -th row
    of each rank id in range [scatter_shard_indx[m] * n_reduce_shards, (scatter_shard_indx[m] + 1) * n_reduce_shards) if scatter_shard_indx is not None, otherwise the output will be all gathered across all reduce ranks.
    When scatter_shard_indx is specified, the caller should ensure that the indices of different shards do not conflict.

    The output buffer for fused comm should be pre-allocated and passed in via fused_comm.out_handles, which contains ipc handles to the output tensors, each with shape (n_rows * n_reduce_shards, n_cols).
    """
    is_input_batched = x.ndim == 3
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert inner_routing_data is None, "routing not supported in batched mode"
        assert fused_comm is None, "fused comm is not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    if inner_routing_data is not None:
        assert routing_data is None
        assert gather_indx is None
        assert scatter_indx is None
        routing_data = RoutingData(
            None, None, inner_routing_data.base.n_expts_tot, 1,
            expected_tokens_per_expt=inner_routing_data.base.expected_tokens_per_expt,
        )
    # canonicalize inputs
    if precision_config is None:
        precision_config = PrecisionConfig()
    if fused_activation is None:
        fused_activation = FusedActivation(FnSpecs.default(), tuple())
    if epilogue is None:
        epilogue = Epilogue(FnSpecs.default(), tuple(), tuple(), False)
    if routing_data is None:
        routing_data = RoutingData(None, None, max(1, w.shape[0]), 1)
    # unpack scales
    w_scale = precision_config.weight_scale
    w_has_mx = w_scale is not None
    is_hopper_fp8 = is_cuda() and not target_info.cuda_capability_geq(10, 0) and bitwidth(w.dtype) == 8
    if is_hopper_fp8: assert w.stride(-2) == 1, "`w` must be column-major when it has data-type FP8 on capability < 10"
    if not isinstance(w, Tensor):
        # TODO: remove this code path; using uint8 for mxfp4 weight will bite us when we want to support uint8 for real
        dtype = FP4 if w.dtype == torch.uint8 else w.dtype
        w = wrap_torch_tensor(w, dtype=dtype)
    if w_has_mx and (torch.cuda.get_device_capability()[0] < 10 or w.storage.layout is not None and not isinstance(w.storage.layout, StridedLayout)):
        assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp and (swizzled or not on >=Blackwell)"
    if w_scale is not None and not isinstance(w_scale, Tensor):
        w_scale = Tensor(w_scale)
    if w_scale is not None:
        w_scale.storage.data = w_scale.data.view(torch.uint8)
        w_scale.dtype = torch.uint8
    x_scale = precision_config.act_scale
    x_has_mx = x_scale is not None
    if x_has_mx: assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if x_scale is not None and not isinstance(x_scale, Tensor):
        x_scale = Tensor(x_scale)
    if not isinstance(x, Tensor):
        x = Tensor(x, dtype=x.dtype)
    x_transpose = x.stride(-1) != 1
    # determine shapes
    has_gather = gather_indx is not None
    has_scatter = scatter_indx is not None
    is_ragged = routing_data.expt_hist is not None
    M = x.shape[-2] if gather_indx is None else gather_indx.src_indx.shape[0]
    if inner_routing_data is not None:
        batch_size = inner_routing_data.base.n_expts_tot
    else:
        batch_size = w.shape[0] if routing_data.expt_hist is None and w.ndim == 3 else 1
    if y_acc_in is not None:
        y_acc_is_y = y_acc_in.data_ptr() == y.data_ptr() and y_acc_in.stride() == y.stride()
    else:
        y_acc_is_y = None
    K = x.shape[-1]
    K_W, N = w.shape[-2:]
    if x.ndim == 3 and w.ndim == 3:
        assert x.shape[0] == w.shape[0]
    # compute optimization flags
    out_dtype = precision_config.out_dtype or x.dtype
    can_use_tma = (
        x.numel() > 0 and x.storage.is_tma_compliant() and
        w.numel() > 0 and w.storage.is_tma_compliant() and
        (w_scale is None or w_scale.storage.is_tma_compliant()) and
        (not is_ragged or x.stride(-1) == 1) and
        # Currently we don't support tma if y is column major; may revisit later if this becomes an issue.
        (y is None or y.stride(-1) == 1) and
        (y_acc_in is None or y_acc_is_y) and
        # If we use inner_routing_data, w must be either padded or row major, otherwise we get
        # unaligned access.
        (inner_routing_data is None or w.stride(-1) == 1 or inner_routing_data.w_is_padded)
    )
    if w_scale is not None and isinstance(w_scale.storage.layout, StridedLayout) and w_scale.storage.data.stride()[-1] != 1:
        # In this case, we need to transpose w_scale. Then the reduction dim
        # becomes the last dim that will be divided by 32. This to be a multiple
        # of 16 to be TMA-compliant requires block_k to be a multiple of 512,
        # which is too big.
        can_use_tma = False
    has_gather_tma = has_gather and target_info.has_tma_gather()
    # hopper w/ mxfp4 doesn't support TMA
    can_use_tma = can_use_tma and (torch.cuda.get_device_capability()[0] > 9 or bitwidth(w.dtype) != 4)
    can_use_split_k = scatter_indx is None and not x_has_mx and not w_has_mx
    opt_flags = make_opt_flags(out_dtype, x.dtype, w.dtype, precision_config,
        batch_size, M, N, w.shape[-2], routing_data,
        can_use_tma, can_use_split_k, epilogue.effective_itemsize,
        x_transpose, y_acc_in is not None,
        inner_routing_data.block_k if inner_routing_data is not None else None,
    )
    if inner_routing_data is not None:
        assert opt_flags.block_k == inner_routing_data.block_k
        assert opt_flags.split_k == 1
        batch_size = inner_routing_data.base.n_expts_tot
        # For unpadded (row major) x, we cannot use tma because memory access isn't aligned.
        x_has_tma = opt_flags.is_persistent and (x.stride(-1) != 1 or inner_routing_data.x_is_padded)
        # If TMA is used, limit is handled automatically, so we can pretend K is "even".
        # (For unpadded input, we assume that the first block_k unused rows are zero-filled,
        # when routing_data.expt_hist.sum() is less than K or K_W.)
        if opt_flags.is_persistent:
            even_K = x_has_tma or inner_routing_data.x_is_padded
        else:
            even_K = inner_routing_data.x_is_padded and inner_routing_data.w_is_padded
    else:
        batch_size = w.shape[0] if routing_data.expt_hist is None and w.ndim == 3 else 1
        assert K == K_W
        x_has_tma = opt_flags.is_persistent and (has_gather_tma or not has_gather)
        even_K = (K % opt_flags.block_k == 0)
    if w_scale is not None and opt_flags.is_persistent and not target_info.has_native_mxfp():
        raise NotImplementedError("Must use non-persistent kernel for simulated MXFP")
    if w_scale is not None and w_scale.storage.layout.name is not None and not opt_flags.is_persistent and target_info.has_native_mxfp():
        raise NotImplementedError("Must use persistent kernel and be TMA-compliant for native MXFP")
    # fused activation
    matmul_fused_activation = fused_activation
    reduce_fused_activation = FusedActivation()
    if opt_flags.split_k > 1:
        matmul_fused_activation, reduce_fused_activation = reduce_fused_activation, matmul_fused_activation
    # allocate output/scratchpad memory
    allocation = init_allocation(x, w, precision_config, fused_activation,
        routing_data, gather_indx, scatter_indx, inner_routing_data, fused_comm.n_reduce_shards if fused_comm is not None else 1, opt_flags)
    memory = apply_allocation(allocation, y)
    # early exit
    if batch_size * M * N == 0:
        ret = memory["output"].squeeze(0)
        if not is_input_batched:
            ret = ret.squeeze(0)
        return ret
    # TMA descriptors require a global memory allocation
    if opt_flags.is_persistent:
        triton.set_allocator(get_per_device_per_stream_alloc_fn(x.device))
    # Intermediate tensors and postprocess kernels for each situation
    has_scratchpad = "matmul" in memory["scratchpad"]
    # Canonical output tensor (matmul scratchpad if present, otherwise final output tensor)
    out_matmul = memory["scratchpad"].get("matmul", memory["output"])
    out_matmul_flex = OutFlexData() if out_matmul.dtype == torch.float32 else precision_config.flex_ctx.out_data
    # Unified mx-scale pointer; when scratchpad exists, prefer its mx buffer
    out_matmul_scale = precision_config.out_scale
    if out_matmul_scale is not None:
        out_matmul_scale = out_matmul_scale.data.view(torch.uint8)
        if has_scratchpad and "mx_out_scale" in memory["scratchpad"]:
            out_matmul_scale = memory["scratchpad"]["mx_out_scale"]
    out_matmul_has_mx = out_matmul_scale is not None and out_matmul.element_size() == 1
    # matrix multiplication
    flex = precision_config.flex_ctx
    bias_stride = None if bias is None else bias.stride(0)
    num_indx = None if scatter_indx is None else scatter_indx.src_indx.shape[0]
    # moe metadata
    block_m = opt_flags.block_m
    expt_data_args = InnerRoutingData.make_kernel_args(inner_routing_data or routing_data, block_m)
    # spmd grid
    grid_m = triton.cdiv(M, opt_flags.block_m)
    if routing_data.expt_data is not None:
        grid_m = routing_data.n_blocks(M, opt_flags.block_m)
    grid_n = triton.cdiv(N, opt_flags.block_n)
    max_grid = batch_size * grid_m * grid_n * opt_flags.split_k
    grid = min(target_info.num_sms() - opt_flags.idle_sms, max_grid) if opt_flags.is_persistent else max_grid
    # canonicalize storage
    has_scatter_tma = scatter_indx is not None and target_info.has_tma_gather()
    y = wrap_torch_tensor(out_matmul.view(math.prod(out_matmul.shape[:-1]), out_matmul.shape[-1]) if has_scatter else out_matmul.view(math.prod(out_matmul.shape[:-2]), *out_matmul.shape[-2:]))
    x_storage = _canonicalize_storage(x.storage, 2 if has_gather_tma else 3, flex.lhs_data)
    w_storage = _canonicalize_storage(w.storage, 3, flex.rhs_data)
    y_storage = _canonicalize_storage(y.storage, 2 if has_scatter_tma else 3, flex.out_data)
    # create tma descriptor for x
    if y_acc_in is not None:
        assert opt_flags.split_k == 1, "y_acc_in + split_k is not supported."
        assert scatter_indx is None, "y_acc_in + scatter is not supported."
        if y_acc_in.ndim == 2:
            y_acc_in = y_acc_in.unsqueeze(0)
        assert y_acc_in.shape == out_matmul.shape[-3:]
        y_acc_strides = y_acc_in.stride()
    else:
        y_acc_strides = (None, None, None)

    x_tma_block_size = [1, opt_flags.block_k] if has_gather_tma else [1, opt_flags.block_m, opt_flags.block_k]
    x_tma_mode = None if not x_has_tma else "ragged" if is_ragged and not has_gather_tma else "dense"
    x_tensor_or_tma = x_storage.make_tma(x_tma_block_size, x_tma_mode) if x_has_tma else x_storage.data
    # create tma descriptor for y
    y_has_tma = (
        opt_flags.is_persistent and (scatter_indx is None or has_scatter_tma)
        and (y_acc_in is None or y_acc_is_y)
    )
    block_n = opt_flags.block_n // opt_flags.epilogue_subtile // matmul_fused_activation.specs.reduction_n
    y_tma_block_size = [1, block_n] if has_scatter_tma else [1, opt_flags.block_m, block_n]
    y_tma_mode = None if not y_has_tma else "ragged" if is_ragged and not has_scatter_tma else "dense"
    y_tensor_or_tma = y_storage.make_tma(y_tma_block_size, y_tma_mode) if y_has_tma else y_storage.data
    # create tma descriptor for w
    w_has_tma = opt_flags.is_persistent
    w_tensor_or_tma = w_storage.make_tma([1, opt_flags.block_k, opt_flags.block_n], "dense") if w_has_tma else w_storage.data
    # create tma descriptor for w_scale
    w_scale_has_tma = opt_flags.is_persistent and w_scale is not None
    # When stride(-2) == stride(-1) == 1, it's ambiguous whether W is transposed
    # (i.e. col-wise). Since this matters when w_has_mx is True and w_transpose
    # is True the fast code path, stride(-2) == 1 takes precedence, e.g., vs.
    # w_transpose = w_storage.data.stride()[-1] != 1
    w_transpose = w_storage.data.stride()[-2] == 1
    if w_scale_has_tma:
        w_scale_storage = w_scale.storage
        scale_block_k = opt_flags.block_k // int(MXFP_BLOCK_SIZE)
        # cancel out the transpose done inside make_tma since
        # BlackwellMXScaleLayout.swizzle_block_shape expects block_shape[1] is
        # the reduction dimension.
        w_scale_tma_block_size = [opt_flags.block_n, scale_block_k] if w_transpose and w_scale.storage.layout.name == "BLACKWELL_SCALE" else [scale_block_k, opt_flags.block_n]
        if isinstance(w_scale.storage.layout, StridedLayout):
            assert w_scale_storage.data.stride()[-1] == 1, "w_scale should be contiguous with StridedLayout"
            w_scale_storage = _canonicalize_storage(w_scale.storage, 3, None)
            w_scale_tma_block_size = [1] + w_scale_tma_block_size
        w_scale_tensor_or_tma = w_scale_storage.make_tma(w_scale_tma_block_size, "dense", is_scale=True)
    else:
        w_scale_tensor_or_tma = w_scale
    # canonicalize strides
    x_strides = [0]*(3 - x_storage.data.ndim) + list(x_storage.data.stride())
    x_scale_strides = x_scale.stride() if x_has_mx else (None, None, None)
    x_scale_strides = (0, ) * (3 - len(x_scale_strides)) + x_scale_strides
    w_scale_strides = w_scale.stride() if w_has_mx and not w_scale_has_tma else (None, None, None)
    w_scale_strides = (0, ) * (3 - len(w_scale_strides)) + w_scale_strides
    out_matmul_scale_strides = out_matmul_scale.stride() if out_matmul_has_mx else (None, None, None, None)
    out_matmul_scale_strides = (0, ) * (4 - len(out_matmul_scale_strides)) + out_matmul_scale_strides
    # launch kernel
    kernels = specializations.get(epilogue=epilogue.specs, activation=matmul_fused_activation.specs)
    if gather_indx is not None:
        gather_src_indx = torch.div(gather_indx.src_indx, routing_data.n_expts_act, rounding_mode='trunc')
    fused_comm_kwargs = {
        "pYPtrs": fused_comm.out_handles,
        "ScatterShardIndx": fused_comm.scatter_shard_indx,
        "reduce_rank": fused_comm.reduce_rank,
        "n_reduce_shards": fused_comm.n_reduce_shards,
    } if fused_comm is not None else {}
    # if routing_data.n_expts_act > 1:
    #     y_storage.data.view(torch.uint8).zero_()
    (kernels._p_matmul_ogs if opt_flags.is_persistent else kernels._matmul_ogs)[(grid,)](
                   y_tensor_or_tma, y_storage.data, *out_matmul.stride(),
                   *((None, out_matmul_scale, None) if out_matmul_has_mx else out_matmul_flex),
                   *out_matmul_scale_strides[-4:],
                   x_tensor_or_tma, x_storage.data, *x_strides, x_transpose,
                   flex.lhs_data.scale,
                   None if x_scale is None else x_scale.data.view(torch.uint8), *x_scale_strides,
                   w_tensor_or_tma, w_storage.data, *w_storage.data.stride(), w_transpose,
                   flex.rhs_data.scale,
                   w_scale_tensor_or_tma, *w_scale_strides,
                   flex.acc_data.reinterpret(y_acc_in), *y_acc_strides,
                   flex.acc_data.scale, y_acc_is_y,
                   bias, bias_stride,
                   x.shape[-2] if routing_data.expt_hist is None else None,
                   N, K, K_W,
                   betas, gammas,
                   None if gather_indx is None else gather_src_indx,
                   None if gather_indx is None else gather_indx.dst_indx,  # Only for launch_metadata
                   None if scatter_indx is None else scatter_indx.src_indx,
                   num_indx,
                   None if scatter_indx is None else scatter_indx.dst_indx,
                   None if scatter_indx is None else scatter_indx.dst_indx.shape[0],
                   *expt_data_args,
                   batch_size, grid_m, grid_n,
                   out_alpha,
                   *matmul_fused_activation.fn_args, matmul_fused_activation.specs.reduction_n,
                   *epilogue.fn_arg_values_matmul,
                   routing_data.n_expts_tot,
                   precision_config.max_num_imprecise_acc,
                   precision_config.allow_tf32,
                   precision_config.flexpoint_saturate_inf,
                   flex.rhs_data.is_per_batch,
                   out_matmul_flex.is_per_batch,
                   flex.acc_data.is_per_batch,
                   opt_flags.block_m,
                   opt_flags.block_n,
                   opt_flags.block_k,
                   opt_flags.group_m,
                   INIT_OUTPUT_TO_ZERO=routing_data.n_expts_act == 1,
                   XCD_SWIZZLE=opt_flags.xcd_swizzle,
                   SWIZZLE_MX_VALUE=w.storage.layout.name,
                   SWIZZLE_MX_SCALE=None if w_scale is None else w_scale.storage.layout.name,
                   EPILOGUE_SUBTILE=opt_flags.epilogue_subtile,
                   SPLIT_K=opt_flags.split_k,
                   EVEN_K=even_K,
                   W_CACHE_MODIFIER=opt_flags.w_cache_modifier,
                   TOKENS_PER_EXPT_FOR_ANNOTATION=routing_data.expected_tokens_per_expt,
                   num_warps=opt_flags.num_warps,
                   num_stages=opt_flags.num_stages,
                   arch=opt_flags.arch,
                   UPCAST_INDICES=should_upcast_indices(x, w, out_matmul),
                   X_TMA_MODE=x_tma_mode,
                   Y_TMA_MODE=y_tma_mode,
                   SWAP_XW=get_swap_xw(precision_config, opt_flags),
                   IS_EPILOGUE_QUANT_MXFP8=epilogue.specs.name == FnName.QUANTIZE_MXFP8.name,
                   NUM_SMS = grid if opt_flags.is_persistent else 0,
                   **fused_comm_kwargs,
                   **opt_flags.target_kernel_kwargs)

    assert not (opt_flags.split_k > 1 and scatter_indx is not None)
    out_final_mx_scale = None
    if opt_flags.split_k > 1:
        assert not out_matmul_has_mx
        postprocess_fn1 = ReducePostprocessFn(specs=reduce_fused_activation.specs, fn_args=reduce_fused_activation.fn_args)
        postprocess_fn2 = None if has_scatter else ReducePostprocessFn(specs=epilogue.specs, fn_args=epilogue.fn_arg_values_finalize)
        y, y_mx_scale = reduce(
            x = out_matmul.view(out_matmul.shape[0], -1, out_matmul.shape[-1]),
            dim = 0,
            # output data/metadata
            y = memory["output"].view(-1, memory["output"].shape[-1]),
            y_dtype = memory["output"].dtype,
            y_flex = precision_config.flex_ctx.out_data,
            y_flex_saturate_inf = precision_config.flexpoint_saturate_inf,
            y_has_mx = precision_config.out_scale is not None,
            # fused functions
            postprocess_fn1 = postprocess_fn1,
            postprocess_fn2 = postprocess_fn2,
        )
        y_shape = out_matmul.shape[1:-1] + (out_matmul.shape[-1] // reduce_fused_activation.specs.reduction_n,)
        out_matmul = y.view(*y_shape).unsqueeze(0)
        if y_mx_scale is not None:
            out_final_mx_scale = y_mx_scale.view(out_matmul.shape[-2], triton.cdiv(out_matmul.shape[-1], 32))
    # TODO: change `matmul_ogs` semantics and move this to another op!
    if scatter_indx is not None and (not is_cuda() or routing_data.n_expts_act > 1): # Matmul ogs kernel fuses scatter already, so only need for n_exps_act > 1.
        mask = (scatter_indx.src_indx != -1).view(out_matmul.shape[-2]//routing_data.n_expts_act, routing_data.n_expts_act, 1)
        out_matmul = out_matmul.view(out_matmul.shape[-2]//routing_data.n_expts_act, routing_data.n_expts_act, -1)
        mask = mask.expand_as(out_matmul)
        out_matmul_scale_shape = out_matmul.shape[:-1] + (triton.cdiv(out_matmul.shape[-1], 32),)
        postprocess_fn = ReducePostprocessFn(specs=epilogue.specs, fn_args=epilogue.fn_arg_values_finalize)
        x_flex = InFlexData(dtype=out_matmul_flex.dtype, scale=out_matmul_flex.expected_scale)
        out_final, out_final_mx_scale = reduce(out_matmul, dim=1, postprocess_fn2=postprocess_fn, x_flex=x_flex, #
                                            mask=mask,
                                            y=memory["output"].squeeze(0).squeeze(0),
                                            x_mxscale=out_matmul_scale.view(*out_matmul_scale_shape) if out_matmul_has_mx else None,
                                            y_has_mx=precision_config.out_scale is not None,
                                            y_flex=precision_config.flex_ctx.out_data,
                                            y_flex_saturate_inf=precision_config.flexpoint_saturate_inf,
                                            )
        out_final = out_final.unsqueeze(0)
    else:
        out_final = out_matmul.squeeze(0)

    if not (is_input_batched or inner_routing_data is not None):
        out_final = out_final.squeeze(0)
    if out_final_mx_scale is not None:
        precision_config.out_scale = out_final_mx_scale
    return out_final

# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------

def matmul_ogs_torch(x, w, bias,
                 routing_data: RoutingData = None,
                 gather_indx: GatherIndx = None,
                 scatter_indx: ScatterIndx = None,
                 precision_config: PrecisionConfig = None,
                 betas = None,
                 gammas = None,
                 inner_routing_data: InnerRoutingData | None = None,
                 round_x = None, round_y = None,
                 ):
    if inner_routing_data is not None:
        assert bias is None, "Not supported yet"
        m, n = x.shape[-2], w.shape[-1]
        block_k = inner_routing_data.block_k
        n_expts_tot = inner_routing_data.base.n_expts_tot
        out = torch.zeros((n_expts_tot, m, n), dtype=torch.float32, device=x.device)
        start_x = start_w = 0
        for expt in range(n_expts_tot):
            k = inner_routing_data.base.expt_hist[expt].item()
            if k > 0:
                out[expt] = matmul_ogs_torch(
                    x[:, start_x:start_x+k], w[start_w:start_w+k, :], None,
                    None, None, None, None, betas, gammas, None, round_x, round_y
                )
            padded_k = triton.cdiv(k, block_k) * block_k
            start_x += padded_k if inner_routing_data.x_is_padded else k
            start_w += padded_k if inner_routing_data.w_is_padded else k
        return out

    is_input_batched = x.ndim == 3
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if is_input_batched:
        assert gather_indx is None, "gather not supported in batched mode"
        assert scatter_indx is None, "scatter not supported in batched mode"
        assert routing_data is None, "routing not supported in batched mode"
        assert w.ndim == 3 and w.shape[0] == x.shape[0]
    if round_x is None:
        round_x = lambda x, idx: x
    if round_y is None:
        round_y = lambda x: x
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    if x.ndim == 2:
        x = x.view(1, *x.shape)
    if routing_data is None:
        routing_data = RoutingData(None, None, w.shape[0], 1)
    n_expts_act = routing_data.n_expts_act
    # memory offsets
    if routing_data.n_expts_tot > 1 and not is_input_batched:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[1]] for _ in range(w.shape[0])]
    # compute
    n_rows = x.shape[1] if gather_indx is None else gather_indx.dst_indx.shape[0]
    y = torch.zeros((x.shape[0], n_rows, w.shape[-1]), device=x.device, dtype=x.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            idx = gather_indx.src_indx[lo:hi] // n_expts_act
        batch = i if is_input_batched else 0
        out = torch.matmul(round_x(x[batch, idx, :], torch.arange(lo, hi, device="cuda")).float(),
                           w[i].float())
        if bias is not None:
            out += bias[i, :] if betas is None else bias[i, :] * betas[lo:hi, None]
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[batch, lo:hi, :] = round_y(out)
    if not is_input_batched:
        y = y.view(y.shape[1], y.shape[2])
    if scatter_indx is None:
        return y
    # accumulate output from all experts
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    for i, (lo, hi) in enumerate(offs):
        dst_idx = scatter_indx.dst_indx[lo:hi] // n_expts_act
        msk = dst_idx != -1
        out[dst_idx[msk], :] += y[lo:hi, :][msk, :].float()
    return out


def post_matmul_comm_torch(y: torch.Tensor, rank: int, n_reduce_shards: int,
                           world_size: int,
                           scatter_shard_indx: torch.Tensor | None = None,
):
    """
    Reference implementation of post matmul communication.

    y: the local matmul output
    rank: the global rank
    n_reduce_shards: the number of reduce shards
    world_size: the world size
    scatter_shard_indx: the shard indices for the scatter. None if all gather.

    Output shape:
    (batch_size, n_rows, n_cols) -> (batch_size, n_rows * n_reduce_shards, n_cols) if batched, otherwise
    (n_rows, n_cols) -> (n_rows * n_reduce_shards, n_cols)
    """
    from torch import distributed as dist
    # if n_reduce_shards == 1:
    #     return y

    ys = [torch.empty_like(y) for _ in range(world_size)]
    dist.all_gather(ys, y)
    out_shape = (*y.shape[:-2], y.shape[-2] * n_reduce_shards, y.shape[-1])

    if scatter_shard_indx is None:
        # all gather
        assert n_reduce_shards == world_size
        return torch.cat(ys, dim=-1).reshape(out_shape)
    else:
        # Note: when multiple ranks scatter to the same destination, the result is undefined.
        scatter_shard_indx_global = torch.empty((world_size, *scatter_shard_indx.shape), device=scatter_shard_indx.device, dtype=scatter_shard_indx.dtype)
        dist.all_gather([scatter_shard_indx_global[i] for i in range(world_size)], scatter_shard_indx)

        assert len(out_shape) == 2, "batched mode not supported"
        result = torch.zeros(out_shape, device=y.device, dtype=y.dtype)
        reduce_shard_id = rank // n_reduce_shards

        for i in range(world_size // n_reduce_shards):
            scatter_mask = scatter_shard_indx_global[i * n_reduce_shards, :] == reduce_shard_id
            for j in range(n_reduce_shards):
                out_slice = result.as_strided(
                    (result.shape[0] // n_reduce_shards, result.shape[1]),
                    (result.stride(0) * n_reduce_shards, result.stride(1)),
                    storage_offset=j * result.stride(0),
                )
                out_slice[scatter_mask, :] = ys[i * n_reduce_shards + j][scatter_mask, :]
        return result
