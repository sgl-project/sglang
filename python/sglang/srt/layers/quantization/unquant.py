from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.kernels.fused_op import BaseFusedOp
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import xpu_moe_ld_padding_elems
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.runtime_context import (
    get_exec,
    get_lora,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
    set_weight_attrs,
    use_intel_amx_backend,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DispatchOutput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUUnquantMoEMethod,
)

_is_cpu_amx_available = cpu_has_amx_support()
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm


class Bf16GemmBackend(Enum):
    AUTO = "auto"
    CUTEDSL = "cutedsl"
    FLASHINFER_PR4266 = "flashinfer_pr4266"
    GEMV = "gemv"
    TORCH = "torch"

    def is_auto(self) -> bool:
        return self == Bf16GemmBackend.AUTO

    def is_cutedsl(self) -> bool:
        return self == Bf16GemmBackend.CUTEDSL

    def is_gemv(self) -> bool:
        return self == Bf16GemmBackend.GEMV

    def is_flashinfer_pr4266(self) -> bool:
        return self == Bf16GemmBackend.FLASHINFER_PR4266

    def is_optimized(self) -> bool:
        return self.is_cutedsl() or self.is_flashinfer_pr4266()


_BF16_GEMM_BACKEND: Optional[Bf16GemmBackend] = None
_cutedsl_bf16_gemm = None
_use_cutedsl_bf16_gemm = None
_hopper_bf16_gemv = None
_use_hopper_bf16_gemv = None
_flashinfer_pr4266_splitk_tactic = None
_flashinfer_pr4266_run_splitk_dense = None
_flashinfer_pr4266_direct_default_tactic = None
_flashinfer_pr4266_prefer_direct = None
_flashinfer_pr4266_run_direct_dense = None
_enable_bf16_splitk_gemm = False

# GB300 TP16 tactics measured under CUDA graph replay with PDL and cold weights.
# Unlisted shapes, including M=64, retain the existing TGV/cuBLAS path.
_FLASHINFER_PR4266_TUNED_TACTICS = {
    (1, 256, 8192): (64, 8, 4, 11),
    (2, 256, 8192): (64, 8, 4, 11),
    (4, 256, 8192): (64, 8, 4, 11),
    (8, 256, 8192): (64, 8, 4, 11),
    (16, 256, 8192): (64, 8, 4, 10),
    (24, 256, 8192): (64, 8, 4, 11),
    (32, 256, 8192): (64, 8, 4, 12),
    (1, 512, 8192): (64, 8, 4, 11),
    (2, 512, 8192): (64, 8, 4, 12),
    (4, 512, 8192): (64, 8, 4, 10),
    (8, 512, 8192): (64, 8, 4, 12),
    (16, 512, 8192): (64, 8, 4, 12),
    (24, 512, 8192): (64, 8, 4, 12),
    (32, 512, 8192): (64, 16, 4, 9),
    (1, 2304, 8192): (128, 8, 4, 6),
    (2, 2304, 8192): (64, 8, 2, 12),
    (4, 2304, 8192): (128, 8, 4, 6),
    (8, 2304, 8192): (64, 8, 4, 10),
    (16, 2304, 8192): (64, 16, 4, 9),
    (24, 2304, 8192): (64, 32, 2, 9),
    (32, 2304, 8192): (64, 32, 2, 9),
    (1, 2560, 8192): (64, 8, 2, 10),
    (2, 2560, 8192): (64, 8, 2, 10),
    (4, 2560, 8192): (64, 8, 2, 10),
    (8, 2560, 8192): (64, 8, 2, 10),
    (16, 2560, 8192): (64, 16, 2, 11),
    (24, 2560, 8192): (64, 32, 2, 9),
    (32, 2560, 8192): (64, 32, 2, 9),
}


def use_flashinfer_pr4266_bf16_gemm(m: int, n: int, k: int) -> bool:
    return (m, n, k) in _FLASHINFER_PR4266_TUNED_TACTICS


def should_enable_bf16_splitk_gemm(backend: Bf16GemmBackend) -> bool:
    """Return whether the optional Split-K path should be initialized."""
    return backend.is_optimized() and envs.SGLANG_ENABLE_BF16_SPLITK_GEMM.get()


def initialize_bf16_gemm_config(server_args: ServerArgs) -> None:
    global _BF16_GEMM_BACKEND
    global _cutedsl_bf16_gemm, _use_cutedsl_bf16_gemm
    global _flashinfer_pr4266_splitk_tactic
    global _flashinfer_pr4266_run_splitk_dense
    global _flashinfer_pr4266_direct_default_tactic
    global _flashinfer_pr4266_prefer_direct
    global _flashinfer_pr4266_run_direct_dense
    global _enable_bf16_splitk_gemm

    from sglang.srt.utils import is_sm100_supported

    backend_str = server_args.bf16_gemm_backend
    if backend_str == "auto" and is_sm100_supported():
        backend_str = (
            "torch"
            if get_exec().deterministic.enable_deterministic_inference
            else "cutedsl"
        )

    backend = Bf16GemmBackend(backend_str)

    if backend.is_gemv():
        if torch.cuda.get_device_capability()[0] != 9:
            raise ValueError("--bf16-gemm-backend gemv requires SM90 (Hopper)")

        global _hopper_bf16_gemv, _use_hopper_bf16_gemv
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import (
            hopper_bf16_gemv,
            use_hopper_bf16_gemv,
        )

        _hopper_bf16_gemv = hopper_bf16_gemv
        _use_hopper_bf16_gemv = use_hopper_bf16_gemv
    elif backend.is_optimized():
        if get_exec().deterministic.enable_deterministic_inference:
            raise ValueError(
                "--bf16-gemm-backend cutedsl is batch-size dependent and cannot "
                "be combined with --enable-deterministic-inference"
            )
        if not is_sm100_supported():
            raise ValueError(
                f"--bf16-gemm-backend {backend.value} requires "
                "SM100/SM103 (Blackwell)"
            )

        from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
            cutedsl_bf16_gemm,
            use_cutedsl_bf16_gemm,
        )

        _cutedsl_bf16_gemm = cutedsl_bf16_gemm
        _use_cutedsl_bf16_gemm = use_cutedsl_bf16_gemm

    _enable_bf16_splitk_gemm = False
    if should_enable_bf16_splitk_gemm(backend):
        from sglang.kernels.ops.gemm.flashinfer_pr4266_dense_bf16_gemm_sm100_direct import (
            default_tactic,
            prefer_direct_bf16_gemm_sm100,
            run_direct_dense,
        )
        from sglang.kernels.ops.gemm.flashinfer_pr4266_dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        _flashinfer_pr4266_splitk_tactic = SplitKTactic
        _flashinfer_pr4266_run_splitk_dense = run_splitk_dense
        _flashinfer_pr4266_direct_default_tactic = default_tactic
        _flashinfer_pr4266_prefer_direct = prefer_direct_bf16_gemm_sm100
        _flashinfer_pr4266_run_direct_dense = run_direct_dense
        _enable_bf16_splitk_gemm = True

    _BF16_GEMM_BACKEND = backend


def _bf16_gemm_dispatch_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _flashinfer_pr4266_bf16_gemm(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    out = torch.empty((x_2d.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    m, n, k = x_2d.shape[0], weight.shape[0], weight.shape[1]
    if bias is None and _flashinfer_pr4266_prefer_direct(m, n, k):
        tactic = _flashinfer_pr4266_direct_default_tactic(m, n, k)
        _flashinfer_pr4266_run_direct_dense(x_2d, weight.T, out, True, tactic)
    else:
        tactic = _flashinfer_pr4266_splitk_tactic(
            *_FLASHINFER_PR4266_TUNED_TACTICS[(m, n, k)]
        )
        _flashinfer_pr4266_run_splitk_dense(
            x_2d,
            weight.T,
            bias,
            out,
            True,
            tactic,
        )
    return out.view(*x.shape[:-1], weight.shape[0])


def _bf16_gemm_dispatch_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    m = x.numel() // x.shape[-1]
    if _enable_bf16_splitk_gemm and use_flashinfer_pr4266_bf16_gemm(
        m, weight.shape[0], weight.shape[1]
    ):
        return _flashinfer_pr4266_bf16_gemm(x, weight, bias)
    if (
        _use_hopper_bf16_gemv is not None
        and bias is None
        and _use_hopper_bf16_gemv(m, weight.shape[0], weight.shape[1])
    ):
        return _hopper_bf16_gemv(x.view(-1, x.shape[-1]), weight).view(
            *x.shape[:-1], -1
        )
    if _use_cutedsl_bf16_gemm is not None and _use_cutedsl_bf16_gemm(
        m, weight.shape[0], weight.shape[1]
    ):
        return _cutedsl_bf16_gemm(x.view(-1, x.shape[-1]), weight, bias).view(
            *x.shape[:-1], -1
        )
    return F.linear(x, weight, bias)


@register_custom_op(fake_impl=_bf16_gemm_dispatch_fake)
def bf16_gemm_dispatch(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return _bf16_gemm_dispatch_impl(x, weight, bias)


def get_bf16_gemm_backend() -> Bf16GemmBackend:
    global _BF16_GEMM_BACKEND
    if _BF16_GEMM_BACKEND is None:
        _BF16_GEMM_BACKEND = Bf16GemmBackend.AUTO
    return _BF16_GEMM_BACKEND


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        elif (
            get_bf16_gemm_backend().is_optimized()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and layer.weight.dtype == torch.bfloat16
            and (bias is None or bias.dtype == torch.bfloat16)
            and not layer.weight.requires_grad
            and (bias is None or not bias.requires_grad)
        ):
            if torch.compiler.is_compiling():
                # The m-dependent kernel heuristic would guard on the symbolic
                # token dim under Dynamo and recompile per shape bucket; the
                # opaque op resolves it at runtime with concrete shapes,
                # keeping the per-shape kernel choice.
                return bf16_gemm_dispatch(x, layer.weight, bias)
            return _bf16_gemm_dispatch_impl(x, layer.weight, bias)

        return F.linear(x, layer.weight, bias)

    def apply_into(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        output: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run an inference-only BF16 linear into caller-owned storage."""
        if (
            get_bf16_gemm_backend().is_cutedsl()
            and x.is_cuda
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and layer.weight.dtype == torch.bfloat16
            and output.dtype == torch.bfloat16
            and output.is_contiguous()
            and output.shape == (x.shape[0], layer.weight.shape[0])
            and (bias is None or bias.dtype == torch.bfloat16)
            and not layer.weight.requires_grad
            and (bias is None or not bias.requires_grad)
            and _use_cutedsl_bf16_gemm(
                x.shape[0], layer.weight.shape[0], layer.weight.shape[1]
            )
        ):
            from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
                cutedsl_bf16_gemm_out,
            )

            return cutedsl_bf16_gemm_out(x, layer.weight, output, bias)

        if x.ndim != 2:
            raise ValueError("caller-owned linear output currently requires a 2D input")
        if output.shape != (x.shape[0], layer.weight.shape[0]):
            raise ValueError(
                f"linear output has shape {output.shape}, expected "
                f"{(x.shape[0], layer.weight.shape[0])}"
            )
        torch.mm(x, layer.weight.t(), out=output)
        if bias is not None:
            output.add_(bias)
        return output


def _use_xpu_moe_ld_padding(use_triton_kernels: bool) -> bool:
    """Whether MoE expert weights should get a padded row stride for XPU.

    is_xpu() only tells us an XPU exists on this machine, not that the weights
    being created land on it -- this can be true while serving on CPU/CUDA.
    create_weights takes no device argument and allocates under the model
    loader's ambient device context, so check that context too: padding a
    non-XPU weight would make it non-contiguous for no benefit, and other
    backends' MoE kernels expect contiguous expert tensors.

    The Triton path stores B transposed and does not read a row stride, so it
    is excluded even on XPU (either via --moe-runner-backend triton or the
    triton_kernels build).
    """
    return (
        is_xpu()
        and not get_moe_runner_backend().is_triton()
        and torch.get_default_device().type == "xpu"
        and not use_triton_kernels
    )


def _empty_xpu_moe_expert_weight(
    num_experts: int,
    n_dim: int,
    k_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate an [E, N, K] XPU expert weight, over-allocating K when padding
    its row stride would avoid L3 set aliasing.

    Some K dims (3072, 7168 in bf16) put every weight row in the same handful
    of L3 sets, which throttles the grouped GEMM's B loads. Over-allocating K
    and returning a narrowed view keeps the logical [E, N, K] shape (so the
    weight loader is unchanged) while giving the rows a non-aliasing stride.
    The Xe20 grouped GEMM reads B's row stride from the tensor, so the padding
    is transparent to it.

    Callers must have checked _use_xpu_moe_ld_padding() first. K dims that are
    already well distributed get no padding and allocate normally.
    """
    pad = xpu_moe_ld_padding_elems(k_dim, dtype.itemsize)
    if pad == 0:
        return torch.empty(num_experts, n_dim, k_dim, dtype=dtype)
    # The view is non-contiguous; only the K slice is ever read or written.
    return torch.empty(num_experts, n_dim, k_dim + pad, dtype=dtype)[:, :, :k_dim]


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, BaseFusedOp):
    """MoE method without quantization."""

    def __init__(
        self,
        use_triton_kernels: bool = False,
        use_flashinfer_trtllm_moe: bool = False,
        use_deep_gemm: bool = False,
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self.use_deep_gemm = use_deep_gemm
        self._cache_permute_indices = dict({})
        # Set by process_weights_after_loading when w13 rows are permuted to
        # interleave gate/up for the fused swiglu up-GEMM epilogue.
        self.w13_swiglu_interleaved = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # XPU only: the sgl-kernel-xpu grouped GEMM honours the weights' row
        # stride, so it can be padded to dodge L3 set aliasing on unlucky K
        # dims. Every other device allocates plainly, exactly as before.
        pad_ld_for_xpu = _use_xpu_moe_ld_padding(self.use_triton_kernels)

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        if pad_ld_for_xpu:
            w13_weight_data = _empty_xpu_moe_expert_weight(
                num_experts, w13_weight_n, w13_weight_k, params_dtype
            )
        else:
            w13_weight_data = torch.empty(
                num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype
            )
        w13_weight = torch.nn.Parameter(w13_weight_data, requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        if pad_ld_for_xpu:
            w2_weight_data = _empty_xpu_moe_expert_weight(
                num_experts, w2_weight_n, w2_weight_k, params_dtype
            )
        else:
            w2_weight_data = torch.empty(
                num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype
            )
        w2_weight = torch.nn.Parameter(w2_weight_data, requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _should_use_aiter_moe = (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and self._aiter_ck_moe_supported(layer)
            and not layer._skip_aiter_moe_shuffle
        )
        if _should_use_aiter_moe:
            copy_or_rebind_param(
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()
            copy_or_rebind_param(
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
            if hasattr(layer, "w13_weight_bias"):
                layer.w13_weight_bias = Parameter(
                    layer.w13_weight_bias.float(), requires_grad=False
                )
            if hasattr(layer, "w2_weight_bias"):
                layer.w2_weight_bias = Parameter(
                    layer.w2_weight_bias.float(), requires_grad=False
                )

        if (
            self.use_deep_gemm
            and layer.w13_weight.dtype == torch.bfloat16
            and (get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_pplx())
            and not _is_npu
            and not _is_hip
            and hasattr(layer, "dispatcher")
        ):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            # The cached indices are GPU tensors. Colocated weight offloading
            # can release their backing memory between reloads, so rebuild them
            # once per post-processing cycle.
            self._cache_permute_indices.clear()

            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                    is_gated_act_gemm=layer.moe_runner_config.is_gated,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )
        if _is_npu:
            # The kernels set the dispatcher output dtype themselves -- they are
            # the ones that know what their gmms expect. NPUUnquantMoEMethod
            # already sets bf16 here, and hardcoding it a second time would
            # clobber a subclass that attached a quantized kernel instead.
            layer.w13_kernel.process_weights_after_loading(layer, "w13")
            layer.w2_kernel.process_weights_after_loading(layer, "w2")

        self._maybe_interleave_w13_for_fused_swiglu(layer)
        return

    def _maybe_interleave_w13_for_fused_swiglu(self, layer: torch.nn.Module) -> None:
        """Permute W13 rows so the triton up-GEMM epilogue can apply the SwiGLU.

        Interleaving puts both operands of ``silu(gate) * up`` in adjacent
        columns of one output tile, so the epilogue can apply the activation
        in-register and store half width -- removing ``intermediate_cache1``
        and the activation launch per MoE layer. Value-neutral: each output
        column is an independent dot product.

        The gate stays conservative because only the fused epilogue understands
        the permuted layout -- every consumer reading W13 or the pre-activation
        buffer in halves layout is excluded here rather than trapped later
        (notably LoRA, whose gate_up delta targets the buffer this eliminates).
        """
        if not envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.get():
            return

        moe_runner_config = layer.moe_runner_config
        if not (
            _is_cuda
            and self._aiter_runner is None
            and self.runner.runner_backend.is_triton()
            and get_moe_a2a_backend().is_none()
            and not self.with_bias
            and layer.w13_weight.dtype == torch.bfloat16
            and moe_runner_config.activation == "silu"
            and moe_runner_config.is_gated
            and moe_runner_config.gemm1_alpha is None
            and moe_runner_config.gemm1_clamp_limit is None
            and moe_runner_config.swiglu_limit is None
            and not moe_runner_config.apply_router_weight_on_input
            # The LoRA MoE hooks read and write the full-width pre-activation
            # buffer in halves layout; both assumptions break here.
            and not get_lora().enable_lora
            and not get_lora().lora_paths
            # EPLB rearranges experts by copying checkpoint-layout weights in.
            and not get_exec().moe.enable_eplb
        ):
            return

        w13 = layer.w13_weight.data
        inter = w13.shape[1] // 2
        idx = torch.empty(w13.shape[1], dtype=torch.long, device=w13.device)
        idx[0::2] = torch.arange(0, inter, device=w13.device)
        idx[1::2] = torch.arange(inter, 2 * inter, device=w13.device)
        # Per-expert, to cap the gather temporary at one expert's slice.
        for e in range(w13.shape[0]):
            w13[e] = w13[e][idx]
        self.w13_swiglu_interleaved = True
        logger.info_once(
            "Interleaved w13 gate/up: the SwiGLU is applied by the MoE up-GEMM epilogue."
        )

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        self,
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        weight_name: str,
    ) -> None:
        """Restore canonical BF16 MoE load shapes before hot weight copy.

        The flashinfer TRT-LLM BF16 postprocess reshapes expert weights into
        block layout. During weight update, checkpoint tensors are in
        canonical layout and need a temporary shape restore for copy.
        """
        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
            return

        expected_shape = None
        if weight_name.endswith(".experts.w13_weight"):
            w13_rows = (
                2 * layer.intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else layer.intermediate_size_per_partition
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)
        elif weight_name.endswith(".experts.w2_weight"):
            expected_shape = (
                layer.num_local_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:
            return

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        if param.data.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."
            )

        param.data = param.data.reshape(expected_shape)

    def _aiter_ck_moe_supported(self, layer) -> bool:
        # aiter CK fused-MoE requires intermediate_size_per_partition to be 128-aligned
        # (GemmSpec=Default; otherwise CK raises "not support this GEMM problem").
        return layer.intermediate_size_per_partition % 128 == 0

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()
                else MoeRunnerBackend.FLASHINFER_TRTLLM
            )
        elif self.use_flashinfer_cutlass:
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

            backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        elif self.use_deep_gemm:
            backend = MoeRunnerBackend.DEEP_GEMM
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        elif _is_npu:
            layer.w13_kernel = NPUUnquantMoEMethod()
            layer.w2_kernel = NPUUnquantMoEMethod()
            moe_runner_config.layer = layer
            backend = MoeRunnerBackend.ASCEND
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

        # aiter CK fused-MoE only supports 128-aligned shapes; otherwise use triton.
        self._aiter_runner: Optional[MoeRunner] = None
        if (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and get_moe_a2a_backend().supports_aiter()
        ):
            if self._aiter_ck_moe_supported(layer):
                self._aiter_runner = MoeRunner(
                    MoeRunnerBackend.AITER, moe_runner_config
                )
            elif get_moe_runner_backend().is_aiter():
                raise ValueError(
                    "moe_runner_backend=aiter is not supported for "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "use --moe-runner-backend triton."
                )
            else:
                logger.warning_once(
                    "aiter CK fused-MoE does not support "
                    f"intermediate_size_per_partition={layer.intermediate_size_per_partition}; "
                    "using triton MoE runner."
                )

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    # forward_native is aliased to forward_cpu at the end of the class body
    # (pre-existing behavior); under torch.compile the dedicated
    # fused_moe_forward_native is installed instead via this hook.
    def _torch_compile_forward(self, num_tokens: int) -> Optional[Callable]:
        # torch.compile on this layer only pays off at bs=1; keep the
        # optimized dispatch otherwise.
        if num_tokens == 1:
            from sglang.srt.layers.moe.fused_moe_native import (
                fused_moe_forward_native,
            )

            return fused_moe_forward_native
        return None

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        x = dispatch_output.hidden_states

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.runner.runner_backend.is_deep_gemm():
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight
            from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

            # Only use_fp8=False when SGLANG_DEEPEP_BF16_DISPATCH is true,
            # otherwise use_fp8=True for FP8 dispatch path
            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=use_fp8,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="bf16",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                output_dtype=x.dtype,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=not layer.should_fuse_routed_scaling_factor_in_topk,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            if self._aiter_runner is not None:
                from sglang.srt.layers.moe.moe_runner.aiter import (
                    AiterMoeQuantInfo,
                )

                quant_info = AiterMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    expert_mask=layer.dispatcher.expert_mask_gpu,
                )
                return self._aiter_runner.run(dispatch_output, quant_info)

            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                fuse_swiglu_interleaved=self.w13_swiglu_interleaved,
            )
            return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                getattr(layer, "w13_weight_bias", None),
                getattr(layer, "w2_weight_bias", None),
                layer.moe_runner_config.gemm1_alpha,
                layer.moe_runner_config.gemm1_clamp_limit,
                True,  # is_vnni
                moe_runner_config.activation,  # activation
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
            fuse_swiglu_interleaved=self.w13_swiglu_interleaved,
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
            "relu2",  # Nemotron-H (NemotronHForCausalLM) uses squared-ReLU.
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if not get_moe_runner_backend().is_triton():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            if moe_runner_config.apply_router_weight_on_input:
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights)
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please drop --moe-runner-backend triton to use \
            the sgl-kernel-xpu path, which supports more activations."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:

        return self.runner.run(dispatch_output, layer)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    def forward_musa(self, *args, **kwargs) -> CombineInput:
        return self.forward_cuda(*args, **kwargs)

    forward_native = forward_cpu
