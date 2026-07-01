from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp, copy_or_rebind_param
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DispatchOutput,
        StandardDispatchOutput,
    )

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUUnquantMoEMethod,
)

_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast


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

        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
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

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
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
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
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
            and get_moe_a2a_backend().is_deepep()
            and get_deepep_mode().enable_low_latency()
            and not _is_npu
            and not _is_hip
            and hasattr(layer, "dispatcher")
        ):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
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
            for weight_name in ["w13_weight", "w2_weight"]:
                weight = getattr(layer, weight_name)
                origin_weight = weight.data.transpose(1, 2)
                new_weight = origin_weight.contiguous()
                origin_weight.untyped_storage().resize_(0)
                weight.data = npu_format_cast(new_weight)
                if hasattr(layer, "dispatcher"):
                    layer.dispatcher.set_quant_config(
                        {"dispatcher_output_dtype": "bf16"}
                    )

        return

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

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

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
        if use_intel_xpu_backend():
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
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:

        backend = self.runner.runner_backend
        return self.runner.run(dispatch_output, layer)

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    def forward_musa(self, *args, **kwargs) -> CombineInput:
        return self.forward_cuda(*args, **kwargs)

    forward_native = forward_cpu
