# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import RoutingMethodType, get_moe_runner_backend
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    can_auto_enable_marlin_fp8,
    cutlass_fp8_supported,
    dispatch_w8a8_block_fp8_linear,
    input_to_float8,
    mxfp8_group_quantize,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_weight_ue8m0_inplace,
    triton_mxfp8_blockscaled_linear,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
    log_info_on_rank0,
    print_warning_once,
    set_weight_attrs,
    use_intel_amx_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.topk import TopKOutput
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_hip_int4 = get_bool_env_var("SGLANG_INT4_WEIGHT") and _is_hip
_use_aiter = envs.SGLANG_USE_AITER.get() and _is_hip

if _use_aiter or _use_hip_int4:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight


ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        use_mxfp8: bool = False,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            log_info_on_rank0(logger, "Detected fp8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.use_mxfp8 = use_mxfp8
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports fp8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        if self.use_mxfp8:
            if weight_block_size is None:
                weight_block_size = [1, 32]
            elif weight_block_size != [1, 32]:
                raise ValueError("MXFP8 requires weight_block_size=[1, 32].")
        self.weight_block_size = weight_block_size

    def get_name(self) -> str:
        return "mxfp8" if self.use_mxfp8 else "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    def get_min_capability(self) -> int:
        return 100 if self.use_mxfp8 else 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        use_mxfp8 = "mxfp8" in quant_method
        is_checkpoint_fp8_serialized = ("fp8" in quant_method) or use_mxfp8
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        packed_modules_mapping = (
            cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        )
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            # Keep both "model." and non-"model." variants for robust prefix matching.
            normalized = []
            for layer in ignored_layers:
                base = layer.removeprefix("model.")
                normalized.append(base)
                normalized.append(f"model.{base}")
            ignored_layers = normalized
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        if use_mxfp8 and weight_block_size is not None:
            logger.warning(
                "MXFP8 ignoring incoming weight_block_size in config.json; it is fixed to [1, 32]."
            )
            weight_block_size = [1, 32]
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            packed_modules_mapping=packed_modules_mapping,
            use_mxfp8=use_mxfp8,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if is_layer_skipped(
                prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
            ):
                return UnquantizedFusedMoEMethod(
                    layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
                )
            return Fp8MoEMethod(self)
        elif isinstance(layer, RadixAttention):
            return Fp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.

    It supports the following quantization schemes:
    - Per-channel weight quantization + per-token activation quantization
    - Per-tensor weight quantization + per-tensor activation quantization
    - Blockwise weight quantization + blockwise activation quantization

    It supports the following checkpoint formats:
    - FP8 checkpoint
    - FP16/BF16 checkpoint. In this case, the weights will be quantized to FP8 during the weight loading.

    Notes:
    - The activation quantization scheme can be static or dynamic. The dynamic activation quantization is more commonly used.
    - On NV platforms, the per-channel weight quantization is used by default, if block quantization is not enabled.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Union[Fp8Config, W4AFp8Config]):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = False
        if _is_cuda:
            force_marlin = get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")
            auto_enable = can_auto_enable_marlin_fp8()
            self.use_marlin = force_marlin or auto_enable

        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
        self.block_quant = (
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()
        self.is_checkpoint_fp8_serialized = (
            self.quant_config.is_checkpoint_fp8_serialized
        )
        self.use_aiter_fp8_per_token = envs.SGLANG_USE_AITER_FP8_PER_TOKEN.get()
        self.use_per_token_if_dynamic = False

    def validate_block_quant_shapes(
        self,
        input_size: int,
        input_size_per_partition: int,
        output_size: int,
        output_size_per_partition: int,
        output_partition_sizes: List[int],
        skip_block_quant_check: bool = False,
    ):
        tp_size = get_tensor_model_parallel_world_size()
        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )

        if skip_block_quant_check:
            print_warning_once(
                "Skipping block quantization checks for weight partition."
            )
        else:
            # Required by row parallel
            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs,
    ):
        # Copy the layer attributes
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.block_quant:
            block_n, block_k = self.quant_config.weight_block_size
            self.validate_block_quant_shapes(
                input_size,
                input_size_per_partition,
                output_size,
                output_size_per_partition,
                output_partition_sizes,
                skip_block_quant_check,
            )

        # Create the weight
        weight_dtype = (
            torch.float8_e4m3fn if self.is_checkpoint_fp8_serialized else params_dtype
        )
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                if hasattr(self.quant_config, "activation_scheme"):
                    assert self.quant_config.activation_scheme == "dynamic"
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    assert self.quant_config.linear_activation_scheme == "dynamic"
                if self.use_mxfp8 and not self.is_checkpoint_fp8_serialized:
                    raise ValueError(
                        "MXFP8 requires fp8-serialized checkpoint for linear layers."
                    )
                scale_dtype = torch.uint8 if self.use_mxfp8 else torch.float32
                scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.empty
                scale = BlockQuantScaleParameter(
                    data=scale_init(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=scale_dtype,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale.format_ue8m0 = self.use_mxfp8
                if scale_dtype != torch.uint8:
                    scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if (
                hasattr(self.quant_config, "activation_scheme")
                and self.quant_config.activation_scheme == "static"
            ) or (
                hasattr(self.quant_config, "linear_activation_scheme")
                and self.quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
        # If ROCm, normalize the weights and scales to e4m3fnuz
        if _is_fp8_fnuz:
            # activation_scheme: dynamic
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
            )
            layer.input_scale = None
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8LinearMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["weight"])
            layer.weight_scale_inv = torch.nn.Parameter(
                layer.weight_scale_inv.data, requires_grad=False
            )
            return
        elif self.use_mxfp8:
            if not self.is_checkpoint_fp8_serialized:
                self._quantize_mxfp8_weights(layer)
                return
            # MXFP8 scales are stored as UE8M0 uint8; no requantization here.
            # Keep parameter object to preserve weight_loader attrs for hot reload.
            layer.weight_scale_inv.requires_grad_(False)
            layer.weight_scale_inv.format_ue8m0 = True
            return
        else:
            # For fp8 linear weights run with deepgemm, the weights and scales need be requantized to ue8m0
            from sglang.srt.layers.quantization.fp8_utils import (
                deepgemm_w8a8_block_fp8_linear_with_fallback,
            )
            from sglang.srt.model_loader.utils import (
                should_deepgemm_weight_requant_ue8m0,
            )

            if (
                should_deepgemm_weight_requant_ue8m0(
                    weight_block_size=getattr(
                        self.quant_config, "weight_block_size", None
                    ),
                )
                and (
                    self.w8a8_block_fp8_linear
                    is deepgemm_w8a8_block_fp8_linear_with_fallback
                )
                and (not layer.weight_scale_inv.format_ue8m0)
            ):
                requant_weight_ue8m0_inplace(
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size,
                )
                layer.weight_scale_inv.format_ue8m0 = True
            weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

        layer.weight.data = weight.data
        layer.weight_scale_inv.data = weight_scale.data

    def _quantize_mxfp8_weights(self, layer: Module) -> None:
        weight = layer.weight.data
        qweight, weight_scale = mxfp8_group_quantize(weight)
        # Keep parameter objects to preserve weight_loader attrs for hot reload.
        layer.weight.data = qweight
        layer.weight.requires_grad_(False)
        if hasattr(layer, "weight_scale_inv") and layer.weight_scale_inv is not None:
            layer.weight_scale_inv.data = weight_scale
            layer.weight_scale_inv.requires_grad_(False)
        else:
            # First-time online MXFP8 quantization (no serialized scales).
            layer.register_parameter(
                "weight_scale_inv", Parameter(weight_scale, requires_grad=False)
            )
        layer.weight_scale_inv.format_ue8m0 = True
        layer.input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.block_quant:
            self.process_weights_after_loading_block_quant(layer)
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            if not self.is_checkpoint_fp8_serialized:
                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):
                    # apply per-channel quantization default as
                    # cutlass sgl-kernel and marlin only support per-channel scale
                    qweight, weight_scale = per_token_group_quant_fp8(
                        layer.weight, layer.weight.shape[-1]
                    )
                    weight_scale = weight_scale.t().contiguous()
                    if _use_aiter and self.use_aiter_fp8_per_token:
                        self.use_per_token_if_dynamic = True
                        qweight = shuffle_weight(qweight.contiguous(), (16, 16))
                else:
                    # per-tensor quantization
                    qweight, weight_scale = input_to_float8(layer.weight)

                # Update the layer with the new values.
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None

            # If checkpoint is fp8, handle that there are N scales for N
            # shards in a fused module
            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.data, requires_grad=False
                    )

                # cutlass sgl-kernel and marlin only support per-channel scale; aiter supports per-channel scale
                if (
                    self.cutlass_fp8_supported
                    or self.use_marlin
                    or (_use_aiter and self.use_aiter_fp8_per_token)
                ):
                    weight = layer.weight
                    weight_scale = convert_to_channelwise(
                        layer.weight_scale, layer.logical_widths
                    )
                    if _use_aiter and self.use_aiter_fp8_per_token:
                        # Otherwise, by default, aiter only uses per-tensor quantization
                        self.use_per_token_if_dynamic = True
                        if _is_fp8_fnuz:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                            )
                        weight = shuffle_weight(weight.contiguous(), (16, 16))
                else:
                    # Dequant -> Quant with max scale so we can run per tensor.
                    weight = layer.weight
                    weight_scale = layer.weight_scale
                    # If ROCm, normalize the weights and scales to e4m3fnuz
                    if _is_fp8_fnuz:
                        weight, weight_scale, input_scale = (
                            normalize_e4m3fn_to_e4m3fnuz(
                                weight=weight,
                                weight_scale=weight_scale,
                                input_scale=layer.input_scale,
                            )
                        )
                        if input_scale is not None:
                            layer.input_scale = Parameter(
                                input_scale, requires_grad=False
                            )

                    weight_scale, weight = requantize_with_max_scale(
                        weight=weight,
                        weight_scale=weight_scale,
                        logical_widths=layer.logical_widths,
                    )

                # Update layer with new values.
                layer.weight = Parameter(weight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if (
                    hasattr(self.quant_config, "activation_scheme")
                    and self.quant_config.activation_scheme == "static"
                ) or (
                    hasattr(self.quant_config, "linear_activation_scheme")
                    and self.quant_config.linear_activation_scheme == "static"
                ):
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

        if self.use_marlin:
            if self.block_quant:
                layer.weight_block_size = self.quant_config.weight_block_size
            prepare_fp8_layer_for_marlin(layer, not self.block_quant)
            # Activations not quantized for marlin.
            del layer.input_scale

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_marlin:
            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        if self.use_mxfp8:
            if isinstance(x, tuple):
                return triton_mxfp8_blockscaled_linear(
                    input=x[0],
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )
            return triton_mxfp8_blockscaled_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        if self.block_quant:
            if use_intel_amx_backend(layer):
                return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                    x,
                    layer.weight,
                    layer.weight_scale_inv,
                    self.quant_config.weight_block_size,
                    bias,
                    x.dtype,
                    True,  # is_vnni
                )

            if isinstance(x, tuple):
                return self.w8a8_block_fp8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
        self.block_quant = (
            self.use_mxfp8 or self.quant_config.weight_block_size is not None
        )
        self.with_bias = False
        if get_moe_runner_backend().is_cutlass():
            assert (
                cutlass_fp8_supported()
            ), "cutlass_fp8 MoE requires CUDA 12.0+ with SM90 or CUDA 12.4+ with SM89"
            assert self.block_quant, "cutlass_fp8 MoE requires block quantization"
            assert is_sm100_supported() or is_sm90_supported()

    @staticmethod
    def is_deepgemm_moe_runner_backend_enabled() -> bool:
        """Check if MoE will actually use DeepGEMM runner for FP8."""
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_deep_gemm():
            return True
        if moe_runner_backend.is_auto():
            return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
                get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
            )
        return False

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.uint32 if _use_hip_int4 else torch.float8_e4m3fn
        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS
        if _is_hip and _use_hip_int4:
            # INT4 MoE weight - INT32 packed
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )

        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIAS (optional, e.g. GPT-OSS)
        if self.with_bias:
            w13_up_dim = (
                2 * intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else intermediate_size_per_partition
            )
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            scale_dtype = torch.uint8 if self.use_mxfp8 else torch.float32
            scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.ones
            w13_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                scale_init(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=scale_dtype,
                ),
                requires_grad=False,
            )
            # w13_weight and w2_weight are always requanted together
            w13_weight_scale.format_ue8m0 = self.use_mxfp8
            w2_weight_scale.format_ue8m0 = self.use_mxfp8
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
            if get_moe_runner_backend().is_cutlass():
                self._ensure_cutlass_buffers_initialized(layer)

        else:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            if _is_hip:  # _use_aiter: TODO: add check back after triton kernel
                # ROCm - using column scaling, duplicate scaling numbers in case per tensor scaling
                w13_weight_scale1 = torch.nn.Parameter(
                    torch.ones(
                        num_experts,
                        2 * intermediate_size_per_partition,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                w2_weight_scale1 = torch.nn.Parameter(
                    torch.ones(num_experts, hidden_size, dtype=torch.float32),
                    requires_grad=False,
                )
                layer.register_parameter("w13_weight_scale1", w13_weight_scale1)
                layer.register_parameter("w2_weight_scale1", w2_weight_scale1)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

            if _is_hip and _use_hip_int4:
                extra_weight_attrs.update(
                    {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
                )
                set_weight_attrs(w13_weight_scale1, extra_weight_attrs)
                set_weight_attrs(w2_weight_scale1, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading_block_quant(self, layer: Module) -> None:
        # If ROCm, normalize the weights and scales to e4m3fnuz
        if _is_fp8_fnuz:
            # activation_scheme: dynamic
            w13_weight, w13_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w13_weight,
                weight_scale=layer.w13_weight_scale_inv,
                input_scale=None,
            )
            w2_weight, w2_weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale_inv,
                input_scale=None,
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            layer.w13_input_scale = None
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            layer.w2_input_scale = None
            if _use_aiter:
                # add this section for MI300
                # Pre-shuffle weights
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
        elif _use_aiter:
            # Pre-shuffle weights
            layer.w13_weight.data = shuffle_weight(
                layer.w13_weight.contiguous(), (16, 16)
            )
            layer.w2_weight.data = shuffle_weight(
                layer.w2_weight.contiguous(), (16, 16)
            )
        elif _is_cpu:
            assert (
                _is_cpu_amx_available
            ), "Fp8MoEMethod on CPU requires that CPU has AMX support"
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
        elif self.use_mxfp8:
            self._process_mxfp8_moe_weights(
                layer, quantize=not self.quant_config.is_checkpoint_fp8_serialized
            )
        else:
            # For fp8 moe run with deepgemm, the expert weights and scales need be requantized to ue8m0
            from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
            from sglang.srt.model_loader.utils import (
                should_deepgemm_weight_requant_ue8m0,
            )

            # Check if MoE will actually use DeepGEMM runner
            will_use_deepgemm = self.is_deepgemm_moe_runner_backend_enabled()

            if (
                should_deepgemm_weight_requant_ue8m0(
                    weight_block_size=getattr(
                        self.quant_config, "weight_block_size", None
                    ),
                )
                and will_use_deepgemm
                and not layer.w13_weight_scale_inv.format_ue8m0
            ):
                assert isinstance(
                    layer, DeepEPMoE
                ), "DeepGemm MoE is only supported with DeepEPMoE"
                weight_block_size = self.quant_config.weight_block_size
                requant_weight_ue8m0_inplace(
                    layer.w13_weight, layer.w13_weight_scale_inv, weight_block_size
                )
                requant_weight_ue8m0_inplace(
                    layer.w2_weight, layer.w2_weight_scale_inv, weight_block_size
                )
                layer.w13_weight_scale_inv.format_ue8m0 = True
                layer.w2_weight_scale_inv.format_ue8m0 = True

    def _process_mxfp8_moe_weights(self, layer: Module, quantize: bool = True) -> None:

        if not (_is_cuda and is_sm100_supported()):
            raise RuntimeError("MXFP8 MoE quantization requires SM100.")

        def _quantize_and_swizzle_with_cutlass_es_kernel(weight: torch.Tensor):
            from sgl_kernel import es_sm100_mxfp8_blockscaled_grouped_quant

            weight = weight.contiguous()
            num_experts, m, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            problem_sizes = torch.empty(
                (num_experts, 3), dtype=torch.int32, device=weight.device
            )
            problem_sizes[:, 0] = m
            problem_sizes[:, 1] = 0
            problem_sizes[:, 2] = k
            expert_offsets = torch.arange(
                0, num_experts * m, m, dtype=torch.int32, device=weight.device
            )
            aligned_m = ((m + 127) // 128) * 128
            blockscale_offsets = torch.arange(
                0,
                num_experts * aligned_m,
                aligned_m,
                dtype=torch.int32,
                device=weight.device,
            )
            qweight = torch.empty_like(weight_flat, dtype=torch.float8_e4m3fn)
            scale = torch.empty(
                (num_experts * aligned_m, k // 32),
                dtype=torch.uint8,
                device=weight.device,
            )
            es_sm100_mxfp8_blockscaled_grouped_quant(
                weight_flat,
                problem_sizes,
                expert_offsets,
                blockscale_offsets,
                qweight,
                scale,
            )
            qweight = qweight.view_as(weight)
            scale = scale.view(num_experts, aligned_m, k // 32)
            if aligned_m != m:
                scale = scale[:, :m, :]
            return qweight, scale

        def _swizzle_mxfp8_sf(scale, num_warps):
            from triton_kernels.tensor import convert_layout, wrap_torch_tensor
            from triton_kernels.tensor_details import layout

            scale_layout, scale_layout_opts = (
                layout.make_default_matmul_mxfp4_w_scale_layout(
                    mx_axis=1, num_warps=num_warps
                )
            )
            scale = scale.transpose(-2, -1)
            scale = convert_layout(
                wrap_torch_tensor(scale), scale_layout, **scale_layout_opts
            )
            return scale

        def _swizzle_with_triton_kernel(
            weight_shape: tuple[int, int, int], scale: torch.Tensor
        ):
            num_experts, m, k = weight_shape
            aligned_m = ((m + 127) // 128) * 128
            scale = scale.view(num_experts, aligned_m, k // 32)
            num_warps = 8
            scale = _swizzle_mxfp8_sf(scale, num_warps)
            scale = scale.data.view(num_experts, aligned_m, k // 32)
            return scale

        def _quantize_and_swizzle_with_triton_kernel(weight: torch.Tensor):

            weight = weight.contiguous()
            _, _, k = weight.shape
            assert k % 32 == 0, f"{k=} must be divisible by 32 for MXFP8"

            weight_flat = weight.view(-1, k).contiguous()
            qweight, scale = mxfp8_group_quantize(weight_flat)
            qweight = qweight.view_as(weight)
            scale = _swizzle_with_triton_kernel(weight.shape, scale)
            return qweight, scale

        if quantize:
            if get_moe_runner_backend().is_cutlass():
                w13_q, w13_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_cutlass_es_kernel(
                    layer.w2_weight.data
                )
            else:
                w13_q, w13_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w13_weight.data
                )
                w2_q, w2_s = _quantize_and_swizzle_with_triton_kernel(
                    layer.w2_weight.data
                )
        else:
            w13_q = layer.w13_weight.data
            w2_q = layer.w2_weight.data
            w13_s = _swizzle_with_triton_kernel(
                layer.w13_weight.data.shape, layer.w13_weight_scale_inv.data
            )
            w2_s = _swizzle_with_triton_kernel(
                layer.w2_weight.data.shape, layer.w2_weight_scale_inv.data
            )

        # Keep parameter objects to preserve weight_loader attrs for hot reload.
        # Prefer in-place copy; rebind only when shape/dtype changes (online quantize).
        def _copy_or_rebind(param: Parameter, new_value: torch.Tensor) -> None:
            if (
                param.data.shape == new_value.shape
                and param.data.dtype == new_value.dtype
            ):
                param.data.copy_(new_value)
            else:
                param.data = new_value

        _copy_or_rebind(layer.w13_weight, w13_q)
        _copy_or_rebind(layer.w2_weight, w2_q)
        _copy_or_rebind(layer.w13_weight_scale_inv, w13_s)
        _copy_or_rebind(layer.w2_weight_scale_inv, w2_s)
        layer.w13_weight.requires_grad_(False)
        layer.w2_weight.requires_grad_(False)
        layer.w13_weight_scale_inv.requires_grad_(False)
        layer.w2_weight_scale_inv.requires_grad_(False)
        layer.w13_weight_scale_inv.format_ue8m0 = True
        layer.w2_weight_scale_inv.format_ue8m0 = True
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if _is_hip and _use_hip_int4:
            self.process_weights_hip_int4(layer)
            return

        # Block quant doesn't need to process weights after loading
        if self.block_quant:
            self.process_weights_after_loading_block_quant(layer)
            return

        # If checkpoint is fp16 or bfloat16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If ROCm, fp8_dtype will be float8_e4m3fnuz (MI300x HW)
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_local_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.num_local_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                    layer.w2_input_scale
                ):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. "
                    )
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False
                )
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False
                )

            # If ROCm, normalize the weights and scales to e4m3fnuz
            if _is_fp8_fnuz:
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

            if _is_hip:
                self.process_weights_hip_scale_padding(layer)

            # Align FP8 weights to FlashInfer per-tensor kernel layout if enabled
            if get_moe_runner_backend().is_flashinfer_trtllm():
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    align_fp8_moe_weights_for_flashinfer_trtllm,
                )

                align_fp8_moe_weights_for_flashinfer_trtllm(layer)
            return

    def process_weights_hip_int4(self, layer: Module):
        # TODO: _use_aiter: add after triton kernel added
        # INT4-FP8 (INT4 MoE Weight, FP8 Compute)
        # Weight Permutation
        layer.w13_weight = torch.nn.Parameter(
            shuffle_weight(layer.w13_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()
        layer.w2_weight = torch.nn.Parameter(
            shuffle_weight(layer.w2_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()

        # INT4-FP8 : offset INT4 w13_weight_scale1 to single w13_weight_scale
        # Fp8 moe kernel needs single fp8 w13_weight_scale for w13 per expert.
        # We won't do requant each expert's fp8 weight (not direct available),
        # instead we adjust half of INT4 w13_weight_scale1 numbers
        assert layer.w13_weight_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        for expert_id in range(layer.num_local_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(2):
                if layer.w13_weight_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (
                        layer.w13_weight_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    layer.w13_weight_scale1[expert_id][
                        start : start + shard_size
                    ] *= int4_rescale
                start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        # special hack to asm_moe, which takes (weight_scale1 * weight_scale) as post GEMM scaling
        # optimal design - shall apply per-column weight_scale1 before GEMM, and weight_scale post
        for expert_id in range(layer.num_local_experts):
            layer.w13_weight_scale1[expert_id] *= max_w13_scales[expert_id]
            layer.w2_weight_scale1[expert_id] *= layer.w2_weight_scale[expert_id]

    def process_weights_hip_scale_padding(self, layer: Module):
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            padding_size,  # Avoid circular import
        )

        if _use_aiter:
            layer.w13_weight = torch.nn.Parameter(
                shuffle_weight(layer.w13_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                shuffle_weight(layer.w2_weight.data, (16, 16)),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

            # ROCm (_use_aiter): using column-wise scaling
            layer.w13_weight_scale1 *= layer.w13_weight_scale.unsqueeze(-1)
            layer.w2_weight_scale1 *= layer.w2_weight_scale.unsqueeze(-1)
        elif get_bool_env_var("SGLANG_MOE_PADDING"):
            # If ROCm, apply weight padding (min. Mem channel contention) only if set
            layer.w13_weight = torch.nn.Parameter(
                F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()
            layer.w2_weight = torch.nn.Parameter(
                F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                requires_grad=False,
            )
            torch.cuda.empty_cache()

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()

        if moe_runner_backend.is_auto():
            if self.is_deepgemm_moe_runner_backend_enabled():
                moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON
        if (
            moe_runner_backend.is_deep_gemm()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_flashinfer_trtllm()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # TODO(cwan): refactor other backends
            pass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        moe_runner_config = self.moe_runner_config

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )

            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.FP8_W8A16,
                layer.w13_weight_scale_inv,  # w1_scale
                layer.w2_weight_scale_inv,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                self.quant_config.weight_block_size,  # block_size
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)

        if _is_hip:
            ret = self.maybe_apply_hip_fused_experts(
                layer,
                x,
                dispatch_output.topk_output,
                moe_runner_config.activation,
                moe_runner_config.no_combine,
            )
            if ret is not None:
                return StandardCombineInput(hidden_states=ret)

        if get_moe_runner_backend().is_cutlass():
            from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                symm_output = torch.empty_like(x)

            topk_weights, topk_ids, _ = dispatch_output.topk_output
            use_mxfp8 = getattr(self.quant_config, "use_mxfp8", False)
            output = cutlass_fused_experts_fp8(
                x,
                layer.w13_weight.transpose(1, 2),
                layer.w2_weight.transpose(1, 2),
                layer.w13_weight_scale_inv.transpose(1, 2),
                layer.w2_weight_scale_inv.transpose(1, 2),
                topk_weights,
                topk_ids,
                self.ab_strides1,
                self.c_strides1,
                self.ab_strides2,
                self.c_strides2,
                self.workspace,
                self.a_ptr,
                self.b_ptr,
                self.out_ptr,
                self.a_scales_ptr,
                self.b_scales_ptr,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                use_fp8_blockscale=True,
                use_mxfp8=use_mxfp8,
                output=symm_output,
                enable_es=(use_mxfp8, use_mxfp8),
            )
            return StandardCombineInput(hidden_states=output)

        if self.runner.runner_backend.is_deep_gemm():

            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

            if self.block_quant:
                block_shape = self.quant_config.weight_block_size
                w13_scale = layer.w13_weight_scale_inv
                w2_scale = layer.w2_weight_scale_inv
            else:
                # Convert per-tensor quant to per-block quant by repeating scales for forward_deepgemm
                scale_block_size = 128
                block_shape = [scale_block_size, scale_block_size]
                w13_scale_n = (w13_weight.shape[1] - 1) // scale_block_size + 1
                w13_scale_k = (w13_weight.shape[2] - 1) // scale_block_size + 1
                w13_scale = (
                    layer.w13_weight_scale.unsqueeze(1)
                    .repeat_interleave(w13_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w13_scale_k, dim=2)
                )
                w2_scale_n = (w2_weight.shape[1] - 1) // scale_block_size + 1
                w2_scale_k = (w2_weight.shape[2] - 1) // scale_block_size + 1
                w2_scale = (
                    layer.w2_weight_scale.unsqueeze(1)
                    .repeat_interleave(w2_scale_n, dim=1)
                    .unsqueeze(2)
                    .repeat_interleave(w2_scale_k, dim=2)
                )
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=True,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                block_shape=block_shape,
            )
        elif self.runner.runner_backend.is_flashinfer_trtllm():
            # FlashInfer TRT-LLM backend only supports fused execution and consumes
            # router logits directly (no separate apply_with_router_logits needed).
            global_num_experts = int(getattr(layer, "num_experts"))
            num_local_experts = int(getattr(layer, "num_local_experts"))
            moe_ep_rank = int(getattr(layer, "moe_ep_rank"))

            quant_info = FlashInferTrtllmFp8MoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                global_num_experts=global_num_experts,
                local_expert_offset=moe_ep_rank * num_local_experts,
                local_num_experts=num_local_experts,
                intermediate_size=layer.w2_weight.shape[2],
                routing_method_type=int(
                    getattr(layer, "routing_method_type", RoutingMethodType.DeepSeekV3)
                ),
                block_quant=self.block_quant,
                weight_block_k=(
                    None
                    if self.quant_config.weight_block_size is None
                    else self.quant_config.weight_block_size[1]
                ),
                w13_weight_scale_inv=(
                    layer.w13_weight_scale_inv if self.block_quant else None
                ),
                w2_weight_scale_inv=(
                    layer.w2_weight_scale_inv if self.block_quant else None
                ),
                w13_input_scale=layer.w13_input_scale if not self.block_quant else None,
                output1_scales_scalar=(
                    getattr(layer, "output1_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output1_scales_gate_scalar=(
                    getattr(layer, "output1_scales_gate_scalar", None)
                    if not self.block_quant
                    else None
                ),
                output2_scales_scalar=(
                    getattr(layer, "output2_scales_scalar", None)
                    if not self.block_quant
                    else None
                ),
            )
        elif self.runner.runner_backend.is_triton():
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                b13=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                use_fp8_w8a8=True,
                w13_scale=(
                    layer.w13_weight_scale_inv
                    if self.block_quant
                    else layer.w13_weight_scale
                ),
                w2_scale=(
                    layer.w2_weight_scale_inv
                    if self.block_quant
                    else layer.w2_weight_scale
                ),
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=self.quant_config.weight_block_size,
            )
        else:
            raise NotImplementedError(
                "Unsupported runner backend: %s" % self.runner.runner_backend
            )

        return self.runner.run(dispatch_output, quant_info)

    def _ensure_cutlass_buffers_initialized(self, layer: Module) -> None:
        if getattr(self, "_cutlass_buffers_ready", False):
            return

        device = layer.w13_weight.device
        num_experts = layer.w13_weight.shape[0]
        hidden_size = layer.w2_weight.shape[1]
        intermediate_size_per_partition = layer.intermediate_size_per_partition

        self.ab_strides1 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.c_strides1 = torch.full(
            (num_experts,),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.ab_strides2 = torch.full(
            (num_experts,),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts,), hidden_size, device=device, dtype=torch.int64
        )
        self.workspace = torch.empty(90000, device=device, dtype=torch.uint8)
        self.a_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.out_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.a_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.b_scales_ptr = torch.empty(num_experts, device=device, dtype=torch.int64)
        self.expert_offsets = torch.empty(
            num_experts + 1, device=device, dtype=torch.int32
        )
        self.problem_sizes1 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )
        self.problem_sizes2 = torch.empty(
            num_experts, 3, device=device, dtype=torch.int32
        )

        self._cutlass_buffers_ready = True

    def maybe_apply_hip_fused_experts(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        activation: str = "silu",
        no_combine: bool = False,
    ) -> Optional[torch.Tensor]:
        topk_weights, topk_ids, _ = topk_output
        if _use_hip_int4:
            # TODO: add triton kernel and add check _use_aiter
            assert not no_combine, f"{no_combine=} is not supported."
            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                quant_type=QuantType.per_Token,
                w1_scale=layer.w13_weight_scale1,
                w2_scale=layer.w2_weight_scale1,
                activation=(
                    ActivationType.Silu if activation == "silu" else ActivationType.Gelu
                ),
            )

        if _use_aiter:
            assert not no_combine, f"{no_combine=} is not supported."
            if self.block_quant:
                return fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    w1_scale=layer.w13_weight_scale_inv,
                    w2_scale=layer.w2_weight_scale_inv,
                    quant_type=QuantType.per_128x128,
                    activation=(
                        ActivationType.Silu
                        if activation == "silu"
                        else ActivationType.Gelu
                    ),
                    expert_mask=layer.expert_mask_gpu,
                )
            else:
                return fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    quant_type=QuantType.per_Token,
                    w1_scale=layer.w13_weight_scale1,
                    w2_scale=layer.w2_weight_scale1,
                    activation=(
                        ActivationType.Silu
                        if activation == "silu"
                        else ActivationType.Gelu
                    ),
                    expert_mask=layer.expert_mask_gpu,
                )
        return None


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
