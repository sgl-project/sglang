# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

try:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )

    MARLIN_FP8_AVAILABLE = True
except ImportError:
    MARLIN_FP8_AVAILABLE = False

    def dummy_func(*args, **kwargs):
        raise ImportError(
            "marlin FP8 requires some operators from vllm. Please install vllm."
        )

    apply_fp8_marlin_linear = prepare_fp8_layer_for_marlin = dummy_func


from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import (
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
    cutlass_fp8_supported,
    dispatch_w8a8_block_fp8_linear,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    log_info_on_rank0,
    print_warning_once,
    set_weight_attrs,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

_is_fp8_fnuz = is_fp8_fnuz()

_use_hip_int4 = get_bool_env_var("SGLANG_INT4_WEIGHT")
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_hip:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.fused_moe_bf16_asm import asm_moe, ck_moe_2stages
    from aiter.ops.shuffle import shuffle_weight

if not (_is_cuda or _is_npu or (_is_cpu and _is_cpu_amx_available)):
    from vllm._custom_ops import scaled_fp8_quant


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
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            log_info_on_rank0(logger, "Detected fp8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
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
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return Fp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (
            get_bool_env_var("SGLANG_FORCE_FP8_MARLIN") and MARLIN_FP8_AVAILABLE
        )
        # Disable marlin for ROCm
        if _is_hip:
            self.use_marlin = False

        self.block_quant = self.quant_config.weight_block_size is not None
        if self.block_quant:
            # Marlin doesn't support block-wise fp8
            self.use_marlin = False

        self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()

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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
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

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
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
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                assert self.quant_config.activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
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
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        if self.block_quant:
            # If ROCm, normalize the weights and scales to e4m3fnuz
            if _is_fp8_fnuz:
                # activation_scheme: dynamic
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=None,
                )

                layer.input_scale = None
            else:
                weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(
                weight_scale, requires_grad=False
            )
            return

        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        # If checkpoint not serialized fp8, quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            if self.cutlass_fp8_supported or self.use_marlin:
                # apply per-channel quantization default, as cutlass sgl-kernel and marlin only support per-channel scale
                qweight, weight_scale = per_token_group_quant_fp8(
                    layer.weight, layer.weight.shape[-1]
                )
                weight_scale = weight_scale.t().contiguous()
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
            layer.weight_scale = torch.nn.Parameter(
                layer.weight_scale.data, requires_grad=False
            )
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = torch.nn.Parameter(
                    layer.input_scale.data, requires_grad=False
                )

            # cutlass sgl-kernel and marlin only support per-channel scale
            if self.cutlass_fp8_supported or self.use_marlin:
                weight = layer.weight
                weight_scale = convert_to_channelwise(
                    layer.weight_scale, layer.logical_widths
                )
            else:
                # Dequant -> Quant with max scale so we can run per tensor.
                weight = layer.weight
                weight_scale = layer.weight_scale
                # If ROCm, normalize the weights and scales to e4m3fnuz
                if _is_fp8_fnuz:
                    weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                        weight=weight,
                        weight_scale=weight_scale,
                        input_scale=layer.input_scale,
                    )
                    if input_scale is not None:
                        layer.input_scale = Parameter(input_scale, requires_grad=False)

                weight_scale, weight = requantize_with_max_scale(
                    weight=weight,
                    weight_scale=weight_scale,
                    logical_widths=layer.logical_widths,
                )

            # Update layer with new values.
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            if self.quant_config.activation_scheme == "static":
                layer.input_scale = Parameter(
                    layer.input_scale.max(), requires_grad=False
                )

        if self.use_marlin:
            prepare_fp8_layer_for_marlin(layer)
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

        if self.block_quant:
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
            use_per_token_if_dynamic=False,
        )


class Fp8MoEMethod:
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
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
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS
        if _is_hip and _use_hip_int4:
            # INT4 MoE weight - INT32 packed
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size,
                    hidden_size // 8,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size // 8, dtype=params_dtype
                ),
                requires_grad=False,
            )
        else:
            w13_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
                ),
                requires_grad=False,
            )
            w2_weight = torch.nn.Parameter(
                torch.empty(
                    num_experts, hidden_size, intermediate_size, dtype=params_dtype
                ),
                requires_grad=False,
            )

        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
            if (
                get_bool_env_var("SGLANG_CUTLASS_MOE")
                and self.cutlass_fp8_supported
                and is_sm100_supported()
            ):
                self.ab_strides1 = torch.full(
                    (num_experts,),
                    hidden_size,
                    device=w13_weight.device,
                    dtype=torch.int64,
                )
                self.c_strides1 = torch.full(
                    (num_experts,),
                    2 * intermediate_size,
                    device=w13_weight.device,
                    dtype=torch.int64,
                )
                self.ab_strides2 = torch.full(
                    (num_experts,),
                    intermediate_size,
                    device=w2_weight.device,
                    dtype=torch.int64,
                )
                self.c_strides2 = torch.full(
                    (num_experts,),
                    hidden_size,
                    device=w2_weight.device,
                    dtype=torch.int64,
                )
                self.workspace = torch.empty(
                    90000, device=w13_weight.device, dtype=torch.uint8
                )
                self.a_ptr = torch.empty(
                    num_experts, device=w13_weight.device, dtype=torch.int64
                )
                self.b_ptr = torch.empty(
                    num_experts, device=w13_weight.device, dtype=torch.int64
                )
                self.out_ptr = torch.empty(
                    num_experts, device=w13_weight.device, dtype=torch.int64
                )
                self.a_scales_ptr = torch.empty(
                    num_experts, device=w13_weight.device, dtype=torch.int64
                )
                self.b_scales_ptr = torch.empty(
                    num_experts, device=w13_weight.device, dtype=torch.int64
                )
                self.expert_offsets = torch.empty(
                    num_experts + 1, device=w13_weight.device, dtype=torch.int32
                )
                self.problem_sizes1 = torch.empty(
                    num_experts, 3, device=w13_weight.device, dtype=torch.int32
                )
                self.problem_sizes2 = torch.empty(
                    num_experts, 3, device=w13_weight.device, dtype=torch.int32
                )

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
                    torch.ones(num_experts, 2 * intermediate_size, dtype=torch.float32),
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

    def process_weights_after_loading(self, layer: Module) -> None:
        if _is_hip and _use_hip_int4:
            self.process_weights_hip_int4(layer)
            return

        # Block quant doesn't need to process weights after loading
        if self.block_quant:
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
                # Pre-shuffle weights
                layer.w13_weight.data = shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
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
                    layer.num_experts, dtype=torch.float32, device=w13_weight.device
                ),
                requires_grad=False,
            )
            for expert in range(layer.num_experts):
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
            for expert_id in range(layer.num_experts):
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
        for expert_id in range(layer.num_experts):
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
        for expert_id in range(layer.num_experts):
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

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        if _is_hip:
            ret = self.maybe_apply_hip_fused_experts(
                layer,
                x,
                topk_weights,
                topk_ids,
                activation,
                no_combine,
            )
            if ret is not None:
                return ret

        if (
            get_bool_env_var("SGLANG_CUTLASS_MOE")
            and self.cutlass_fp8_supported
            and self.block_quant
            and is_sm100_supported()
        ):
            from sglang.srt.layers.moe.cutlass_moe import cutlass_fused_experts_fp8

            return cutlass_fused_experts_fp8(
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
            )
        # Expert fusion with FP8 quantization
        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace and not no_combine,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=True,
            w1_scale=(
                layer.w13_weight_scale_inv
                if self.block_quant
                else layer.w13_weight_scale
            ),
            w2_scale=(
                layer.w2_weight_scale_inv if self.block_quant else layer.w2_weight_scale
            ),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )

    def maybe_apply_hip_fused_experts(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        no_combine: bool = False,
    ) -> Optional[torch.Tensor]:
        if _use_hip_int4:
            # TODO: add triton kernel and add check _use_aiter
            assert not no_combine, f"{no_combine=} is not supported."
            return ck_moe_2stages(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                QuantType.per_Token,
                layer.w13_weight_scale1,
                layer.w2_weight_scale1,
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
                    expert_mask=None,
                )
            else:
                return ck_moe_2stages(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    QuantType.per_Token,
                    layer.w13_weight_scale1,
                    layer.w2_weight_scale1,
                    activation=(
                        ActivationType.Silu
                        if activation == "silu"
                        else ActivationType.Gelu
                    ),
                )
        return None


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
