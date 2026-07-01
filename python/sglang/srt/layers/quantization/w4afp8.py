from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        StandardDispatchOutput,
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class W4AFp8Config(QuantizationConfig):
    """Config class for MIXED_PRECISION W4AFp8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = True,
        is_checkpoint_w4afp8_serialized: bool = True,
        linear_activation_scheme: str = "dynamic",
        moe_activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_checkpoint_w4afp8_serialized = is_checkpoint_w4afp8_serialized
        if is_checkpoint_w4afp8_serialized:
            logger.warning("Detected w4afp8 checkpoint. Please note that")
        if moe_activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {moe_activation_scheme}")
        self.linear_activation_scheme = linear_activation_scheme
        self.moe_activation_scheme = moe_activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = [128, 128]
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "w4afp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W4AFp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        is_checkpoint_w4afp8_serialized = "w4afp8" in quant_method
        linear_activation_scheme = "dynamic"
        moe_activation_scheme = "dynamic"
        weight_block_size = [128, 128]
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            is_checkpoint_w4afp8_serialized=is_checkpoint_w4afp8_serialized,
            linear_activation_scheme=linear_activation_scheme,
            moe_activation_scheme=moe_activation_scheme,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W4AFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def interleave_scales(scales: torch.Tensor) -> torch.Tensor:
    """Interleave scales in groups of 4 similar to TRT-LLM implementation."""
    s_shape = scales.shape
    # Reshape to separate groups of 4
    alignment = 4 if s_shape[2] % 4 == 0 else 1
    scales_interleaved = scales.reshape(
        s_shape[0], s_shape[1], (s_shape[2] // alignment), alignment
    )
    # Permute dimensions to interleave
    scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
    # Reshape back to original dimensions but with interleaved values
    scales_interleaved = scales_interleaved.reshape(
        s_shape[0], s_shape[2] // alignment, s_shape[1] * alignment
    )
    return scales_interleaved.contiguous()


class W4AFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: W4AFp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        assert "weight_loader" in extra_weight_attrs

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition * 2,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Input scales
        w13_input_scale = torch.nn.Parameter(
            torch.ones((num_experts, 2), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        # Pre-populate the strides
        device = layer.w13_weight.device

        self.a_strides1 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides1 = torch.full(
            (num_experts, 3),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.a_strides2 = torch.full(
            (num_experts, 3),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2

        self.expert_offsets = torch.empty(
            (num_experts + 1), dtype=torch.int32, device=device
        )
        self.problem_sizes1 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )
        self.problem_sizes2 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )

        return

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer SM90 W4A8 expects fused W13 as [up_proj, gate_proj].
        return True

    @staticmethod
    def _get_flashinfer_w4a8_helpers():
        try:
            from flashinfer.fused_moe import (
                cutlass_fused_moe,
                interleave_moe_weights_for_sm90_mixed_gemm,
            )
            from flashinfer.fused_moe.core import ActivationType
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "W4AFP8 MoE requires flashinfer>=0.6.12 with PR 3084 "
                "SM90 mixed-input helpers."
            ) from exc

        return cutlass_fused_moe, interleave_moe_weights_for_sm90_mixed_gemm, ActivationType

    @staticmethod
    def _to_flashinfer_scale_dtype(scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if scale.dtype == dtype:
            return scale
        return scale.to(torch.bfloat16).view(dtype)

    @staticmethod
    def _dynamic_per_tensor_fp8_scale(x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8_scale

            scale = torch.empty(1, dtype=torch.float32, device=x.device)
            per_tensor_quant_fp8_scale(x.contiguous(), scale)
        else:
            scale = x.abs().max().to(torch.float32).reshape(1) / 448.0
        return torch.clamp(scale, min=1.0e-12)

    def process_weights_after_loading(self, layer: Module) -> None:
        _, interleave_moe_weights, _ = self._get_flashinfer_w4a8_helpers()

        if not layer.w13_weight.is_cuda or not layer.w2_weight.is_cuda:
            raise RuntimeError("FlashInfer W4AFP8 MoE requires CUDA weights.")

        w13_weight = interleave_moe_weights(
            layer.w13_weight.detach().contiguous().view(torch.uint8), "int4"
        )
        w2_weight = interleave_moe_weights(
            layer.w2_weight.detach().contiguous().view(torch.uint8), "int4"
        )
        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)

        w13_weight_scale = interleave_scales(layer.w13_weight_scale.to(torch.float32))
        w2_weight_scale = interleave_scales(layer.w2_weight_scale.to(torch.float32))
        layer.w13_weight_scale = Parameter(
            w13_weight_scale.to(torch.bfloat16).contiguous(), requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            w2_weight_scale.to(torch.bfloat16).contiguous(), requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        flashinfer_cutlass_fused_moe, _, ActivationType = (
            self._get_flashinfer_w4a8_helpers()
        )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor or 1.0
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor

        num_experts = layer.w13_weight.shape[0]
        hidden_size = x.shape[-1]
        intermediate_size = layer.w2_weight.shape[-1] * 2
        output_dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else x.dtype

        # Runtime per-tensor scale for the first FP8 activation. FlashInfer's
        # W4A8 fused API also needs a GEMM2 activation scale before entering the
        # kernel; the true dynamic GEMM2 scale depends on the internal SwiGLU
        # result, so use neutral scale 1.0 for this smoke-test path.
        a1_scale = self._dynamic_per_tensor_fp8_scale(x)
        a2_scale = torch.ones_like(a1_scale)

        fc31_act_scale = (
            torch.ones(
                (num_experts, hidden_size),
                dtype=x.dtype,
                device=x.device,
            )
            / a1_scale
        ).contiguous()
        fc2_act_scale = (
            torch.ones(
                (num_experts, intermediate_size, 1),
                dtype=x.dtype,
                device=x.device,
            )
            / a2_scale
        ).contiguous()

        empty = torch.empty(0, dtype=x.dtype, device=x.device)
        quant_scales = (
            self._to_flashinfer_scale_dtype(layer.w13_weight_scale, x.dtype),
            self._to_flashinfer_scale_dtype(layer.w2_weight_scale, x.dtype),
            self._to_flashinfer_scale_dtype(fc31_act_scale, x.dtype),
            self._to_flashinfer_scale_dtype(fc2_act_scale, x.dtype),
            empty,
            empty,
            a1_scale.expand(num_experts).contiguous(),
            a2_scale.expand(num_experts).contiguous(),
        )

        output = torch.empty(
            (x.shape[0], hidden_size),
            dtype=output_dtype,
            device=x.device,
        )
        flashinfer_cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            output_dtype=output_dtype,
            quant_scales=quant_scales,
            output=output,
            use_w4_group_scaling=True,
            use_packed_weights=True,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=max(1, int(x.shape[0])),
        )
        return StandardCombineInput(hidden_states=output)

    def apply_deepep_ll(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPLLDispatchOutput,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "W4AFP8 FlashInfer SM90 mixed-input MoE currently supports only "
            "standard dispatch. Disable DeepEP for this checkpoint."
        )

    def apply_deepep_normal(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPNormalDispatchOutput,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "W4AFP8 FlashInfer SM90 mixed-input MoE currently supports only "
            "standard dispatch. Disable DeepEP for this checkpoint."
        )
