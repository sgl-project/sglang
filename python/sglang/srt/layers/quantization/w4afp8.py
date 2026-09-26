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
        moe_activation_scheme: str = "static",
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
        return [torch.bfloat16, torch.float8_e4m3fn]

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
        moe_activation_scheme = "static"
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
        self.use_flashinfer = False

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

        from sglang.srt.layers.moe.utils import get_moe_runner_backend

        self.use_flashinfer = get_moe_runner_backend().is_flashinfer_cutlass()
        self.load_up_proj_weight_first = self.use_flashinfer
        if self.use_flashinfer:
            self._validate_flashinfer_config()
            from sglang.srt.layers.moe.utils import get_moe_a2a_backend

            if get_moe_a2a_backend().is_deepep() and layer.moe_ep_size <= 1:
                raise ValueError(
                    "FlashInfer W4AFP8 DeepEP requires expert parallel size > 1."
                )
            if params_dtype != torch.bfloat16:
                raise ValueError("FlashInfer W4AFP8 requires BF16 model activations.")
            if hidden_size % 128 or intermediate_size_per_partition % 128:
                raise ValueError(
                    "FlashInfer W4AFP8 requires group-128 aligned GEMM dimensions."
                )

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
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
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
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
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

        if self.use_flashinfer:
            return

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

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.use_flashinfer:
            self._process_flashinfer_weights(layer)
            if hasattr(layer, "dispatcher"):
                layer.dispatcher.set_quant_config(
                    {
                        "normal_dispatcher_output_dtype": "bf16",
                        "low_latency_dispatcher_output_dtype": "bf16",
                    }
                )
            return

        dtype = torch.bfloat16
        device = layer.w2_weight.device

        # Interleave w13_weight_scale (gate_up_proj)
        w13_weight_scale = layer.w13_weight_scale_inv.to(dtype)
        w13_weight_scale = interleave_scales(w13_weight_scale)
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)

        # Interleave w2_weight_scale (down_proj)
        w2_weight_scale = layer.w2_weight_scale_inv.to(dtype)
        w2_weight_scale = interleave_scales(w2_weight_scale)
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale, requires_grad=False)

        # Process input scales
        w13_input_scale_max = layer.w13_input_scale.max().to(torch.float32).item()
        new_w13_input_scale = torch.tensor(
            [w13_input_scale_max],
            dtype=torch.float32,
            device=device,
        )
        layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)

        w2_input_scale_max = layer.w2_input_scale.max().to(torch.float32).item()
        new_w2_input_scale = torch.tensor(
            [w2_input_scale_max], dtype=torch.float32, device=device
        )
        layer.w2_input_scale = Parameter(new_w2_input_scale, requires_grad=False)

        if hasattr(layer, "dispatcher"):
            # The normal kernel requantizes BF16 inputs with the checkpoint's
            # static activation scale. The low-latency kernel instead consumes
            # DeepEP's FP8 payload together with its per-token-group scales.
            layer.dispatcher.set_quant_config(
                {
                    "normal_dispatcher_output_dtype": "bf16",
                    "low_latency_dispatcher_output_dtype": "fp8",
                }
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer:
            from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass  # noqa: F401
            from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
            from sglang.srt.layers.moe.utils import MoeRunnerBackend

            if (
                moe_runner_config.activation != "silu"
                or not moe_runner_config.is_gated
                or moe_runner_config.apply_router_weight_on_input
                or moe_runner_config.no_combine
                or any(
                    getattr(moe_runner_config, name) is not None
                    for name in (
                        "gemm1_alpha",
                        "gemm1_beta",
                        "gemm1_clamp_limit",
                        "swiglu_limit",
                    )
                )
            ):
                raise ValueError(
                    "FlashInfer W4AFP8 requires gated SiLU with output router weights "
                    "and no activation overrides."
                )
            self.runner = MoeRunner(
                MoeRunnerBackend.FLASHINFER_CUTLASS, moe_runner_config
            )

    def _validate_flashinfer_config(self) -> None:
        import inspect

        from packaging.version import Version

        from sglang.srt.layers.moe.utils import get_moe_a2a_backend
        from sglang.srt.runtime_context import get_exec
        from sglang.srt.utils import is_flashinfer_available

        a2a = get_moe_a2a_backend()
        if a2a.is_deepep():
            if get_exec().moe.deepep_dispatcher_output_dtype not in ("auto", "bf16"):
                raise ValueError("FlashInfer W4AFP8 DeepEP requires BF16 dispatch.")
        elif not a2a.is_none():
            raise ValueError(
                "FlashInfer W4AFP8 supports A2A backends none or deepep only."
            )
        if (
            self.quant_config.group_size != 128
            or self.quant_config.moe_activation_scheme != "static"
        ):
            raise ValueError(
                "FlashInfer W4AFP8 requires group-128 weights and static activation scales."
            )
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise ValueError("FlashInfer W4AFP8 requires an SM90 GPU.")
        if not is_flashinfer_available():
            raise RuntimeError(
                "FlashInfer W4AFP8 requires flashinfer-python >= 0.6.18."
            )
        import flashinfer
        from flashinfer import fused_moe

        if Version(flashinfer.__version__) < Version("0.6.18") or not all(
            callable(getattr(fused_moe, name, None))
            for name in (
                "cutlass_fused_moe",
                "interleave_moe_weights_for_sm90_mixed_gemm",
                "interleave_moe_scales_for_sm90_mixed_gemm",
            )
        ):
            raise RuntimeError(
                "FlashInfer W4AFP8 requires the mixed-GEMM APIs from flashinfer-python >= 0.6.18."
            )

        required_kwargs = {
            "use_w4_group_scaling",
            "use_packed_weights",
            "use_fused_finalize",
        }
        if not required_kwargs.issubset(
            inspect.signature(fused_moe.cutlass_fused_moe).parameters
        ):
            raise RuntimeError(
                "FlashInfer cutlass_fused_moe lacks the packed W4AFP8 API."
            )

    def _process_flashinfer_weights(self, layer: Module) -> None:
        if getattr(layer, "_flashinfer_w4afp8_prepared", False):
            return

        from flashinfer.fused_moe import (
            interleave_moe_scales_for_sm90_mixed_gemm,
            interleave_moe_weights_for_sm90_mixed_gemm,
        )

        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
        )

        # The loader already placed FC1 in [up; gate] order, including scales.
        hidden = layer.w13_weight.shape[-1] * 2
        intermediate = layer.w2_weight.shape[-1] * 2
        experts = layer.w2_weight.shape[0]
        for name in ("w13_weight", "w2_weight"):
            packed = getattr(layer, name).data.view(torch.uint8)
            setattr(
                layer,
                name,
                Parameter(
                    interleave_moe_weights_for_sm90_mixed_gemm(packed, "int4"),
                    requires_grad=False,
                ),
            )
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            scales = getattr(layer, name).to(torch.bfloat16)
            setattr(
                layer,
                name,
                Parameter(
                    interleave_moe_scales_for_sm90_mixed_gemm(scales, 128),
                    requires_grad=False,
                ),
            )
        # Match the native path: one static scale per GEMM over loaded experts.
        for name in ("w13_input_scale", "w2_input_scale"):
            scale = getattr(layer, name).max().float().reshape(1)
            if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
                raise ValueError(
                    "FlashInfer W4AFP8 activation scales must be finite and positive."
                )
            setattr(layer, name, Parameter(scale, requires_grad=False))
        a1, a2 = layer.w13_input_scale, layer.w2_input_scale
        self.flashinfer_quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="w4afp8",
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            quant_scales=[
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                a1.reciprocal().to(torch.bfloat16).expand(hidden).contiguous(),
                a2.reciprocal().to(torch.bfloat16).expand(intermediate).contiguous(),
                torch.empty(0, dtype=torch.bfloat16, device=a1.device),
                torch.empty(0, dtype=torch.bfloat16, device=a2.device),
                a1.expand(experts).contiguous(),
                a2.expand(experts).contiguous(),
            ],
            moe_tp_size=layer.moe_tp_size,
            moe_tp_rank=layer.moe_tp_rank,
            moe_ep_size=layer.moe_ep_size,
            moe_ep_rank=layer.moe_ep_rank,
        )
        layer._flashinfer_w4afp8_prepared = True

    def apply(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if self.use_flashinfer:
            return self.runner.run(dispatch_output, self.flashinfer_quant_info)

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        output = cutlass_w4a8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
        )
        return StandardCombineInput(hidden_states=output)

    def apply_deepep_ll(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPLLDispatchOutput,
    ) -> torch.Tensor:
        hidden_states, hidden_scales, topk_ids, _, masked_m, expected_m = (
            dispatch_output
        )

        if hidden_scales is None:
            raise RuntimeError(
                "W4AFP8 DeepEP low-latency requires FP8 dispatcher output "
                "with per-token-group scales."
            )

        from sglang.srt.layers.moe.cutlass_w4a8_moe import (
            cutlass_w4a8_moe_deepep_ll,
        )

        output = cutlass_w4a8_moe_deepep_ll(
            hidden_states,
            hidden_scales,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_ids,
            masked_m,
            layer.quant_method.a_strides1,
            layer.quant_method.b_strides1,
            layer.quant_method.c_strides1,
            layer.quant_method.a_strides2,
            layer.quant_method.b_strides2,
            layer.quant_method.c_strides2,
            layer.quant_method.s_strides13,
            layer.quant_method.s_strides2,
            layer.quant_method.expert_offsets,
            layer.quant_method.problem_sizes1,
            layer.quant_method.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            expected_m=expected_m,
        )

        return output

    def apply_deepep_normal(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPNormalDispatchOutput,
    ) -> torch.Tensor:
        hidden_states, topk_idx, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                "W4AFP8 DeepEP normal requires BF16 dispatcher output, "
                f"but got {hidden_states.dtype}."
            )

        num_tokens = hidden_states.shape[0]
        if num_tokens > 0:
            from sglang.srt.layers.moe.cutlass_w4a8_moe import (
                cutlass_w4a8_moe_deepep_normal,
            )

            return cutlass_w4a8_moe_deepep_normal(
                hidden_states,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                topk_weights,
                topk_idx,
                self.a_strides1,
                self.b_strides1,
                self.c_strides1,
                self.a_strides2,
                self.b_strides2,
                self.c_strides2,
                self.s_strides13,
                self.s_strides2,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                layer.w13_input_scale,
                layer.w2_input_scale,
            )
        else:
            return hidden_states
