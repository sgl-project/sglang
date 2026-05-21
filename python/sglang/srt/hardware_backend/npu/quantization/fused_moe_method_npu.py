from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class _NPUFusedMoEMethodBase(FusedMoEMethodBase):
    """Base class for NPU fused MoE methods with common helpers."""

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config

    def _set_dispatcher_output_dtype(self, layer: torch.nn.Module, dtype: str) -> None:
        """Set dispatcher output dtype if the layer has a dispatcher."""
        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": dtype})

    def _grouped_matmul(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        original_dtype: torch.dtype,
        **scale_args,
    ) -> torch.Tensor:
        """Execute grouped matmul with given scale arguments."""
        weight = getattr(layer, f"{weight_prefix}_weight")
        return torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[weight],
            **scale_args,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]


# ---------------------------------------------------------------------------
#  NPUW4A4Int4DynamicMoEMethod
# ---------------------------------------------------------------------------
class NPUW4A4Int4DynamicMoEMethod(_NPUFusedMoEMethodBase):
    """W4A4 dynamic MoE – weights are int4, activations are int4."""

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale")
        offset: Optional[torch.Tensor] = getattr(
            layer, f"{weight_prefix}_weight_offset", None
        )

        weight.data = npu_format_cast(weight.data.transpose(1, 2))
        weight.data = self._pack_to_int32(weight.data.to(torch.int32)).contiguous()

        # Pack scale into uint64 format (two fp32 values per int64)
        scale_np = scale.data.cpu().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            torch.nn.Parameter(
                scale_uint64_tensor.squeeze(-1).contiguous(), requires_grad=False
            ),
        )

        if offset is not None:
            setattr(
                layer,
                f"{weight_prefix}_weight_offset",
                torch.nn.Parameter(
                    offset.data.squeeze(-1).contiguous(), requires_grad=False
                ),
            )

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def _pack_to_int32(self, weight: torch.Tensor) -> torch.Tensor:
        # pack 8 int4 into one int32; each int32 represents a single int4 value
        assert (
            weight.shape[-1] % 8 == 0
        ), "the last dim of weight must be divisible by 8 for int4 packing"
        new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(weight.flatten(0, 1))
        new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        return new_weight

    def apply(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        original_dtype: torch.dtype,
        weight_prefix: str,
    ) -> torch.Tensor:
        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        scale_args = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        return self._grouped_matmul(
            layer,
            weight_prefix,
            hidden_states,
            expert_tokens,
            original_dtype,
            **scale_args,
        )

    def quant_activations(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(
            hidden_states, dst_type=torch.quint4x2
        )
        return hidden_states, pertoken_scale


# ---------------------------------------------------------------------------
#  NPUW8A8Int8DynamicMoEMethod
# ---------------------------------------------------------------------------
class NPUW8A8Int8DynamicMoEMethod(_NPUFusedMoEMethodBase):
    """W8A8 dynamic MoE – weights are int8, activations in int8."""

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        weight = getattr(layer, f"{weight_prefix}_weight")
        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        offset = getattr(layer, f"{weight_prefix}_weight_offset", None)

        weight.data = npu_format_cast(weight.data.transpose(1, 2)).contiguous()

        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            torch.nn.Parameter(
                scale.data.squeeze(-1).to(dtype=torch.bfloat16).contiguous(),
                requires_grad=False,
            ),
        )

        if offset is not None:
            setattr(
                layer,
                f"{weight_prefix}_weight_offset",
                torch.nn.Parameter(
                    offset.data.squeeze(-1).contiguous(), requires_grad=False
                ),
            )

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "int8")

    def apply(
        self,
        layer,
        hidden_states,
        expert_tokens,
        pertoken_scale,
        original_dtype,
        weight_prefix,
    ):
        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        scale_args = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        return self._grouped_matmul(
            layer,
            weight_prefix,
            hidden_states,
            expert_tokens,
            original_dtype,
            **scale_args,
        )

    def quant_activations(
        self,
        hidden_states,
    ):
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(
            hidden_states, dst_type=torch.int8
        )
        return hidden_states, pertoken_scale


# ---------------------------------------------------------------------------
#  NPUW4A8Int8DynamicMoEMethod
# ---------------------------------------------------------------------------
class NPUW4A8Int8DynamicMoEMethod(_NPUFusedMoEMethodBase):
    """W4A8 dynamic MoE – weights are int4, activations quantized to int8."""

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
        is_per_channel_weight: bool = False,
        activation_use_clip: bool = False,
    ):
        super().__init__(quant_config)
        self.is_per_channel_weight = is_per_channel_weight
        self.activation_use_clip = activation_use_clip

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._process_weight_group(layer, weight_prefix)

        # Dispatcher dtype is set after the final weight group (w2) is processed
        if weight_prefix == "w2":
            self._set_dispatcher_output_dtype(layer, "int8")

    def _process_weight_group(self, layer, prefix):
        weight = getattr(layer, f"{prefix}_weight")
        scale = getattr(layer, f"{prefix}_weight_scale")
        scale_second = getattr(layer, f"{prefix}_weight_scale_second", None)
        bias = getattr(layer, f"{prefix}_bias", None)

        weight.data = npu_format_cast(weight.data.transpose(1, 2).contiguous())
        weight.data = self._pack_to_int32(weight.data)
        setattr(
            layer, f"{prefix}_weight", torch.nn.Parameter(weight, requires_grad=False)
        )

        if not self.activation_use_clip:
            scale_processed, bias = self._process_scale(
                weight, scale, scale_second, self.is_per_channel_weight
            )
            setattr(
                layer,
                f"{prefix}_weight_scale",
                torch.nn.Parameter(
                    scale_processed.squeeze(-1).contiguous(), requires_grad=False
                ),
            )
            if scale_second is not None:
                delattr(layer, f"{prefix}_weight_scale_second")
                delattr(layer, f"{prefix}_weight_offset_second")
        else:
            scale_processed = scale.data.squeeze(-1).contiguous().unsqueeze(1)
            setattr(
                layer,
                f"{prefix}_weight_scale",
                torch.nn.Parameter(scale_processed, requires_grad=False),
            )
            if bias is not None:
                setattr(
                    layer,
                    f"{prefix}_scale_bias",
                    torch.nn.Parameter(
                        bias.data.transpose(1, 2).contiguous().sum(dim=1),
                        requires_grad=False,
                    ),
                )

    def _process_scale(self, weight, scale, per_group_scale, is_per_channel):
        scale = scale.transpose(1, 2).contiguous()
        if is_per_channel:
            scale_np = scale.cpu().numpy()
            scale_np.dtype = np.uint32
            scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
            return scale_uint64_tensor, None

        # Per‑group: multiply channel and group scales, then pack into uint64
        per_group_scale = per_group_scale.transpose(1, 2).contiguous()
        group_num, k, n = weight.shape
        n = n * 2  # packed weight halves the column dimension
        per_group_scale = per_group_scale.reshape(group_num, -1, n)
        group_num, quantgroup_num, n = per_group_scale.shape

        scale_fp32 = (scale * per_group_scale).to(torch.float16).to(torch.float32)
        scale_fp32_np = scale_fp32.cpu().numpy()
        scale_fp32_np.dtype = np.uint32
        sscale_uint64 = np.zeros((group_num, quantgroup_num, n * 2), dtype=np.uint32)
        sscale_uint64[..., ::2] = scale_fp32_np
        sscale_uint64_tensor = (
            torch.from_numpy(sscale_uint64.view(np.int64).copy())
            .reshape(group_num, quantgroup_num, n)
            .npu()
        )
        return sscale_uint64_tensor, None

    def _pack_to_int32(self, weight: torch.Tensor):
        assert weight.shape[-1] % 4 == 0, "last dim must be divisible by 4"
        return weight.view(torch.int32).contiguous()

    def quant_activations(self, hidden_states):
        hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(
            hidden_states, dst_type=torch.int8
        )
        return hidden_states, pertoken_scale

    def apply(
        self,
        layer,
        hidden_states,
        expert_tokens,
        pertoken_scale,
        original_dtype,
        weight_prefix,
    ):
        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        bias = getattr(layer, f"{weight_prefix}_scale_bias", None)
        scale_args = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        if bias is not None:
            scale_args["bias"] = [bias]
        return self._grouped_matmul(
            layer,
            weight_prefix,
            hidden_states,
            expert_tokens,
            original_dtype,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUW4A16Int4DynamicMoEMethod
# ---------------------------------------------------------------------------
class NPUW4A16Int4DynamicMoEMethod(_NPUFusedMoEMethodBase):
    """W4A16 dynamic MoE – weights are int4, activations stay in BF16."""

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        weight = getattr(layer, f"{weight_prefix}_weight")
        weight.data = weight.data.transpose(1, 2).contiguous()
        weight.data = self._pack_to_int32(weight.data)
        setattr(
            layer,
            f"{weight_prefix}_weight",
            torch.nn.Parameter(weight, requires_grad=False),
        )

        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        offset = getattr(layer, f"{weight_prefix}_weight_offset")
        scale.data = scale.data.transpose(-1, -2).contiguous()
        offset.data = offset.data.transpose(-1, -2).contiguous()
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            torch.nn.Parameter(scale, requires_grad=False),
        )
        setattr(
            layer,
            f"{weight_prefix}_weight_offset",
            torch.nn.Parameter(offset, requires_grad=False),
        )

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def _pack_to_int32(self, weight: torch.Tensor):
        if weight.dtype == torch.int32:
            if weight.shape[-1] % 8 == 0:
                new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
                    weight.flatten(0, 1)
                )
                return new_weight.view(weight.shape[0], weight.shape[1], -1)
        elif weight.dtype == torch.int8:
            assert weight.shape[-1] % 4 == 0, "last dim must be divisible by 4"
            return weight.view(torch.int32).contiguous()
        raise ValueError(f"Unsupported weight dtype: {weight.dtype}")

    def quant_activations(self, hidden_states):
        return hidden_states, None

    def apply(
        self,
        layer,
        hidden_states,
        expert_tokens,
        pertoken_scale,  # ignored (always None)
        original_dtype,
        weight_type,
    ):
        scale = getattr(layer, f"{weight_type}_weight_scale")
        offset = getattr(layer, f"{weight_type}_weight_offset")
        scale_args = {
            "antiquant_scale": [scale],
            "antiquant_offset": [offset],
        }
        return self._grouped_matmul(
            layer,
            weight_type,
            hidden_states,
            expert_tokens,
            original_dtype,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUW16A16DynamicMoEMethod
# ---------------------------------------------------------------------------
class NPUW16A16DynamicMoEMethod(_NPUFusedMoEMethodBase):
    """W16A16 dynamic MoE – all computations in BF16, no quantization."""

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        weight = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(weight.data.transpose(1, 2).contiguous())
        setattr(
            layer,
            f"{weight_prefix}_weight",
            torch.nn.Parameter(weight, requires_grad=False),
        )

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def quant_activations(self, hidden_states):
        return hidden_states, None

    def apply(
        self,
        layer,
        hidden_states,
        expert_tokens,
        pertoken_scale,  # ignored
        original_dtype,
        weight_type,
    ):
        return self._grouped_matmul(
            layer, weight_type, hidden_states, expert_tokens, original_dtype
        )
