from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
import torch_npu

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

import logging

from sglang.srt.hardware_backend.npu.moe.hidden_states_quant import HiddenStatesDynamicQuant
from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul

logger = logging.getLogger(__name__)


# DEPRECATED METHOD
# TODO: Remove in future realeses
def fused_moe_npu(
    x,
    w1,
    w2,
    topk_output,
    moe_runner_config,
):
    logger.warning_once(
        f"The fused_moe_npu method deprecated and will be removed in future releases"
    )
    topk_weights, topk_ids, _ = topk_output
    original_dtype = x.dtype
    num_tokens = x.shape[0]
    topk_weights = topk_weights.to(x.dtype)
    topk_ids = topk_ids.to(torch.int32)
    num_experts = w1.shape[0]
    top_k = topk_weights.shape[-1]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )

    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )

    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )

    expert_tokens = expert_tokens.to(torch.int64)

    # gmm1: gate_up_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1.permute(0, 2, 1)],
        bias=None,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # act_fn:
    if moe_runner_config.activation == "silu":
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    else:
        from sglang.srt.layers.activation import GeluAndMul

        hidden_states = GeluAndMul()(hidden_states)

    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2.permute(0, 2, 1)],
        bias=None,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    return final_hidden_states


class _NPUMoEMethodBase(FusedMoEMethodBase):
    """Base class for NPU MoE methods with common helpers."""

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        super().__init__()
        self.quant_config = quant_config

    @staticmethod
    def _set_dispatcher_output_dtype(layer: torch.nn.Module, dtype) -> None:
        """Set dispatcher output dtype if the layer has a dispatcher."""
        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": dtype})

    @staticmethod
    def _validate_weight_prefix(layer: torch.nn.Module, weight_prefix: str) -> None:
        """Ensure the required attributes exist on the layer for the given prefix."""
        required = [f"{weight_prefix}_weight"]
        for attr in required:
            if not hasattr(layer, attr):
                raise AttributeError(
                    f"Layer {layer} is missing required attribute '{attr}' for "
                    f"weight_prefix='{weight_prefix}'"
                )


# ---------------------------------------------------------------------------
#  NPUW4A4Int4DynamicMoEMethod
# ---------------------------------------------------------------------------
class NPUW4A4Int4MoEMethod(_NPUMoEMethodBase):
    """W4A4 dynamic MoE – weights are int4, activations are int4."""

    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(
            quant_dtype=torch.quint4x2
        )

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        # Process scale
        scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale")
        scale_np = scale.data.cpu().contiguous().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        processed_scale = torch.nn.Parameter(
            scale_uint64_tensor.squeeze(-1), requires_grad=False
        )
        setattr(layer, f"{weight_prefix}_weight_scale", processed_scale)

        # Process offset
        offset: Optional[torch.Tensor] = getattr(
            layer, f"{weight_prefix}_weight_offset", None
        )
        if offset is not None:
            processed_offset = torch.nn.Parameter(
                offset.data.squeeze(-1), requires_grad=False
            )
            setattr(layer, f"{weight_prefix}_weight_offset", processed_offset)

        # Process weight
        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        if not envs.SGLANG_NPU_W4A4_NEW_PACKING.get():
            weight.data = self._w4a4_pack_int4(weight.data)
        weight.data = weight.data.transpose(-2, -1).contiguous()
        weight.data = npu_format_cast(weight.data)
        weight.data = self._pack_to_int32(weight.data)

        # Set DeepEP dispatcher output dtype
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def _pack_int4(self, weight) -> torch.Tensor:
        """
        Pack int4 weight to int8 weight
        @param weight: torch.Tensor, int4 weight
        @return: torch.Tensor, int8 weight
        """
        weight = weight.to(torch.int8)
        e = 0  # number of experts
        if len(weight.shape) == 2:
            k, n = weight.shape
        elif len(weight.shape) == 3:
            e, k, n = weight.shape
        n_new = n // 2 + n % 2

        if n_new != n // 2:
            raise AssertionError("n dimension should be even")

        weight = weight.reshape(-1, 2)
        weight0 = weight[:, :1]
        weight1 = weight[:, 1:]

        weight1_4 = torch.bitwise_left_shift(weight1, 4)
        weight2_4 = weight0 & 0b00001111

        weight_add = torch.bitwise_or(weight1_4, weight2_4)
        if e == 0:
            weight_res = weight_add.reshape(k, n_new)
        else:
            weight_res = weight_add.reshape(e, k, n_new)
        return weight_res

    def _w4a4_pack_int4(self, save_quant_weight):
        """
        Pack int4 weight to int8 weight
        @param save_quant_weight: torch.Tensor, int4 weight
        @return: torch.Tensor, int8 weight
        """
        weight = save_quant_weight.transpose(-1, -2).contiguous()
        packed_weight_tensor = self._pack_int4(weight)
        packed_weight_tensor = packed_weight_tensor.transpose(-1, -2).contiguous()
        return packed_weight_tensor

    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32
        return weight.contiguous().view(torch.int32)

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUW4A4Mxfp4MoEMethod
# ---------------------------------------------------------------------------
class NPUW4A4Mxfp4MoEMethod(_NPUMoEMethodBase):
    """W4A4 MXFP4 MoE – weights & activations float4_e2m1fn_x2, scales float8_e8m0fnu."""

    def __init__(self, group_size: int = 32):
        super().__init__(quant_config=None)
        self.group_size = group_size
        self.matmul = GroupedMatmul()
        # Activation quantizer for float4
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.float4_e2m1fn_x2)

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        weight = getattr(layer, f"{weight_prefix}_weight")   # [E, N, K_packed] uint8
        scale  = getattr(layer, f"{weight_prefix}_weight_scale")  # [E, N, K_groups] uint8

        # 1) Weight: transpose to [E, K_packed, N] – no npu_format_cast needed
        weight.data = weight.data.transpose(1, 2).contiguous()

        # 2) Scale: pack into [E, K_groups//2, N, 2]
        g, n, k = scale.shape                     # k = hidden_size // group_size
        scale.data = scale.data.reshape(g, n, k // 2, 2).transpose(1, 2).contiguous()

        # 3) Store on layer and on self (for matmul lookup)
        setattr(layer, f"{weight_prefix}_weight",
                torch.nn.Parameter(weight.data, requires_grad=False))
        setattr(layer, f"{weight_prefix}_weight_scale",
                torch.nn.Parameter(scale.data, requires_grad=False))

        setattr(self, f"{weight_prefix}_weight", weight.data)
        setattr(self, f"{weight_prefix}_weight_scale", scale.data)

        # 4) Output dtype – reuse bf16 for now
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        weight_scale = getattr(self, f"{weight_prefix}_weight_scale", None)

        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(
                hidden_states
            )

        scale_args: Dict[str, Any] = {
            "scale": [weight_scale],
            "per_token_scale": [pertoken_scale],
            "scale_dtype": torch.float8_e8m0fnu,
            "per_token_scale_dtype": torch.float8_e8m0fnu,
        }

        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,          # float4_e2m1fn_x2
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUW4A8Int8MoEMethod
# ---------------------------------------------------------------------------
class NPUW4A8Int8MoEMethod(_NPUMoEMethodBase):
    """W4A8 MoE – weights are int4, activations quantized to int8."""

    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
        is_per_channel_weight: bool = False,
        activation_use_clip: bool = False,
    ):
        super().__init__(quant_config)
        self.is_per_channel_weight = is_per_channel_weight
        self.activation_use_clip = activation_use_clip
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.int8)

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        # Process scale (and bias if needed)
        scale = getattr(layer, f"{weight_prefix}_weight_scale")
        scale_second = getattr(layer, f"{weight_prefix}_weight_scale_second", None)
        bias = getattr(layer, f"{weight_prefix}_bias", None)

        if not self.activation_use_clip:
            # Process scale according to per-channel or per-group
            processed_scale = self._process_scale(
                getattr(layer, f"{weight_prefix}_weight"),
                scale,
                scale_second,
                self.is_per_channel_weight,
            )
            setattr(
                layer,
                f"{weight_prefix}_weight_scale",
                torch.nn.Parameter(processed_scale.squeeze(-1), requires_grad=False),
            )
            if scale_second is not None:
                delattr(layer, f"{weight_prefix}_weight_scale_second")
                delattr(layer, f"{weight_prefix}_weight_offset_second")
        else:
            # With clip: simple squeeze + unsqueeze
            processed_scale = scale.data.squeeze(-1).unsqueeze(1).contiguous()
            setattr(
                layer,
                f"{weight_prefix}_weight_scale",
                torch.nn.Parameter(processed_scale, requires_grad=False),
            )
            if bias is not None:
                setattr(
                    layer,
                    f"{weight_prefix}_scale_bias",
                    torch.nn.Parameter(
                        bias.data.transpose(1, 2).sum(dim=1).contiguous(),
                        requires_grad=False,
                    ),
                )

        # Process weight
        weight = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(weight.data.transpose(1, 2))
        weight.data = self._pack_to_int32(weight.data)
        setattr(
            layer,
            f"{weight_prefix}_weight",
            torch.nn.Parameter(weight, requires_grad=False),
        )

        # Set dispatcher output dtype
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "int8")

    def _process_scale(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        per_group_scale: Optional[torch.Tensor],
        is_per_channel: bool,
    ) -> torch.Tensor:
        scale = scale.transpose(1, 2).contiguous()
        if is_per_channel:
            scale_np = scale.cpu().contiguous().numpy()
            scale_np.dtype = np.uint32
            scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
            return scale_uint64_tensor

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
        return sscale_uint64_tensor

    def _pack_to_int32(self, weight: torch.Tensor) -> torch.Tensor:
        # pack 4 int8 (representing 8 int4) into int32
        assert weight.shape[-1] % 4 == 0, (
            f"Last dimension of weight must be divisible by 4 for int8→int32 packing, "
            f"got shape {weight.shape}"
        )
        return weight.contiguous().view(torch.int32)

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }

        bias = None
        if self.activation_use_clip:
            bias = getattr(quant_info, f"{weight_prefix}_scale_bias", None)
        if bias is not None:
            scale_args["bias"] = [bias]

        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUW4A8Mxfp8MoEMethod
# ---------------------------------------------------------------------------
class NPUW4A8Mxfp4MoEMethod(_NPUMoEMethodBase):
    """W4A8 MXFP4 MoE – weights float4_e2m1fn_x2 (packed), activations float8_e4m3fn."""

    def __init__(self, group_size: int = 32):
        super().__init__(quant_config=None)
        self.group_size = group_size
        self.matmul = GroupedMatmul()
        # Activation quantizer stays the same – we still quantize to float8_e4m3fn
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.float8_e4m3fn)


    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        # 1) Fetch the packed weight and scale
        weight = getattr(layer, f"{weight_prefix}_weight")   # shape: [E, N, K_packed] uint8
        scale  = getattr(layer, f"{weight_prefix}_weight_scale")  # shape: [E, N, K_groups] uint8

        # 2) Cast weight to a hardware format that treats each byte as
        #    two float4_e2m1fn values, but presents as float8_e4m3fn.
        #    Format 29 is the Ascend ND format for this packed layout.
        weight.data = torch.ops.npu.npu_format_cast(
            weight.data,
            29,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=torch.float4_e2m1fn_x2,
        )

        # 3) Transpose weight: [E, N, K_packed] -> [E, K_packed, N]
        weight.data = weight.data.transpose(1, 2).contiguous()

        # 4) Pack scale groups and transpose:
        #    Original scale shape: [E, N, K_groups] where K_groups = K_orig // group_size
        #    Two groups are packed per byte -> reshape to [E, N, K_groups//2, 2]
        #    Then transpose to [E, K_groups//2, N, 2]
        g, n, k = scale.shape
        scale.data = scale.data.reshape(g, n, k // 2, 2).transpose(1, 2).contiguous()

        # 5) Store on the layer (and also on self for the matmul)
        setattr(layer, f"{weight_prefix}_weight",
                torch.nn.Parameter(weight.data, requires_grad=False))
        setattr(layer, f"{weight_prefix}_weight_scale",
                torch.nn.Parameter(scale.data, requires_grad=False))

        # Crucial: also store on self (quant_info) so that GroupedMatmul can find them
        setattr(self, f"{weight_prefix}_weight", weight.data)
        setattr(self, f"{weight_prefix}_weight_scale", scale.data)

        # 6) Dispatcher output dtype (unchanged)
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")


    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        weight_scale = getattr(self, f"{weight_prefix}_weight_scale", None)

        # Dynamic MXFP8 activation quantisation
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(
                hidden_states
            )

        scale_args: Dict[str, Any] = {
            "scale": [weight_scale],
            "per_token_scale": [pertoken_scale],
            "scale_dtype": torch.float8_e8m0fnu,
            "per_token_scale_dtype": torch.float8_e8m0fnu,
        }

        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,          # float8_e4m3fn
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUW8A8Int8MoEMethod
# ---------------------------------------------------------------------------
class NPUW8A8Int8MoEMethod(_NPUMoEMethodBase):
    """W8A8 MoE – weights are int8, activations in int8."""

    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.int8)

    @staticmethod
    def maybe_apply_fuseep_weights(layer: torch.nn.Module) -> bool:
        """Apply the FuseEP weight layout if --moe-a2a-backend is ascend_fuseep.

        Returns True when the FuseEP layout was (or has already been) applied,
        so that the caller can skip its own ``process_weights_after_loading`` body.
        """
        from sglang.srt.layers.moe import get_moe_a2a_backend

        if not get_moe_a2a_backend().is_ascend_fuseep():
            return False

        # Guard against double processing when called for multiple prefixes.
        if getattr(layer, "_fuseep_weights_processed", False):
            return True

        from sglang.srt.hardware_backend.npu.moe.fuseep import process_fuseep_weights

        for prefix in ("w13", "w2"):
            process_fuseep_weights(layer, prefix)
        layer._fuseep_weights_processed = True
        return True

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        # If the FuseEP weight layout is used, process weights via
        # maybe_apply_fuseep_weights and skip the rest of this method.
        if self.maybe_apply_fuseep_weights(layer):
            return

        self._validate_weight_prefix(layer, weight_prefix)

        # Process scale
        scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale")
        processed_scale = torch.nn.Parameter(
            scale.data.squeeze(-1).to(dtype=torch.bfloat16), requires_grad=False
        )
        setattr(layer, f"{weight_prefix}_weight_scale", processed_scale)

        # Process offset
        offset: Optional[torch.Tensor] = getattr(
            layer, f"{weight_prefix}_weight_offset", None
        )
        if offset is not None:
            processed_offset = torch.nn.Parameter(
                offset.data.squeeze(-1), requires_grad=False
            )
            setattr(layer, f"{weight_prefix}_weight_offset", processed_offset)

        # Process weight
        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(weight.data.transpose(1, 2))
        setattr(
            layer,
            f"{weight_prefix}_weight",
            torch.nn.Parameter(weight, requires_grad=False),
        )

        # Set dispatcher output dtype
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "int8")

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUW8A8Mxfp8MoEMethod
# ---------------------------------------------------------------------------
class NPUW8A8Mxfp8MoEMethod(_NPUMoEMethodBase):
    """W8A8 MXFP8 MoE – weights float8_e4m3fn, scales uint8 (per‑group)."""

    def __init__(self, group_size: int = 32):
        super().__init__(quant_config=None)
        self.group_size = group_size
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.float8_e4m3fn)

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")   # [E, N, K]
        scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale")  # [E, N, num_groups ]

        # 1) Weight: transpose to [E, K, N]
        weight.data = weight.data.transpose(1, 2).contiguous()

        # 2) Scale: pad if number of num_groups is odd, then pack into [E, num_groups //2, N, 2]
        num_groups = scale.shape[-1]
        scale.data = scale.data.reshape(scale.shape[0], scale.shape[1], num_groups  // 2, 2)
        # transpose: [E, N, num_groups //2, 2] -> [E, num_groups //2, N, 2]
        scale.data = scale.data.transpose(1, 2).contiguous()

        setattr(
            layer,
            f"{weight_prefix}_weight",
            torch.nn.Parameter(weight.data, requires_grad=False),
        )
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            torch.nn.Parameter(scale.data, requires_grad=False),
        )

        # Set dispatcher output dtype – mxfp8_e4m3fn
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16") # add "mxfp8_e4m3fn" in future

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        weight_scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
    
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(
                hidden_states
            )
    
        scale_args: Dict[str, Any] = {
            "scale": [weight_scale],
            "per_token_scale": [pertoken_scale],
            "scale_dtype": torch.float8_e8m0fnu,
            "per_token_scale_dtype": torch.float8_e8m0fnu,
        }
    
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,          # float8_e4m3fn
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUWNA16Int4MoEMethod
# ---------------------------------------------------------------------------
class NPUWNA16Int4MoEMethod(_NPUMoEMethodBase):
    """W4A16 MoE – weights are int4, activations stay in BF16."""

    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        # Process scale
        scale = getattr(layer, f"{weight_prefix}_weight_scale")  # shape [E, N, 1]
        scale = scale.data.transpose(-1, -2).contiguous()  # [E, N, 1] -> [E, 1, N]
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            torch.nn.Parameter(scale, requires_grad=False),
        )

        # Process offset
        offset = getattr(layer, f"{weight_prefix}_weight_offset", None)
        if offset is not None:
            offset = offset.data.transpose(-1, -2).contiguous()
            setattr(
                layer,
                f"{weight_prefix}_weight_offset",
                torch.nn.Parameter(offset, requires_grad=False),
            )

        # Process weight: unpack, transpose, repack
        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        unpacked_weight = (
            self._unpack_from_int32(weight.data.flatten(0, 1), 4)
            .view(weight.shape[0], weight.shape[1], -1)
            .transpose(1, 2)
            .int()
        )
        weight.data = self._pack_to_int32(unpacked_weight)
        setattr(
            layer,
            f"{weight_prefix}_weight",
            torch.nn.Parameter(weight, requires_grad=False),
        )

        # Set dispatcher output dtype
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def _pack_to_int32(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.dim() == 3
        if weight.dtype == torch.int32:
            assert weight.shape[-1] % 8 == 0, (
                f"Last dimension of int32 weight must be divisible by 8 for int4 packing, "
                f"got {weight.shape}"
            )
            new_weight = torch.ops.npu.npu_convert_weight_to_int4pack(
                weight.flatten(0, 1)
            )
            new_weight = new_weight.view(weight.shape[0], weight.shape[1], -1)
        elif weight.dtype == torch.int8:
            assert weight.shape[-1] % 4 == 0, (
                f"Last dimension of int8 weight must be divisible by 4 for int32 packing, "
                f"got {weight.shape}"
            )
            new_weight = weight.contiguous().view(torch.int32)
        else:
            raise ValueError(f"Unsupported weight dtype for packing: {weight.dtype}")
        return new_weight.contiguous()

    def _unpack_from_int32(
        self,
        value: torch.Tensor,
        num_bits: int,
        shape: Optional[torch.Size] = None,
        packed_dim: int = 1,
    ) -> torch.Tensor:
        """
        Unpacks a tensor of packed int32 weights into individual int8s,
        maintaining the original bit range.
        """
        if value.dtype is not torch.int32:
            raise ValueError(
                f"Expected {torch.int32} but got {value.dtype}, Aborting unpack."
            )
        if num_bits > 8:
            raise ValueError("Unpacking is only supported for less than 8 bits")

        pack_factor = 32 // num_bits
        mask = (1 << num_bits) - 1

        if packed_dim == 1:
            unpacked = torch.zeros(
                (value.shape[0], value.shape[1] * pack_factor),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[:, i::pack_factor] = (value >> (num_bits * i)) & mask
            if shape is not None:
                original_row_size = int(shape[1])
                unpacked = unpacked[:, :original_row_size]
        else:
            unpacked = torch.zeros(
                (value.shape[0] * pack_factor, value.shape[1]),
                device=value.device,
                dtype=torch.int32,
            )
            for i in range(pack_factor):
                unpacked[i::pack_factor, :] = (value >> (num_bits * i)) & mask
            if shape is not None:
                original_row_size = int(shape[0])
                unpacked = unpacked[:original_row_size, :]

        offset = pow(2, num_bits) // 2
        unpacked = (unpacked - offset).to(torch.int8)
        return unpacked

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,  # not used, but kept for interface consistency
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        scale = getattr(quant_info, f"{weight_prefix}_weight_scale", None)
        offset = getattr(quant_info, f"{weight_prefix}_weight_offset", None)
        scale_args: Dict[str, Any] = {
            "antiquant_scale": [scale],
            "antiquant_offset": [offset] if offset is not None else [],
        }
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            **scale_args,
        )


# ---------------------------------------------------------------------------
# NPUUnquantMoEMethod
# ---------------------------------------------------------------------------
class NPUUnquantMoEMethod(_NPUMoEMethodBase):
    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()
        self.hidden_states_quantizer = None

    def process_weights_after_loading(self, layer, weight_prefix):
        self._validate_weight_prefix(layer, weight_prefix)
        weight_name = f"{weight_prefix}_weight"

        online_quant = "w4a8_mxfp8"   # or from server_args / env

        if online_quant == "w8a8_int8":
            self._apply_online_w8a8(layer, weight_prefix, weight_name)
        elif online_quant == "w8a8_mxfp8":
            self._apply_online_mxfp8(layer, weight_prefix, weight_name)
        elif online_quant == "w4a8_mxfp8":        # W4A8 MXFP4
            self._apply_online_mxfp4_w4a8(layer, weight_prefix, weight_name)
        elif online_quant == "w4a4_mxfp4":        # W4A4 MXFP4
            self._apply_online_mxfp4_w4a4(layer, weight_prefix, weight_name)
        else:
            # Pure BF16: just store the transposed weight on self
            weight = getattr(layer, weight_name)
            formatted = npu_format_cast(weight.data.transpose(1, 2))
            layer.__setattr__(weight_name,
                              torch.nn.Parameter(formatted, requires_grad=False))
            setattr(self, weight_name, formatted)   # also store on self
            if weight_prefix == "w13":
                self._set_dispatcher_output_dtype(layer, "bf16")

    # --------------- W8A8 path (unchanged) ---------------
    def _apply_online_w8a8(self, layer, weight_prefix, weight_name):
        weight_fp = getattr(layer, weight_name)
        qw, weight_scale = torch.ops.npu.npu_dynamic_quant(weight_fp)
        qw_npu = npu_format_cast(qw.transpose(-2, -1))

        setattr(layer, weight_name,
                torch.nn.Parameter(qw_npu, requires_grad=False))
        layer.register_parameter(
            f"{weight_name}_scale",
            torch.nn.Parameter(weight_scale, requires_grad=False))

        setattr(self, weight_name, qw_npu)
        setattr(self, f"{weight_name}_scale", weight_scale)

        torch.npu.empty_cache()

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "int8")

        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch.int8)

    # --------------- MXFP8 path (unchanged) ---------------
    def _apply_online_mxfp8(self, layer, weight_prefix, weight_name):
        weight_fp = getattr(layer, weight_name)

        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=torch_npu.float8_e4m3fn
        )
        # Transpose and force contiguous
        qw_t = qw.transpose(1, 2).contiguous()          # [E, K, N]
        w_scale_t = w_scale.transpose(1, 2).contiguous() # [E, ceil(K/64), N, 2]

        # Store on the layer (for parameter saving etc.)
        setattr(layer, weight_name,
                torch.nn.Parameter(qw_t, requires_grad=False))
        layer.register_parameter(
            f"{weight_name}_scale",
            torch.nn.Parameter(w_scale_t, requires_grad=False),
        )

        setattr(self, weight_name, qw_t)
        setattr(self, f"{weight_name}_scale", w_scale_t)

        torch.npu.empty_cache()
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=torch_npu.float8_e4m3fn)

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    # --------------- W4A8 MXFP4 path ---------------
    def _apply_online_mxfp4_w4a8(self, layer, weight_prefix, weight_name):
        self._apply_online_mxfp4_common(
            layer, weight_prefix, weight_name,
            act_quant_dtype=torch.float8_e4m3fn
        )

    # --------------- W4A4 MXFP4 path ---------------
    def _apply_online_mxfp4_w4a4(self, layer, weight_prefix, weight_name):
        self._apply_online_mxfp4_common(
            layer, weight_prefix, weight_name,
            act_quant_dtype=torch_npu.float4_e2m1fn_x2
        )

    # --------------- shared MXFP4 weight processing (W4A8 & W4A4) ---------------
    def _apply_online_mxfp4_common(self, layer, weight_prefix, weight_name, act_quant_dtype):
        weight_fp = getattr(layer, weight_name)               # [E, N, K]
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        fp4_dtype = torch_npu.float4_e2m1fn_x2
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=fp4_dtype, round_mode="round"
        )
        # qw: [E, N, K//2] (packed FP4)
        # w_scale: [E, N, ceil(K/64), 2] (or 2D legacy shape)

        qw_t = qw_nz.transpose(1, 2)                     # [E, K_packed, N]

        # Pack scale to [E, K_groups//2, N, 2]
        g, n, k = w_scale.shape
        if w_scale.dim() == 2:                           # fallback
            w_scale = w_scale.reshape(g, n, -1, 2)
        w_scale_t = w_scale.reshape(g, n, -1, 2).transpose(1, 2).contiguous()

        # Store on layer and self
        setattr(layer, weight_name,
                torch.nn.Parameter(qw_t, requires_grad=False))
        layer.register_parameter(
            f"{weight_name}_scale",
            torch.nn.Parameter(w_scale_t, requires_grad=False))
        setattr(self, weight_name, qw_t)
        setattr(self, f"{weight_name}_scale", w_scale_t)
        torch.npu.empty_cache()

        # Set activation quantizer according to the mode (W4A8 or W4A4)
        self.hidden_states_quantizer = HiddenStatesDynamicQuant(quant_dtype=act_quant_dtype)

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    # apply method (works for all quant modes)
    def apply(self, quant_info, hidden_states, expert_tokens,
              pertoken_scale, output_dtype, weight_prefix, group_list_type):
        weight_scale = getattr(self, f"{weight_prefix}_weight_scale", None)
        if weight_scale is None:
            # fallback pure BF16
            return self.matmul.forward(
                quant_info, weight_prefix, hidden_states, expert_tokens,
                output_dtype, group_list_type=group_list_type)

        if self.hidden_states_quantizer is not None and pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(hidden_states)

        scale_args = {
            "scale": [weight_scale],
            "per_token_scale": [pertoken_scale],
            "scale_dtype": torch_npu.float8_e8m0fnu,
            "per_token_scale_dtype": torch_npu.float8_e8m0fnu,
        }
        return self.matmul.forward(
            quant_info, weight_prefix, hidden_states, expert_tokens,
            output_dtype, group_list_type=group_list_type, **scale_args)
