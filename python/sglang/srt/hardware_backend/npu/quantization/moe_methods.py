from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

import logging

from sglang.srt.hardware_backend.npu.moe.hidden_states_quant import (
    HiddenStatesDynamicQuant,
)
from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _get_float8_e8m0fnu_dtype,
)

logger = logging.getLogger(__name__)


def _require_e8m0_dtype():
    """Resolve the e8m0 block-scale dtype, failing loudly if it is unavailable.

    The MXFP8 ops take the scale dtype explicitly; passing None silently changes
    how they interpret the scales, so a missing dtype must raise rather than
    propagate.
    """
    e8m0_dtype = _get_float8_e8m0fnu_dtype()
    if e8m0_dtype is None:
        raise RuntimeError(
            "float8_e8m0fnu dtype not found — MXFP8 MoE requires Ascend A5 with a "
            "torch build exposing torch.float8_e8m0fnu (torch_npu >= 2.9)."
        )
    return e8m0_dtype


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

    @staticmethod
    def _get_bias_args(
        quant_info: "AscendQuantInfo", weight_prefix: str
    ) -> Dict[str, Any]:
        bias = getattr(quant_info, f"{weight_prefix}_scale_bias", None)
        if bias is None:
            bias = getattr(quant_info, f"{weight_prefix}_weight_bias", None)
        return {"bias": [bias]} if bias is not None else {}


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
            hidden_states, pertoken_scale = self.hidden_states_quantizer.__call__(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
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
    def maybe_process_fuseep_weights(layer: torch.nn.Module) -> bool:
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
        if self.maybe_process_fuseep_weights(layer):
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
            hidden_states, pertoken_scale = self.hidden_states_quantizer.__call__(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
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
            hidden_states, pertoken_scale = self.hidden_states_quantizer.__call__(
                hidden_states
            )
        scale_args: Dict[str, Any] = {
            "scale": [scale],
            "per_token_scale": [pertoken_scale],
        }
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))

        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
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
        scale_args.update(self._get_bias_args(quant_info, weight_prefix))
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUWUnquantMoEMethod
# ---------------------------------------------------------------------------
class NPUUnquantMoEMethod(_NPUMoEMethodBase):
    """Unquant MoE – all computations in BF16, no quantization."""

    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight")
        weight.data = npu_format_cast(weight)
        setattr(
            layer,
            f"{weight_prefix}_weight",
            torch.nn.Parameter(weight, requires_grad=False),
        )

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "bf16")

    def apply(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,  # ignored
        output_dtype: torch.dtype,
        weight_prefix: str,
        group_list_type,
    ) -> torch.Tensor:
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=False,
            **self._get_bias_args(quant_info, weight_prefix),
        )


# ---------------------------------------------------------------------------
#  NPUMXFP8MoEMethod
# ---------------------------------------------------------------------------
class NPUMXFP8MoEMethod(_NPUMoEMethodBase):
    """MXFP8 MoE on Ascend A5 – float8_e4m3fn weights with e8m0 block scales.

    Serves both the online config path (``--quantization mxfp8``, weights
    quantised at load time) and the offline ModelSlim ``W8A8_MXFP8`` scheme
    (weights already quantised); ``process_weights_after_loading`` tells the two
    apart by weight dtype.

    Unlike the int8/int4 methods this one does not quantise activations itself:
    the dispatcher's ``npu_moe_init_routing_v2(quant_mode=3)`` emits the e4m3
    payload and e8m0 scale as part of routing, and gmm1 re-quantises its own
    output. Consequently gmm1 is a single fused kernel rather than a matmul plus
    a separate activation, so the runner calls ``apply_fused_gmm1_swiglu`` for
    w13 and ``apply`` only for w2.
    """

    def __init__(self):
        super().__init__(quant_config=None)
        self.matmul = GroupedMatmul()

    @staticmethod
    def _quantize_weight_online(
        weight: torch.Tensor, weight_prefix: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantise BF16/FP16 expert weights ``[E, N, K]`` to MXFP8 at load time.

        Returns the e4m3 payload ``[E, N, K]`` and its e8m0 block scale
        ``[E, N, K//64, 2]`` (block_size=32, already pair-split by the op).
        """
        if weight.dtype not in (torch.float16, torch.bfloat16):
            logger.warning(
                "NPUMXFP8MoEMethod: %s_weight dtype %s is not float16/bfloat16; "
                "casting to bfloat16 before MXFP8 quantisation.",
                weight_prefix,
                weight.dtype,
            )
            weight = weight.to(torch.bfloat16)
        # cpu offload may have moved the weight back to host memory.
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        return torch.ops.npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn)

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)

        weight: torch.Tensor = getattr(layer, f"{weight_prefix}_weight").data
        if weight.dtype == torch.float8_e4m3fn:
            # Offline (ModelSlim) path: the checkpoint already holds e4m3 weights
            # and {prefix}_weight_scale holds uint8 block scales [E, N, K//32].
            # Only re-layout: split the flat scale axis into pairs to match what
            # npu_dynamic_mx_quant produces online.
            scale: torch.Tensor = getattr(layer, f"{weight_prefix}_weight_scale").data
            scale = scale.reshape(scale.shape[0], -1, scale.shape[-1] // 2, 2)
        else:
            weight, scale = self._quantize_weight_online(weight, weight_prefix)

        # Both paths hand the grouped matmul weight [E, K, N] and scale
        # [E, K//64, N, 2] as strided transpose views — DO NOT call
        # .contiguous(). The reduction loop scans the K-dim per output column,
        # which the [E, N, K] row-major source gives at stride 1 through the
        # transpose view (matches NPUMXFP8LinearMethod, msmodelslim's offline
        # layout and vllm-ascend's AscendW8A8MXFP8DynamicFusedMoEMethod).
        # .contiguous() would physically reorder to [E, K, N] row-major, making
        # the inner-loop stride = N and tanking HBM bandwidth.
        setattr(
            layer,
            f"{weight_prefix}_weight",
            Parameter(weight.transpose(1, 2), requires_grad=False),
        )
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            Parameter(scale.transpose(1, 2), requires_grad=False),
        )

        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "mxfp8")

    def apply_fused_gmm1_swiglu(
        self,
        quant_info: "AscendQuantInfo",
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        pertoken_scale: torch.Tensor,
        group_list_type,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gate/up projection, swiglu and requantisation in one kernel (gmm1).

        Returns the e4m3 activations and their e8m0 block scale, i.e. exactly
        what the w2 gmm needs, which is why the runner skips its activation step
        for MXFP8.
        """
        e8m0_dtype = _require_e8m0_dtype()
        # gmm1 wants a cumulative group_list while gmm2 keeps the COUNT form the
        # dispatcher produces (group_list_type=1). The asymmetry is intentional.
        group_list = expert_tokens.cumsum(0) if group_list_type == 1 else expert_tokens
        return torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
            x=hidden_states,
            weight=[quant_info.w13_weight],
            group_list=group_list,
            weight_scale=[quant_info.w13_weight_scale],
            x_scale=pertoken_scale,
            dequant_mode=2,
            quant_mode=2,
            dequant_dtype=torch.float32,
            quant_dtype=torch.float8_e4m3fn,
            # e4m3 is implicit for these two — it is not in the op's QUANT_DTYPES.
            x_dtype=None,
            weight_dtype=None,
            weight_scale_dtype=e8m0_dtype,
            x_scale_dtype=e8m0_dtype,
        )

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
        if weight_prefix != "w2":
            raise ValueError(
                f"NPUMXFP8MoEMethod.apply only serves the w2 gmm, got "
                f"'{weight_prefix}'. gmm1 fuses swiglu into a single op and must "
                f"go through apply_fused_gmm1_swiglu, which returns a scale too."
            )

        e8m0_dtype = _require_e8m0_dtype()
        scale_args: Dict[str, Any] = {
            "scale": [getattr(quant_info, f"{weight_prefix}_weight_scale", None)],
            "per_token_scale": [pertoken_scale],
            "scale_dtype": e8m0_dtype,
            "per_token_scale_dtype": e8m0_dtype,
            "x_dtype": None,
            "weight_dtype": None,
        }
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **scale_args,
        )


# ---------------------------------------------------------------------------
#  NPUMXFP8FusedMoEMethod
# ---------------------------------------------------------------------------
class NPUMXFP8FusedMoEMethod(FusedMoEMethodBase):
    """Online MXFP8 FusedMoE entry point (``--quantization mxfp8`` on A5).

    Owns the BF16 weight placeholders and the runner wiring; the per-gmm work
    lives in ``NPUMXFP8MoEMethod``, which quantises the weights to MXFP8 at load
    time. The offline ModelSlim path reuses that same kernel via
    ``ModelSlimMXFP8MoEScheme`` instead of this class.

    TP only: MoE EP / DeepEP is not supported on this path.
    """

    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        super().__init__()
        self.quant_config = quant_config
        self.w13_kernel = NPUMXFP8MoEMethod()
        self.w2_kernel = NPUMXFP8MoEMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.utils import set_weight_attrs

        # Weights load in params_dtype (BF16/FP16) and are quantised to MXFP8 in
        # process_weights_after_loading.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.w13_kernel.process_weights_after_loading(layer, "w13")
        self.w2_kernel.process_weights_after_loading(layer, "w2")

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        from sglang.srt.layers.moe.moe_runner import MoeRunner
        from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend

        layer.w13_kernel = self.w13_kernel
        layer.w2_kernel = self.w2_kernel
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_weight_scale=layer.w13_weight_scale,
            w2_weight_scale=layer.w2_weight_scale,
        )
        return self.runner.run(dispatch_output, quant_info)
