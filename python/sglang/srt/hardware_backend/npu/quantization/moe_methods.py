from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, npu_format_cast
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

import logging

from sglang.srt.hardware_backend.npu.moe.matmul import (
    GroupedMatmul,
    GroupedMatmulSwigluQuant,
)
from sglang.srt.hardware_backend.npu.moe.quant import HiddenStatesDynamicQuant
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _get_float4_e2m1fn_x2_dtype,
    _get_float8_e8m0fnu_dtype,
)

logger = logging.getLogger(__name__)


_E8M0_DTYPE = None


def _require_e8m0_dtype():
    """Resolve the e8m0 block-scale dtype, failing loudly if it is unavailable.

    The grouped matmuls validate their scale-dtype arguments against torch_npu's
    own dtype enum (``torch_npu.float8_e8m0fnu``, 293 on A5) and reject the torch
    dtype object with "weight_scale_dtype only supports float8_e8m0fnu or None,
    but the actual value is Float8_e8m0fnu" — hence torch_npu first, torch only
    as a fallback. Dense ``npu_quant_matmul`` accepts either, which is why
    ``_get_float8_e8m0fnu_dtype`` reads it off torch.

    The MXFP8 ops take the scale dtype explicitly; passing None silently changes
    how they interpret the scales, so a missing dtype must raise rather than
    propagate.

    torch_npu is imported lazily (and cached) so this module stays importable on
    CUDA/CPU/AMD/XPU CI.
    """
    global _E8M0_DTYPE
    if _E8M0_DTYPE is None:
        from sglang.srt.utils import is_npu

        if is_npu():
            import torch_npu

            _E8M0_DTYPE = getattr(torch_npu, "float8_e8m0fnu", None)
        if _E8M0_DTYPE is None:
            _E8M0_DTYPE = _get_float8_e8m0fnu_dtype()
        if _E8M0_DTYPE is None:
            raise RuntimeError(
                "float8_e8m0fnu dtype not found — MXFP8 MoE requires Ascend A5 "
                "with a torch_npu build exposing float8_e8m0fnu (torch_npu >= 2.9)."
            )
    return _E8M0_DTYPE


def _require_fp4_dtype():
    fp4_dtype = _get_float4_e2m1fn_x2_dtype()
    if fp4_dtype is None:
        raise RuntimeError(
            "float4_e2m1fn_x2 dtype not found — MXFP4 MoE requires Ascend A5 "
            "with a torch_npu build exposing float4_e2m1fn_x2."
        )
    return fp4_dtype


def _pack_mxfp_weight_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert checkpoint ``[E, N, K/32]`` scales to ``[E, K/64, N, 2]``."""
    if scale.ndim != 3:
        raise ValueError(
            f"MXFP expert weight scale must be 3D [E, N, K/32], got {scale.shape}."
        )
    if scale.shape[-1] % 2 != 0:
        raise ValueError(
            "MXFP expert weight scale block dimension must be even for pair "
            f"packing, got {scale.shape[-1]}."
        )
    return scale.reshape(
        scale.shape[0], scale.shape[1], scale.shape[2] // 2, 2
    ).transpose(1, 2)


def _normalize_mxfp_input_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 3:
        return scale
    if scale.ndim != 2 or scale.shape[-1] % 2 != 0:
        raise ValueError(
            "MXFP activation scale must be [tokens, K/32] or "
            f"[tokens, K/64, 2], got {scale.shape}."
        )
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


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
        weight.data = npu_format_cast(weight.data.transpose(1, 2).contiguous())

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
            self._update_bias(layer, weight_prefix)
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
                        bias.data.contiguous(),
                        requires_grad=False,
                    ),
                )

        # Process weight
        weight = getattr(layer, f"{weight_prefix}_weight")
        weight.data = weight.data.transpose(1, 2).contiguous()
        weight.data = npu_format_cast(weight.data)
        weight.data = self._pack_to_int32(weight.data)

        # Set dispatcher output dtype
        if weight_prefix == "w13":
            self._set_dispatcher_output_dtype(layer, "int8")

    @staticmethod
    def _update_bias(
        layer: torch.nn.Module,
        weight_prefix: str,
    ) -> None:
        scale_bias_name = f"{weight_prefix}_scale_bias"
        if hasattr(layer, scale_bias_name):
            scale_bias = getattr(layer, scale_bias_name)
            scale_bias.data = scale_bias.data.transpose(1, 2).contiguous().sum(dim=1)

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
        return weight.view(torch.int32).contiguous()

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
#  Packed MXFP4 ModelSlim MoE methods
# ---------------------------------------------------------------------------
class _NPUPackedMXFP4MoEMethod(_NPUMoEMethodBase):
    """Common packed-FP4 weight layout used by W4A4 and W4A8 ModelSlim MoE."""

    activation_is_fp4 = False
    deepep_dispatcher_dtype = "mxfp8"

    def __init__(self, weight_prefix: str) -> None:
        super().__init__(quant_config=None)
        if weight_prefix not in ("w13", "w2"):
            raise ValueError(
                f"weight_prefix must be 'w13' or 'w2', got {weight_prefix!r}."
            )
        self.weight_prefix = weight_prefix
        self.matmul = GroupedMatmul()

    def _format_weight(self, weight: torch.Tensor, fp4_dtype) -> torch.Tensor:
        if self.activation_is_fp4:
            return npu_format_cast(weight)
        return npu_format_cast(
            weight,
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )

    def process_weights_after_loading(
        self, layer: torch.nn.Module, weight_prefix: str
    ) -> None:
        self._validate_weight_prefix(layer, weight_prefix)
        fp4_dtype = _require_fp4_dtype()
        weight = getattr(layer, f"{weight_prefix}_weight").data
        weight_scale = getattr(layer, f"{weight_prefix}_weight_scale").data
        expected_scale_blocks = (weight.shape[-1] * 2 + 31) // 32
        if weight_scale.shape[:2] != weight.shape[:2] or (
            weight_scale.shape[-1] != expected_scale_blocks
        ):
            raise ValueError(
                f"{weight_prefix} MXFP4 scale shape {tuple(weight_scale.shape)} "
                f"does not match packed weight shape {tuple(weight.shape)}."
            )

        weight = self._format_weight(weight, fp4_dtype).transpose(1, 2)
        weight_scale = _pack_mxfp_weight_scale(weight_scale)
        setattr(
            layer,
            f"{weight_prefix}_weight",
            Parameter(weight, requires_grad=False),
        )
        setattr(
            layer,
            f"{weight_prefix}_weight_scale",
            Parameter(weight_scale, requires_grad=False),
        )

        if weight_prefix == "w13":
            from sglang.srt.layers.moe import get_moe_a2a_backend

            if get_moe_a2a_backend().is_deepep():
                dispatcher_dtype = self.deepep_dispatcher_dtype
            elif self.activation_is_fp4:
                # AscendTP has no MXFP4 routing mode. It sends BF16 and the
                # grouped-matmul method performs dynamic MXFP4 quantization.
                dispatcher_dtype = "bf16"
            else:
                dispatcher_dtype = "mxfp8"
            self._set_dispatcher_output_dtype(layer, dispatcher_dtype)

    @staticmethod
    def _weight_scale(
        quant_info: "AscendQuantInfo", weight_prefix: str
    ) -> torch.Tensor:
        scale = (
            quant_info.w13_weight_scale
            if weight_prefix == "w13"
            else quant_info.w2_weight_scale
        )
        if scale is None:
            raise RuntimeError(
                f"{weight_prefix}_weight_scale is required for MXFP4 MoE; "
                "unit-scale fallback is not allowed."
            )
        return scale

    def _quantize_input(
        self,
        hidden_states: torch.Tensor,
        input_scale: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_scale is not None:
            return hidden_states, _normalize_mxfp_input_scale(input_scale)
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "Pre-quantized MXFP MoE activations require their UE8M0 scale; "
                f"got dtype={hidden_states.dtype} with no scale."
            )

        if self.activation_is_fp4:
            return torch.ops.npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=_require_fp4_dtype(),
                round_mode="round",
            )
        return torch.ops.npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn
        )

    def _scale_args(
        self, weight_scale: torch.Tensor, input_scale: torch.Tensor
    ) -> Dict[str, Any]:
        e8m0_dtype = _require_e8m0_dtype()
        fp4_dtype = _require_fp4_dtype()
        if self.activation_is_fp4:
            return {
                "scale": [weight_scale],
                "per_token_scale": [input_scale],
                "scale_dtype": e8m0_dtype,
                "per_token_scale_dtype": e8m0_dtype,
                "x_dtype": fp4_dtype,
                "weight_dtype": fp4_dtype,
            }
        return {
            "antiquant_scale": [weight_scale],
            "per_token_scale": [input_scale],
            "per_token_scale_dtype": e8m0_dtype,
            "x_dtype": torch.float8_e4m3fn,
            "weight_dtype": fp4_dtype,
        }

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
        weight_scale = self._weight_scale(quant_info, weight_prefix)
        hidden_states, input_scale = self._quantize_input(
            hidden_states, pertoken_scale
        )
        return self.matmul.forward(
            quant_info,
            weight_prefix,
            hidden_states,
            expert_tokens,
            output_dtype,
            group_list_type=group_list_type,
            transposed=True,
            **self._scale_args(weight_scale, input_scale),
        )


class NPUW4A4MXFP4MoEMethod(_NPUPackedMXFP4MoEMethod):
    """W4A4 MXFP4 grouped matmul for pre-quantized ModelSlim experts."""

    activation_is_fp4 = True
    deepep_dispatcher_dtype = "mxfp4"


class NPUW4A8MXFPMoEMethod(_NPUPackedMXFP4MoEMethod):
    """W4A8 grouped matmul with packed MXFP4 weights and MXFP8 activations."""


# ---------------------------------------------------------------------------
#  NPUMXFP8MoEMethod
# ---------------------------------------------------------------------------
class NPUMXFP8MoEMethod(_NPUMoEMethodBase):
    """MXFP8 MoE on Ascend A5 – float8_e4m3fn weights with e8m0 block scales.

    Serves both the online config path (``--quantization mxfp8``, weights
    quantised at load time) and the offline ModelSlim ``W8A8_MXFP8`` scheme
    (weights already quantised); ``process_weights_after_loading`` tells the two
    apart by weight dtype.

    gmm1 re-quantises its own output, so it is a single fused kernel rather than
    a matmul plus a separate activation: the runner calls
    ``apply_fused_gmm1_swiglu`` for w13 and ``apply`` only for w2 — hence the
    per-prefix matmul chosen here.

    Where the *activation* quant happens depends on the dispatcher. AscendTP
    uses ``npu_moe_init_routing_v2(quant_mode=3)``; A5 DeepEP requests MXFP8 on
    the wire. Both emit an e4m3 payload plus an e8m0 block scale. The method can
    still dynamically quantize BF16 input when no quantizing dispatcher is used.
    """

    def __init__(self, weight_prefix: str):
        super().__init__(quant_config=None)
        if weight_prefix == "w13":
            self.matmul = GroupedMatmulSwigluQuant()
            self.hidden_states_quantizer = HiddenStatesDynamicQuant(
                quant_dtype=torch.float8_e4m3fn
            )
        else:
            self.matmul = GroupedMatmul()
            self.hidden_states_quantizer = None

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

        # FRACTAL_NZ before the transpose, never after. gmm1 asserts that weight
        # and weight_scale carry the SAME transpose flag (CheckMXTranspose: "the
        # transposition of weightScale/weight should be equal"), and the cast
        # yields a physically retiled — hence non-transposed — tensor. Casting
        # the [E, K, N] view would therefore leave the weight at false against a
        # true scale and fail outright, which is why this cannot copy the int8
        # MoE methods above (they transpose first, but carry no MX scale to keep
        # in sync). Same order as the dense W4A8 path in linear_method_npu.py.
        #
        # A5 measurement, Qwen3-30B-A3B shapes, 128 experts (see
        # llm/probe_mxfp8_moe_nz.py): +1.4% decode, +3.8% prefill against a 0.2-
        # 0.3% noise floor, bit-identical outputs. Set
        # SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT to fall back to plain ND.
        weight = npu_format_cast(weight)

        # Both paths hand the grouped matmul weight [E, K, N] and scale
        # [E, K//64, N, 2] as strided transpose views — DO NOT call
        # .contiguous(). Beyond breaking the transpose-flag match above, it
        # measures slower on the same probe: making both sides contiguous costs
        # 6.2% on decode. This matches NPUMXFP8LinearMethod, msmodelslim's
        # offline layout and vllm-ascend's AscendW8A8MXFP8DynamicFusedMoEMethod.
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
        pertoken_scale: Optional[torch.Tensor],
        group_list_type,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gate/up projection, swiglu and requantisation in one kernel (gmm1).

        Returns the e4m3 activations and their e8m0 block scale, i.e. exactly
        what the w2 gmm needs, which is why the runner skips its activation step
        for MXFP8.

        ``pertoken_scale`` is None when the dispatcher handed over unquantised
        hidden states (the DeepEP path), in which case the activation quant that
        ascend_tp fuses into routing is done here instead. Both dispatchers
        therefore reach the kernel below with the same e4m3 + e8m0 input.
        """
        if pertoken_scale is None:
            hidden_states, pertoken_scale = self.hidden_states_quantizer(hidden_states)
        else:
            pertoken_scale = _normalize_mxfp_input_scale(pertoken_scale)

        e8m0_dtype = _require_e8m0_dtype()
        return self.matmul.forward(
            quant_info,
            "w13",
            hidden_states,
            expert_tokens,
            group_list_type=group_list_type,
            transposed=True,
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
