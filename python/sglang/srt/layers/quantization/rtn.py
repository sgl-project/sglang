# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright © 2025, Oracle and/or its affiliates.

import logging
import os
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.rtn_utils import (
    fix_weights,
    rtn_dequantize,
    rtn_quantize,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


# By default, use 8 bit as target precision, but it can be overridden by setting the RTN_NUM_BITS envvar
NUM_BITS = os.getenv("RTN_NUM_BITS", "8")


# By default, use group size of 128 parameters, but it can be overridden by setting the RTN_GROUP_SIZE envvar
GROUP_SIZE = os.getenv("RTN_GROUP_SIZE", "128")


class RTNConfig(QuantizationConfig):
    """Config class for RTN."""

    def __init__(
        self,
        weight_bits: int = int(NUM_BITS),
        group_size: int = int(GROUP_SIZE),
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size

        if self.weight_bits != 4 and self.weight_bits != 8:
            raise ValueError(
                "Currently, only 4-bit or 8-bit weight quantization is "
                f"supported for RTN, but got {self.weight_bits} bits."
            )

    def __repr__(self) -> str:
        return (
            f"RTNConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "rtn"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RTNConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    def is_marlin_compatible(self) -> bool:
        from sglang.srt.layers.quantization.gptq import GPTQMarlinConfig
        from sglang.srt.layers.quantization.marlin_utils import check_marlin_supported

        # Check bits/sym
        # RTN is symmetric (centered)
        if (self.weight_bits, True) not in GPTQMarlinConfig.TYPE_MAP:
            return False

        quant_type = GPTQMarlinConfig.TYPE_MAP[(self.weight_bits, True)]
        # Check group size
        return check_marlin_supported(quant_type, self.group_size)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.marlin_utils import (
            check_marlin_supports_layer,
            check_moe_marlin_supports_layer,
        )

        if isinstance(layer, LinearBase):
            if self.is_marlin_compatible() and check_marlin_supports_layer(
                layer, self.group_size
            ):
                return RTNMarlinLinearMethod(self)
            return RTNLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if self.is_marlin_compatible() and check_moe_marlin_supports_layer(
                layer, self.group_size
            ):
                return RTNMarlinMoEMethod(self)
            return RTNMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class RTNTensor:
    """A wrapper over Tensor that enables quantization on-the-fly by
    overloading the copy_ method.
    """

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.data = data
        self.scale = scale
        self.quant_config = quant_config

    def dim(self):
        """Return number of dimensions, matching torch.Tensor.dim() behavior"""
        return len(self.shape)

    @property
    def __class__(self):
        """Make isinstance(rtn_tensor, torch.Tensor) return True"""
        return torch.Tensor

    def narrow(self, dim, start, length):
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        return RTNTensor(
            self.data.narrow(dim, start // factor, length // factor),
            self.scale.narrow(dim, start, length),
            self.quant_config,
        )

    def __getitem__(self, key):
        return RTNTensor(self.data[key], self.scale[key], self.quant_config)

    @property
    def shape(self):
        shape = self.data.shape
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        batch_present = len(shape) == 3
        if batch_present:
            return torch.Size((shape[0], shape[1] * factor, shape[2]))
        else:
            return torch.Size((shape[0] * factor, shape[1]))

    def copy_(self, loaded_weight: torch.Tensor) -> None:
        qweight, weight_scale = rtn_quantize(
            loaded_weight.cuda(),
            self.quant_config.weight_bits,
            self.quant_config.group_size,
        )

        self.data.copy_(qweight)
        self.scale.data.copy_(weight_scale)


class RTNParameter(Parameter):
    """A wrapper over Parameter that returns RTNTensor (a wrapper over Tensor)
    when its data is accessed. We need this wrapper for the data loading phase
    only, so we can intercept a weight copying function (torch.Tensor.copy_)
    and apply quantization on-the-fly.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(
        self, data: torch.Tensor, scale: torch.Tensor, quant_config: RTNConfig
    ) -> None:
        self.scale = scale
        self.quant_config = quant_config

    @property
    def data(self):
        return RTNTensor(super().data, self.scale, self.quant_config)


class RTNLinearMethod(LinearMethodBase):
    """Linear method for RTN.

    Args:
        quant_config: The RTN quantization config.
    """

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        num_groups_per_col = (
            input_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )

        scale = Parameter(
            torch.empty(
                output_size_per_partition, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        factor = 1 if self.quant_config.weight_bits == 8 else 2

        weight = RTNParameter(
            data=torch.empty(
                output_size_per_partition // factor,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=scale,
            quant_config=self.quant_config,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("scale", scale)
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fix_weights(layer, "weight")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.weight
        scale = layer.scale

        weight = rtn_dequantize(qweight, scale)
        out = F.linear(x, weight)
        del weight
        if bias is not None:
            out.add_(bias)

        return out


class RTNMoEMethod:
    """MoE method for RTN."""

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

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        factor = 1 if self.quant_config.weight_bits == 8 else 2

        # Fused gate_up_proj (column parallel)
        num_groups_per_col = (
            hidden_size // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w13_scale = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                num_groups_per_col,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scale", w13_scale)

        w13_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition // factor,
                hidden_size,
                dtype=torch.uint8,
            ),
            scale=w13_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        num_groups_per_col = (
            intermediate_size_per_partition // self.quant_config.group_size
            if self.quant_config.group_size != -1
            else 1
        )
        w2_scale = Parameter(
            torch.zeros(
                num_experts, hidden_size, num_groups_per_col, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_scale", w2_scale)

        w2_weight = RTNParameter(
            data=torch.empty(
                num_experts,
                hidden_size // factor,
                intermediate_size_per_partition,
                dtype=torch.uint8,
            ),
            scale=w2_scale,
            quant_config=self.quant_config,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_bits = self.quant_config.weight_bits
        fix_weights(layer, "w13_weight", weight_bits == 4)
        fix_weights(layer, "w2_weight", weight_bits == 4)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        enable_eplb: bool = False,
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        if enable_eplb:
            raise NotImplementedError("EPLB not supported for `RTNMoEMethod` yet.")

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

        weight_bits = self.quant_config.weight_bits
        group_size = self.quant_config.group_size

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            use_int4_w4a16=weight_bits == 4,
            use_int8_w8a16=weight_bits == 8,
            w1_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            block_shape=[0, group_size],
            routed_scaling_factor=routed_scaling_factor,
        )


# Copied from sglang/srt/layers/quantization/gptq.py to avoid circular import
def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    from sgl_kernel import gptq_marlin_repack

    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = gptq_marlin_repack(b_q_weight[e], perm[e], size_k, size_n, num_bits)
    return output


def pack_columns(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits
    q_res = torch.zeros(
        (size_k, size_n // pack_factor), dtype=torch.int32, device=q_w.device
    )
    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor].to(torch.int32) << (num_bits * i)
    return q_res


class RTNMarlinLinearMethod(LinearMethodBase):
    """Linear method for RTN with Marlin kernels."""

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # We misuse RTNLinearMethod to create the weights, so that we can use
        # the same loading logic as RTNLinearMethod
        self.rtn_method = RTNLinearMethod(self.quant_config)
        self.rtn_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sgl_kernel import gptq_marlin_repack

        from sglang.srt.layers.quantization.gptq import GPTQMarlinConfig
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_empty_g_idx,
            marlin_make_workspace,
            marlin_permute_scales,
        )
        from sglang.srt.layers.quantization.utils import replace_parameter

        device = layer.weight.device

        # 1. Get unpacked weights from RTNParameter
        # RTNParameter (layer.weight) stores quantized data.
        # We get the raw data tensor.
        qweight = layer.weight.data.data
        if self.quant_config.weight_bits == 4:
            # Unpack 4-bit weights
            # Shape: (out, in/2) -> (out, in)
            from sglang.srt.layers.quantization.rtn_utils import rtn_unpack

            qweight = rtn_unpack(qweight, 4)

        # Transpose to (in, out) for Marlin/GPTQ packing
        qweight = qweight.t().contiguous()

        # 2. Pack to GPTQ format (int32)
        # Shape: (in, out/pack_factor) if we were packing output?
        # Wait, GPTQ packs along the second dimension (output_dim) in (input, output) layout??
        # Let's verify pack_cols.
        # pack_cols(q_w, ... size_k, size_n).
        # q_w is (size_k, size_n).
        # q_res is (size_k, size_n // pack_factor).
        # So it packs along size_n (output dimension).
        # GPTQLinearMethod creates:
        # qweight = (input / pack, output).
        # And gptq_gemm expects (input / pack, output).
        # BUT Marlin expects (k/16, n*bits/2) (after repack).
        # And gptq_marlin_repack input format:
        # It takes `b_q_weight` which comes from GPTQ layer.
        # GPTQ layer qweight is `(input_size // pack, output_size)`.
        # So it packs along input dimension??
        # Let's re-read GPTQLinearMethod.create_weights.
        # qweight shape: (input_size_per_partition // pack_factor, output_size_per_partition).
        # This means packing is along input dimension (dim 0).
        # BUT pack_cols in utils.py implements:
        # q_res |= q_w[:, i::pack_factor] << num_bits * i
        # This packs along dim 1 (columns).
        # CONTRAINDICATION.

        # In GPTQLinearMethod.create_weights:
        # qweight = PackedvLLMParameter(..., packed_dim=0, ...)
        # So the STORAGE is packed along dim 0.
        # But `pack_cols` packs along dim 1.
        # This implies `q_w` passed to `pack_cols` must be transposed?
        # `gptq_quantize_weights` calls `quantize_weights`.
        # `quantize_weights` returns `w_q` of shape `(groupsize, -1)` then restored to `(size_k, size_n)`.
        # `gptq_quantize_weights` returns `w_q` (size_k, size_n).
        # Then where is packing?
        # `gptq.py` doesn't use `pack_cols` in `apply` or `create_weights` directly (it just registers param).
        # The loading uses `weight_loader` which presumably handles packing if format is 'gptq'.
        # For RTN, we have unpacked weights.

        # If GPTQ stores (input_size // pack, output_size), then packing is along input.
        # `pack_rows` packs along dim 0.
        # `rtn_quantize` with pack=True packs along dim 2 (output dim if batch present, input dim if not??).
        # RTN w: (out, in). Packed: (out, in/2). Packing along IN (dim 1).

        # So RTN packing matches packing along input dimension for (out, in) tensor.
        # GPTQ qweight: (in // pack, out).
        # So if we transpose RTN -> (in // 2, out).
        # It matches (in // pack, out).
        # EXCEPT `pack_factor` for GPTQ 4-bit is 8.
        # RTN pack factor is 2.

        # So we definitely need to repack.

        # We need to create `qweight` of shape `(in // 8, out)`.
        # `qweight_unpacked` is `(in, out)`.
        # We need to pack rows (dim 0).

        # `pack_rows` in `utils.py`:
        # q_res |= q_w[i::pack_factor, :] << num_bits * i
        # This packs along rows.

        # So I need `pack_rows` equivalent.

        input_size = qweight.shape[0]
        output_size = qweight.shape[1]

        def pack_rows_torch(q_w, num_bits):
            pack_factor = 32 // num_bits
            q_res = torch.zeros(
                (q_w.shape[0] // pack_factor, q_w.shape[1]),
                dtype=torch.int32,
                device=q_w.device,
            )
            for i in range(pack_factor):
                q_res |= q_w[i::pack_factor, :].to(torch.int32) << (num_bits * i)
            return q_res

        qweight_gptq = pack_rows_torch(qweight, self.quant_config.weight_bits)

        # 3. Call gptq_marlin_repack
        # It expects `b_q_weight` (gptq format), perm (empty for rtn).

        part_shape = (input_size, output_size)

        # Create empty sort indices (identity)
        g_idx_sort_indices = marlin_make_empty_g_idx(device)

        marlin_qweight = gptq_marlin_repack(
            qweight_gptq.contiguous(),
            g_idx_sort_indices,
            input_size,
            output_size,
            self.quant_config.weight_bits,
        )

        # 4. Scales
        scales = layer.scale.data
        # RTN Scales: (out, in // group).
        # Marlin Scales expects: (in // group, out) permuted.

        scales = scales.t().contiguous()
        marlin_scales = marlin_permute_scales(
            scales, input_size, output_size, self.quant_config.group_size
        )

        # 5. Replace params
        replace_parameter(layer, "qweight", marlin_qweight)
        replace_parameter(layer, "scales", marlin_scales)
        replace_parameter(layer, "g_idx", marlin_make_empty_g_idx(device))
        replace_parameter(layer, "g_idx_sort_indices", g_idx_sort_indices)
        replace_parameter(
            layer, "qzeros", marlin_make_empty_g_idx(device)
        )  # No zeros for RTN

        # Clear old params
        if hasattr(layer, "weight"):
            del layer.weight
        if hasattr(layer, "scale"):
            del layer.scale

        # Workspace
        self.workspace = marlin_make_workspace(device)

        # Metadata
        self.is_k_full = True  # RTN usually no act order

        # Type
        self.quant_type = GPTQMarlinConfig.TYPE_MAP[
            (self.quant_config.weight_bits, True)
        ]

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.quantization.marlin_utils import apply_gptq_marlin_linear

        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.qzeros,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=self.quant_type,
            input_size_per_partition=layer.qweight.shape[0] * 16,
            output_size_per_partition=layer.output_size_per_partition,
            is_k_full=self.is_k_full,
            bias=bias,
        )


class RTNMarlinMoEMethod:
    """MoE method for RTN with Marlin kernels."""

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.rtn_method = RTNMoEMethod(self.quant_config)
        self.rtn_method.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_moe_permute_scales,
        )
        from sglang.srt.layers.quantization.rtn_utils import rtn_unpack
        from sglang.srt.layers.quantization.utils import replace_parameter

        # Helper to process weights
        def process_weight(rtn_param, is_gate_up):
            # rtn_param data: packed (experts, out, in/2)
            qweight_packed = rtn_param.data.data
            if self.quant_config.weight_bits == 4:
                qweight = rtn_unpack(qweight_packed, 4)
            else:
                qweight = qweight_packed

            # Unpacked: (experts, out, in)
            # Expected by gptq_marlin_moe_repack: (experts, in//pack, out) ??
            # gptq_marlin_moe_repack calls gptq_marlin_repack per expert.
            # So we need to provide GPTQ packed format per expert.
            # Transpose to (experts, in, out)
            qweight = qweight.transpose(1, 2).contiguous()

            num_experts, inp, out = qweight.shape

            # Pack rows (in dim). (experts, in/pack, out)
            pack_factor = 32 // self.quant_config.weight_bits
            q_res = torch.zeros(
                (num_experts, inp // pack_factor, out),
                dtype=torch.int32,
                device=rtn_param.device,
            )
            for i in range(pack_factor):
                q_res |= qweight[:, i::pack_factor, :].to(torch.int32) << (
                    self.quant_config.weight_bits * i
                )
            pass
            # Wait, pack_rows_torch I wrote earlier was 2D. I need batch support.

        # Implementation of batch pack_rows
        def pack_rows_batch(q_w, num_bits):
            # q_w: (batch, rows, cols)
            pack_factor = 32 // num_bits
            q_res = torch.zeros(
                (q_w.shape[0], q_w.shape[1] // pack_factor, q_w.shape[2]),
                dtype=torch.int32,
                device=q_w.device,
            )
            for i in range(pack_factor):
                q_res |= q_w[:, i::pack_factor, :].to(torch.int32) << (num_bits * i)
            return q_res

        # Processing w13
        w13_packed = layer.w13_weight.data.data
        if self.quant_config.weight_bits == 4:
            w13_unpacked = rtn_unpack(w13_packed, 4)
        else:
            w13_unpacked = w13_packed

        w13_t = w13_unpacked.transpose(1, 2).contiguous()  # (E, in, out)
        w13_gptq = pack_rows_batch(w13_t, self.quant_config.weight_bits)

        # Sort indices (empty/identity)
        device = w13_packed.device
        num_experts = w13_packed.shape[0]
        w13_size_k = w13_t.shape[1]
        w13_size_n = w13_t.shape[2]
        w13_sort_indices = torch.empty(
            (num_experts, w13_size_k), dtype=torch.int32, device=device
        )
        for e in range(num_experts):
            w13_sort_indices[e] = torch.arange(
                w13_size_k, dtype=torch.int32, device=device
            )

        marlin_w13_qweight = gptq_marlin_moe_repack(
            w13_gptq,
            w13_sort_indices,
            w13_size_k,
            w13_size_n,
            self.quant_config.weight_bits,
        )

        # Processing w2
        w2_packed = layer.w2_weight.data.data
        if self.quant_config.weight_bits == 4:
            w2_unpacked = rtn_unpack(w2_packed, 4)
        else:
            w2_unpacked = w2_packed

        w2_t = w2_unpacked.transpose(1, 2).contiguous()
        w2_gptq = pack_rows_batch(w2_t, self.quant_config.weight_bits)
        w2_size_k = w2_t.shape[1]
        w2_size_n = w2_t.shape[2]
        w2_sort_indices = torch.empty(
            (num_experts, w2_size_k), dtype=torch.int32, device=device
        )
        for e in range(num_experts):
            w2_sort_indices[e] = torch.arange(
                w2_size_k, dtype=torch.int32, device=device
            )

        marlin_w2_qweight = gptq_marlin_moe_repack(
            w2_gptq,
            w2_sort_indices,
            w2_size_k,
            w2_size_n,
            self.quant_config.weight_bits,
        )

        # Scales
        # RTN scales: (experts, out, in//group)
        # Marlin scales: (experts, in//group, out) permuted
        w13_scales = layer.w13_scale.transpose(1, 2).contiguous()
        marlin_w13_scales = marlin_moe_permute_scales(
            w13_scales, w13_size_k, w13_size_n, self.quant_config.group_size
        )

        w2_scales = layer.w2_scale.transpose(1, 2).contiguous()
        marlin_w2_scales = marlin_moe_permute_scales(
            w2_scales, w2_size_k, w2_size_n, self.quant_config.group_size
        )

        # Replace
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)
        replace_parameter(layer, "w13_scales", marlin_w13_scales)
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

        replace_parameter(layer, "w13_g_idx_sort_indices", w13_sort_indices)
        replace_parameter(layer, "w2_g_idx_sort_indices", w2_sort_indices)

        # Set g_idx, zeros to None/Empty
        replace_parameter(
            layer,
            "w13_g_idx",
            torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            ),
        )
        replace_parameter(
            layer,
            "w2_g_idx",
            torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            ),
        )

        # Remove old
        del layer.w13_weight
        del layer.w13_scale
        del layer.w2_weight
        del layer.w2_scale

        self.is_k_full = True

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe import (
            MoeRunner,
            MoeRunnerBackend,
            get_moe_runner_backend,
        )

        assert get_moe_runner_backend().is_auto()
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        enable_eplb: bool = False,
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
        from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
        from sglang.srt.layers.moe.topk import select_experts

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

        # Helper stub to simulate DispatchOutput since we don't have dispatch yet?
        # MoeRunner.run expects DispatchOutput.
        # But wait, MarlinRunner (fused_experts_none_to_marlin) takes dispatch_output.
        # usually run_moe calls dispatcher.
        # But `apply` here replaces `fused_experts`.
        # `fused_experts` bundles everything.

        # But `MoeRunner` interface runs dispatch + combine?
        # Check `sglang/srt/layers/moe/moe_runner/runner.py`.
        # `run(self, dispatch_output, quant_info)`.
        # So we needed to dispatch first.
        # But `RTNMoEMethod.apply` calls `fused_experts` which usually assumes `x` is token states.

        # `fused_experts` in `fused_moe_triton.py` DOES NOT dispatch?
        # No, `fused_experts` takes `topk_weights`, `topk_ids` etc.
        # It is the kernel wrapper.

        # `runner.run` calls the implementation.
        # `fused_experts_none_to_marlin` takes `StandardDispatchOutput`.
        # We can construct `StandardDispatchOutput` wrapping `x`, `topk`, etc.

        dispatch_output = StandardDispatchOutput(
            hidden_states=x,
            topk_output=type(
                "TopKOutput",
                (),
                {
                    "topk_weights": topk_weights,
                    "topk_ids": topk_ids,
                    "router_logits": router_logits,
                },
            )(),
            router_logits=router_logits,
        )
        # Hacky topk_output.
        # Actually `StandardDispatchOutput` expects `TopKOutput`.
        from sglang.srt.layers.moe.topk import TopKOutput

        topk_out = TopKOutput(topk_weights, topk_ids, router_logits)
        dispatch_output = StandardDispatchOutput(
            x, topk_out, router_logits
        )  # router_logits argument?

        # `MoeRunner` calls `self.backend_func(dispatch_output, quant_info, self.config)`.

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_qweight,
            w2_qweight=layer.w2_qweight,
            w13_scales=layer.w13_scales,
            w2_scales=layer.w2_scales,
            w13_g_idx=layer.w13_g_idx,
            w2_g_idx=layer.w2_g_idx,
            w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
            w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
            weight_bits=self.quant_config.weight_bits,
            is_k_full=self.is_k_full,
        )

        # We need to manually invoke the runner or fused function?
        # `RTNMoEMethod` creates a runner? No, `RTNMoEMethod.apply` is the implementation.
        # `FusedMoE` layer calls `method.apply`.

        return self.runner.run(dispatch_output, quant_info).hidden_states
