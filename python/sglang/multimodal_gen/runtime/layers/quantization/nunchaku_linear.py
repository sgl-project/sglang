# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
    from nunchaku.ops.gemv import awq_gemv_w4a16_cuda
    from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
except ImportError:
    svdq_gemm_w4a4_cuda = None
    awq_gemv_w4a16_cuda = None
    svdq_quantize_w4a4_act_fuse_lora_cuda = None


class NunchakuSVDQLinearMethod(LinearMethodBase):
    def __init__(
        self,
        precision: str = "int4",
        rank: int = 32,
        act_unsigned: bool = False,
    ):
        self.precision = precision
        self.rank = rank
        self.act_unsigned = act_unsigned

        if precision == "nvfp4":
            self.group_size = 16
        else:
            self.group_size = 64

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})

        num_groups = input_size_per_partition // self.group_size
        if self.precision == "nvfp4":
            scale_dtype = torch.float8_e4m3fn
        else:
            scale_dtype = params_dtype
        wscales = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=scale_dtype),
            requires_grad=False,
        )

        smooth_factor = Parameter(
            torch.empty(input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        smooth_factor_orig = Parameter(
            torch.empty(input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        proj_down = Parameter(
            torch.empty(input_size_per_partition, self.rank, dtype=params_dtype),
            requires_grad=False,
        )
        proj_up = Parameter(
            torch.empty(output_size_per_partition, self.rank, dtype=params_dtype),
            requires_grad=False,
        )

        if self.precision == "nvfp4":
            wcscales = Parameter(
                torch.empty(
                    output_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            wtscale = Parameter(
                torch.empty(1, dtype=params_dtype),
                requires_grad=False,
            )
        else:
            wcscales = None
            wtscale = None

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("smooth_factor", smooth_factor)
        layer.register_parameter("smooth_factor_orig", smooth_factor_orig)
        layer.register_parameter("proj_down", proj_down)
        layer.register_parameter("proj_up", proj_up)
        if wcscales is not None:
            layer.register_parameter("wcscales", wcscales)
        if wtscale is not None:
            layer.register_parameter("wtscale", wtscale)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.precision = self.precision
        layer.rank = self.rank
        layer.group_size = self.group_size
        layer.act_unsigned = self.act_unsigned

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is not None:
            set_weight_attrs(qweight, {"weight_loader": weight_loader})
            set_weight_attrs(wscales, {"weight_loader": weight_loader})
            set_weight_attrs(smooth_factor, {"weight_loader": weight_loader})
            set_weight_attrs(smooth_factor_orig, {"weight_loader": weight_loader})
            set_weight_attrs(proj_down, {"weight_loader": weight_loader})
            set_weight_attrs(proj_up, {"weight_loader": weight_loader})
            if wcscales is not None:
                set_weight_attrs(wcscales, {"weight_loader": weight_loader})
            if wtscale is not None:
                set_weight_attrs(wtscale, {"weight_loader": weight_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.wscales = Parameter(layer.wscales.data, requires_grad=False)
        layer.smooth_factor = Parameter(layer.smooth_factor.data, requires_grad=False)
        layer.smooth_factor_orig = Parameter(
            layer.smooth_factor_orig.data, requires_grad=False
        )
        layer.proj_down = Parameter(layer.proj_down.data, requires_grad=False)
        layer.proj_up = Parameter(layer.proj_up.data, requires_grad=False)
        if hasattr(layer, "wcscales") and layer.wcscales is not None:
            layer.wcscales = Parameter(layer.wcscales.data, requires_grad=False)
        if hasattr(layer, "wtscale") and layer.wtscale is not None:
            layer.wtscale = Parameter(layer.wtscale.data, requires_grad=False)

        alpha: float | None = None
        wtscale = getattr(layer, "wtscale", None)
        if wtscale is not None:
            if isinstance(wtscale, Parameter):
                wtscale = wtscale.data
            if isinstance(wtscale, torch.Tensor):
                alpha = float(wtscale.detach().cpu().item())
            else:
                alpha = float(wtscale)
        layer._nunchaku_alpha = alpha

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        quantized_x, ascales, lora_act_out = svdq_quantize_w4a4_act_fuse_lora_cuda(
            x_2d,
            lora_down=layer.proj_down,
            smooth=layer.smooth_factor,
            fp4=layer.precision == "nvfp4",
            pad_size=256,
        )
        out_2d = torch.empty(
            x_2d.shape[0],
            layer.output_size_per_partition,
            dtype=x_2d.dtype,
            device=x_2d.device,
        )
        alpha: float | None = getattr(layer, "_nunchaku_alpha", None)
        wcscales = getattr(layer, "wcscales", None)

        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=layer.qweight,
            out=out_2d,
            ascales=ascales,
            wscales=layer.wscales,
            lora_act_in=lora_act_out,
            lora_up=layer.proj_up,
            bias=bias,
            fp4=layer.precision == "nvfp4",
            alpha=alpha,
            wcscales=wcscales,
            act_unsigned=getattr(layer, "act_unsigned", False),
        )
        out = out_2d.reshape(*orig_shape[:-1], layer.output_size_per_partition)
        return out


class NunchakuAWQLinearMethod(LinearMethodBase):
    def __init__(self, group_size: int = 64):
        self.group_size = group_size
        self.pack_factor = 8

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)

        qweight = Parameter(
            torch.empty(
                output_size_per_partition // 4,
                input_size_per_partition // 2,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})

        num_groups = input_size_per_partition // self.group_size
        wscales = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        wzeros = Parameter(
            torch.empty(num_groups, output_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("wscales", wscales)
        layer.register_parameter("wzeros", wzeros)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.group_size = self.group_size
        layer.pack_factor = self.pack_factor

        weight_loader = extra_weight_attrs.get("weight_loader")
        if weight_loader is not None:
            set_weight_attrs(qweight, {"weight_loader": weight_loader})
            set_weight_attrs(wscales, {"weight_loader": weight_loader})
            set_weight_attrs(wzeros, {"weight_loader": weight_loader})

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.wscales = Parameter(layer.wscales.data, requires_grad=False)
        layer.wzeros = Parameter(layer.wzeros.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])

        in_features = layer.input_size_per_partition
        out_features = layer.output_size_per_partition
        out_2d = awq_gemv_w4a16_cuda(
            in_feats=x_2d,
            kernel=layer.qweight,
            scaling_factors=layer.wscales,
            zeros=layer.wzeros,
            m=x_2d.shape[0],
            n=out_features,
            k=in_features,
            group_size=layer.group_size,
        )
        if bias is not None:
            view_shape = [1] * (out_2d.ndim - 1) + [-1]
            out_2d.add_(bias.view(view_shape))

        out = out_2d.reshape(*orig_shape[:-1], out_features)
        return out
