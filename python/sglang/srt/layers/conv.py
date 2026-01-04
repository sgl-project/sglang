import torch
import torch.nn as nn

from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.utils import cpu_has_amx_support, is_cpu, use_intel_amx_backend

_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
if _is_cpu and _is_cpu_amx_available:
    conv3d_embed = torch.ops.sgl_kernel.conv3d_embed_cpu


class Conv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if use_intel_amx_backend(self) and self.bias is not None:
            return conv3d_embed(
                input,
                self.weight,
                self.bias,
                is_vnni=True,
            )
        return super().forward(input)
