# SPDX-License-Identifier: Apache-2.0
# 3D convolution for the MiniMax H3 visual VAE.
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        padding_mode_t = "constant" if padding_mode_t == "zeros" else padding_mode_t
        self.pad_mode = padding_mode
        self.pad_mode_t = padding_mode_t or ("constant" if causal else "replicate")
        self.causal = causal

    def _apply_temporal_padding(self, x):
        B, C, D, H, W = x.shape
        if D > 1:
            pad_size = (
                0,
                0,
                0,
                0,
                self.padding[0] * 2 if self.causal else self.padding[0],
                0 if self.causal else self.padding[0],
            )
            return F.pad(x, pad_size, mode=self.pad_mode_t)
        else:
            if self.pad_mode_t == "constant":
                assert self.causal, "Zeros padding is only supported for causal mode"
                return F.pad(
                    x,
                    (0, 0, 0, 0, self.kernel_size[0] - 1, 0),
                    mode="constant",
                )
            else:
                return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def _apply_padding(self, x):
        if sum(self.padding) == 0:
            return x

        if self.padding[1] or self.padding[2]:
            x = F.pad(
                x,
                (
                    self.padding[2],
                    self.padding[2],
                    self.padding[1],
                    self.padding[1],
                    0,
                    0,
                ),
                mode=self.pad_mode,
            )

        x = self._apply_temporal_padding(x)
        return x

    def forward(self, x, *, pre_padded=False):
        if (
            not pre_padded
            and not torch.is_grad_enabled()
            and x.device.type == "npu"
            and x.ndim == 5
            and x.shape[2] == 1
            and self.dilation[0] == 1
        ):
            if sum(self.padding) == 0 and self.kernel_size[0] == 1:
                return F.conv2d(
                    x.squeeze(2),
                    self.weight.squeeze(2),
                    self.bias,
                    stride=self.stride[1:],
                    padding=0,
                    dilation=self.dilation[1:],
                    groups=self.groups,
                ).unsqueeze(2)
            if sum(self.padding) != 0 and self.causal and self.pad_mode_t == "constant":
                x2 = x.squeeze(2)
                if self.padding[1] or self.padding[2]:
                    x2 = F.pad(
                        x2,
                        (
                            self.padding[2],
                            self.padding[2],
                            self.padding[1],
                            self.padding[1],
                        ),
                        mode=self.pad_mode,
                    )
                return F.conv2d(
                    x2,
                    self.weight[:, :, -1, :, :],
                    self.bias,
                    stride=self.stride[1:],
                    padding=0,
                    dilation=self.dilation[1:],
                    groups=self.groups,
                ).unsqueeze(2)

        if sum(self.padding) == 0:
            return super().forward(x)

        if not pre_padded:
            x = self._apply_padding(x)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )
