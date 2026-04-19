"""
Conv2d/Conv3d layers with unfold+linear optimization for patch embeddings.

When kernel_size == stride, padding == 0, dilation == 1, groups == 1, the conv
is equivalent to unfold + F.linear, which is significantly faster on CUDA and
also avoids the PyTorch 2.9.1 + CuDNN < 9.15 Conv3d bug
(https://github.com/pytorch/pytorch/issues/168167).
"""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

_VALID_PADDING_STRINGS = {"same", "valid"}
_VALID_PADDING_MODES = {"zeros", "reflect", "replicate", "circular"}


def _tuplify(val, n: int) -> tuple:
    if isinstance(val, (list, tuple)):
        assert len(val) == n
        return tuple(val)
    return (val,) * n


def _check_enable_linear(
    kernel_size: tuple,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> bool:
    """Check if conv can be replaced with unfold + F.linear."""
    return (
        kernel_size == stride
        and all(p == 0 for p in padding)
        and all(d == 1 for d in dilation)
        and groups == 1
    )


def _reverse_repeat_tuple(t: tuple) -> tuple:
    """(1, 2, 3) -> (3, 3, 2, 2, 1, 1). Used for F.pad with non-zeros padding_mode."""
    return tuple(x for x in reversed(t) for _ in range(2))


def _compute_same_padding_for_pad(kernel_size: tuple, dilation: tuple) -> tuple:
    """Compute _reversed_padding_repeated_twice for padding='same'.

    This mirrors PyTorch's nn.Conv*d behavior: pre-compute the exact pad
    amounts so that F.pad can be called before F.conv*d(padding=0).
    """
    pad = []
    for k, d in zip(reversed(kernel_size), reversed(dilation)):
        total = d * (k - 1)
        pad.append(total // 2)
        pad.append(total - total // 2)
    return tuple(pad)


def _validate_conv_args(
    in_channels: int,
    out_channels: int,
    groups: int,
    padding,
    padding_mode: str,
    stride: tuple,
) -> None:
    if in_channels % groups != 0:
        raise ValueError(
            f"in_channels ({in_channels}) must be divisible by groups ({groups})"
        )
    if out_channels % groups != 0:
        raise ValueError(
            f"out_channels ({out_channels}) must be divisible by groups ({groups})"
        )
    if padding_mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {_VALID_PADDING_MODES}, got '{padding_mode}'"
        )
    if isinstance(padding, str):
        if padding not in _VALID_PADDING_STRINGS:
            raise ValueError(
                f"padding must be one of {_VALID_PADDING_STRINGS}, got '{padding}'"
            )
        if padding == "same" and any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")


class Conv2dLayer(MultiPlatformOp):
    """Drop-in replacement for nn.Conv2d. Linear optimization disabled by default."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        disable_linear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuplify(kernel_size, 2)
        self.stride = _tuplify(stride, 2)
        self.dilation = _tuplify(dilation, 2)
        self.groups = groups
        self.padding_mode = padding_mode

        _validate_conv_args(
            in_channels, out_channels, groups, padding, padding_mode, self.stride
        )

        if isinstance(padding, str):
            self.padding = (0, 0) if padding == "valid" else padding
        else:
            self.padding = _tuplify(padding, 2)

        # Pre-compute pad tuple for padding_mode != "zeros" (mirrors nn.Conv2d).
        # When padding="same", we need numeric values for F.pad;
        # when padding is already numeric, _reverse_repeat_tuple handles it.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = _compute_same_padding_for_pad(
                self.kernel_size, self.dilation
            )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding)

        padding_tuple = self.padding if isinstance(self.padding, tuple) else (1, 1)
        self.enable_linear = not disable_linear and _check_enable_linear(
            self.kernel_size, self.stride, padding_tuple, self.dilation, groups
        )

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        K1, K2 = self.kernel_size
        x = x.unfold(2, K1, K1).unfold(3, K2, K2)
        N, _, Hp, Wp = x.shape[:4]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(N, Hp, Wp, -1)
        x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)
        return x.permute(0, 3, 1, 2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)


class Conv3dLayer(MultiPlatformOp):
    """Drop-in replacement for nn.Conv3d with automatic linear optimization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int], str] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        disable_linear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuplify(kernel_size, 3)
        self.stride = _tuplify(stride, 3)
        self.dilation = _tuplify(dilation, 3)
        self.groups = groups
        self.padding_mode = padding_mode

        _validate_conv_args(
            in_channels, out_channels, groups, padding, padding_mode, self.stride
        )

        if isinstance(padding, str):
            self.padding = (0, 0, 0) if padding == "valid" else padding
        else:
            self.padding = _tuplify(padding, 3)

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = _compute_same_padding_for_pad(
                self.kernel_size, self.dilation
            )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding)

        padding_tuple = self.padding if isinstance(self.padding, tuple) else (1, 1, 1)
        self.enable_linear = not disable_linear and _check_enable_linear(
            self.kernel_size, self.stride, padding_tuple, self.dilation, groups
        )

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        K1, K2, K3 = self.kernel_size
        x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)
        N, Dp, Hp, Wp = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(N, Dp, Hp, Wp, -1)
        x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)
        return x.permute(0, 4, 1, 2, 3)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight,
                self.bias,
                self.stride,
                (0, 0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_linear:
            return self._forward_mulmat(x)
        return self._forward_conv(x)
