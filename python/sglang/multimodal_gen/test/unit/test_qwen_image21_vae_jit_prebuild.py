"""The Qwen-Image 2.1 VAE mount builds its first decode's JIT modules together, before the first request."""

import torch
from torch import nn

from sglang.kernels.ops.diffusion.norm import bias_residual_add_jit
from sglang.kernels.ops.diffusion.norm import channel_rmsnorm_silu_nhwc_jit as nhwc
from sglang.multimodal_gen.runtime.models.vaes import qwen_image21_vae_cuda_opt
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import VaeFastPathGate
from sglang.multimodal_gen.runtime.models.vaes.qwen_image21_vae_cuda_opt import (
    FoldedPadConv2d,
    FusedChannelRMSNormSiLU,
    _build_first_decode_jit_modules,
    _first_decode_jit_builders,
)


class _Norm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))
        self.scale = channels**0.5


class _Conv(nn.Module):
    def __init__(self, channels: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        self._padding = (1, 1, 1, 1)


class _Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, gate, bias: bool = True) -> None:
        super().__init__()
        self.norm1 = FusedChannelRMSNormSiLU(_Norm(c_in), gate)
        self.conv1 = FoldedPadConv2d(_Conv(c_out, bias))
        self.norm2 = FusedChannelRMSNormSiLU(_Norm(c_out), gate)


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        gate = VaeFastPathGate()
        self.blocks = nn.ModuleList(
            [
                _Block(1280, 1280, gate),
                _Block(1280, 1280, gate),
                _Block(1280, 640, gate),
                _Block(320, 320, gate, bias=False),
            ]
        )
        self.norm_out = FusedChannelRMSNormSiLU(_Norm(160), gate)


def test_one_builder_per_module_the_default_decode_reaches():
    builders = _first_decode_jit_builders(_Decoder(), _Block)

    nhwc_keys = {b.args for b in builders if b.func is nhwc._jit_module}
    assert nhwc_keys == {
        (*nhwc._specialization(1280), False),
        (*nhwc._specialization(1280), True),
        (*nhwc._specialization(640), True),
        # no conv bias to fuse, and the output norm never takes one
        (*nhwc._specialization(320), False),
        (*nhwc._specialization(160), False),
    }
    others = [b for b in builders if b.func is not nhwc._jit_module]
    assert [(b.func, b.args) for b in others] == [
        (bias_residual_add_jit._jit_module, (True,))
    ]
    assert len(builders) == len(nhwc_keys) + 1


def test_builders_skip_widths_the_kernel_does_not_take():
    assert nhwc.channel_rmsnorm_silu_nhwc_builders([(3, False), (4096, True)]) == []


def test_a_failed_prebuild_is_left_to_the_first_use(monkeypatch):
    calls = []

    def failing_compile(builders):
        calls.append(len(list(builders)))
        raise RuntimeError("no compiler")

    monkeypatch.setattr(qwen_image21_vae_cuda_opt, "parallel_compile", failing_compile)
    _build_first_decode_jit_modules(_Decoder(), _Block)
    assert calls == [6]
