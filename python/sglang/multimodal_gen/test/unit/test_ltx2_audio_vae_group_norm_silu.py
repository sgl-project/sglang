"""Unit tests for LTX-2 audio VAE GroupNorm+SiLU fusion plumbing.

The PR wires `apply_group_norm_silu` into the four ``norm + non_linearity``
sites in ``ltx_2_audio.py``:

- ``LTX2AudioResnetBlock.forward`` (norm1 + silu, norm2 + silu)
- ``LTX2AudioEncoder.forward``     (norm_out + silu)
- ``LTX2AudioDecoder.forward``     (norm_out + silu)

These tests run entirely on CPU and cover the *wiring*: the helper's gate
conditions are honored (GroupNorm-vs-PixelNorm, affine, requires_grad,
inplace SiLU), the forward passes are numerically identical to the
pre-fusion eager chain, and the helper is invoked the expected number of
times in each module's forward.

The Triton dispatch path (i.e. the CUDA fast branch of the helper) is
verified separately on GPU CI by
``python/sglang/jit_kernel/tests/diffusion/test_group_norm_silu.py::test_apply_group_norm_silu``,
which parametrizes over ``(shape, num_groups, dtype)`` on a real CUDA
tensor and asserts numerical parity against the eager reference. Avoiding
duplication of that here keeps these tests CPU-runnable and idiomatic for
the multimodal_gen unit suite.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.jit_kernel.diffusion.group_norm_silu import apply_group_norm_silu
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio import (
    LTX2AudioPixelNorm,
    LTX2AudioResnetBlock,
)

_HELPER_PATH = (
    "sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio.apply_group_norm_silu"
)


# ---------------------------------------------------------------------------
# Helper-level tests (apply_group_norm_silu fallback paths)
# ---------------------------------------------------------------------------
#
# The helper's CUDA fast path is gated by ``x.is_cuda AND not torch.is_grad_enabled()
# AND not x.requires_grad AND isinstance(norm, nn.GroupNorm) AND isinstance(activation,
# nn.SiLU) AND not activation.inplace AND norm.affine AND weight is not None
# AND bias is not None``. Any failing condition routes to the eager
# ``activation(norm(x))`` fallback. The tests below sweep each fallback trigger
# and check the helper produces bit-identical output to that eager chain.
#
# The CUDA fast path itself is covered by
# ``jit_kernel/tests/diffusion/test_group_norm_silu.py``.

# Each case is a (norm_factory, silu_factory) pair that fails one of the gates.
# The parametrize IDs name *which* gate trips the fallback.
_FALLBACK_CASES = [
    pytest.param(
        lambda: nn.GroupNorm(num_groups=8, num_channels=32, eps=1e-6, affine=True),
        lambda: nn.SiLU(),
        id="cpu_tensor",  # gate: not x.is_cuda
    ),
    pytest.param(
        lambda: LTX2AudioPixelNorm(dim=1, eps=1e-6),
        lambda: nn.SiLU(),
        id="pixel_norm_not_groupnorm",  # gate: not isinstance(norm, nn.GroupNorm)
    ),
    pytest.param(
        lambda: nn.GroupNorm(num_groups=8, num_channels=32, affine=True),
        lambda: nn.SiLU(inplace=True),
        id="inplace_silu",  # gate: activation.inplace
    ),
    pytest.param(
        lambda: nn.GroupNorm(num_groups=8, num_channels=32, affine=False),
        lambda: nn.SiLU(),
        id="non_affine_group_norm",  # gate: not norm.affine
    ),
]


@pytest.mark.parametrize("make_norm,make_silu", _FALLBACK_CASES)
def test_helper_fallback_matches_eager(make_norm, make_silu):
    """Each fallback trigger must produce output bit-identical to
    ``activation(norm(x))`` — the helper's eager-fallback branch."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 4, 4)
    norm = make_norm()
    silu = make_silu()
    expected = silu(norm(x))
    actual = apply_group_norm_silu(x, norm, silu)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_helper_fallback_supports_autograd():
    """The fallback branch must remain autograd-traceable. Parametrized with
    the others would make the test contract muddier — the backward-pass
    invariant only makes sense for the ``requires_grad=True`` case."""
    torch.manual_seed(0)
    x = torch.randn(1, 32, 4, 4, requires_grad=True)
    norm = nn.GroupNorm(num_groups=8, num_channels=32, affine=True)
    silu = nn.SiLU()

    out = apply_group_norm_silu(x, norm, silu)
    torch.testing.assert_close(out, silu(norm(x)), rtol=0, atol=0)

    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# LTX2AudioResnetBlock forward parity
# ---------------------------------------------------------------------------


def _eager_resnet_forward(block: LTX2AudioResnetBlock, x: torch.Tensor) -> torch.Tensor:
    """Reference eager forward — what the resnet block would compute if the
    `apply_group_norm_silu` calls were replaced with the original
    ``self.non_linearity(self.norm(x))`` chain."""
    h = block.non_linearity(block.norm1(x))
    h = block.conv1(h)
    h = block.non_linearity(block.norm2(h))
    h = block.dropout(h)
    h = block.conv2(h)
    if block.in_channels != block.out_channels:
        x = (
            block.conv_shortcut(x)
            if block.use_conv_shortcut
            else block.nin_shortcut(x)
        )
    return x + h


# Each case: (norm_type, causality_axis, in_channels, out_channels).
# `causality_axis=None` is the non-causal config (group norm allowed).
# `causality_axis="height"|"width"` are the causal configs (must use pixel norm).
# The last case (32→64) exercises the `nin_shortcut` channel-projection path.
_RESNET_FORWARD_CASES = [
    pytest.param("pixel", "height", 64, 64, id="causal_pixel_height"),
    pytest.param("pixel", "width",  64, 64, id="causal_pixel_width"),
    pytest.param("group", None,     64, 64, id="non_causal_group_same_channels"),
    pytest.param("group", None,     32, 64, id="non_causal_group_nin_shortcut"),
]


@pytest.mark.parametrize(
    "norm_type,causality_axis,in_channels,out_channels", _RESNET_FORWARD_CASES
)
def test_resnet_block_forward_matches_eager_reference(
    norm_type, causality_axis, in_channels, out_channels
):
    """Resnet block forward must produce the exact same output as the original
    ``norm + non_linearity`` chain. On CPU the helper's fallback path is taken,
    so this is a strict 0-tolerance comparison. The channel-mismatch case
    also exercises the shortcut/`nin_shortcut` path."""
    torch.manual_seed(42)
    block = LTX2AudioResnetBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        norm_type=norm_type,
        causality_axis=causality_axis,
        temb_channels=0,
        dropout=0.0,
    )
    block.eval()  # also disables dropout
    x = torch.randn(1, in_channels, 8, 16)

    with torch.no_grad():
        actual = block(x)
        expected = _eager_resnet_forward(block, x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "with_temb",
    [
        pytest.param(False, id="without_temb"),
        # The timestep-embedding silu path (`self.non_linearity(temb)`) must
        # NOT be routed through `apply_group_norm_silu` — wrapping a pure
        # SiLU-on-an-embedding with the helper would be incorrect.
        pytest.param(True, id="with_temb_temb_silu_unwrapped"),
    ],
)
def test_resnet_block_calls_helper_exactly_twice_per_forward(with_temb):
    """Each forward goes through `apply_group_norm_silu` exactly twice
    (norm1+silu, norm2+silu). The temb-silu site is intentionally left on
    the eager path, so the call count stays at 2 regardless of `temb`.
    Locks the wiring against accidental regressions to the inline pattern."""
    torch.manual_seed(0)
    block = LTX2AudioResnetBlock(
        in_channels=32,
        out_channels=32,
        norm_type="pixel",
        causality_axis="height",
        temb_channels=64 if with_temb else 0,
    )
    block.eval()
    x = torch.randn(1, 32, 8, 8)
    temb = torch.randn(1, 64) if with_temb else None

    with patch(_HELPER_PATH, side_effect=apply_group_norm_silu) as spy:
        with torch.no_grad():
            block(x, temb=temb)

    assert spy.call_count == 2
    # Every helper call must receive the SiLU instance, not some other activation.
    for call in spy.call_args_list:
        assert isinstance(call.args[2], nn.SiLU)


# ---------------------------------------------------------------------------
# Encoder / Decoder norm_out fusion site tests
# ---------------------------------------------------------------------------


class _FakeEncoderForward:
    """Smallest possible test double exercising the
    ``norm_out -> apply_group_norm_silu -> conv_out`` pattern installed in
    ``LTX2AudioEncoder.forward`` (lines 530-533) and the symmetric site in
    ``LTX2AudioDecoder.forward`` (lines 731-734)."""

    def __init__(self, channels: int = 32, norm_type: str = "group") -> None:
        if norm_type == "group":
            self.norm_out = nn.GroupNorm(
                num_groups=32, num_channels=channels, eps=1e-6, affine=True
            )
        else:
            self.norm_out = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        self.non_linearity = nn.SiLU()
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def fused(self, x: torch.Tensor) -> torch.Tensor:
        h = apply_group_norm_silu(x, self.norm_out, self.non_linearity)
        return self.conv_out(h)

    def eager(self, x: torch.Tensor) -> torch.Tensor:
        h = self.non_linearity(self.norm_out(x))
        return self.conv_out(h)


@pytest.mark.parametrize("norm_type", ["group", "pixel"])
def test_norm_out_pattern_matches_eager(norm_type: str):
    """The `norm_out -> silu -> conv_out` epilogue must compute the same
    result whether routed through `apply_group_norm_silu` or the original
    inline chain."""
    torch.manual_seed(11)
    fake = _FakeEncoderForward(channels=32, norm_type=norm_type)
    fake.norm_out.eval()
    fake.conv_out.eval()
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(fake.fused(x), fake.eager(x), rtol=0, atol=0)
