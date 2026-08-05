from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    DupUp3D,
    forward_context,
    residual_up_block_forward,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "factor_t", "first_chunk", "channels_last"),
    (
        (1024, 1024, 2, False, False),
        (1024, 1024, 2, True, True),
        (1024, 512, 2, False, False),
        (512, 256, 1, False, False),
    ),
)
def test_add_into_matches_materialized_shortcut(
    in_channels: int,
    out_channels: int,
    factor_t: int,
    first_chunk: bool,
    channels_last: bool,
):
    torch.manual_seed(0)
    dup = DupUp3D(in_channels, out_channels, factor_t=factor_t, factor_s=2)
    frames = 1 if first_chunk else 3
    x = torch.randn(
        1,
        in_channels,
        frames,
        2,
        3,
        dtype=torch.bfloat16,
        device=_DEVICE,
    )
    with forward_context(first_chunk_arg=first_chunk):
        shortcut = dup(x)
    x_main = torch.randn_like(shortcut)
    if channels_last:
        x_main = x_main.contiguous(memory_format=torch.channels_last_3d)
    expected = x_main + shortcut
    destination = x_main.clone(memory_format=torch.preserve_format)

    assert dup.can_fuse_add(destination, x, first_chunk)
    actual = dup.add_into_(destination, x, first_chunk)

    assert actual is destination
    assert torch.equal(actual, expected)
    if channels_last:
        assert actual.stride() == x_main.stride()
        assert actual.is_contiguous(memory_format=torch.channels_last_3d)


def test_residual_up_block_uses_fused_shortcut():
    torch.manual_seed(0)
    dup = DupUp3D(64, 32, factor_t=2, factor_s=2)
    x = torch.randn(1, 64, 1, 2, 3, device=_DEVICE)
    with forward_context(first_chunk_arg=True):
        shortcut = dup(x)
    x_main = torch.randn_like(shortcut)
    expected = x_main + shortcut
    block = SimpleNamespace(
        avg_shortcut=dup,
        resnets=(),
        upsampler=lambda _: x_main.clone(memory_format=torch.preserve_format),
    )

    with (
        patch.object(
            dup,
            "forward",
            side_effect=AssertionError("materialized shortcut path was used"),
        ),
        forward_context(first_chunk_arg=True),
    ):
        actual = residual_up_block_forward(block, x)

    assert torch.equal(actual, expected)
