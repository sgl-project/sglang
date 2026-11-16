# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.multimodal_gen.runtime.models.vaes import wanvae
from sglang.multimodal_gen.runtime.models.vaes.resample import AvgDown3D, DupUp3D


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("factor_t,factor_s", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_resample_layout_and_causal_crop(device, dtype, factor_t, factor_s):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    out_channels = 2 * factor_t * factor_s**2
    down = AvgDown3D(4, out_channels, factor_t, factor_s)
    up = DupUp3D(out_channels, 4, factor_t, factor_s)
    causal_up = wanvae.DupUp3D(out_channels, 4, factor_t, factor_s)
    for frames in (1, 3, 4):
        x = torch.arange(2 * 4 * frames * 4 * 6, device=device).reshape(
            2, 4, frames, 4, 6
        )
        x = (x % 31).to(dtype)
        for layout in (torch.contiguous_format, torch.channels_last_3d):
            x = x.contiguous(memory_format=layout)
            padded = F.pad(x, (0, 0, 0, 0, (-frames) % factor_t, 0))
            packed = rearrange(
                padded,
                "b c (t ft) (h fh) (w fw) -> b (c ft fh fw) t h w",
                ft=factor_t,
                fh=factor_s,
                fw=factor_s,
            )
            expected = packed.reshape(2, out_channels, 2, *packed.shape[2:]).mean(2)
            reduced = down(x)
            torch.testing.assert_close(reduced, expected, rtol=0, atol=0)
            expanded = rearrange(
                reduced.repeat_interleave(2, dim=1),
                "b (c ft fh fw) t h w -> b c (t ft) (h fh) (w fw)",
                ft=factor_t,
                fh=factor_s,
                fw=factor_s,
            )
            for first in (False, True, False):
                expected = expanded[:, :, factor_t - 1 :] if first else expanded
                token = wanvae.first_chunk.set(first)
                try:
                    torch.testing.assert_close(
                        causal_up(reduced), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        up(reduced, first), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(up(reduced), expanded, rtol=0, atol=0)
                finally:
                    wanvae.first_chunk.reset(token)
    assert not down.state_dict() and not up.state_dict() and not causal_up.state_dict()
