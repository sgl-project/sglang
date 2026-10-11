# SPDX-License-Identifier: Apache-2.0
"""LTX's frame callback must preserve the existing decode and final crop."""

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.ltx_video import (
    LTXVideoVAEArchConfig,
    LTXVideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import AutoencoderKLLTX2Video


def tiny_vae(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    arch = LTXVideoVAEArchConfig(
        latent_channels=4,
        spatial_compression_ratio=2,
        temporal_compression_ratio=2,
        block_out_channels=[8],
        down_block_types=["LTX2VideoDownBlock3D"],
        spatio_temporal_scaling=[True],
        layers_per_block=[1, 1],
        downsample_type=["spatiotemporal"],
        patch_size=1,
        decoder_block_out_channels=[8],
        decoder_spatio_temporal_scaling=[True],
        decoder_layers_per_block=[1, 1],
        decoder_inject_noise=[False, False],
        upsample_residual=[True],
        upsample_factor=[2],
    )
    vae = AutoencoderKLLTX2Video(LTXVideoVAEConfig(arch_config=arch))
    vae.tile_sample_min_num_frames = 8
    vae.tile_sample_stride_num_frames = 4
    return vae.to(device=device, dtype=dtype).eval()


def legacy_temporal_decode(vae, z):
    # Preserve the original two-loop algorithm, independently of the new callback.
    ratio = vae.temporal_compression_ratio
    stride = vae.tile_sample_stride_num_frames
    rows = []
    for start in range(0, z.shape[2], stride // ratio):
        tile = vae.decoder(z[:, :, start : start + 8 // ratio + 1], None)
        rows.append(tile if start == 0 else tile[:, :, :-1])
    kept = []
    for index, tile in enumerate(rows):
        if index:
            tile = vae.blend_t(rows[index - 1], tile, 8 - stride)
        kept.append(tile[:, :, : stride + (index == 0)])
    return torch.cat(kept, dim=2)[:, :, : (z.shape[2] - 1) * ratio + 1]


@pytest.mark.parametrize("batch_size,slicing", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("temporal", [False, True])
@torch.no_grad()
def test_frames_are_final_and_match_legacy_decode(batch_size, slicing, temporal):
    vae = tiny_vae()
    vae.use_framewise_decoding = temporal
    vae.use_slicing = slicing
    z = torch.randn(batch_size, 4, 7, 4, 4)
    expected = legacy_temporal_decode(vae, z) if temporal else vae.decode(z).sample
    chunks = []
    snapshots = []

    def consume(frames):
        chunks.append(frames)
        snapshots.append(frames.clone())

    actual = vae.decode(z, return_dict=False, on_frames=consume)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), actual, rtol=0, atol=0)
    for frames, snapshot in zip(chunks, snapshots):
        torch.testing.assert_close(frames, snapshot, rtol=0, atol=0)
    assert len(chunks) > 1 if temporal and batch_size == 1 else len(chunks) == 1
