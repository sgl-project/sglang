# SPDX-License-Identifier: Apache-2.0
"""Splitting the diffusion decoder's tiles across ranks must not move the output.

The tiles of a diffusion decode are independent, so they can be shared out over
the ranks that would otherwise each decode all of them. That is only a
performance optimization if the result is bit-for-bit what the single-rank,
tile-by-tile decode produced -- which means the noise has to keep coming off
one generator in one global tile order, however few tiles a rank ends up
decoding.

The distributed cases run over gloo on CPU, so they need no GPU.
"""

import math
import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX2VideoDiffusionDecoder3d,
    LTX2VideoDiffusionDecoderModel,
    LTX2VideoVaePixelShuffleUpsampler,
    _all_gather_tiles,
    _tile_intervals,
)

PARALLEL_STATE = "sglang.multimodal_gen.runtime.distributed.parallel_state"


class _Upsampler:
    """Just the stride; the stub reproduces the shuffle itself."""

    def __init__(self, stride):
        self.stride = stride


class _StubDecoder:
    """A decoder with the real stage geometry and arithmetic values.

    `tiled_decode` only needs the shapes to line up and the values to depend on
    both the tile's content and its noise -- that is enough for a misplaced
    tile or a misdrawn noise to show up as a different output.
    """

    # Under test: the shape prediction that lets a rank size a tile's noise
    # without paying for the tile's context.
    stage_4_output_extent = LTX2VideoDiffusionDecoder3d.stage_4_output_extent

    def __init__(self, num_inference_steps=1):
        self.upsamples = [_Upsampler((2, 1, 1)), _Upsampler((2, 2, 2))]
        self.temporal_compression_ratio = math.prod(u.stride[0] for u in self.upsamples)
        self.trailing_pad_latent_frames = 2
        self.patch_size = 2
        self.out_channels = 3
        self.context_channels = 4
        self.default_num_inference_steps = num_inference_steps
        self.model_output_type = "x0"
        self.drawn_noise = []

    @property
    def ghost(self) -> int:
        return self.trailing_pad_latent_frames * math.prod(
            u.stride[0] for u in self.upsamples[:-1]
        )

    def forward_stages_1_to_3(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = hidden_states.mean(dim=1)
        trailing = features[:, -1:].repeat(1, self.ghost, 1, 1)
        features = torch.cat([features, trailing], dim=1)
        channels = torch.arange(
            self.context_channels, dtype=features.dtype, device=features.device
        )
        return features.unsqueeze(-1) + channels

    def forward_stage_4(
        self,
        hidden_states: torch.Tensor,
        drop_leading_frame: bool = True,
        crop_trailing_ghost: bool = True,
    ) -> torch.Tensor:
        stride_t, stride_h, stride_w = self.upsamples[-1].stride
        out = (
            hidden_states.repeat_interleave(stride_t, dim=1)
            .repeat_interleave(stride_h, dim=2)
            .repeat_interleave(stride_w, dim=3)
        )
        if stride_t == 2 and drop_leading_frame:
            out = out[:, 1:]
        num_pad = self.trailing_pad_latent_frames
        if crop_trailing_ghost and num_pad > 0:
            out = out[:, : -num_pad * self.temporal_compression_ratio]
        return out

    def denoise(self, context, x_t, num_inference_steps):
        self.drawn_noise.append(x_t.clone())
        pixels = context[..., : self.out_channels].permute(0, 4, 1, 2, 3)
        pixels = pixels.repeat_interleave(self.patch_size, dim=3).repeat_interleave(
            self.patch_size, dim=4
        )
        return pixels + x_t * float(num_inference_steps)


def _build_model(num_inference_steps: int = 1) -> LTX2VideoDiffusionDecoderModel:
    model = LTX2VideoDiffusionDecoderModel.__new__(LTX2VideoDiffusionDecoderModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        arch_config=SimpleNamespace(
            decoder_stage_kernels=[(3, 3, 3)],
            decoder_stage5_kernel=(3, 3, 3),
        )
    )
    model.decoder = _StubDecoder(num_inference_steps)
    model.use_tiling = True
    model.use_parallel_tiling = False
    # Divided by (scale_t, scale_h, scale_w) = (2, 4, 4) inside `tiled_decode`,
    # so: 4-cell tiles on a 2-cell stride, on every axis.
    model.tile_sample_min_num_frames = 8
    model.tile_sample_stride_num_frames = 4
    model.tile_sample_min_height = 16
    model.tile_sample_stride_height = 8
    model.tile_sample_min_width = 16
    model.tile_sample_stride_width = 8
    return model


def _latent(shape, dtype=torch.float32) -> torch.Tensor:
    numel = math.prod(shape)
    return torch.arange(numel, dtype=torch.float32).reshape(shape).div_(numel).to(dtype)


# A 2 x 2 x 3 = 12-tile grid, and the width axis is ragged: 7 cells over a
# 2-cell stride leaves a last tile of 3 where the others are 4.
MANY_TILES = (1, 4, 6, 6, 7)
# One tile on every axis -- fewer tiles than ranks, so some rank decodes none.
ONE_TILE = (1, 4, 4, 4, 4)


def _decode(model, latent, seed, shard=(0, 1, None)):
    model._tile_shard = lambda: shard
    model.decoder.drawn_noise = []
    generator = torch.Generator().manual_seed(seed)
    return model.tiled_decode(latent, generator=generator)


def _serial_reference(model, latent, seed):
    """The pre-parallel decode: one generator, consumed tile by tile.

    Deliberately a second implementation of the tile walk rather than a call
    into the one under test -- it is the thing the optimization must not have
    changed.
    """
    decoder = model.decoder
    stride_t, stride_h, stride_w = decoder.upsamples[-1].stride
    patch = decoder.patch_size
    scale_t, scale_h, scale_w = stride_t, stride_h * patch, stride_w * patch
    features = decoder.forward_stages_1_to_3(latent)
    num_frames = features.shape[1] - decoder.ghost
    axes = [
        _tile_intervals(num_frames, 8 // scale_t, 4 // scale_t, 3),
        _tile_intervals(features.shape[2], 16 // scale_h, 8 // scale_h, 3),
        _tile_intervals(features.shape[3], 16 // scale_w, 8 // scale_w, 3),
    ]
    generator = torch.Generator().manual_seed(seed)
    noise = []
    for t0, t1 in axes[0]:
        for h0, h1 in axes[1]:
            for w0, w1 in axes[2]:
                is_trailing = t1 == num_frames
                context = decoder.forward_stage_4(
                    features[
                        :,
                        t0 : features.shape[1] if is_trailing else t1,
                        h0:h1,
                        w0:w1,
                    ],
                    drop_leading_frame=t0 == 0,
                    crop_trailing_ghost=is_trailing,
                )
                noise.append(
                    torch.randn(
                        (
                            latent.shape[0],
                            decoder.out_channels,
                            context.shape[1],
                            context.shape[2] * patch,
                            context.shape[3] * patch,
                        ),
                        generator=generator,
                        dtype=latent.dtype,
                    )
                )
    return noise


class TestStage4OutputExtent(unittest.TestCase):
    """The predicted extent has to be what stage 4 actually produces."""

    def _decoder(self, stride):
        decoder = _StubDecoder()
        decoder.upsamples = [_Upsampler(stride)]
        decoder.temporal_compression_ratio = stride[0]
        return decoder

    def test_matches_the_real_upsampler(self):
        for stride in [(2, 2, 2), (1, 2, 2), (2, 1, 1)]:
            for drop, crop in [(True, True), (True, False), (False, True)]:
                with self.subTest(stride=stride, drop=drop, crop=crop):
                    decoder = self._decoder(stride)
                    upsampler = LTX2VideoVaePixelShuffleUpsampler(8, stride)
                    hidden = torch.zeros(1, 6, 5, 4, 8)
                    with torch.no_grad():
                        out = upsampler(hidden, drop_leading_frame=drop)
                    num_pad = decoder.trailing_pad_latent_frames
                    if crop and num_pad > 0:
                        out = out[:, : -num_pad * decoder.temporal_compression_ratio]
                    self.assertEqual(
                        decoder.stage_4_output_extent(
                            6, 5, 4, drop_leading_frame=drop, crop_trailing_ghost=crop
                        ),
                        tuple(out.shape[1:4]),
                    )


class TestSerialDecodeIsUnchanged(unittest.TestCase):
    """A single rank must still see the noise the pre-parallel decode saw."""

    def test_noise_matches_the_pre_parallel_stream(self):
        for shape in (MANY_TILES, ONE_TILE):
            with self.subTest(tiles=shape):
                model = _build_model()
                latent = _latent(shape)
                _decode(model, latent, seed=7)
                expected = _serial_reference(model, latent, seed=7)
                self.assertEqual(len(model.decoder.drawn_noise), len(expected))
                for got, want in zip(model.decoder.drawn_noise, expected):
                    self.assertTrue(torch.equal(got, want))

    def test_the_request_seed_still_changes_the_result(self):
        model = _build_model()
        latent = _latent(MANY_TILES)
        self.assertFalse(
            torch.equal(_decode(model, latent, seed=7), _decode(model, latent, seed=8))
        )


class TestTileShard(unittest.TestCase):
    """Which ranks the tiles are split over."""

    def test_off_when_parallel_tiling_is_disabled(self):
        model = _build_model()
        model.use_parallel_tiling = False
        self.assertEqual(model._tile_shard(), (0, 1, None))

    def test_off_before_the_parallel_state_exists(self):
        model = _build_model()
        model.use_parallel_tiling = True
        with patch(
            f"{PARALLEL_STATE}.model_parallel_is_initialized", return_value=False
        ):
            self.assertEqual(model._tile_shard(), (0, 1, None))

    def test_uses_the_decode_parallel_group_not_the_sp_group(self):
        # The decoder is replicated over TP/SP/PP/CFG, so sharding only over SP
        # would leave the TP and CFG ranks redecoding the whole volume.
        model = _build_model()
        model.use_parallel_tiling = True
        group = object()
        coordinator = SimpleNamespace(device_group=group)
        with patch(
            f"{PARALLEL_STATE}.model_parallel_is_initialized", return_value=True
        ):
            with patch(
                f"{PARALLEL_STATE}.get_decode_parallel_world_size", return_value=4
            ):
                with patch(
                    f"{PARALLEL_STATE}.get_decode_parallel_rank", return_value=2
                ):
                    with patch(
                        f"{PARALLEL_STATE}.get_decode_parallel_group_coordinator",
                        return_value=coordinator,
                    ):
                        self.assertEqual(model._tile_shard(), (2, 4, group))

    def test_off_at_a_single_decode_rank(self):
        model = _build_model()
        model.use_parallel_tiling = True
        with patch(
            f"{PARALLEL_STATE}.model_parallel_is_initialized", return_value=True
        ):
            with patch(
                f"{PARALLEL_STATE}.get_decode_parallel_world_size", return_value=1
            ):
                self.assertEqual(model._tile_shard(), (0, 1, None))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _check_gather(group, rank, world_size, dtype, tile_shapes):
    """Every rank must rebuild the same tile list, whatever it held itself."""
    tiles = [
        torch.full(shape, float(i + 1), dtype=dtype)
        for i, shape in enumerate(tile_shapes)
    ]
    local_indices = list(range(rank, len(tiles), world_size))
    gathered = _all_gather_tiles(
        [tiles[i] for i in local_indices],
        local_indices,
        len(tiles),
        group,
        torch.device("cpu"),
    )
    assert len(gathered) == len(tiles)
    for want, got in zip(tiles, gathered):
        assert got is not None, "a tile went missing from the gather"
        assert got.dtype == dtype, f"{got.dtype} != {dtype}"
        assert torch.equal(got, want)


def _check_sharded_decode(group, rank, world_size, dtype, latent_shape, steps):
    model = _build_model(steps)
    latent = _latent(latent_shape, dtype)
    serial = _decode(model, latent, seed=11)
    serial_noise = list(model.decoder.drawn_noise)
    sharded = _decode(model, latent, seed=11, shard=(rank, world_size, group))
    assert torch.equal(serial, sharded), "sharding the tiles moved the output"
    # ...and this rank did less work to get there, on the tiles it owns.
    owned = list(range(rank, len(serial_noise), world_size))
    assert len(model.decoder.drawn_noise) == len(owned)
    for index, got in zip(owned, model.decoder.drawn_noise):
        assert torch.equal(got, serial_noise[index])


def _distributed_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        # A group narrower than the world, so an even split gets covered too.
        pair = dist.new_group([0, 1])
        groups = [(dist.group.WORLD, rank, world_size)]
        if rank < 2:
            groups.append((pair, rank, 2))

        for group, group_rank, group_size in groups:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Ragged tiles, and more tiles than ranks.
                _check_gather(
                    group,
                    group_rank,
                    group_size,
                    dtype,
                    [(2, 3, 4), (2, 3, 4), (2, 3, 2), (1, 3, 4)],
                )
                # Fewer tiles than ranks: the tail ranks send nothing, and have
                # only the metadata to tell them what dtype to send it as.
                _check_gather(group, group_rank, group_size, dtype, [(2, 3, 4)])

                for latent_shape in (MANY_TILES, ONE_TILE):
                    for steps in (1, 2):
                        _check_sharded_decode(
                            group,
                            group_rank,
                            group_size,
                            dtype,
                            latent_shape,
                            steps,
                        )
    finally:
        dist.destroy_process_group()


class TestShardedDecode(unittest.TestCase):
    """The real gather, over a real process group."""

    def test_three_ranks_agree_with_the_serial_decode(self):
        mp.spawn(
            _distributed_worker,
            args=(3, _free_port()),
            nprocs=3,
            join=True,
        )


class TestTileIntervals(unittest.TestCase):
    def test_intervals_cover_the_axis(self):
        intervals = _tile_intervals(30, 12, 8, 4)
        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][1], 30)

    def test_a_short_remnant_is_merged_into_the_previous_tile(self):
        # 25 with stride 8 would leave a 1-long trailing tile, which is below
        # the neighborhood kernel and cannot be decoded on its own.
        for start, end in _tile_intervals(25, 12, 8, 4):
            self.assertGreaterEqual(end - start, 4)

    def test_an_axis_shorter_than_one_tile_stays_whole(self):
        self.assertEqual(_tile_intervals(5, 12, 8, 4), [(0, 5)])

    def test_a_round_robin_split_covers_every_tile_exactly_once(self):
        # How `tiled_decode` assigns tiles: rank r takes r, r+W, r+2W, ...
        for total in (1, 7, 14, 15):
            for world_size in (1, 2, 3, 4):
                assigned = [
                    i
                    for rank in range(world_size)
                    for i in range(rank, total, world_size)
                ]
                self.assertEqual(sorted(assigned), list(range(total)))


if __name__ == "__main__":
    unittest.main()
