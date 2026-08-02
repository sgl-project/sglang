# SPDX-License-Identifier: Apache-2.0
# MiniMax H3 visual VAE: 3D causal CNN encoder + ViT3D decoder (inference-only bundle).
import math
import os
from typing import List, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models import ModelMixin
from diffusers.utils import logging
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import (
    get_decode_parallel_group_coordinator,
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
    model_parallel_is_initialized,
)

from .processor import (
    VAEProcessor,
    get_denormalize_transform,
    get_normalize_transform,
)
from .vae_cnn import EncoderFCN3D
from .vae_vit import ViT3DDecoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _resolve_temporal_cat_dtype():
    raw = (
        os.environ.get("MINIMAX_H3_VAE_DECODER_TEMPORAL_CAT_DTYPE", "").strip().lower()
    )
    if raw in ("", "0", "false", "no", "off", "none", "keep", "default"):
        return None
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if raw not in mapping:
        raise ValueError(
            "MINIMAX_H3_VAE_DECODER_TEMPORAL_CAT_DTYPE must be one of "
            "fp16|bf16|fp32|keep, got %r" % raw
        )
    return mapping[raw]


def _resolve_temporal_stream_cat():
    raw = (
        os.environ.get("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", "1")
        .strip()
        .lower()
    )
    return raw not in ("0", "false", "no", "off", "disable", "disabled")


def get_tile_parallel_state():
    if not dist.is_initialized() or not model_parallel_is_initialized():
        return 0, 1
    return get_decode_parallel_rank(), get_decode_parallel_world_size()


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, upcast_fp32=True):
        if upcast_fp32:
            parameters = parameters.to(dtype=torch.float32)

        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = self.logvar.mul(0.5).exp_()

    @torch.compiler.disable
    def sample(self, generator=None):
        noise = torch.randn(self.mean.shape, generator=generator)
        noise = noise.to(device=self.parameters.device)
        return noise.mul_(self.std).add_(self.mean)


class ClsTokenAggregator:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.cls_tokens = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cls_tokens and hasattr(self.vae.encoder, "loss_info"):
            self.vae.encoder.loss_info["cls_token"] = torch.stack(
                self.cls_tokens, dim=0
            ).mean(dim=0)
        return False

    def collect(self):
        if (
            hasattr(self.vae.encoder, "loss_info")
            and "cls_token" in self.vae.encoder.loss_info
        ):
            self.cls_tokens.append(self.vae.encoder.loss_info["cls_token"].clone())

    def collect_stacked(self, num_tiles, sample_batch_size):
        if (
            hasattr(self.vae.encoder, "loss_info")
            and "cls_token" in self.vae.encoder.loss_info
        ):
            cls_token = self.vae.encoder.loss_info["cls_token"]
            cls_token = cls_token.unflatten(0, (num_tiles, sample_batch_size))
            self.cls_tokens.extend(token.clone() for token in cls_token)


class AutoencoderKL(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    Abstract shared base for the MiniMax H3 visual VAE.

    This class only carries the shared inference machinery (temporal
    chunking, tiling, encode/decode entry points). Instantiate the concrete
    subclass ``AutoencoderKLLegacy`` via ``from_pretrained`` instead.
    """

    _compilable_modules = ["encoder", "decoder"]
    _deprecated_kwargs = [
        "clip_length",
        "token_drop",
        "isolated_first_frame",
        "isolated_last_frame",
        "isolated_key_frame",
        "encoder_tiling",
        "decoder_tiling",
        "parallel_tiling",
        "stack_tiling",
        "tile_size",
        "tile_overlap_min",
        "decoder_tile_size",
        "decoder_tile_overlap_min",
        "latent_patch_size",
        "crop_mode",
        "encoder_parallel",
        "decoder_parallel",
        "chunk_dim",
    ]  # legacy config keys accepted by from_pretrained for checkpoint compatibility

    def setup_forward(self, **kwargs):
        self.clip_length = kwargs.get("clip_length", 17)
        self.token_drop = kwargs.get("token_drop", 0)
        self.frame_drop = self.token_drop * self.vae_ratio_t
        self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0
        )
        self.isolated_first_frame = kwargs.get("isolated_first_frame", False)
        self.isolated_last_frame = kwargs.get("isolated_last_frame", False)
        self.isolated_key_frame = kwargs.get("isolated_key_frame", False)

        self.encoder_tiling = kwargs.get("encoder_tiling", False)
        self.decoder_tiling = kwargs.get("decoder_tiling", False)
        self.stack_tiling = kwargs.get("stack_tiling", False)
        self.tile_size = kwargs.get("tile_size", 256)
        self.tile_overlap_min = kwargs.get("tile_overlap_min", 64)
        self.decoder_tile_size = kwargs.get("decoder_tile_size", self.tile_size)
        self.decoder_tile_overlap_min = kwargs.get(
            "decoder_tile_overlap_min", self.tile_overlap_min
        )
        self._blend_weight_cache = {}
        self.latent_patch_size = kwargs.get("latent_patch_size", 1)
        self.crop_mode = kwargs.get("crop_mode", "top_left")
        self.pixel_norm_type = kwargs.get("pixel_norm_type", "imagenet")

        if kwargs.get("encoder_parallel", False) or kwargs.get(
            "decoder_parallel", False
        ):
            raise ValueError(
                "MiniMax H3 VAE spatial sharding is unsupported; use complete-tile "
                "parallel decode instead"
            )
        parallel_tiling = kwargs.get("parallel_tiling", False)
        if hasattr(self, "parallel_tiling") and parallel_tiling != self.parallel_tiling:
            logger.warning(
                "Do not support changing parallel tiling after initialization"
            )
        else:
            self.parallel_tiling = parallel_tiling

        processor_kwargs = {
            "vae_ratio": self.vae_ratio,
            "vae_ratio_t": self.vae_ratio_t,
            "clip_length": self.clip_length,
            "frame_overlap": self.frame_overlap,
            "token_overlap": self.token_overlap,
            "tokens_chunk_size": self.tokens_chunk_size,
            "isolated_last_frame": self.isolated_last_frame,
            "latent_patch_size": self.latent_patch_size,
            "crop_mode": self.crop_mode,
            "pixel_norm_type": self.pixel_norm_type,
            "transform": self.transform,
            "transform_rev": self.transform_rev,
            "use_3d_conv": self.use_3d_conv,
        }
        if hasattr(self, "processor"):
            for key, value in processor_kwargs.items():
                setattr(self.processor, key, value)
        else:
            self.processor = VAEProcessor(**processor_kwargs)

    def split_tiles(self, input_len, is_decoder=False):
        tile_size = self.decoder_tile_size if is_decoder else self.tile_size
        tile_overlap_min = (
            self.decoder_tile_overlap_min if is_decoder else self.tile_overlap_min
        )

        if tile_size >= input_len:
            return [0], [input_len], []

        N = math.ceil(input_len / tile_size)
        while True:
            overlaps = [tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len

            if remaining < 0:
                N += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])

        tile_len = [tile_size] * N
        return tile_start_idx, tile_len, overlaps

    def blend(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)

        cache_key = (blend_extent, b.device, b.dtype)
        weights = self._blend_weight_cache.get(cache_key)
        if weights is None:
            positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
            weights = (1 - positions / blend_extent, positions / blend_extent)
            self._blend_weight_cache[cache_key] = weights
        weight_a, weight_b = weights

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        a_overlap = a[tuple(slice_a)]

        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        b_overlap = b[tuple(slice_b)]

        blended = a_overlap * weight_a
        blended.add_(b_overlap * weight_b)

        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            b_rest = b[tuple(slice_b_rest)]
            return torch.cat([blended, b_rest], dim=dim)
        else:
            return blended

    def _assemble_tiles(self, rows, y_overlap, x_overlap):
        output_height = sum(
            row[0].shape[-2] - (y_overlap[i] if i < len(rows) - 1 else 0)
            for i, row in enumerate(rows)
        )
        output_width = sum(
            tile.shape[-1] - (x_overlap[j] if j < len(rows[0]) - 1 else 0)
            for j, tile in enumerate(rows[0])
        )
        output = rows[0][0].new_empty(
            (*rows[0][0].shape[:-2], output_height, output_width)
        )

        y_offset = 0
        # released VAE blends vertically before horizontally; preserve that
        # order while writing each cropped tile into the final tensor
        for i, row in enumerate(rows):
            row_height = row[0].shape[-2] - (y_overlap[i] if i < len(rows) - 1 else 0)
            x_offset = 0
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -x_overlap[j]]

                tile_height, tile_width = tile.shape[-2:]
                output[
                    ...,
                    y_offset : y_offset + tile_height,
                    x_offset : x_offset + tile_width,
                ].copy_(tile)
                x_offset += tile_width
            y_offset += row_height

        return output

    def _all_gather_tiled_results(self, tasks, num_tiles):
        tile_rank, tile_world_size = get_tile_parallel_state()

        if not tasks:
            raise ValueError(f"Found empty tasks on tile rank {tile_rank}")

        expected_tasks = (
            num_tiles - tile_rank + tile_world_size - 1
        ) // tile_world_size
        if len(tasks) != expected_tasks:
            raise ValueError(
                f"Expected {expected_tasks} tiled tasks on rank {tile_rank}, "
                f"got {len(tasks)} for num_tiles={num_tiles}, "
                f"world_size={tile_world_size}"
            )

        max_tasks = (num_tiles + tile_world_size - 1) // tile_world_size
        if len(tasks) == max_tasks:
            stacked = torch.stack(tasks, dim=0)
        else:
            stacked = tasks[0].new_empty((max_tasks, *tasks[0].shape))
            torch.stack(tasks, dim=0, out=stacked[: len(tasks)])
            stacked[len(tasks) :].zero_()

        # Round-robin tile ownership makes every rank's task count known from
        # num_tiles and world size. Pad only the leading task dimension and use a
        # single equal-shape all-gather; the previous path paid for a barrier,
        # a shape all-gather, and then a padded data all-gather per temporal
        # clip.
        gathered = get_decode_parallel_group_coordinator().all_gather(
            stacked, separate_tensors=True
        )

        results = [None] * num_tiles
        for rank, rank_tensors in enumerate(gathered):
            num_rank_tasks = (num_tiles - rank + tile_world_size - 1) // tile_world_size
            for k in range(num_rank_tasks):
                global_idx = k * tile_world_size + rank
                results[global_idx] = rank_tensors[k]

        return results

    def _local_tile_indices(self, num_tiles, tile_rank, tile_world_size):
        return list(range(tile_rank, num_tiles, tile_world_size))

    def _run_tile_tasks(
        self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None
    ):
        if stack_tiling and tile_indices:
            sample_batch_size = tiles[0].shape[0]
            tile_batch = torch.cat([tiles[idx] for idx in tile_indices], dim=0)
            output_batch = forward_fn(tile_batch)
            output_tiles = output_batch.unflatten(
                0, (len(tile_indices), sample_batch_size)
            ).unbind(dim=0)
            if cls_agg is not None:
                cls_agg.collect_stacked(len(tile_indices), sample_batch_size)
            return list(output_tiles)

        tasks = []
        for idx in tile_indices:
            tasks.append(forward_fn(tiles[idx]))
            if cls_agg is not None:
                cls_agg.collect()
        return tasks

    def tiled_encode(self, x):
        if self.parallel_tiling:  # Fast online encoding for large videos
            tile_rank, tile_world_size = get_tile_parallel_state()
        else:
            tile_rank, tile_world_size = 0, 1

        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height, False)
        x_idx, x_len, x_overlap = self.split_tiles(width, False)

        i_max, j_max = len(y_idx), len(x_idx)
        num_tiles = i_max * j_max
        if tile_world_size > num_tiles:
            # Every rank executes this replicated path. When the canvas has
            # fewer tiles than ranks, run locally instead of assigning
            # empty task lists that cannot participate in the tensor gather.
            tile_rank, tile_world_size = 0, 1

        x_tiles = []
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                x_tiles.append(tile)

        with ClsTokenAggregator(self) as agg:
            local_tile_indices = self._local_tile_indices(
                num_tiles, tile_rank, tile_world_size
            )
            stack_tiling = self.stack_tiling and not (
                self.training and getattr(self.encoder, "mask_enabled", False)
            )
            encoded_tasks = self._run_tile_tasks(
                x_tiles, local_tile_indices, self.encode, stack_tiling, agg
            )

            if tile_world_size > 1:
                all_encoded = self._all_gather_tiled_results(encoded_tasks, num_tiles)
                if agg.cls_tokens:
                    agg.cls_tokens = self._all_gather_tiled_results(
                        agg.cls_tokens, num_tiles
                    )
            else:
                all_encoded = encoded_tasks

        rows = [[None for _ in range(j_max)] for _ in range(i_max)]
        for idx, encoded in enumerate(all_encoded):
            i, j = idx // j_max, idx % j_max
            rows[i][j] = encoded.to(x.device)

        latent_y_overlap = [
            tile_overlap // self.vae_ratio for tile_overlap in y_overlap
        ]
        latent_x_overlap = [
            tile_overlap // self.vae_ratio for tile_overlap in x_overlap
        ]

        z = self._assemble_tiles(rows, latent_y_overlap, latent_x_overlap)

        return z

    def tiled_decode(self, z):
        if self.parallel_tiling:  # Fast online decoding for large videos
            tile_rank, tile_world_size = get_tile_parallel_state()
        else:
            tile_rank, tile_world_size = 0, 1

        height, width = (
            z.shape[-2] * self.vae_ratio,
            z.shape[-1] * self.vae_ratio,
        )
        y_idx, y_len, y_overlap = self.split_tiles(height, True)
        x_idx, x_len, x_overlap = self.split_tiles(width, True)

        i_max, j_max = len(y_idx), len(x_idx)
        num_tiles = i_max * j_max
        if tile_world_size > num_tiles:
            # See tiled_encode: small canvases fall back to replicated local
            # tiling so no decode rank is assigned an empty task list.
            tile_rank, tile_world_size = 0, 1

        z_tiles = []
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            i_pos, i_len = (
                i_pos // self.vae_ratio,
                i_len // self.vae_ratio,
            )
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                j_pos, j_len = (j_pos // self.vae_ratio, j_len // self.vae_ratio)
                tile = z[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                z_tiles.append(tile)

        local_tile_indices = self._local_tile_indices(
            num_tiles, tile_rank, tile_world_size
        )
        stack_tiling = self.stack_tiling and not (
            self.training and getattr(self.decoder, "mask_enabled", False)
        )
        decoded_tasks = self._run_tile_tasks(
            z_tiles, local_tile_indices, self.decode, stack_tiling
        )

        if tile_world_size > 1:
            all_decoded = self._all_gather_tiled_results(decoded_tasks, num_tiles)
        else:
            all_decoded = decoded_tasks

        rows = [[None for _ in range(j_max)] for _ in range(i_max)]
        for idx, decoded in enumerate(all_decoded):
            i, j = idx // j_max, idx % j_max
            rows[i][j] = decoded.to(z.device)

        dec = self._assemble_tiles(rows, y_overlap, x_overlap)
        return dec

    def _adaptive_encode(self, x):
        if self.encoder_tiling:
            return self.tiled_encode(x)
        else:
            return self.encode(x)

    def _adaptive_decode(self, z):
        if self.decoder_tiling:
            return self.tiled_decode(z)
        else:
            return self.decode(z)

    def trim_code(self, z, target_codes):
        if target_codes < z.shape[2]:
            if self.causal_encoder:
                z = z[:, :, -target_codes:, :, :]
            else:
                start_frame = (z.shape[2] - target_codes) // 2
                z = z[:, :, start_frame : start_frame + target_codes, :, :]
        return z

    def trim_output(self, dec, target_frames):
        if target_frames < dec.shape[2]:
            if self.causal_encoder:  # This is defined by encoder, not decoder
                dec = dec[:, :, -target_frames:, :, :]
            else:
                start_frame = (dec.shape[2] - target_frames) // 2
                dec = dec[:, :, start_frame : start_frame + target_frames, :, :]
        return dec

    def encode_temporal(self, x):
        offset_frame = (
            1 if self.isolated_first_frame and self.frame_pre_padding == 0 else 0
        )

        frame_num = x.shape[2]
        pad_size = (offset_frame - frame_num) % self.clip_length
        padded_frame_num = frame_num + pad_size
        num_chunks = (padded_frame_num - offset_frame) // self.clip_length

        z_list = []
        for i in range(num_chunks):
            start_idx = i * self.clip_length + offset_frame
            end_idx = (i + 1) * self.clip_length + offset_frame
            clip_x = x[:, :, start_idx : min(end_idx, frame_num), :, :]
            if end_idx > frame_num:
                pad_frames = x[:, :, -1:].expand(-1, -1, end_idx - frame_num, -1, -1)
                clip_x = torch.cat([clip_x, pad_frames], dim=2)

            if self.isolated_key_frame:
                key_frame = clip_x[:, :, :1, :, :]
                z_key = self._adaptive_encode(key_frame)

                if clip_x.shape[2] > 1:
                    video_frames = clip_x[:, :, 1:, :, :]
                    z_video = self._adaptive_encode(video_frames)
                    z = torch.cat([z_key, z_video], dim=2)
                else:
                    z = z_key
            else:
                z = self._adaptive_encode(clip_x)

            z_list.append(z)

        z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, : -self.token_drop]

        if self.isolated_first_frame:
            input_first_frame = x[:, :, :1, :, :]
            z_first_frame = self._adaptive_encode(input_first_frame)

            if self.frame_pre_padding == 0:
                z = torch.cat([z_first_frame, z], dim=2)
            else:
                z = torch.cat([z_first_frame, z[:, :, 1:, :, :]], dim=2)

        if self.isolated_last_frame:
            last_frame_idx = padded_frame_num - self.frame_drop + offset_frame
            if last_frame_idx >= frame_num:
                input_last_frame = x[:, :, -1:, :, :]
            else:
                input_last_frame = x[:, :, last_frame_idx : last_frame_idx + 1, :, :]
            z_last_frame = self._adaptive_encode(input_last_frame)
            z = torch.cat([z, z_last_frame], dim=2)

        return z

    def _decode_temporal_pad_frames(self, z, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return int(pad_tokens) * int(self.vae_ratio_t)

        z_len_before_pad = z.shape[2] - pad_tokens
        return sum(
            (
                intra_tail
                if (z_len_before_pad + k) % self.tokens_chunk_size == 0
                else self.vae_ratio_t
            )
            for k in range(pad_tokens)
        )

    def _decode_temporal_output_frame_plan(
        self, z, z_head, z_tail, num_chunks, pad_tokens
    ):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0

        if z_head is not None:
            total_frames += 1

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(
                0, min(t_end_idx, z.shape[2]) - min(t_start_idx, z.shape[2])
            )
            if i == 0 and z_head is not None:
                clip_token_len += z_head.shape[2]
            if i == num_chunks - 1 and z_tail is not None:
                clip_token_len += z_tail.shape[2]

            clip_frame_len = clip_token_len * self.vae_ratio_t
            if i == 0 and z_head is not None:
                clip_frame_len = max(0, clip_frame_len - self.vae_ratio_t)
            if i == num_chunks - 1 and z_tail is not None:
                clip_frame_len = max(0, clip_frame_len - self.vae_ratio_t)

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames

        total_frames += final_overlap_frames
        if z_tail is not None:
            total_frames += 1

        pad_frames = self._decode_temporal_pad_frames(z, pad_tokens)
        return int(total_frames), int(pad_frames), int(total_frames - pad_frames)

    def _decode_temporal_streaming(
        self, z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype
    ):
        total_frames, pad_frames, output_frames = (
            self._decode_temporal_output_frame_plan(
                z, z_head, z_tail, num_chunks, pad_tokens
            )
        )
        if output_frames <= 0:
            raise ValueError(
                f"decode_temporal streaming planned non-positive output_frames={output_frames} "
                f"total_frames={total_frames} pad_frames={pad_frames}"
            )

        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        dec = None
        dec_overlap = None
        write_pos = 0
        logical_frames = 0
        dropped_frames = 0
        decoded_count = 0

        def write_part(part):
            nonlocal dec, write_pos, logical_frames, dropped_frames
            part_frames = int(part.shape[2])
            if part_frames <= 0:
                return
            logical_frames += part_frames
            if dec is None:
                out_shape = list(part.shape)
                out_shape[2] = output_frames
                dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)

            remaining = int(dec.shape[2]) - write_pos
            copy_frames = min(part_frames, max(0, remaining))
            if copy_frames > 0:
                dec[:, :, write_pos : write_pos + copy_frames, :, :].copy_(
                    part[:, :, :copy_frames, :, :]
                )
                write_pos += copy_frames
            dropped_frames += part_frames - copy_frames

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

            if i == 0 and z_head is not None:
                clip_z = torch.cat([z_head, clip_z], dim=2)

            if i == num_chunks - 1 and z_tail is not None:
                clip_z = torch.cat([clip_z, z_tail], dim=2)

            clip_dec = self._adaptive_decode(clip_z)
            decoded_count += 1
            if temporal_cat_dtype is not None and clip_dec.dtype != temporal_cat_dtype:
                clip_dec = clip_dec.to(temporal_cat_dtype)
            if clip_dec.device != z.device:
                clip_dec = clip_dec.to(z.device)

            dec_tail = None
            if i == 0 and z_head is not None:
                write_part(
                    clip_dec[:, :, self.vae_ratio_t - 1 : self.vae_ratio_t, :, :]
                )
                clip_dec = clip_dec[:, :, self.vae_ratio_t :, :, :]

            if i == num_chunks - 1 and z_tail is not None:
                dec_tail = clip_dec[:, :, -1:, :, :]
                clip_dec = clip_dec[:, :, : -self.vae_ratio_t, :, :]

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding :, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                        dec_overlap = None
                    write_part(clip_dec_chunk)
                else:
                    # Break the view's reference to the full decoded clip so earlier
                    # temporal chunks can be released before the final output exists.
                    dec_overlap = clip_dec_chunk.contiguous()

            if i == num_chunks - 1:
                if dec_overlap is not None:
                    write_part(dec_overlap)
                    dec_overlap = None
                if dec_tail is not None:
                    write_part(dec_tail)

            del clip_dec, clip_z

        if dec is None:
            raise RuntimeError("decode_temporal streaming produced no output tensor")
        if (
            logical_frames != total_frames
            or dropped_frames != pad_frames
            or write_pos != output_frames
        ):
            raise RuntimeError(
                "decode_temporal streaming frame plan mismatch: "
                f"logical_frames={logical_frames} total_frames={total_frames} "
                f"dropped_frames={dropped_frames} pad_frames={pad_frames} "
                f"write_pos={write_pos} output_frames={output_frames}"
            )

        return dec

    def decode_temporal(self, z):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t

        isolated_token_num = 0
        if self.isolated_first_frame and self.frame_pre_padding == 0:
            isolated_token_num = isolated_token_num + 1
        if self.isolated_last_frame:
            isolated_token_num = isolated_token_num + 1

        pseudo_total_tokens = z.shape[2] - isolated_token_num + self.token_drop

        pad_tokens = 0
        remainder = pseudo_total_tokens % self.tokens_chunk_size
        if remainder != 0:
            if self.training:
                raise ValueError(f"Temporal token length {z.shape[2]} is wrong!")
            else:
                pad_tokens = self.tokens_chunk_size - remainder
                pseudo_total_tokens = pseudo_total_tokens + pad_tokens

        pseudo_num_chunks = pseudo_total_tokens // self.tokens_chunk_size
        num_chunks = pseudo_num_chunks - int(self.token_drop > 0)

        z_head = None
        if self.isolated_first_frame and self.frame_pre_padding == 0:
            z_head = z[:, :, :1, :, :]
            z = z[:, :, 1:, :, :]

        z_tail = None
        if self.isolated_last_frame:
            z_tail = z[:, :, -1:, :, :]
            z = z[:, :, :-1, :, :]

        if pad_tokens > 0:
            pad_z = z[:, :, -1:, :, :].expand(-1, -1, pad_tokens, -1, -1)
            z = torch.cat([z, pad_z], dim=2)

        temporal_cat_dtype = _resolve_temporal_cat_dtype()
        if not self.training and _resolve_temporal_stream_cat():
            return self._decode_temporal_streaming(
                z, z_head, z_tail, num_chunks, pad_tokens, temporal_cat_dtype
            )

        decoded_tasks = []
        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

            if i == 0 and z_head is not None:
                clip_z = torch.cat([z_head, clip_z], dim=2)

            if i == num_chunks - 1 and z_tail is not None:
                clip_z = torch.cat([clip_z, z_tail], dim=2)

            clip_dec = self._adaptive_decode(clip_z)
            if temporal_cat_dtype is not None and clip_dec.dtype != temporal_cat_dtype:
                clip_dec = clip_dec.to(temporal_cat_dtype)

            decoded_tasks.append((i, clip_dec))

        clip_dec_list = [clip_dec.to(z.device) for _, clip_dec in decoded_tasks]

        dec_list = []
        dec_overlap = None

        dec_head = None
        if z_head is not None:
            dec_head = clip_dec_list[0][
                :, :, self.vae_ratio_t - 1 : self.vae_ratio_t, :, :
            ]
            clip_dec_list[0] = clip_dec_list[0][:, :, self.vae_ratio_t :, :, :]

        dec_tail = None
        if z_tail is not None:
            dec_tail = clip_dec_list[-1][:, :, -1:, :, :]
            clip_dec_list[-1] = clip_dec_list[-1][:, :, : -self.vae_ratio_t, :, :]

        if dec_head is not None:
            dec_list.append(dec_head)

        for i in range(num_chunks):
            for j in range(int(self.token_drop > 0) + 1):
                clip_dec = clip_dec_list[i]

                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding :, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                    dec_list.append(clip_dec_chunk)
                else:
                    dec_overlap = clip_dec_chunk

        if dec_overlap is not None:
            dec_list.append(dec_overlap)

        if dec_tail is not None:
            dec_list.append(dec_tail)

        dec = torch.cat(dec_list, dim=2)

        pad_frames = self._decode_temporal_pad_frames(z, pad_tokens)
        if pad_frames > 0:
            dec = dec[:, :, :-pad_frames, :, :]

        return dec

    def decode_base(self, z, frame_num=None, process_image=False):
        if process_image or not self.use_3d_conv:
            if not self.use_3d_conv and z.ndim == 5:
                z = z.squeeze(2)

            recon = self._adaptive_decode(z)
        else:
            recon = self.decode_temporal(z)

        if self.use_3d_conv:
            if frame_num is not None:
                target_frames = frame_num
            else:
                target_frames = recon.shape[2]

            recon = self.trim_output(recon, target_frames)
            if process_image:
                recon = recon.squeeze(2)

        return recon

    #########################################################
    # following methods are for inference
    #########################################################

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[List[np.ndarray], List[torch.Tensor]],
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
    ) -> List[torch.Tensor]:
        """encode images into latents

        Args:
            images (Union[List[np.ndarray], List[torch.Tensor]]):
                List of images, single input will be wrapped in a list.
                If input is a list of np.ndarray, it should be in shape B * (H, W, 3), dtype uint8.
                If input is a list of torch.Tensor, it should be in shape B * (3, H, W), dtype float32.
            transform_input (bool, optional):
                Whether to transform input using ImageNet std/mean. Defaults to False.
                If input is a list of np.ndarray, it will always be set to True.
            use_fp16_latent (bool, optional):
                Whether to use fp16 latent. Defaults to False.
            verbose (bool, optional):
                Whether to print debug information. Defaults to False.

        Returns:
            List[torch.Tensor]:
                List of image latents.
                If self.use_3d_conv is True, it should be in shape B * (D, 1, H', W').
                Otherwise, it should be in shape B * (D, H', W').
        """

        images = self.processor._ensure_list(images)
        runtime_owned = False

        if isinstance(images[0], Image.Image):
            images = [np.array(image) for image in images]

        if isinstance(images[0], np.ndarray):
            device = next(self.parameters()).device
            images = self.processor.convert_numpy_to_tensor(images, device)
            images = torch.split(images, 1, dim=0)
            transform_input = True
            runtime_owned = True

        if transform_input:
            images = [
                image.unsqueeze(0) if image.ndim == 3 else image for image in images
            ]
            images = [
                self.processor.transform_tensor(image, runtime_owned=runtime_owned)
                for image in images
            ]

        prepared = []
        for image_tensor in images:
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            _, _, h, w = image_tensor.shape
            new_h, new_w = self.processor._align_to_total_patch_size(h, w)
            image_tensor = self.processor._crop_to_align(image_tensor, new_h, new_w)
            prepared.append(image_tensor)

        if len(prepared) > 1 and len(set(t.shape for t in prepared)) == 1:
            stacked = torch.cat(prepared, dim=0)
            if verbose:
                logger.info(f"batch encode input shape {tuple(stacked.shape)}")
            all_latents = self.encode_base(stacked, True)
            image_latents = [
                all_latents[i].contiguous() for i in range(all_latents.shape[0])
            ]
        else:
            image_latents = []
            for image_tensor in prepared:
                if verbose:
                    logger.info(f"input shape {tuple(image_tensor.shape)}")
                image_latent = self.encode_base(image_tensor, True)
                image_latents.append(image_latent.squeeze(0).contiguous())

        if use_fp16_latent:
            image_latents = [lat.to(torch.float16) for lat in image_latents]

        if verbose:
            for lat in image_latents:
                logger.info(f"image latent shape {tuple(lat.shape)}")

        return image_latents

    @torch.no_grad()
    def encode_videos(
        self,
        videos: Union[List[np.ndarray], List[torch.Tensor]],
        transform_input: bool = False,
        use_fp16_latent: bool = False,
        verbose: bool = False,
        encode_prefix: bool = False,
    ) -> List[torch.Tensor]:
        """encode videos into latents

        Args:
            videos (Union[List[np.ndarray], List[torch.Tensor]]):
                List of videos, single input will be wrapped in a list.
                If input is a list of np.ndarray, it should be in shape B * (T, H, W, 3), dtype uint8.
                If input is a list of torch.Tensor, it should be in shape B * (3, T, H, W), dtype float32.
            transform_input (bool, optional):
                Whether to transform input using ImageNet std/mean. Defaults to False.
                If input is a list of np.ndarray, it will always be set to True.
            use_fp16_latent (bool, optional):
                Whether to use fp16 latent. Defaults to False.
            verbose (bool, optional):
                Whether to print debug information. Defaults to False.
            encode_prefix (bool, optional):
                Continuation (prefix) mode: prepend normalized
                black frames to token alignment, append black frames to chunk
                alignment, encode with token_drop disabled, then discard only
                the trailing padding tokens. Returns both latents and leading
                pad-frame counts. Defaults to False.

        Returns:
            List[torch.Tensor]:
                List of video latents, shape B * (D, T', H', W').
                With encode_prefix=True, returns
                (List[torch.Tensor], List[int]).
        """

        videos = self.processor._ensure_list(videos)
        runtime_owned = False

        if isinstance(videos[0], np.ndarray):
            device = next(self.parameters()).device
            videos = [
                self.processor.convert_numpy_to_tensor(video, device)
                for video in videos
            ]
            transform_input = True
            runtime_owned = True

        if transform_input:
            videos = [
                self.processor.transform_tensor(video, runtime_owned=runtime_owned)
                for video in videos
            ]
            videos = [video.transpose(0, 1) for video in videos]

        if encode_prefix:
            if self.isolated_last_frame:
                raise ValueError("encode_prefix does not support isolated_last_frame")

            video_latents = []
            prefix_pad_frames = []
            for video in videos:
                if video.ndim == 4:
                    video = video.unsqueeze(0)
                _, _, _, h, w = video.shape
                new_h, new_w = self.processor._align_to_total_patch_size(h, w)
                video = self.processor._crop_to_align(
                    video, new_h, new_w, is_video=True
                )

                model_alignment = (
                    self.token_drop,
                    self.frame_drop,
                    self.token_overlap,
                    self.frame_overlap,
                )
                processor_alignment = (
                    self.processor.token_overlap,
                    self.processor.frame_overlap,
                )
                self.token_drop = 0
                self.frame_drop = 0
                self.token_overlap = 0
                self.frame_overlap = 0
                self.processor.token_overlap = 0
                self.processor.frame_overlap = 0
                try:
                    orig_frames = video.shape[2]
                    leading, trailing, drop_tokens = (
                        self.processor.align_video_length_2pass(orig_frames)
                    )
                    _, _, _, cropped_h, cropped_w = video.shape
                    if leading > 0:
                        black = self.processor.transform(
                            video.new_zeros(leading, 3, cropped_h, cropped_w)
                        )
                        black = black.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        video = torch.cat([black, video], dim=2)
                    if trailing > 0:
                        black = self.processor.transform(
                            video.new_zeros(trailing, 3, cropped_h, cropped_w)
                        )
                        black = black.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        video = torch.cat([video, black], dim=2)

                    if verbose:
                        logger.info(
                            f"[encode_prefix] {orig_frames} frames -> "
                            f"pad leading={leading}, trailing={trailing} -> "
                            f"{video.shape[2]} frames"
                        )

                    video_latent = self.encode_base(video, False)
                    if drop_tokens > 0:
                        video_latent = video_latent[:, :, :-drop_tokens, :, :]
                    prefix_pad_frames.append(leading)
                finally:
                    (
                        self.token_drop,
                        self.frame_drop,
                        self.token_overlap,
                        self.frame_overlap,
                    ) = model_alignment
                    (
                        self.processor.token_overlap,
                        self.processor.frame_overlap,
                    ) = processor_alignment

                video_latents.append(video_latent.squeeze(0).contiguous())

            if use_fp16_latent:
                video_latents = [lat.to(torch.float16) for lat in video_latents]
            if verbose:
                for latent in video_latents:
                    logger.info(f"video latent shape {tuple(latent.shape)}")
            return video_latents, prefix_pad_frames

        prepared = []
        for video in videos:
            if video.ndim == 4:
                video = video.unsqueeze(0)
            used_frame_length = self.processor.get_suitable_video_length(
                video.shape[2], verbose
            )
            _, _, _, h, w = video.shape
            new_h, new_w = self.processor._align_to_total_patch_size(h, w)
            video = video[:, :, :used_frame_length, :, :]
            video = self.processor._crop_to_align(video, new_h, new_w, is_video=True)
            prepared.append(video)

        if len(prepared) > 1 and len(set(t.shape for t in prepared)) == 1:
            stacked = torch.cat(prepared, dim=0)
            if verbose:
                logger.info(f"batch encode input shape {tuple(stacked.shape)}")
            all_latents = self.encode_base(stacked, False)
            video_latents = [
                all_latents[i].contiguous() for i in range(all_latents.shape[0])
            ]
        else:
            video_latents = []
            for video in prepared:
                if verbose:
                    logger.info(f"input shape {tuple(video.shape)}")
                video_latent = self.encode_base(video, False)
                video_latents.append(video_latent.squeeze(0).contiguous())

        if use_fp16_latent:
            video_latents = [lat.to(torch.float16) for lat in video_latents]

        if verbose:
            for lat in video_latents:
                logger.info(f"video latent shape {tuple(lat.shape)}")

        return video_latents


# ============================================================================
# Legacy CNN VAE
# ============================================================================


class AutoencoderKLLegacy(AutoencoderKL):
    r"""
    A VAE model (legacy CNN-based) for encoding pixels into latents and decoding latent representations into pixels.
    """

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=16,
        z_channels=16,
        use_3d_conv=False,
        # cnn vae
        zq_ch_encoder=None,
        zq_ch_decoder=None,
        num_res_blocks=2,
        num_res_blocks_decoder=None,
        ch_mult=[1, 2, 2, 4, 4, 8],
        space_down=[2, 2, 2, 2, 1, 1],
        space_up=[1, 2, 2, 2, 2, 1],
        time_down=None,
        time_up=None,
        padding_mode="zeros",
        padding_mode_t=None,
        use_t_isolated_gn=False,
        causal_encoder=True,
        causal_decoder=True,
        use_vit_decoder=False,
        vit_decoder_kwargs=None,
        # stats
        shift_factor=0.0,
        scaling_factor=1.0,
        # pixel normalization
        pixel_norm_type="imagenet",
        # others
        **kwargs,
    ):
        ModelMixin.__init__(self)  # NOTE: avoid wrong @register_to_config

        if not use_3d_conv or not use_vit_decoder:
            raise NotImplementedError(
                "this release only supports use_3d_conv=True with use_vit_decoder=True"
            )

        self.transform = get_normalize_transform(pixel_norm_type)
        self.transform_rev = get_denormalize_transform(pixel_norm_type)

        self.use_3d_conv = use_3d_conv
        self.causal_encoder = causal_encoder
        self.causal_decoder = causal_decoder
        self.slidedec = self.causal_encoder and not self.causal_decoder

        # some registered parameters for simplicity
        self.vae_ratio = int(np.cumprod(space_down)[-1])
        self.vae_ratio_t = int(np.cumprod(time_down)[-1]) if time_down else 1
        self.config["vae_ratio"] = self.vae_ratio
        self.config["vae_ratio_t"] = self.vae_ratio_t

        # Configure inference-time chunking and tiling.
        self.setup_forward(**kwargs)

        # init encoder
        encoder_config = {
            "double_z": True,
            "z_channels": z_channels,
            "zq_ch": zq_ch_encoder,
            "in_channels": in_channels,
            "ch": ch,
            "num_res_blocks": num_res_blocks,
            "ch_mult": ch_mult,
            "space_down": space_down,
            "time_down": time_down,
            "padding_mode": padding_mode,
            "padding_mode_t": padding_mode_t,
            "causal": causal_encoder,
            "use_t_isolated_gn": use_t_isolated_gn,
        }
        self.encoder = EncoderFCN3D(**encoder_config)

        # init pointwise quant/post_quant conv
        self.quant_conv = nn.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, 1)

        self.use_vit_decoder = use_vit_decoder

        # init decoder
        vit_kwargs = {
            "patch_size": self.vae_ratio,
            "in_channels": z_channels,
            "out_channels": out_ch,
            **(vit_decoder_kwargs or {}),
        }
        vit_kwargs.setdefault("patch_size_t", self.vae_ratio_t)
        vit_kwargs.setdefault("t_causal", causal_decoder)
        self.decoder = ViT3DDecoder(**vit_kwargs)

    @torch.no_grad()
    def encode(self, x):
        return self.quant_conv(self.encoder(x))

    @torch.no_grad()
    def decode(self, z):
        z2 = self.post_quant_conv(z)
        if self.use_vit_decoder:
            return self.decoder(z2)
        return self.decoder(z2, z)

    def encode_base(self, input, process_image=False):
        if self.use_3d_conv and input.ndim == 4:
            input = input.unsqueeze(2)

        if process_image or not self.use_3d_conv:
            moments = self._adaptive_encode(input)
        else:
            moments = self.encode_temporal(input)

        z = DiagonalGaussianDistribution(moments).sample()

        if process_image and self.use_3d_conv:
            z = self.trim_code(z, 1)

        return z
