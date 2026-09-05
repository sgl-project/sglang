from typing import List, Optional, Tuple, Union

import torch
from transformers.configuration_utils import PretrainedConfig

from sglang.kernels.ops.attention.rotary_triton import (
    triton_ernie45_rope_fused_inplace,
)
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding


class BailingMRotaryEmbedding(RotaryEmbedding):
    """Bailing multimodal RoPE with centered height and width positions."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        video_rope: bool = False,
    ) -> None:
        # Bailing positions are bounded by the checkpoint context on both sides:
        # text/time grow positive while centered height/width can be negative.
        position_start = -max_position_embeddings if video_rope else 0
        cache_length = max_position_embeddings * (2 if video_rope else 1)
        super().__init__(
            head_size,
            rotary_dim,
            cache_length,
            base,
            is_neox_style,
            dtype,
            position_start=position_start,
        )

        if mrope_section is not None:
            if sum(mrope_section) != rotary_dim // 2:
                raise ValueError(
                    "mrope_section must sum to rotary_dim // 2; "
                    f"got {mrope_section=} and {rotary_dim=}"
                )
            # The checkpoint stores [time, height, width], while the shared
            # Ernie4.5 kernel consumes [height, width, time].
            mrope_section = [mrope_section[1], mrope_section[2], mrope_section[0]]
        self.mrope_section = mrope_section

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim not in (1, 2):
            raise ValueError(
                f"Bailing mRoPE expects 1D or 2D positions, got {positions.shape=}"
            )
        positions = positions - self.position_start
        if positions.ndim == 2:
            if fused_set_kv_buffer_arg is not None:
                raise ValueError(
                    "fused_set_kv_buffer_arg is not supported for Bailing mRoPE"
                )
            if self.mrope_section is None:
                raise ValueError("mrope_section is required for 2D Bailing positions")

            query_shape = query.shape
            key_shape = key.shape
            if query.ndim == 3:
                query = query.reshape(query_shape[0], -1)
                key = key.reshape(key_shape[0], -1)
            triton_ernie45_rope_fused_inplace(
                query,
                key,
                self.cos_sin_cache,
                positions,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
                self.is_neox_style,
            )
            if query_shape != query.shape:
                query = query.view(query_shape)
                key = key.view(key_shape)
            return query, key
        return RotaryEmbedding.forward(self, positions, query, key)

    @staticmethod
    def _text_config(hf_config: PretrainedConfig) -> PretrainedConfig:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is None:
            text_config = getattr(hf_config, "llm_config", None)
        if text_config is None:
            raise ValueError("Bailing VL config must define text_config or llm_config")
        return text_config

    @staticmethod
    def _validate_position_bounds(
        positions: torch.Tensor, text_config: PretrainedConfig
    ) -> None:
        if positions.numel() == 0:
            return
        bound = text_config.max_position_embeddings
        min_position = int(positions.min().item())
        max_position = int(positions.max().item())
        if min_position < -bound or max_position >= bound:
            raise ValueError(
                "Bailing mRoPE position exceeds the checkpoint bounds: "
                f"min={min_position}, max={max_position}, allowed=[{-bound}, {bound})"
            )

    @classmethod
    def bailing_3drope_get_input_positions_tensor(
        cls,
        input_ids: torch.Tensor,
        hf_config: PretrainedConfig,
        image_grid_thw: Union[List[List[int]], torch.Tensor, None],
        video_grid_thw: Union[List[List[int]], torch.Tensor, None],
        second_per_grid_ts: Optional[List[float]] = None,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del context_len, seq_len, audio_feature_lengths, use_audio_in_video
        scale_factor = 2.0
        second_per_grid_ts = None
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        text_config = cls._text_config(hf_config)
        image_patch_id = text_config.image_patch_token
        video_patch_id = text_config.video_patch_token
        image_start_token_id = text_config.image_start_token
        video_start_token_id = text_config.video_start_token
        use_interleaved_frame_timestamp = getattr(
            text_config, "use_interleaved_frame_timestamp", False
        )

        if image_grid_thw is None and video_grid_thw is None:
            position_ids = (
                torch.arange(input_ids.numel(), device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, 1, -1)
            )
            cls._validate_position_bounds(position_ids, text_config)
            position_delta = torch.zeros(
                [1, 1], device=input_ids.device, dtype=input_ids.dtype
            )
            return position_ids, position_delta

        if video_grid_thw is not None and use_interleaved_frame_timestamp:
            video_grid_thw = torch.as_tensor(video_grid_thw).clone()
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        image_count = 0
        video_count = 0
        if image_grid_thw is not None:
            starts = torch.argwhere(input_ids == image_start_token_id).squeeze(1)
            starts = starts[starts + 1 < input_ids.numel()]
            if starts.numel() > 0:
                image_count = int((input_ids[starts + 1] == image_patch_id).sum())
        if video_grid_thw is not None:
            start_token = (
                image_start_token_id
                if use_interleaved_frame_timestamp
                else video_start_token_id
            )
            starts = torch.argwhere(input_ids == start_token).squeeze(1)
            starts = starts[starts + 1 < input_ids.numel()]
            if starts.numel() > 0:
                video_count = int((input_ids[starts + 1] == video_patch_id).sum())

        input_tokens = input_ids.tolist()
        position_chunks = []
        start = 0
        image_index = video_index = 0
        remaining_images = image_count
        remaining_videos = video_count
        device = input_ids.device

        for _ in range(image_count + video_count):
            image_start = (
                input_tokens.index(image_patch_id, start)
                if image_patch_id in input_tokens[start:] and remaining_images > 0
                else len(input_tokens) + 1
            )
            video_start = (
                input_tokens.index(video_patch_id, start)
                if video_patch_id in input_tokens[start:] and remaining_videos > 0
                else len(input_tokens) + 1
            )
            if image_start < video_start:
                t, h, w = torch.as_tensor(image_grid_thw[image_index]).tolist()
                seconds_per_grid = 0.0
                image_index += 1
                remaining_images -= 1
                media_start = image_start
            else:
                t, h, w = torch.as_tensor(video_grid_thw[video_index]).tolist()
                seconds_per_grid = (
                    second_per_grid_ts[video_index]
                    if second_per_grid_ts is not None
                    else 1.0
                )
                video_index += 1
                remaining_videos -= 1
                media_start = video_start

            grid_t = int(t)
            grid_h = int(h) // spatial_merge_size
            grid_w = int(w) // spatial_merge_size
            text_len = media_start - start
            position_start = (
                int(position_chunks[-1][0].max().item()) + 1 if position_chunks else 0
            )
            position_chunks.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                + position_start
            )

            time_index = (
                torch.arange(grid_t, device=device)
                .view(-1, 1)
                .expand(-1, grid_h * grid_w)
                .flatten()
            )
            height_index = (
                torch.arange(grid_h, device=device)
                .view(1, -1, 1)
                .expand(grid_t, -1, grid_w)
                .flatten()
                - (grid_h - 1) // 2
            )
            width_index = (
                torch.arange(grid_w, device=device)
                .view(1, 1, -1)
                .expand(grid_t, grid_h, -1)
                .flatten()
                - (grid_w - 1) // 2
            )
            if second_per_grid_ts is not None:
                time_index = time_index * seconds_per_grid * scale_factor
            else:
                time_index = time_index * scale_factor
            time_index = time_index + text_len + position_start
            position_chunks.append(
                torch.stack(
                    [time_index, height_index + time_index, width_index + time_index]
                )
            )
            start = media_start + grid_t * grid_h * grid_w

        if start < len(input_tokens):
            position_start = (
                int(position_chunks[-1][0].max().item()) + 1 if position_chunks else 0
            )
            text_len = len(input_tokens) - start
            position_chunks.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                + position_start
            )

        positions = torch.cat(position_chunks, dim=1).reshape(3, -1)
        if positions.shape[1] != input_ids.numel():
            raise ValueError(
                "Bailing mRoPE media grids do not match the prompt token spans: "
                f"positions={positions.shape[1]}, tokens={input_ids.numel()}"
            )
        cls._validate_position_bounds(positions, text_config)
        position_delta = (
            (positions[0].max() + 1 - input_ids.numel())
            .reshape(1, 1)
            .to(dtype=input_ids.dtype)
        )
        return positions.unsqueeze(1).to(dtype=input_ids.dtype), position_delta
