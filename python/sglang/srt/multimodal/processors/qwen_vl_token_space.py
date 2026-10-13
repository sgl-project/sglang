from typing import List, Optional

import torch
from transformers import BatchFeature

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.processors.token_space_processor import (
    MultimodalProcessorOutputBuilder,
)
from sglang.srt.multimodal.transport.cuda_ipc import (
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
)
from sglang.srt.runtime_context import get_mm, get_parallel


class QwenVLMultimodalProcessorOutputBuilder(MultimodalProcessorOutputBuilder):
    """Qwen VL serving output: padded ids, vision boundaries and M-RoPE positions."""

    def __init__(self, hf_config):
        model_type = hf_config.model_type
        if model_type == "qwen3_omni_moe":
            hf_config = hf_config.thinker_config
        super().__init__(hf_config)
        self.model_type = model_type
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)
        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self._spatial_merge_size = hf_config.vision_config.spatial_merge_size
        self._tokens_per_second = getattr(
            hf_config.vision_config, "tokens_per_second", None
        )

    def compute_mrope_positions(self, input_ids, mm_items):
        image_grid_thw = self._concat_mm_item_grid(
            mm_items, "image_grid_thw", Modality.IMAGE
        )
        video_grid_thw = self._concat_mm_item_grid(
            mm_items, "video_grid_thw", Modality.VIDEO
        )

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=self._tokens_per_second,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    @staticmethod
    def _get_processor_output_value(ret, key):
        if ret is None:
            return None
        return ret.get(key) if hasattr(ret, "get") else getattr(ret, key, None)

    def _get_precomputed_mrope_from_output(self, ret):
        mrope_positions = self._get_processor_output_value(ret, "mrope_positions")
        mrope_position_delta = self._get_processor_output_value(
            ret, "mrope_position_delta"
        )
        if mrope_positions is None or mrope_position_delta is None:
            return None

        mrope_positions = torch.as_tensor(mrope_positions)
        if mrope_positions.ndim == 3:
            if mrope_positions.shape[1] != 1:
                return None
            mrope_positions = mrope_positions.squeeze(1)
        if mrope_positions.ndim != 2 or mrope_positions.shape[0] != 3:
            return None

        mrope_position_delta = torch.as_tensor(mrope_position_delta)
        if mrope_position_delta.ndim <= 1:
            mrope_position_delta = mrope_position_delta.reshape(-1, 1)
        return mrope_positions, mrope_position_delta

    @staticmethod
    def _as_grid_batch(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.unsqueeze(0) if value.ndim == 1 else value
        tensor = torch.as_tensor(value, dtype=torch.long)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    def _compute_image_only_mrope_positions_from_offsets(
        self,
        input_len: int,
        mm_items: List[MultimodalDataItem],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """instead of calling get_rope_index, build mrope position from mm_items.offsets and image_grid_thw of each image
        basically a simplified version of get_rope_index for image-only reqs
        """
        if self.model_type not in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen4_exp",
            "intern_s2_preview",
            "interns2_mobius",
            "cosmos3_omni",
        ):
            return None

        image_items = [item for item in mm_items if item.is_image()]
        if not image_items or len(image_items) != len(mm_items):
            return None

        spatial_merge_size = self._spatial_merge_size
        sorted_items = sorted(image_items, key=lambda item: item.offsets[0][0])
        position_segments = []
        st = 0
        next_pos = 0

        for item in sorted_items:
            if item.offsets is None or len(item.offsets) != 1:
                return None

            start, end = item.offsets[0]
            if start < st or end >= input_len:
                return None

            text_len = start - st
            if text_len > 0:
                position_segments.append(
                    torch.arange(text_len, dtype=dtype, device=device)
                    .view(1, -1)
                    .expand(3, -1)
                    + next_pos
                )
                next_pos += text_len

            grid = self._as_grid_batch(item.model_specific_data.get("image_grid_thw"))
            if grid is None or grid.shape[0] != 1:
                return None
            t, h, w = [int(x) for x in grid[0].tolist()]
            llm_grid_t = t
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w
            if num_image_tokens != end - start + 1:
                return None

            t_index = (
                torch.arange(llm_grid_t, dtype=dtype, device=device)
                .view(-1, 1)
                .expand(llm_grid_t, llm_grid_h * llm_grid_w)
                .reshape(-1)
            )
            h_index = (
                torch.arange(llm_grid_h, dtype=dtype, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            w_index = (
                torch.arange(llm_grid_w, dtype=dtype, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, llm_grid_w)
                .reshape(-1)
            )
            position_segments.append(
                torch.stack([t_index, h_index, w_index]) + next_pos
            )
            next_pos += max(llm_grid_t, llm_grid_h, llm_grid_w)
            st = end + 1

        if st < input_len:
            text_len = input_len - st
            position_segments.append(
                torch.arange(text_len, dtype=dtype, device=device)
                .view(1, -1)
                .expand(3, -1)
                + next_pos
            )

        mrope_positions = torch.cat(position_segments, dim=1).unsqueeze(1)
        mrope_position_delta = (mrope_positions.max() + 1 - input_len).reshape(1, 1)
        return mrope_positions, mrope_position_delta

    @classmethod
    def _concat_mm_item_grid(cls, mm_items: list[MultimodalDataItem], key, modality):
        grids = []
        for item in mm_items:
            if not item.is_modality(modality):
                continue
            grid = cls._as_grid_batch(item.model_specific_data.get(key))
            if grid is not None:
                grids.append(grid)
        if not grids:
            return None
        if len(grids) == 1:
            return grids[0]
        return torch.cat(grids, dim=0)

    @classmethod
    def _get_grid_from_output_or_items(
        cls, ret, mm_items, key, modality, input_data=None
    ):
        grid = cls._get_processor_output_value(ret, key)
        if grid is None:
            grid = cls._concat_mm_item_grid(mm_items, key, modality)
        if grid is None and input_data and isinstance(input_data[0], dict):
            grid = input_data[0].get(key)
        return grid

    def _compute_mrope_positions(
        self,
        ret,
        mm_items,
        input_ids,
        *,
        image_data,
        video_data,
        second_per_grid_ts,
        audio_feature_lengths,
    ):
        image_grid_thw = self._get_grid_from_output_or_items(
            ret, mm_items, "image_grid_thw", Modality.IMAGE, image_data
        )
        video_grid_thw = self._get_grid_from_output_or_items(
            ret,
            mm_items,
            "video_grid_thw",
            Modality.VIDEO,
            video_data,
        )

        mrope_result = self._get_precomputed_mrope_from_output(ret)
        if mrope_result is None:
            if (
                video_grid_thw is None
                and second_per_grid_ts is None
                and audio_feature_lengths is None
            ):
                mrope_result = self._compute_image_only_mrope_positions_from_offsets(
                    input_len=input_ids.numel(),
                    mm_items=mm_items,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
        if mrope_result is None:
            mrope_result = MRotaryEmbedding.get_rope_index(
                spatial_merge_size=self._spatial_merge_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                model_type=self.model_type,
                tokens_per_second=self._tokens_per_second,
                # use the expanded token ids
                input_ids=input_ids.unsqueeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                use_audio_in_video=False,
                audio_seqlens=audio_feature_lengths,
                audio_token_id=getattr(self.hf_config, "audio_token_id", None),
                audio_start_token_id=self.audio_start_token_id,
                position_id_per_seconds=getattr(
                    self.hf_config, "position_id_per_seconds", None
                ),
            )

        mrope_positions, mrope_position_delta = mrope_result
        if mrope_positions.ndim == 3:
            mrope_positions = mrope_positions.squeeze(1)
        return mrope_positions, mrope_position_delta

    def _mark_cuda_ipc_features_for_deferred_reconstruction(
        self, mm_items, keep_mm_features_on_device
    ):
        supports_deferred_reconstruction = get_mm().mm_enable_dp_encoder or (
            get_parallel().tp_size == 1
            and self.model_type in ("qwen3_vl", "qwen3_vl_moe")
        )
        if not (
            keep_mm_features_on_device
            and supports_deferred_reconstruction
            and self.model_type
            in ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe")
        ):
            return
        for item in mm_items:
            if item.is_image() or item.is_video():
                item.model_specific_data[DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY] = (
                    True
                )

    def _build_mm_output(
        self,
        ret: BatchFeature,
        mm_items: list[MultimodalDataItem],
        input_ids: torch.Tensor,
        *,
        keep_mm_features_on_device,
        base_input_ids: Optional[list[int]] = None,
        image_data=None,
        video_data=None,
        audio_feature_lengths=None,
    ) -> MultimodalProcessorOutput:
        self._mark_cuda_ipc_features_for_deferred_reconstruction(
            mm_items, keep_mm_features_on_device
        )

        second_per_grid_ts = self._get_processor_output_value(ret, "second_per_grid_ts")
        if second_per_grid_ts is None:
            second_per_grid_ts = self._get_processor_output_value(
                ret, "video_second_per_grid"
            )

        input_ids = input_ids.flatten()
        if (
            isinstance(base_input_ids, list)
            and len(base_input_ids) == input_ids.numel()
        ):
            # reuse preprocess input if it already carries list of input_ids
            input_ids_list = base_input_ids
        else:
            input_ids_list = input_ids.tolist()

        # look for if padded_input_ids already exists before computing
        padded_input_ids = self._get_processor_output_value(ret, "padded_input_ids")
        if padded_input_ids is None:
            padded_input_ids = MultimodalProcessorOutput.build_padded_input_ids(
                input_ids_list, mm_items
            )
        elif isinstance(padded_input_ids, torch.Tensor):
            # reuse existing padded_input_ids
            padded_input_ids = padded_input_ids.flatten().tolist()
        else:
            padded_input_ids = list(padded_input_ids)

        mrope_positions, mrope_position_delta = self._compute_mrope_positions(
            ret,
            mm_items,
            input_ids,
            image_data=image_data,
            video_data=video_data,
            second_per_grid_ts=second_per_grid_ts,
            audio_feature_lengths=audio_feature_lengths,
        )
        output = MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=padded_input_ids,
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
        return output
