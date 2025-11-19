from typing import List, Union

from decord import VideoReader

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.models.glm4v_moe import Glm4vMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class Glm4vImageProcessor(SGLangBaseProcessor):
    models = [Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # GLM-V specific tokens
        self.IMAGE_TOKEN = "<|image|>"
        self.VIDEO_TOKEN = "<|video|>"
        self.IMAGE_START_TOKEN = "<|begin_of_image|>"
        self.IMAGE_END_TOKEN = "<|end_of_image|>"
        self.VIDEO_START_TOKEN = "<|begin_of_video|>"
        self.VIDEO_END_TOKEN = "<|end_of_video|>"

        # Token IDs
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id
        self.IMAGE_START_TOKEN_ID = hf_config.image_start_token_id
        self.IMAGE_END_TOKEN_ID = hf_config.image_end_token_id
        self.VIDEO_START_TOKEN_ID = hf_config.video_start_token_id
        self.VIDEO_END_TOKEN_ID = hf_config.video_end_token_id

        # Vision config
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 112 * 112
        self.MAX_PIXELS = 30000 * 28 * 28 * 2

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            video_token=self.VIDEO_TOKEN,
            # Note: For GLM4v videos, it uses the video token before tokenization but uses image token after tokenization
            video_token_id=self.IM_TOKEN_ID,
        ).build(_processor)

    # adapted from https://github.com/huggingface/transformers/blob/369c99d0cea403b77bd0aef818527106453fd9fc/src/transformers/video_utils.py#L312
    async def preprocess_video(self, vr: VideoReader):
        """
        Preprocess video using VideoReader from Decord backend.

        Args:
            vr (VideoReader): VideoReader object from decord

        Returns:
            tuple: A tuple containing processed frames and metadata
        """
        video_fps = vr.get_avg_fps()
        total_num_frames = len(vr)
        duration = total_num_frames / video_fps if video_fps else 0

        # Extract all frames
        indices = list(range(total_num_frames))
        frames = vr.get_batch(indices).asnumpy()

        # Return metadata as dict so transformers can properly create VideoMetadata objects
        metadata = {
            "total_num_frames": int(total_num_frames),
            "fps": float(video_fps),
            "duration": float(duration),
            "video_backend": "decord",
            "frames_indices": indices,
        }

        return frames, metadata

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
        )

        video_metadata = None

        if base_output.videos:
            videos_processed = [
                await self.preprocess_video(video) for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))
            # transformer requires the video inputs to be under this format
            base_output.videos = [base_output.videos]
            video_metadata = [video_metadata]

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens, video_metadata=video_metadata
        )

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            attention_mask=getattr(ret, "attention_mask", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_inputs = {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

        return mm_inputs
