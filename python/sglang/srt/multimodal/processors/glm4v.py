import re
from typing import List, Union
from decord import VideoReader
from transformers.video_utils import VideoMetadata

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration
from sglang.srt.models.glm4v_moe import Glm4v_moeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)


class Glm4vImageProcessor(SGLangBaseProcessor):
    models = [Glm4vForConditionalGeneration, Glm4v_moeForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # GLM-4V specific tokens
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
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28

        # Setup regex patterns
        self.IMAGE_TOKEN_REGEX = re.compile(
            r"<\|begin_of_image\|>(?:<\|image\|>)+<\|end_of_image\|>"
        )
        self.VIDEO_TOKEN_REGEX = re.compile(
            r"<\|begin_of_video\|>(?:<\|video\|>)+<\|end_of_video\|>"
        )

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            image_token_regex=self.IMAGE_TOKEN_REGEX,
            video_token=self.VIDEO_TOKEN,
            video_token_id=self.VIDEO_TOKEN_ID,
            video_token_regex=self.VIDEO_TOKEN_REGEX,
        ).build(_processor)

    def preprocess_video(self, vr: VideoReader):
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

        metadata = VideoMetadata(
            total_num_frames=int(total_num_frames),
            fps=float(video_fps),
            duration=float(duration),
            video_backend="decord",
        )

        # Extract all frames
        indices = list(range(total_num_frames))
        frames = vr.get_batch(indices).asnumpy()
        metadata.frames_indices = indices

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

        # videos_processed = []
        # video_outputs = {}
        # if base_output.videos:
        #     # devide the base output into image_base_output and video_base_output
        #     video_base_output = BaseMultiModalProcessorOutput(
        #         input_text="<|begin_of_video|><|video|><|end_of_video|>",
        #         videos=base_output.videos,
        #     )
        #     base_output.videos = []

        #     # Process videos standalone first
        #     video_grid_thw_lst = []

        #     for video_vr in video_base_output.videos:
        #         # Process individual video
        #         frames, metadata = self.preprocess_video(video_vr)

        #         video_base_output.videos = [[frames]]

        #         video_mm_items, video_input_ids, video_ret = self.process_and_combine_mm_data(
        #             video_base_output, self.mm_tokens, video_metadata=[[metadata]]
        #         )

        #         # # Replace image token IDs with video token IDs in the processed video
        #         # video_input_ids[video_input_ids == self.mm_tokens.image_token_id] = self.mm_tokens.video_token_id
        #         # # Get video placeholder text
        #         # video_placeholder = self._processor.tokenizer.batch_decode(video_input_ids.unsqueeze(0))[0]

        #         # # Replace first remaining video token with the processed placeholder
        #         # input_text = input_text.replace(
        #         #     "<|begin_of_video|><|video|><|end_of_video|>",
        #         #     video_placeholder,
        #         #     1  # Replace first occurrence (next unprocessed video)
        #         # )

        #         # Collect video grid info
        #         if hasattr(video_ret, "video_grid_thw") and video_ret.video_grid_thw is not None:
        #             grid_t = len(video_ret.video_grid_thw)
        #             _, grid_h, grid_w = video_ret.video_grid_thw[0]
        #             grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])
        #             video_grid_thw_lst.append(grid_thw)

        #         videos_processed.append(video_mm_items)

        #     # Combine video outputs
        #     if video_grid_thw_lst:
        #         video_outputs["video_grid_thw"] = torch.cat(video_grid_thw_lst)

        #     base_output.input_text = input_text

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # # Combine video items with image items
        # if videos_processed:
        #     for video_items in videos_processed:
        #         mm_items.extend(video_items)

        input_ids = input_ids.flatten()
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids,
            hf_config=self.hf_config,
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
        )
        mrope_positions = mrope_positions.squeeze(1)
        print(f"!!!DEBUG: mm_items: {mm_items}")
        print(f"!!!DEBUG: mrope_positions: {mrope_positions}")

        mm_inputs = {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

        return mm_inputs
