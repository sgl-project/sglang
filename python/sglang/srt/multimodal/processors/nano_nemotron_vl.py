from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from sglang.srt.multimodal.internvl_utils import image_to_pixel_values
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

if TYPE_CHECKING:
    from decord import VideoReader

Image.MAX_IMAGE_PIXELS = None
DEFAULT_NUM_TILES = 12
NUM_VIDEO_FRAMES = 32  # TODO: allow num frames to be configurable
NUM_VIDEO_TILES = 1


class NanoNemotronVLImageProcessor(BaseMultimodalProcessor):
    models = [NemotronH_Nano_VL_V2]

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)
        self.image_size = hf_config.image_size
        self.VIDEO_CONTEXT_TOKEN = hf_config.video_context_token
        self.IMG_CONTEXT_TOKEN = hf_config.img_context_token
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.num_image_token = int(
            (self.image_size // hf_config.patch_size) ** 2
            * (hf_config.downsample_ratio**2)
        )
        if hasattr(self._processor, "tokenizer"):
            tokenizer = self._processor.tokenizer
        else:
            tokenizer = self._processor
        self.tokenizer = tokenizer

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMG_CONTEXT_TOKEN,
            image_token_id=tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
            video_token=self.VIDEO_CONTEXT_TOKEN,
            video_token_id=tokenizer.convert_tokens_to_ids(self.VIDEO_CONTEXT_TOKEN),
        ).build(_image_processor)

        # Normalization config (mean/std) and tiling behavior
        self.norm_mean = hf_config.norm_mean
        self.norm_std = hf_config.norm_std
        self.use_thumbnail = hf_config.use_thumbnail

    def preprocess_image(
        self, image: Image.Image, *, max_num_tiles: int = DEFAULT_NUM_TILES
    ) -> torch.Tensor:
        return image_to_pixel_values(
            image,
            input_size=self.image_size,
            max_num_tiles=max_num_tiles,
            use_thumbnail=self.use_thumbnail,
            mean=self.norm_mean,
            std=self.norm_std,
        ).to(dtype=torch.bfloat16)

    def render_image(self, *, num_tiles: int):
        return f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * self.num_image_token * num_tiles}{self.IMG_END_TOKEN}"

    def render_frame(self, frame_index: int, *, timestamp: float):
        return f"Frame {frame_index + 1} sampled at {timestamp:.2f} seconds: {self.render_image(num_tiles=NUM_VIDEO_TILES)}"

    @staticmethod
    def parse_video(video: "VideoReader") -> tuple[np.ndarray, list[float]]:
        frames = np.linspace(0, len(video) - 1, NUM_VIDEO_FRAMES, dtype=int)
        video_array = video.get_batch(frames).asnumpy()
        timestamps = video.get_frame_timestamp(frames)[:, 0]
        return video_array, timestamps

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        image_feature = None
        if base_output.images:
            preprocessed_images = [
                self.preprocess_image(image) for image in base_output.images
            ]
            rendered_images = [
                self.render_image(num_tiles=image.shape[0])
                for image in preprocessed_images
            ]
            input_text = input_text.replace(
                self.IMG_CONTEXT_TOKEN, "".join(rendered_images), 1
            )
            image_feature = torch.cat(preprocessed_images, dim=0)

        video_feature = None
        if base_output.videos:
            preprocessed_videos = []
            for video in base_output.videos:
                video_array, timestamps = self.parse_video(video)
                frames_tensors = [
                    self.preprocess_image(
                        Image.fromarray(frame, mode="RGB"),
                        max_num_tiles=NUM_VIDEO_TILES,
                    )
                    for frame in video_array
                ]
                preprocessed_video = torch.cat(frames_tensors, dim=0)
                preprocessed_videos.append(preprocessed_video)
                rendered_frames = [
                    self.render_frame(frame_index=i, timestamp=timestamp)
                    for i, timestamp in enumerate(timestamps)
                ]
                input_text = input_text.replace(
                    self.VIDEO_CONTEXT_TOKEN, "".join(rendered_frames), 1
                )
            video_feature = torch.cat(preprocessed_videos, dim=0)

        input_ids_tensor = self.tokenizer(input_text, return_tensors="pt")[
            "input_ids"
        ].flatten()

        items = []

        for modality, feature in {
            (Modality.IMAGE, image_feature),
            (Modality.VIDEO, video_feature),
        }:
            if feature is not None:
                token_id = self.mm_tokens.get_token_id_by_modality(modality)
                items.append(
                    MultimodalDataItem(
                        feature=feature,
                        modality=modality,
                        offsets=self.get_mm_items_offset(
                            input_ids_tensor.to("cuda"), token_id
                        ),
                    )
                )

        return {
            "input_ids": input_ids_tensor.tolist(),
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
            "video_token_id": self.mm_tokens.video_token_id,
        }
