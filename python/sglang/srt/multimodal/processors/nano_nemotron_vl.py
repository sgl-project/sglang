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

    def select_frames(
        self, video: "VideoReader", num_frames: int = 32
    ):  # TODO: allow num frames to be configurable
        total_frames = len(video)
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    def preprocess_video(self, video: "VideoReader"):
        frames = self.select_frames(video)
        video_array = video.get_batch(frames).asnumpy()
        assert isinstance(video_array, np.ndarray), "Video array must be a numpy array"
        frames_tensors = [
            self.preprocess_image(Image.fromarray(frame, mode="RGB"), max_num_tiles=1)
            for frame in video_array
        ]
        processed_video = torch.cat(frames_tensors, dim=0)
        fps = video.get_avg_fps()
        assert isinstance(fps, float), "FPS must be a float"
        timestamps = [frame / fps for frame in frames]
        return processed_video, timestamps

    def enhance_input_text(self, input_text: str, context_token: str, seqs: list[str]):
        placeholder = "<<TEMP_PLACEHOLDER>>"
        input_text = input_text.replace(context_token, placeholder)
        for seq in seqs:
            input_text = input_text.replace(placeholder, seq, 1)
        input_text = input_text.replace(placeholder, context_token)
        return input_text

    def produce_image_feature(self, images: list[Image.Image], input_text: str):
        preprocessed_images = [self.preprocess_image(image) for image in images]
        seqs = [
            f"{self.IMG_START_TOKEN}{self.IMG_CONTEXT_TOKEN * self.num_image_token * tiles.shape[0]}{self.IMG_END_TOKEN}"
            for tiles in preprocessed_images
        ]
        concatenated_images = torch.cat(preprocessed_images, dim=0)
        input_text = self.enhance_input_text(input_text, self.IMG_CONTEXT_TOKEN, seqs)
        return concatenated_images, input_text

    def produce_video_feature(self, videos: "list[VideoReader]", input_text: str):
        preprocessed_videos = []
        for video in videos:
            preprocessed_video, timestamps = self.preprocess_video(video)
            preprocessed_videos.append(preprocessed_video)
            seqs = [
                f"Frame {i + 1} sampled at {timestamp:.2f} seconds: {self.IMG_START_TOKEN}{self.VIDEO_CONTEXT_TOKEN * self.num_image_token}{self.IMG_END_TOKEN}"
                for i, timestamp in enumerate(timestamps)
            ]
            input_text = self.enhance_input_text(
                input_text, self.VIDEO_CONTEXT_TOKEN, seqs
            )
        concatenated_videos = torch.cat(preprocessed_videos, dim=0)
        return concatenated_videos, input_text

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
        image_feature, video_feature = None, None
        if base_output.images:
            image_feature, input_text = self.produce_image_feature(
                base_output.images, input_text
            )

        if base_output.videos:
            video_feature, input_text = self.produce_video_feature(
                base_output.videos, input_text
            )

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
