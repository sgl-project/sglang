import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import PIL
from decord import VideoReader, cpu
from PIL import Image

from sglang.srt.utils import encode_video, load_audio, load_image, logger


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image and video, in given order
    images: Optional[list[PIL.Image]] = None

    # audios
    audios: Optional[list[np.ndarray]] = None

    def normalize(self):
        for field_name in ["image_sizes", "images", "audios"]:
            field = getattr(self, field_name, None)
            if field is not None and isinstance(field, list) and len(field) == 0:
                setattr(self, field_name, None)


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[str] = None
    video_token: Optional[str] = None
    audio_token: Optional[str] = None

    def collect(self) -> list[str]:
        return [
            token
            for token in [self.image_token, self.video_token, self.audio_token]
            if token
        ]


class BaseMultimodalProcessor(ABC):
    models = []

    def __init__(self, hf_config, server_args, _processor):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        """
        process multimodal data with transformers AutoProcessor
        """
        if images is not None:
            kwargs["images"] = images
        if videos is not None:
            kwargs["videos"] = videos
        if audios is not None:
            kwargs["audios"] = audios

        processor = self._processor
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        return result

    @abstractmethod
    async def process_mm_data_async(
        self, image_data, input_text, max_req_input_len, **kwargs
    ):
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input
        """
        # Before processing inputs
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                # Estimate frames for the video
                vr = VideoReader(path, ctx=cpu(0))
                num_frames = len(vr)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        max_req_input_len: int,
        image_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """
        if isinstance(multimodal_tokens.image_token, int):
            multimodal_tokens.image_token = (
                self._processor.tokenizer.convert_ids_to_tokens(
                    multimodal_tokens.image_token
                )
            )
        else:
            multimodal_tokens.image_token = multimodal_tokens.image_token

        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._processor.tokenizer.decode(prompt)
        else:
            prompt = prompt
        if return_text:
            import re

            pattern = (
                "("
                + "|".join(re.escape(sep) for sep in multimodal_tokens.collect())
                + ")"
            )
            # split text into list of normal text and special tokens
            text_parts = re.split(pattern, prompt)

        # TODO(mick): load from server_args, env, or sampling_params
        MAX_NUM_FRAMES = 30
        estimated_frames_list = self.get_estimated_frames_list(image_data=image_data)
        total_frame_count = sum(estimated_frames_list)
        # a heuristic value, suggesting the maximum fraction of frames to embed from all visual inputs.
        # e.g., 0.1 suggests that 1 frame out of 10 input frames should be used
        scaling_factor = min(1.0, MAX_NUM_FRAMES / max(1, total_frame_count))

        assert len(image_data) == len(estimated_frames_list)

        image_index, audio_index = 0, 0
        hashes, image_sizes, images, audios = [], [], [], []
        new_text = ""
        for index, text_part in enumerate(text_parts):
            try:
                if text_part == multimodal_tokens.image_token:
                    # load as image
                    if len(images) >= MAX_NUM_FRAMES:
                        frames_to_process = 0
                    else:
                        estimated_frames = estimated_frames_list[image_index]
                        frames_to_process = max(
                            1, int(estimated_frames * scaling_factor)
                        )

                    if frames_to_process == 0:
                        frames = []
                    else:
                        image_file = image_data[image_index]
                        if isinstance(image_file, str) and image_file.startswith(
                            "video:"
                        ):
                            # video
                            path = image_file[len("video:") :]
                            frames = encode_video(
                                path, frame_count_limit=frames_to_process
                            )
                        else:
                            # image
                            raw_image, _size = load_image(image_file)
                            if discard_alpha_channel:
                                raw_image = raw_image.convert("RGB")
                            frames = [raw_image]
                        if len(frames) == 0:
                            continue

                    image_sizes += frames[0].size * len(frames)

                    # Generate a hashable value for the image file
                    if isinstance(image_file, Image.Image):
                        # For PIL.Image objects, use the ID as a hashable value
                        hash_value = hash(id(image_file))
                    else:
                        # For other types (strings, etc.), use the regular hash
                        hash_value = hash(image_file)

                    hashes += [hash_value] * len(frames)
                    images += frames
                    image_index += 1
                    if frames_to_process != 0:
                        new_text += multimodal_tokens.image_token * len(frames)
                    assert frames_to_process == len(frames)
                elif text_part == multimodal_tokens.audio_token:
                    # load as audio
                    audio_file = audio_data[audio_index]
                    audio = load_audio(audio_file)
                    hashes += [hash(audio_file)]
                    audios += [audio]
                    audio_index += 1
                    new_text += multimodal_tokens.audio_token
                else:
                    # TODO(mick): handle video
                    # normal text
                    new_text += text_part

            except Exception as e:
                logger.error(f"An exception occurred while loading images: {e}")
                raise RuntimeError(f"An exception occurred while loading images: {e}")

        out = BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            input_text=new_text,
        )
        out.normalize()
        return out
