import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from typing import Optional

import PIL
import transformers
from decord import VideoReader, cpu
from PIL import Image

from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import load_image
from sglang.utils import logger

global global_processor


def get_global_processor():
    global global_processor
    return global_processor


def init_global_processor(sglang_image_processor, server_args: ServerArgs):
    """Init the global processor for multi-modal models."""
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = sglang_image_processor._build_processor(server_args=server_args)


@dataclasses.dataclass
class BaseImageProcessorOutput:
    image_hashes: list[int]
    image_sizes: list[tuple[int, int]]
    all_frames: [PIL.Image]
    # input_text, with each frame of video/image represented as an image_token
    input_text: str


class BaseImageProcessor(ABC):
    def __init__(self, hf_config, server_args, _processor):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.executor = concurrent.futures.ProcessPoolExecutor(
            initializer=init_global_processor,
            mp_context=mp.get_context("fork"),
            initargs=(
                self,
                server_args,
            ),
            max_workers=int(os.environ.get("SGLANG_CPU_COUNT", os.cpu_count())),
        )

    def _build_processor(self, server_args):
        """Init the global processor for multi modal models."""
        from sglang.srt.hf_transformers_utils import get_processor

        return get_processor(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )

    @abstractmethod
    async def process_images_async(
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

    @staticmethod
    def encode_video(video_path, frame_count_limit=None):
        if not os.path.exists(video_path):
            logger.error(f"Video {video_path} does not exist")
            return []

        if frame_count_limit == 0:
            return []

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_indices = [i for i in range(0, len(vr), sample_fps)]
        if frame_count_limit is not None and len(frame_indices) > frame_count_limit:
            frame_indices = uniform_sample(frame_indices, frame_count_limit)

        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        return frames

    def load_images(
        self,
        input_ids: list,
        image_data,
        image_token: str,
        max_req_input_len: int,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
    ) -> BaseImageProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:

            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """
        image_hashes, image_sizes = [], []
        all_frames = []
        new_text_parts = []

        if isinstance(input_ids, list) and return_text:
            assert len(input_ids) and isinstance(input_ids[0], int)
            input_text = self._processor.tokenizer.decode(input_ids)
        else:
            input_text = input_ids

        if return_text:
            text_parts = input_text.split(image_token)

        # TODO(mick): load from server_args, env, or sampling_params
        MAX_NUM_FRAMES = 30
        estimated_frames_list = self.get_estimated_frames_list(image_data=image_data)
        total_frame_count = sum(estimated_frames_list)
        # a heuristic value, suggesting the maximum fraction of frames to embed from all visual inputs.
        # e.g., 0.1 suggests that 1 frame out of 10 input frames should be used
        scaling_factor = min(1.0, MAX_NUM_FRAMES / total_frame_count)

        assert len(image_data) == len(estimated_frames_list)

        # Process each input with allocated frames
        for image_index, (image, estimated_frames) in enumerate(
            zip(image_data, estimated_frames_list)
        ):
            if len(all_frames) >= MAX_NUM_FRAMES:
                max_frames_to_process = 0
            else:
                max_frames_to_process = max(1, int(estimated_frames * scaling_factor))

            if max_frames_to_process == 0:
                frames = []
            else:
                try:
                    if isinstance(image, str) and image.startswith("video:"):
                        path = image[len("video:") :]
                        frames = BaseImageProcessor.encode_video(
                            path, frame_count_limit=max_frames_to_process
                        )
                    else:
                        raw_image, _size = load_image(image)
                        if discard_alpha_channel:
                            raw_image = raw_image.convert("RGB")
                        frames = [raw_image]
                    assert len(frames) != 0
                except FileNotFoundError as e:
                    print(e)
                    return None

                image_sizes += [frames[0].size] * len(frames)
                image_hashes += [hash(image)] * len(frames)
                all_frames += frames

            if return_text:
                new_text_parts.append(text_parts[image_index])
            if max_frames_to_process != 0:
                new_text_parts.append(image_token * len(frames))
            assert max_frames_to_process >= len(frames)
        if return_text:
            new_text_parts.append(text_parts[-1])

        input_text = "".join(new_text_parts)
        return BaseImageProcessorOutput(
            image_hashes, image_sizes, all_frames, input_text
        )


class DummyImageProcessor(BaseImageProcessor):
    def __init__(self):
        pass

    async def process_images_async(self, *args, **kwargs):
        return None


def init_global_processor(
    sglang_image_processor: BaseImageProcessor, server_args: ServerArgs
):
    """Init the global processor for multi-modal models."""
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = sglang_image_processor._build_processor(server_args=server_args)
