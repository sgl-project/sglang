import asyncio
import math
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.image_processor import BaseImageProcessor
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.internvl import InternVLChatModel
from sglang.srt.utils import load_image


# Compatible with InternVL
class InternVLImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _image_processor):
        super().__init__(hf_config, server_args, _image_processor)
        self._image_processor = _image_processor
        image_size = hf_config.force_image_size or hf_config.vision_config.image_size
        patch_size = hf_config.vision_config.patch_size

        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (hf_config.downsample_ratio**2)
        )

        tokenizer = self._processor
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(
            self.IMG_CONTEXT_TOKEN
        )

    @staticmethod
    def _process_single_image_task(
        image_data: Union[str, bytes],
        image_processor=None,
    ):
        pass

    @staticmethod
    def build_transform(input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def resize_image(img, size):
            return img.resize((size, size), Image.Resampling.BICUBIC)

        def to_tensor(img):
            # Convert PIL Image to numpy array
            img_array = np.array(img).astype(np.float32) / 255.0
            # Convert HWC to CHW format
            img_array = img_array.transpose(2, 0, 1)
            return torch.from_numpy(img_array)

        def normalize(tensor, mean, std):
            mean = torch.tensor(mean).view(-1, 1, 1)
            std = torch.tensor(std).view(-1, 1, 1)
            return (tensor - mean) / std

        def transform(img):
            img = img.convert("RGB") if img.mode != "RGB" else img
            img = resize_image(img, input_size)
            tensor = to_tensor(img)
            tensor = normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
            return tensor

        return transform

    @staticmethod
    def dynamic_preprocess(
        image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):

        def find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, image_size
        ):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array(
            [
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_segments)
            ]
        )
        return frame_indices

    @staticmethod
    def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = InternVLImageProcessor.build_transform(input_size=input_size)
        frame_indices = InternVLImageProcessor.get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            img = InternVLImageProcessor.dynamic_preprocess(
                img, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None

        tokenizer = self._processor
        if isinstance(input_ids, list):
            assert len(input_ids) and isinstance(input_ids[0], int)
            input_text = tokenizer.decode(input_ids)
        else:
            input_text = input_ids

        image_hashes, image_sizes = [], []

        all_frames = []

        def load_image_internvl(image_file, input_size=448, max_num=12):
            image, _size = load_image(image_file)
            image = image.convert("RGB")
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
            transform = InternVLImageProcessor.build_transform(input_size=input_size)
            images = InternVLImageProcessor.dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        num_patches_list = []

        # Process each input with allocated frames
        for image_index, (image) in enumerate(image_data):
            try:
                if isinstance(image, str) and image.startswith("video:"):
                    path = image[len("video:") :]
                    pixel_values, num_patches_list_video = (
                        InternVLImageProcessor.load_video(path)
                    )

                    frames = [pixel_values.to(torch.bfloat16)]
                    num_patches_list += num_patches_list_video
                else:
                    raw_image = load_image_internvl(image)
                    frames = [raw_image.to(torch.bfloat16)]
                    num_patches = raw_image.shape[0]
                    num_patches_list += [num_patches]

            except FileNotFoundError as e:
                print(e)
                return None
            image_hashes += [hash(image)] * len(frames)
            all_frames += frames

        pixel_values = torch.cat(all_frames, dim=0)
        for idx, num_patches in enumerate(num_patches_list):
            image_tokens = (
                self.IMG_START_TOKEN
                + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + self.IMG_END_TOKEN
            )
            input_text = input_text.replace("<image>", image_tokens, 1)
        return {
            "input_ids": tokenizer(input_text, return_tensors="pt")["input_ids"]
            .flatten()
            .tolist(),
            "pixel_values": pixel_values,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.img_context_token_id,
            "image_hashes": image_hashes,
            "image_sizes": image_sizes,
            "modalities": request_obj.modalities or ["image"],
        }


ImageProcessorMapping = {
    InternVLChatModel: InternVLImageProcessor,
}
