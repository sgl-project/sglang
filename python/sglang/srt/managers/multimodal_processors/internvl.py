# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from sglang.srt.managers.multimodal_processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.internvl import InternVLChatModel


class InternVLImageProcessor(BaseMultimodalProcessor):
    models = [InternVLChatModel]

    def __init__(self, hf_config, server_args, _image_processor):
        super().__init__(hf_config, server_args, _image_processor)
        image_size = hf_config.force_image_size or hf_config.vision_config.image_size
        patch_size = hf_config.vision_config.patch_size

        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_TOKEN = "<image>"
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

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, max_req_input_len, **kwargs
    ):
        if not image_data:
            return None

        # Ensure image_data is a list
        if isinstance(image_data, str):
            image_data = [image_data]

        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(image_token=self.IMG_TOKEN),
            max_req_input_len=max_req_input_len,
            discard_alpha_channel=True,
        )

        def process_image_internvl(image, input_size=448, max_num=12):
            transform = InternVLImageProcessor.build_transform(input_size=input_size)
            images = InternVLImageProcessor.dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        num_patches_list = []
        pixel_values = []
        # Process each input with allocated frames
        for image_index, (image) in enumerate(base_output.images):
            try:
                # TODO: video input
                raw_image = process_image_internvl(image)
                pixel_value = [raw_image.to(torch.bfloat16).cuda()]
                pixel_values += pixel_value
                num_patches = raw_image.shape[0]
                num_patches_list += [num_patches]

            except FileNotFoundError as e:
                print(e)
                return None

        pixel_values = torch.cat(pixel_values, dim=0)

        for idx, num_patches in enumerate(num_patches_list):
            image_tokens = (
                self.IMG_START_TOKEN
                + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + self.IMG_END_TOKEN
            )
            input_text = input_text.replace("<image>", image_tokens, 1)

        tokenizer = self._processor
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids,
            mm_token_id=self.img_context_token_id,
        )
        items = [
            MultimodalDataItem(
                pixel_values=pixel_values,
                modality=Modality.IMAGE,
                image_offsets=image_offsets,
            )
        ]

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.img_context_token_id,
        }
