# Adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py

from functools import lru_cache

import numpy as np
import torch
from decord import VideoReader, cpu, gpu
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.interns1 import InternS1ForConditionalGeneration
from sglang.srt.models.internvl import InternVLChatModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class InternVLImageProcessor(BaseMultimodalProcessor):
    models = [InternVLChatModel, InternS1ForConditionalGeneration]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_normalize_tensors(device="cuda", dtype=torch.float32):
        mean = torch.tensor(
            InternVLImageProcessor.IMAGENET_MEAN, device=device, dtype=dtype
        ).view(-1, 1, 1)
        std = torch.tensor(
            InternVLImageProcessor.IMAGENET_STD, device=device, dtype=dtype
        ).view(-1, 1, 1)
        return mean, std

    def __init__(self, hf_config, server_args, _image_processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _image_processor, *args, **kwargs)
        image_size = (
            getattr(hf_config, "force_image_size", None)
            or hf_config.vision_config.image_size
        )
        patch_size = hf_config.vision_config.patch_size
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]

        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (hf_config.downsample_ratio**2)
        )
        if hasattr(self._processor, "tokenizer"):
            tokenizer = self._processor.tokenizer
        else:
            tokenizer = self._processor
        self.tokenizer = tokenizer

        self.img_start_token_id = tokenizer.convert_tokens_to_ids(self.IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<IMG_CONTEXT>",
            image_token_id=tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
        ).build(_image_processor)

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
        try:
            vr = VideoReader(video_path, ctx=gpu(0), num_threads=1)
            use_gpu = True
        except (RuntimeError, OSError) as e:
            print(
                f"[WARNING] Load video on gpu decoding failed: {e}. Falling back to CPU."
            )
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            use_gpu = False

        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list = []
        num_patches_list = []
        frame_indices = InternVLImageProcessor.get_index(
            bound, fps, max_frame, first_idx=0, num_segments=num_segments
        )

        mean, std = InternVLImageProcessor._get_normalize_tensors(device="cuda")

        for frame_index in frame_indices:
            # Load frame
            frame = vr[frame_index]
            if use_gpu:
                img = frame.cuda().permute(2, 0, 1).float() / 255.0
            else:
                img_np = frame.asnumpy()
                img = torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0

            img = (img - mean) / std

            tiles = InternVLImageProcessor.dynamic_preprocess(
                img, image_size=input_size, max_num=max_num, use_thumbnail=True
            )

            pixel_values_list.append(tiles)
            num_patches_list.append(tiles.shape[0])

        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values, num_patches_list

    @staticmethod
    def dynamic_preprocess(tensor, image_size=448, max_num=12, use_thumbnail=False):
        C, H, W = tensor.shape
        aspect_ratio = W / H

        # Generate all possible aspect ratios
        target_ratios = set(
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find closest ratio
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)

        for x, y in target_ratios:
            target_ar = x / y
            diff = abs(aspect_ratio - target_ar)
            blocks = x * y
            best_blocks = best_ratio[0] * best_ratio[1]

            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best_ratio = (x, y)
            elif diff == best_ratio_diff and blocks > best_blocks:
                best_ratio = (x, y)

        target_w, target_h = image_size * best_ratio[0], image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        # Resize on GPU
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # Split into tiles
        tiles = []
        for i in range(blocks):
            x = (i % best_ratio[0]) * image_size
            y = (i // best_ratio[0]) * image_size
            tile = resized[:, y : y + image_size, x : x + image_size]
            tiles.append(tile)

        # Add thumbnail if needed
        if use_thumbnail and len(tiles) > 1:
            thumb = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            tiles.append(thumb)

        return torch.stack(tiles).to(torch.bfloat16)

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        num_patches_list = []
        pixel_values = []

        mean, std = InternVLImageProcessor._get_normalize_tensors(device="cuda")

        # Process each input with allocated frames
        for image_index, image in enumerate(base_output.images):
            try:
                # TODO: video input
                # Convert PIL to GPU tensor
                if isinstance(image, Image.Image):
                    img_np = np.array(image.convert("RGB"))
                    tensor = (
                        torch.from_numpy(img_np).permute(2, 0, 1).cuda().float() / 255.0
                    )
                else:
                    tensor = image.cuda()  # assume already tensor

                tensor = (tensor - mean) / std
                tiles = self.dynamic_preprocess(
                    tensor, image_size=448, max_num=12, use_thumbnail=True
                )

                pixel_values.append(tiles)
                num_patches_list.append(tiles.shape[0])

            except Exception as e:
                print(f"[Error] Failed to process image {image_index}: {e}")
                return None

        # Concatenate all
        pixel_values = torch.cat(pixel_values, dim=0)

        original_placeholder = "<<<__IMG_CONTEXT_PLACEHOLDER__>>>"

        input_text = base_output.input_text.replace(
            self.IMG_CONTEXT_TOKEN, original_placeholder
        )

        input_text_updated = input_text
        for num_patches in num_patches_list:
            image_tokens = (
                self.IMG_START_TOKEN
                + self.IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + self.IMG_END_TOKEN
            )
            input_text_updated = input_text_updated.replace(
                original_placeholder, image_tokens, 1
            )

        input_text_updated = input_text_updated.replace(
            original_placeholder, self.IMG_CONTEXT_TOKEN
        )

        # Tokenize
        input_ids_tensor = self.tokenizer(input_text_updated, return_tensors="pt")[
            "input_ids"
        ].flatten()
        input_ids = input_ids_tensor.tolist()

        # Get image token offsets
        image_offsets = self.get_mm_items_offset(
            input_ids=input_ids_tensor.to("cuda"),
            mm_token_id=self.mm_tokens.image_token_id,
        )

        items = [
            MultimodalDataItem(
                feature=pixel_values,
                modality=Modality.IMAGE,
                offsets=image_offsets,
            )
        ]

        return {
            "input_ids": input_ids,
            "mm_items": items,
            "im_start_id": self.img_start_token_id,
            "im_end_id": self.img_end_token_id,
            "im_token_id": self.mm_tokens.image_token_id,
        }
