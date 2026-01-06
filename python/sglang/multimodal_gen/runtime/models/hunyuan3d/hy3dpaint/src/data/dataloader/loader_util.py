#!/usr/bin/env python3

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import cv2
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageChops


class BaseDataset(Dataset):
    def __init__(self, json_path, num_view=4, image_size=512):
        self.data = list()
        self.num_view = num_view
        self.image_size = image_size
        if isinstance(json_path, str):
            json_path = [json_path]
        for jp in json_path:
            with open(jp) as f:
                self.data.extend(json.load(f))
        print("============= length of dataset %d =============" % len(self.data))

    def __len__(self):
        return len(self.data)

    def load_image(self, pil_img, color, image_size=None):
        if image_size is None:
            image_size = self.image_size
        if isinstance(pil_img, str):
            pil_img = Image.open(pil_img)
        else:
            pil_img = pil_img
        if pil_img.mode == "L":
            pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize((image_size, image_size))
        image = np.asarray(pil_img, dtype=np.float32) / 255.0
        if image.shape[2] == 3:
            image = image[:, :, :3]
            alpha = np.ones_like(image)
        else:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def _apply_scaling(self, image, scale_factor, width, height, bg_color, scale_width=True):
        """Apply scaling to image with proper cropping or padding."""
        if scale_width:
            new_width = int(width * scale_factor)
            new_height = height
        else:
            new_width = width
            new_height = int(height * scale_factor)

        image = image.resize((new_width, new_height), resample=Image.BILINEAR)

        if scale_factor > 1.0:
            # Crop to original size
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            image = image.crop((left, top, left + width, top + height))
        else:
            # Pad to original size
            pad_width = (width - new_width) // 2
            pad_height = (height - new_height) // 2
            image = ImageOps.expand(
                image,
                (
                    pad_width,
                    pad_height,
                    width - new_width - pad_width,
                    height - new_height - pad_height,
                ),
                fill=bg_color,
            )
        return image

    def _apply_rotation(self, image, bg_color):
        """Apply random rotation to image."""
        original_size = image.size
        angle = random.uniform(-30, 30)
        image = image.convert("RGBA")
        rotated_image = image.rotate(angle, resample=Image.BILINEAR, expand=True)

        # Create background with bg_color
        background = Image.new("RGBA", rotated_image.size, (bg_color[0], bg_color[1], bg_color[2], 255))
        background.paste(rotated_image, (0, 0), rotated_image)
        image = background.convert("RGB")

        # Crop to original size
        left = (image.width - original_size[0]) // 2
        top = (image.height - original_size[1]) // 2
        right = left + original_size[0]
        bottom = top + original_size[1]

        return image.crop((left, top, right, bottom))

    def _apply_translation(self, image, bg_color):
        """Apply random translation to image."""
        max_dx = 0.1 * image.size[0]
        max_dy = 0.1 * image.size[1]
        dx = int(random.uniform(-max_dx, max_dx))
        dy = int(random.uniform(-max_dy, max_dy))

        image = ImageChops.offset(image, dx, dy)

        # Fill edges
        width, height = image.size
        if dx > 0:
            image.paste(bg_color, (0, 0, dx, height))
        elif dx < 0:
            image.paste(bg_color, (width + dx, 0, width, height))

        if dy > 0:
            image.paste(bg_color, (0, 0, width, dy))
        elif dy < 0:
            image.paste(bg_color, (0, height + dy, width, height))

        return image

    def _apply_perspective(self, image, bg_color):
        """Apply random perspective transformation to image."""
        image_np = np.array(image)
        height, width = image_np.shape[:2]

        # Define original and new points
        original_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        perspective_scale = 0.2

        new_points = np.float32(
            [
                [random.uniform(0, width * perspective_scale), random.uniform(0, height * perspective_scale)],
                [random.uniform(width * (1 - perspective_scale), width), random.uniform(0, height * perspective_scale)],
                [
                    random.uniform(width * (1 - perspective_scale), width),
                    random.uniform(height * (1 - perspective_scale), height),
                ],
                [
                    random.uniform(0, width * perspective_scale),
                    random.uniform(height * (1 - perspective_scale), height),
                ],
            ]
        )

        matrix = cv2.getPerspectiveTransform(original_points, new_points)
        image_np = cv2.warpPerspective(
            image_np, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color
        )

        return Image.fromarray(image_np)

    def augment_image(
        self,
        image,
        bg_color,
        identity_prob=0.5,
        rotate_prob=0.3,
        scale_prob=0.5,
        translate_prob=0.5,
        perspective_prob=0.3,
    ):
        if random.random() < identity_prob:
            return image

        # Convert torch tensors back to PIL images for augmentation
        image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        bg_color = (int(bg_color[0] * 255), int(bg_color[1] * 255), int(bg_color[2] * 255))

        # Random rotation
        if random.random() < rotate_prob:
            image = self._apply_rotation(image, bg_color)

        # Random scaling
        if random.random() < scale_prob:
            width, height = image.size
            scale_factor = random.uniform(0.8, 1.2)

            if random.random() < 0.5:
                # Scale both dimensions proportionally
                image = self._apply_scaling(image, scale_factor, width, height, bg_color, scale_width=True)
                image = self._apply_scaling(image, scale_factor, width, height, bg_color, scale_width=False)
            else:
                # Scale width then height independently
                scale_factor_w = random.uniform(0.8, 1.2)
                scale_factor_h = random.uniform(0.8, 1.2)
                image = self._apply_scaling(image, scale_factor_w, width, height, bg_color, scale_width=True)
                image = self._apply_scaling(image, scale_factor_h, width, height, bg_color, scale_width=False)

        # Random translation
        if random.random() < translate_prob:
            image = self._apply_translation(image, bg_color)

        # Random perspective
        if random.random() < perspective_prob:
            image = self._apply_perspective(image, bg_color)

        # Convert back to torch tensors
        image = image.convert("RGB")
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        return image
