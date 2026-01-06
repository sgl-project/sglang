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

import cv2
import numpy as np
import torch
from PIL import Image
from einops import repeat, rearrange


def array_to_tensor(np_array):
    image_pt = torch.tensor(np_array).float()
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    image_pts = repeat(image_pt, "c h w -> b c h w", b=1)
    return image_pts


class ImageProcessorV2:
    def __init__(self, size=512, border_ratio=None):
        self.size = size
        self.border_ratio = border_ratio

    @staticmethod
    def recenter(image, border_ratio: float = 0.2):
        """ recenter an image to leave some empty space at the image border.

        Args:
            image (ndarray): input image, float/uint8 [H, W, 3/4]
            mask (ndarray): alpha mask, bool [H, W]
            border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

        Returns:
            ndarray: output image, float/uint8 [H, W, 3/4]
        """

        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask = np.ones_like(image[..., 0:1]) * 255
            image = np.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]

        H, W, C = image.shape

        size = max(H, W)
        result = np.zeros((size, size, C), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        if h == 0 or w == 0:
            raise ValueError('input image is empty')
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2

        y2_min = (size - w2) // 2
        y2_max = y2_min + w2

        result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2),
                                                          interpolation=cv2.INTER_AREA)

        bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255

        mask = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)

        mask = mask * 255
        result = result.clip(0, 255).astype(np.uint8)
        mask = mask.clip(0, 255).astype(np.uint8)
        return result, mask

    def load_image(self, image, border_ratio=0.15, to_tensor=True):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image, mask = self.recenter(image, border_ratio=border_ratio)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = image.convert("RGBA")
            image = np.asarray(image)
            image, mask = self.recenter(image, border_ratio=border_ratio)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., np.newaxis]

        if to_tensor:
            image = array_to_tensor(image)
            mask = array_to_tensor(mask)
        return image, mask

    def __call__(self, image, border_ratio=0.15, to_tensor=True, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio
        image, mask = self.load_image(image, border_ratio=border_ratio, to_tensor=to_tensor)
        outputs = {
            'image': image,
            'mask': mask
        }
        return outputs


class MVImageProcessorV2(ImageProcessorV2):
    """
    view order: front, front clockwise 90, back, front clockwise 270
    """
    return_view_idx = True

    def __init__(self, size=512, border_ratio=None):
        super().__init__(size, border_ratio)
        self.view2idx = {
            'front': 0,
            'left': 1,
            'back': 2,
            'right': 3
        }

    def __call__(self, image_dict, border_ratio=0.15, to_tensor=True, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio

        images = []
        masks = []
        view_idxs = []
        for idx, (view_tag, image) in enumerate(image_dict.items()):
            view_idxs.append(self.view2idx[view_tag])
            image, mask = self.load_image(image, border_ratio=border_ratio, to_tensor=to_tensor)
            images.append(image)
            masks.append(mask)

        zipped_lists = zip(view_idxs, images, masks)
        sorted_zipped_lists = sorted(zipped_lists)
        view_idxs, images, masks = zip(*sorted_zipped_lists)

        image = torch.cat(images, 0).unsqueeze(0)
        mask = torch.cat(masks, 0).unsqueeze(0)
        outputs = {
            'image': image,
            'mask': mask,
            'view_idxs': view_idxs
        }
        return outputs


IMAGE_PROCESSORS = {
    "v2": ImageProcessorV2,
    'mv_v2': MVImageProcessorV2,
}

DEFAULT_IMAGEPROCESSOR = 'v2'
