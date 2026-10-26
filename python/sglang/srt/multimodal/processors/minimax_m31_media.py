"""M3.1 resize policy with the public MiniMax image/video processors."""

import numpy as np
import torch
from PIL import Image
from transformers import MiniMaxM3VLImageProcessor, MiniMaxM3VLVideoProcessor


def _resize_shape(height, width, max_size, factor):
    def align(value):
        return (value + factor - 1) // factor * factor

    height, width = align(height), align(width)
    max_height, max_width = max_size
    if height > max_height or width > max_width:
        new_height = min(height * max_height // height, height * max_width // width)
        new_width = min(width * max_height // height, width * max_width // width)
        height = align(max(new_height, factor))
        width = align(max(new_width, factor))
    return height, width


class _MiniMaxM31Resize:
    def __init__(self, *, max_size, **kwargs):
        self.max_size = tuple(max_size)
        super().__init__(**kwargs)

    def resize_frames(self, tensor, size):
        height, width = tensor.shape[-2:]
        out_height, out_width = size
        if (height, width) == (out_height, out_width):
            return tensor
        # Match the checkpoint's PIL bicubic interpolation.
        frames = tensor.reshape(-1, tensor.shape[-3], height, width)
        resized = []
        for frame in frames:
            pixels = frame.permute(1, 2, 0).cpu().to(torch.uint8)
            image = Image.fromarray(pixels.numpy())
            image = image.resize((out_width, out_height), Image.Resampling.BICUBIC)
            resized.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
        return (
            torch.stack(resized)
            .reshape(*tensor.shape[:-2], out_height, out_width)
            .to(device=tensor.device, dtype=tensor.dtype)
        )

    def _resize_media(self, media, patch_size, merge_size):
        result = []
        for tensor in media:
            height, width = tensor.shape[-2:]
            out_height, out_width = _resize_shape(
                height, width, self.max_size, patch_size * merge_size
            )
            result.append(self.resize_frames(tensor, (out_height, out_width)))
        return result


class MiniMaxM31ImageProcessor(_MiniMaxM31Resize, MiniMaxM3VLImageProcessor):
    def _preprocess(self, images, do_resize, **kwargs):
        if do_resize:
            images = self._resize_media(
                images, kwargs["patch_size"], kwargs["merge_size"]
            )
        return super()._preprocess(images=images, do_resize=False, **kwargs)

    def get_number_of_image_patches(self, height, width, images_kwargs=None):
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        height, width = _resize_shape(
            height, width, self.max_size, patch_size * merge_size
        )
        return height // patch_size * (width // patch_size)


class MiniMaxM31VideoProcessor(_MiniMaxM31Resize, MiniMaxM3VLVideoProcessor):
    def _preprocess(self, videos, do_resize, **kwargs):
        if do_resize:
            videos = self._resize_media(
                videos, kwargs["patch_size"], kwargs["merge_size"]
            )
        return super()._preprocess(videos=videos, do_resize=False, **kwargs)
