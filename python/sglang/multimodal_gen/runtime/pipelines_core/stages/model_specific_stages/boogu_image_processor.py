# SPDX-License-Identifier: Apache-2.0
import warnings

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import (
    PipelineImageInput,
    VaeImageProcessor,
    is_valid_image_imagelist,
)


class BooguImageProcessor(VaeImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        resample: str = "lanczos",
        max_pixels: int | None = None,
        max_side_length: int | None = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_rgb: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_rgb=do_convert_rgb,
            do_convert_grayscale=do_convert_grayscale,
        )
        self.max_pixels = max_pixels
        self.max_side_length = max_side_length

    def get_new_height_width(
        self,
        image: PIL.Image.Image | np.ndarray | torch.Tensor,
        height: int | None = None,
        width: int | None = None,
        max_pixels: int | None = None,
        max_side_length: int | None = None,
    ) -> tuple[int, int]:
        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            else:
                height = image.shape[1]

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            else:
                width = image.shape[2]

        if max_side_length is None:
            max_side_length = self.max_side_length

        if max_pixels is None:
            max_pixels = self.max_pixels

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Image height and width must be positive, got height={height}, width={width}"
            )

        max_side_length_ratio = 1.0
        if max_side_length is not None:
            if height > width:
                max_side_length_ratio = max_side_length / height
            else:
                max_side_length_ratio = max_side_length / width

        cur_pixels = height * width
        max_pixels_ratio = (
            (max_pixels / cur_pixels) ** 0.5 if max_pixels is not None else 1.0
        )
        ratio = min(max_pixels_ratio, max_side_length_ratio, 1.0)

        new_height, new_width = (
            int(height * ratio)
            // self.config.vae_scale_factor
            * self.config.vae_scale_factor,
            int(width * ratio)
            // self.config.vae_scale_factor
            * self.config.vae_scale_factor,
        )
        return new_height, new_width

    def preprocess(
        self,
        image: PipelineImageInput,
        height: int | None = None,
        width: int | None = None,
        max_pixels: int | None = None,
        max_side_length: int | None = None,
        resize_mode: str = "default",
        crops_coords: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        if (
            self.config.do_convert_grayscale
            and isinstance(image, (torch.Tensor, np.ndarray))
            and image.ndim == 3
        ):
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(1)
            else:
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if (
            isinstance(image, list)
            and isinstance(image[0], np.ndarray)
            and image[0].ndim == 4
        ):
            warnings.warn(
                "Passing `image` as a list of 4d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if (
            isinstance(image, list)
            and isinstance(image[0], torch.Tensor)
            and image[0].ndim == 4
        ):
            warnings.warn(
                "Passing `image` as a list of 4d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            supported_formats_str = ", ".join(str(x) for x in supported_formats)
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support {supported_formats_str}"
            )

        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            if self.config.do_resize:
                height, width = self.get_new_height_width(
                    image[0], height, width, max_pixels, max_side_length
                )
                image = [
                    self.resize(i, height, width, resize_mode=resize_mode)
                    for i in image
                ]
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            image = self.pil_to_numpy(image)
            image = self.numpy_to_pt(image)

        elif isinstance(image[0], np.ndarray):
            image = (
                np.concatenate(image, axis=0)
                if image[0].ndim == 4
                else np.stack(image, axis=0)
            )
            image = self.numpy_to_pt(image)
            height, width = self.get_new_height_width(
                image, height, width, max_pixels, max_side_length
            )
            if self.config.do_resize:
                image = self.resize(image, height, width)

        elif isinstance(image[0], torch.Tensor):
            image = (
                torch.cat(image, axis=0)
                if image[0].ndim == 4
                else torch.stack(image, axis=0)
            )

            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)

            channel = image.shape[1]
            if channel == self.config.vae_latent_channels:
                return image

            height, width = self.get_new_height_width(
                image, height, width, max_pixels, max_side_length
            )
            if self.config.do_resize:
                image = self.resize(image, height, width)

        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range "
                f"for image tensor is [0,1] when passing as pytorch tensor or numpy Array. You passed `image` with "
                f"value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image
