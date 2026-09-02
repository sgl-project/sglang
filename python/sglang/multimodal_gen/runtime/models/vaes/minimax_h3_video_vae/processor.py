# SPDX-License-Identifier: Apache-2.0
# Tensor pre/post-processing for the MiniMax H3 visual VAE.
import math
from typing import Tuple

import numpy as np
import torch
from diffusers.utils import logging
from einops import rearrange
from torchvision.transforms import Normalize

NORM_CONFIGS = {
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "simple": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "raw": {
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    },
}


def get_norm_constants(
    norm_type: str = "imagenet",
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if norm_type not in NORM_CONFIGS:
        raise ValueError(
            f"Unknown norm_type: {norm_type}. Must be one of {list(NORM_CONFIGS.keys())}"
        )
    config = NORM_CONFIGS[norm_type]
    return config["mean"], config["std"]


def get_normalize_transform(
    norm_type: str = "imagenet", *, inplace: bool = False
) -> Normalize:
    mean, std = get_norm_constants(norm_type)
    return Normalize(mean, std, inplace=inplace)


def get_denormalize_transform(norm_type: str = "imagenet") -> Normalize:
    mean, std = get_norm_constants(norm_type)
    inv_mean = tuple(-m / s for m, s in zip(mean, std))
    inv_std = tuple(1.0 / s for s in std)
    return Normalize(inv_mean, inv_std)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class VAEProcessor:

    def __init__(
        self,
        *,
        vae_ratio,
        vae_ratio_t,
        clip_length,
        frame_overlap,
        token_overlap,
        tokens_chunk_size,
        isolated_last_frame,
        latent_patch_size,
        crop_mode,
        pixel_norm_type="imagenet",
        transform=None,
        transform_rev=None,
        use_3d_conv=False,
    ):
        self.vae_ratio = vae_ratio
        self.vae_ratio_t = vae_ratio_t
        self.clip_length = clip_length
        self.frame_overlap = frame_overlap
        self.token_overlap = token_overlap
        self.tokens_chunk_size = tokens_chunk_size
        self.isolated_last_frame = isolated_last_frame
        self.latent_patch_size = latent_patch_size
        self.crop_mode = crop_mode
        self.transform = transform or get_normalize_transform(pixel_norm_type)
        self._runtime_owned_transform = (
            get_normalize_transform(pixel_norm_type, inplace=True)
            if transform is None
            else None
        )
        self.transform_rev = transform_rev or get_denormalize_transform(pixel_norm_type)
        self.use_3d_conv = use_3d_conv

    def _ensure_list(self, data):
        return data if isinstance(data, list) else [data]

    def _align_to_total_patch_size(self, h, w):
        total_patch_size = self.latent_patch_size * self.vae_ratio
        new_h = (h // total_patch_size) * total_patch_size
        new_w = (w // total_patch_size) * total_patch_size
        return new_h, new_w

    def _crop_to_align(self, tensor, new_h, new_w, is_video=False):
        if is_video:
            _, _, _, h, w = tensor.shape
        else:
            _, _, h, w = tensor.shape

        if self.crop_mode == "center":
            top = (h - new_h) // 2
            left = (w - new_w) // 2
        else:
            top = 0
            left = 0

        if is_video:
            return tensor[:, :, :, top : top + new_h, left : left + new_w]
        else:
            return tensor[:, :, top : top + new_h, left : left + new_w]

    def _align_target_token(self, T, mode):
        intra_tail = self.clip_length % self.vae_ratio_t
        min_frames = intra_tail or self.vae_ratio_t
        full_chunks = T // self.clip_length
        remainder = T % self.clip_length

        if remainder == 0:
            return max(T, min_frames)

        if mode == "pad":
            aligned_r = (
                math.ceil((remainder - intra_tail) / self.vae_ratio_t)
                * self.vae_ratio_t
                + intra_tail
            )
            if aligned_r > self.clip_length:
                return (full_chunks + 1) * self.clip_length + intra_tail
            return full_chunks * self.clip_length + aligned_r
        else:  # trim
            k = (remainder - intra_tail) // self.vae_ratio_t
            if k >= 0:
                target = (
                    full_chunks * self.clip_length + k * self.vae_ratio_t + intra_tail
                )
                return max(target, min_frames)
            elif full_chunks > 0:
                return full_chunks * self.clip_length
            else:
                return min_frames

    def _align_target(self, T, mode, granularity):
        if granularity == "chunk":
            step = self.clip_length
            tail = self.frame_overlap
            if self.isolated_last_frame:
                tail += 1

            k = math.ceil((T - tail) / step) if mode == "pad" else (T - tail) // step
            return max(k, 1) * step + tail

        isolated_extra = 1 if self.isolated_last_frame else 0
        return self._align_target_token(T - isolated_extra, mode) + isolated_extra

    def align_video_length(self, video_length, mode="pad", granularity="chunk"):
        target = self._align_target(video_length, mode, granularity)
        delta = target - video_length
        if delta > 0 and mode == "trim":
            raise ValueError(
                f"Cannot trim {video_length} frames to valid length {target}: "
                f"not enough frames (granularity={granularity})"
            )
        return delta

    def align_video_length_2pass(self, video_length):
        """Return the leading/trailing frame pads and trailing latent drop.

        This is the continuation-prefix (2-pass) alignment.  The caller temporarily disables the model's normal token
        drop and keeps these mirrored processor fields at zero.
        """
        if self.isolated_last_frame:
            raise ValueError(
                "align_video_length_2pass does not support isolated_last_frame"
            )
        if self.token_overlap != 0 or self.frame_overlap != 0:
            raise ValueError("align_video_length_2pass requires token_drop=0 alignment")

        leading = self.align_video_length(video_length, mode="pad", granularity="token")
        token_aligned = video_length + leading
        trailing = self.align_video_length(
            token_aligned, mode="pad", granularity="chunk"
        )

        if trailing > 0:
            intra_tail = self.clip_length % self.vae_ratio_t
            full_chunks = token_aligned // self.clip_length
            remainder = token_aligned % self.clip_length
            real_tokens = full_chunks * self.tokens_chunk_size
            if remainder > 0:
                real_tokens += (remainder - intra_tail) // self.vae_ratio_t + 1
            drop_tokens = self.get_latent_length(token_aligned + trailing) - real_tokens
        else:
            drop_tokens = 0

        return leading, trailing, drop_tokens

    def get_suitable_video_length(self, video_length, verbose=False):
        used_frame_length = video_length + self.align_video_length(
            video_length, mode="trim", granularity="chunk"
        )
        if verbose:
            logger.info(
                f"Pick first {used_frame_length} frames from {video_length}-frame video"
            )
        return used_frame_length

    def get_latent_length(self, video_length):
        tail_frame = self.frame_overlap
        tail_token = self.token_overlap
        if self.isolated_last_frame:
            tail_frame += 1
            tail_token += 1

        video_length = self.get_suitable_video_length(video_length)
        latent_length = (
            int((video_length - tail_frame) // self.clip_length)
            * self.tokens_chunk_size
            + tail_token
        )
        return latent_length

    def transform_tensor(self, tensor, *, runtime_owned=False):
        B, T = None, None
        if tensor.ndim == 5:
            if tensor.shape[2] == 3:
                tensor = tensor.transpose(1, 2)
            B, _, T, _, _ = tensor.shape
            tensor = rearrange(tensor, "b c t h w -> (b t) c h w")
        elif tensor.ndim == 4:
            if tensor.shape[0] == 3:
                tensor = tensor.transpose(0, 1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        transform = (
            self._runtime_owned_transform
            if runtime_owned and self._runtime_owned_transform is not None
            else self.transform
        )
        tensor = transform(tensor)

        if B is not None and T is not None:
            tensor = rearrange(tensor, "(b t) c h w -> b c t h w", b=B, t=T)

        return tensor.contiguous()

    def revert_tensor(self, tensor):
        B, T = None, None
        if self.use_3d_conv:
            tensor = tensor.unsqueeze(2) if tensor.ndim == 4 else tensor
            B, _, T, _, _ = tensor.shape
            tensor = rearrange(tensor, "b c t h w -> (b t) c h w")
        tensor_rev = self.transform_rev(tensor).clamp_(0, 1)
        if B is not None:
            tensor_rev = rearrange(tensor_rev, "(b t) c h w -> b c t h w", b=B, t=T)
        return tensor_rev.contiguous()

    @staticmethod
    def convert_numpy_to_tensor(numpy_array, device=None):
        if isinstance(numpy_array, list):
            numpy_array = np.stack(numpy_array, axis=0)
        tensor = torch.from_numpy(numpy_array)
        # Keep decoded uint8 pixels compact across the host-to-device copy.
        # Casting the full video on CPU quadruples both the temporary host
        # allocation and transfer volume for no loss of information.
        if device is not None:
            tensor = tensor.to(device)
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor.to(torch.float32).div_(255.0)
