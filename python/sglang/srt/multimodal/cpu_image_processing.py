import functools
from collections.abc import Sequence

import numpy as np
import torch


@functools.lru_cache(maxsize=32)
def _build_uint8_normalization_lut_cached(
    image_mean: tuple[float, ...], image_std: tuple[float, ...]
) -> np.ndarray:
    mean = np.asarray(image_mean, dtype=np.float64)
    std = np.asarray(image_std, dtype=np.float64)
    if np.any(std == 0):
        raise ValueError("image normalization standard deviation must be nonzero")

    values = np.arange(256, dtype=np.uint8)
    table = np.empty((3, 256), dtype=np.float32)
    std_inv = 1.0 / std
    for channel in range(3):
        normalized = (values / 255.0).astype(np.float32)
        normalized -= mean[channel]
        normalized *= std_inv[channel]
        table[channel] = normalized
    table.flags.writeable = False
    return table


def build_uint8_normalization_lut(
    image_mean: Sequence[float], image_std: Sequence[float]
) -> np.ndarray:
    """Build an exact uint8 lookup table for common HF normalization.

    The operation order intentionally matches processors that evaluate
    ``(uint8 / 255.0).astype(float32)``, then subtract a float64 mean and
    multiply by a float64 reciprocal standard deviation in place.  Keeping
    this contract explicit lets model adapters reuse the fast path without
    silently changing preprocessing numerics.
    """
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("uint8 image normalization requires three channels")
    return _build_uint8_normalization_lut_cached(
        tuple(float(value) for value in image_mean),
        tuple(float(value) for value in image_std),
    )


def normalize_and_navit_patchify_uint8(
    pixel_values: np.ndarray,
    *,
    patch_size: int,
    normalization_lut: np.ndarray,
) -> np.ndarray:
    """Fuse exact uint8 normalization with NaViT patchification on CPU.

    ``pixel_values`` may be HWC or THWC.  The returned layout is
    ``(num_patches, channels, patch_height, patch_width)``.  Applying the LUT
    after the uint8 layout transform avoids materializing a full float image
    and then copying it again during patchification.
    """
    if pixel_values.dtype != np.uint8:
        raise ValueError(
            f"NaViT CPU preprocessing expects uint8 input, got {pixel_values.dtype}"
        )
    if pixel_values.ndim == 3:
        pixel_values = pixel_values[None, ...]
    if pixel_values.ndim != 4 or pixel_values.shape[-1] != 3:
        raise ValueError(
            "NaViT CPU preprocessing expects HWC or THWC input with three channels"
        )
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if normalization_lut.shape != (3, 256) or normalization_lut.dtype != np.float32:
        raise ValueError(
            "normalization_lut must be a float32 array with shape (3, 256)"
        )

    frames, height, width, channels = pixel_values.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image shape {(height, width)} is not divisible by patch size {patch_size}"
        )

    patches = pixel_values.reshape(
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        channels,
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4).reshape(
        -1, channels, patch_size, patch_size
    )

    output = np.empty(patches.shape, dtype=np.float32)
    for channel in range(channels):
        output[:, channel] = normalization_lut[channel][patches[:, channel]]
    return output


def materialize_exact_navit_image_features(
    medias: Sequence[dict], image_processor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Resize and patchify image media with checkpoint-exact CPU numerics.

    Model adapters remain responsible for canonicalizing each media object to
    the checkpoint's expected color mode. The shared path only requires the
    small NaViT capability contract exposed by both Kimi-K2.5 and Kimi-K3
    processors, so future compatible checkpoints need no encoder-server branch.
    """
    config = image_processor.media_proc_cfg
    patch_size = int(config["patch_size"])
    normalization_lut = build_uint8_normalization_lut(
        config["image_mean"], config["image_std"]
    )
    features = []
    grids = []
    for media in medias:
        resize_config = image_processor.get_resize_config(media)
        image_np = image_processor.resize_image(
            media["image"],
            resize_config["new_width"],
            resize_config["new_height"],
            resize_config["pad_width"],
            resize_config["pad_height"],
        )
        features.append(
            torch.from_numpy(
                normalize_and_navit_patchify_uint8(
                    image_np,
                    patch_size=patch_size,
                    normalization_lut=normalization_lut,
                )
            )
        )
        grids.append(
            [
                1,
                image_np.shape[0] // patch_size,
                image_np.shape[1] // patch_size,
            ]
        )
    return features, torch.tensor(grids, dtype=torch.int64)


def supports_exact_navit_cpu_preprocessing(image_processor) -> bool:
    """Whether a checkpoint processor exposes the shared exact fast path."""
    config = getattr(image_processor, "media_proc_cfg", None)
    return (
        isinstance(config, dict)
        and callable(getattr(image_processor, "get_resize_config", None))
        and callable(getattr(image_processor, "resize_image", None))
        and all(name in config for name in ("patch_size", "image_mean", "image_std"))
    )
