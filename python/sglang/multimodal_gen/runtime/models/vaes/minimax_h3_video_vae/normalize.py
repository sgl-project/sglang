# SPDX-License-Identifier: Apache-2.0
# Pixel normalization transforms for the MiniMax H3 visual VAE.
from typing import Tuple

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
