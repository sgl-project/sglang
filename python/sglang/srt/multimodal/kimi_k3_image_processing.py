from typing import Union

import numpy as np
import torch
from PIL import Image

DEFERRED_PREPROCESSING_KEY = "kimi_k3_deferred_preprocessing"


def to_chw_uint8(
    image: Union[torch.Tensor, Image.Image],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(image, Image.Image):
        has_alpha = image.mode != "RGB" and (
            "A" in image.getbands() or "transparency" in image.info
        )
        array = np.array(image.convert("RGBA" if has_alpha else "RGB"), copy=True)
        image = torch.from_numpy(array).permute(2, 0, 1)

    if image.dtype != torch.uint8:
        raise ValueError(
            f"Kimi-K3 preprocessing expects raw uint8 pixels, got {image.dtype}"
        )
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if device is not None:
        image = image.to(device)
    return image


def fill_transparent_bg(x: torch.Tensor, bg_config: Union[dict, None]) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    rgb = x[:, :3]
    if bg_config is None:
        return rgb

    _, _, height, width = x.shape
    pattern = bg_config.get("pattern", "black")
    if pattern == "chessboard":
        square = bg_config.get("chessboard_square_size", 16)
        white = float(bg_config.get("chessboard_white_value", 255))
        gray = float(bg_config.get("chessboard_gray_value", 200))
        top_left = bg_config.get("chessboard_square_on_top_left", True)
        ys = torch.arange(height, device=x.device) // square
        xs = torch.arange(width, device=x.device) // square
        parity = (ys.unsqueeze(1) + xs.unsqueeze(0)) % 2
        background = torch.where(parity == (1 if top_left else 0), gray, white)
        background = background.unsqueeze(0).expand(3, height, width)
    elif pattern == "white":
        background = torch.full((3, height, width), 255.0, device=x.device)
    elif pattern == "black":
        background = torch.zeros(3, height, width, device=x.device)
    elif pattern == "gray":
        background = torch.full((3, height, width), 128.0, device=x.device)
    else:
        raise ValueError(f"Invalid background pattern: {pattern}")

    alpha = (x[:, 3:4] / 255.0).clamp(0.0, 1.0)
    return (alpha * rgb + (1.0 - alpha) * background).clamp(0.0, 255.0).floor_()


def normalization_tensors(
    image_mean: list[float],
    image_std: list[float],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.tensor(
        [1.0 / (255.0 * std) for std in image_std],
        device=device,
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    bias = torch.tensor(
        [-mean / std for mean, std in zip(image_mean, image_std)],
        device=device,
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    return scale, bias
