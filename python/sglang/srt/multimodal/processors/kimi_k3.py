"""Kimi K3 multimodal processor.

GPU image preprocessing dedicated to K3: unlike the K2.5 wrapper it keeps
the alpha channel through the bicubic resize and then composites RGBA
images onto the checkpoint-configured background
(``transparent_bg_config`` with ``transparent_bg_fill_stage ==
"after_resize"`` in preprocessor_config.json), instead of dropping alpha
at load time.
"""

import re
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from sglang.kernels.ops.mm.process import normalize_and_patchify
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.kimi_k25 import (
    KimiGPUProcessorWrapper,
    _get_image_dimensions,
    _grid_thw_from_resize_config,
    _resize_bicubic_if_needed,
    navit_resize_config,
)


def _k3_to_cuda_chw(image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        arr = np.asarray(image.convert("RGBA" if has_alpha else "RGB"))
        return torch.from_numpy(arr).permute(2, 0, 1).cuda()

    image = image.cuda()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image


def _chessboard_background(
    height: int, width: int, cfg: dict, device: torch.device
) -> torch.Tensor:
    square = cfg.get("chessboard_square_size", 16)
    white = float(cfg.get("chessboard_white_value", 255))
    gray = float(cfg.get("chessboard_gray_value", 200))
    on_top_left = cfg.get("chessboard_square_on_top_left", True)

    ys = torch.arange(height, device=device) // square
    xs = torch.arange(width, device=device) // square
    parity = (ys.unsqueeze(1) + xs.unsqueeze(0)) % 2
    gray_parity = 1 if on_top_left else 0
    bg = torch.where(parity == gray_parity, gray, white)
    return bg.unsqueeze(0).expand(3, height, width)


def _fill_transparent_bg(x: torch.Tensor, bg_cfg: Union[dict, None]) -> torch.Tensor:
    """Composite a resized (1, 4, H, W) float image in [0, 255] onto the
    configured background; 3-channel input passes through."""
    if x.shape[1] == 3:
        return x
    rgb = x[:, :3]
    if bg_cfg is None:
        return rgb

    _, _, height, width = x.shape
    pattern = bg_cfg.get("pattern", "black")
    if pattern == "chessboard":
        bg = _chessboard_background(height, width, bg_cfg, x.device)
    elif pattern == "white":
        bg = torch.full((3, height, width), 255.0, device=x.device)
    elif pattern == "black":
        bg = torch.zeros(3, height, width, device=x.device)
    elif pattern == "gray":
        bg = torch.full((3, height, width), 128.0, device=x.device)
    else:
        raise ValueError(f"Invalid background pattern: {pattern}")

    alpha = (x[:, 3:4] / 255.0).clamp(0.0, 1.0)
    return (alpha * rgb + (1.0 - alpha) * bg).clamp(0.0, 255.0)


def _k3_process_single_image(
    image: Union[torch.Tensor, Image.Image],
    config: dict,
    image_scale: torch.Tensor,
    image_bias: torch.Tensor,
    patch_size: int,
    transparent_bg_config: Union[dict, None],
) -> torch.Tensor:
    image = _k3_to_cuda_chw(image)

    new_h, new_w = config["new_height"], config["new_width"]
    padded_h = new_h + config["pad_height"]
    padded_w = new_w + config["pad_width"]

    x = _resize_bicubic_if_needed(image.unsqueeze(0), new_h, new_w)
    x = _fill_transparent_bg(x, transparent_bg_config)

    return normalize_and_patchify(
        x, image_scale, image_bias, patch_size, padded_h, padded_w
    ).squeeze(0)


class KimiK3GPUProcessorWrapper(KimiGPUProcessorWrapper):
    def __init__(self, *args, transparent_bg_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._transparent_bg_config = transparent_bg_config

    def _gpu_call(self, text, images, original_input_ids=None):
        input_text = text[0] if isinstance(text, list) else text

        resize_configs = []
        for image in images:
            w, h = _get_image_dimensions(image)
            resize_configs.append(
                navit_resize_config(
                    w,
                    h,
                    self._patch_size,
                    self._merge_kernel_size,
                    self._in_patch_limit,
                    self._patch_limit_on_one_side,
                    self._fixed_output_tokens,
                )
            )

        input_ids = self._prepare_input_ids(
            input_text, resize_configs, original_input_ids
        )

        image_scale, image_bias = self._get_gpu_norm_tensors()
        patches = []
        grids = []
        for image, config in zip(images, resize_configs):
            patches.append(
                _k3_process_single_image(
                    image,
                    config,
                    image_scale,
                    image_bias,
                    self._patch_size,
                    self._transparent_bg_config,
                )
            )
            grids.append(_grid_thw_from_resize_config(config, self._patch_size))

        pixel_values = torch.cat(patches, dim=0)
        grid_thws = torch.tensor(grids, dtype=torch.int64)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thws,
        }


class KimiK3ImageProcessor(KimiGridMMDataMixin, SGLangBaseProcessor):
    models = [KimiK3ForConditionalGeneration]
    gpu_image_decode = True
    prefer_tokenized_input = True
    precompute_hash_before_cpu_transfer = True
    auto_mm_processor_worker_num = 2
    auto_mm_io_worker_num = 16
    supports_mm_processor_concurrency = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        mm_tokens = MultimodalSpecialTokens(
            image_token="<|media_pad|>",
            image_token_id=hf_config.media_placeholder_token_id,
            image_token_regex=re.compile(r"(?:<\|media_pad\|>)+"),
        ).build(_processor)

        media_proc_cfg = _processor.media_processor.media_proc_cfg

        processor = KimiK3GPUProcessorWrapper(
            _processor,
            image_token=mm_tokens.image_token,
            image_token_id=mm_tokens.image_token_id,
            patch_size=media_proc_cfg["patch_size"],
            merge_kernel_size=media_proc_cfg["merge_kernel_size"],
            in_patch_limit=media_proc_cfg["in_patch_limit"],
            patch_limit_on_one_side=media_proc_cfg["patch_limit_on_one_side"],
            fixed_output_tokens=media_proc_cfg.get("fixed_output_tokens"),
            image_mean=media_proc_cfg["image_mean"],
            image_std=media_proc_cfg["image_std"],
            transparent_bg_config=media_proc_cfg.get("transparent_bg_config"),
        )
        super().__init__(hf_config, server_args, processor, *args, **kwargs)
        self.mm_tokens = mm_tokens

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        if getattr(request_obj, "video_data", None) or kwargs.get("audio_data"):
            raise ValueError("Kimi-K3 supports image input only")

        expected_image_count = len(image_data or [])
        if isinstance(input_text, (list, torch.Tensor)):
            input_ids = np.asarray(
                input_text.detach().flatten().cpu()
                if isinstance(input_text, torch.Tensor)
                else input_text,
                dtype=np.int64,
            )
            placeholder_count = int(
                np.count_nonzero(input_ids == self.mm_tokens.image_token_id)
            )
            if placeholder_count != expected_image_count:
                raise ValueError(
                    "Kimi-K3 image placeholders must map one-to-one to image data: "
                    f"expected {expected_image_count}, found {placeholder_count} token(s)"
                )

            # Keep structural media tokens distinct from user text that happens to
            # spell ``<|media_pad|>``. Decoding the whole prompt and matching the
            # resulting string would lose that distinction and could bind an image
            # to user-provided text instead of the renderer-inserted token.
            base_output = await self.fast_load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
                discard_alpha_channel=False,
            )

        if len(base_output.images) != expected_image_count:
            raise ValueError(
                "Kimi-K3 image placeholders must map one-to-one to image data: "
                f"expected {expected_image_count}, loaded {len(base_output.images)}"
            )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output,
            self.mm_tokens,
            sglang_original_input_ids=base_output.input_ids,
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        return self._build_kimi_mm_data_from_grids(
            prompt=prompt,
            embeddings=embeddings,
            image_token_id=self.mm_tokens.image_token_id,
            img_grid_thw=img_grid_thw,
        )
