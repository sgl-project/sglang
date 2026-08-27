# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""sglang multimodal processor for MiniCPM-V 4.6.

Ports per-image preprocessing + chat-template expansion sglang-side because
no working HF ``MiniCPMV4_6Processor`` is reachable yet: transformers main
does not ship one until 5.7+, and the released 4.6 checkpoints ship only a
tokenizer (no remote-code processor), so ``AutoProcessor.from_pretrained``
falls through to a bare tokenizer. Once a real processor is loadable, this
module collapses to a thin wrapper that delegates to it.
"""

from __future__ import annotations

import math
from itertools import chain
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as F
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.minicpmv import MiniCPMV4_6ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

IMAGENET_STANDARD_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STANDARD_STD = (0.5, 0.5, 0.5)

# Inner per-feature pad sentinel: prevents the next per-image
# ``replace(image_token, ...)`` from clobbering a previous expansion's inner
# pads. Swapped back to the real pad token once per modality after splicing.
_PAD_PLACEHOLDER = "<|placeholder|>"


def _ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def _to_chw_tensor(image) -> torch.Tensor:
    """PIL / torch / numpy -> ``(C, H, W)`` float32 in ``[0, 255]``.

    Image inputs from ``load_mm_data`` are PIL; video frames from sglang's
    video decoder come back as numpy arrays.
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() != 3:
            raise ValueError(f"expected 3-D image tensor, got {image.shape}")
        if image.shape[0] not in (1, 3, 4):
            image = image.permute(2, 0, 1).contiguous()
        if image.shape[0] == 4:
            image = image[:3]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image.float()

    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return F.pil_to_tensor(image).float()

    import numpy as np

    if isinstance(image, np.ndarray):
        t = torch.from_numpy(image)
        if t.dim() == 3 and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1).contiguous()
        if t.shape[0] == 4:
            t = t[:3]
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t.float()

    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.resize(
        image,
        size=[height, width],
        interpolation=F.InterpolationMode.BICUBIC,
        antialias=True,
    )


def _divide_to_patches(
    image: torch.Tensor, patch_h: int, patch_w: int
) -> List[torch.Tensor]:
    _, H, W = image.shape
    if H % patch_h != 0 or W % patch_w != 0:
        raise ValueError(f"image ({H}, {W}) not divisible by ({patch_h}, {patch_w})")
    rows = H // patch_h
    cols = W // patch_w
    patches: List[torch.Tensor] = []
    for r in range(rows):
        for c in range(cols):
            patches.append(
                image[
                    :, r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w
                ]
            )
    return patches


def _reshape_by_patch(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``(C, H, W) -> (C, P, H*W/P)`` NaViT packing."""
    C = image.shape[0]
    patches = torch.nn.functional.unfold(
        image.unsqueeze(0), (patch_size, patch_size), stride=(patch_size, patch_size)
    )
    patches = patches.reshape(C, patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(C, patch_size, -1)
    return patches


def _flatten_patches(
    per_item_pv: List[List[torch.Tensor]],
    per_item_ts: List[List[List[int]]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-item per-patch -> flat per-patch (source first, slices row-major)."""
    flat_pv = list(chain.from_iterable(per_item_pv))
    flat_ts = [
        torch.tensor(ts, dtype=torch.int32) for ts in chain.from_iterable(per_item_ts)
    ]
    return flat_pv, flat_ts


class MiniCPMV4_6ImageProcessor:
    """Per-image preprocessing.

    Pipeline: pick a slice grid (rows x cols, up to ``max_slice_nums``); resize
    source and (optionally) tiles to multiples of ``patch_size * 4`` (factor 4
    = the two successive 2x2 spatial merges: mid-ViT merger + DownsampleMLP);
    rescale, normalize, and NaViT-pack each tile into ``(C, P, H*W/P)``.
    """

    def __init__(
        self,
        max_slice_nums: int = 9,
        scale_resolution: int = 448,
        patch_size: int = 14,
        slice_mode: bool = True,
        downsample_mode: str = "16x",
        use_image_id: bool = True,
        image_mean: Sequence[float] = IMAGENET_STANDARD_MEAN,
        image_std: Sequence[float] = IMAGENET_STANDARD_STD,
        rescale_factor: float = 1.0 / 255.0,
    ) -> None:
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.slice_mode = slice_mode
        self.downsample_mode = downsample_mode
        self.use_image_id = use_image_id
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.rescale_factor = rescale_factor

    def _find_best_resize(
        self,
        image_size: Tuple[int, int],
        allow_upscale: bool = False,
    ) -> Tuple[int, int]:
        height, width = image_size
        scale = self.scale_resolution
        # factor 4 = two successive 2x2 spatial merges (mid-ViT + DownsampleMLP)
        divisor = self.patch_size * 4
        if (height * width > scale * scale) or allow_upscale:
            aspect_ratio = width / height
            height = int(scale / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        best_w = _ensure_divide(width, divisor)
        best_h = _ensure_divide(height, divisor)
        return best_h, best_w

    def _get_refine_size(
        self,
        image_size: Tuple[int, int],
        grid: Tuple[int, int],
        allow_upscale: bool = False,
    ) -> Tuple[int, int]:
        height, width = image_size
        grid_y, grid_x = grid
        refine_w = _ensure_divide(width, grid_x)
        refine_h = _ensure_divide(height, grid_y)
        bh, bw = self._find_best_resize(
            (refine_h // grid_y, refine_w // grid_x),
            allow_upscale=allow_upscale,
        )
        return bh * grid_y, bw * grid_x

    def _get_sliced_grid(
        self, image_size: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        original_h, original_w = image_size
        scale = self.scale_resolution
        log_ratio = math.log(original_w / original_h)
        ratio = original_w * original_h / (scale * scale)
        multiple = min(math.ceil(ratio), self.max_slice_nums)
        if multiple <= 1:
            return None

        best_grid = (1, 1)
        min_error = float("inf")
        for num_slices in (multiple - 1, multiple, multiple + 1):
            if num_slices == 1 or num_slices > self.max_slice_nums:
                continue
            for num_rows in range(1, num_slices + 1):
                if num_slices % num_rows != 0:
                    continue
                num_cols = num_slices // num_rows
                error = abs(log_ratio - math.log(num_rows / num_cols))
                if error < min_error:
                    # Ref returns ``[cols, rows]``; preserve the convention so
                    # downstream code matches HF.
                    best_grid = (num_cols, num_rows)
                    min_error = error
        return best_grid

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        t = t * self.rescale_factor
        return (t - self.image_mean.to(t.dtype)) / self.image_std.to(t.dtype)

    def __call__(self, images: List) -> dict:
        return self.preprocess(images)

    def preprocess(self, images: List) -> dict:
        """Returns ``{pixel_values, tgt_sizes, grids, num_patches_per_image}``.

        Per image, ``pixel_values[i]`` is a list whose first entry is the
        source patch and remaining entries are slice tiles in row-major grid
        order. ``grids[i]`` is ``[cols, rows]`` (zeros if no slicing).
        """
        per_image_pv: List[List[torch.Tensor]] = []
        per_image_ts: List[List[List[int]]] = []
        all_grids: List[List[int]] = []
        num_patches_per_image: List[int] = []

        for image in images:
            chw = _to_chw_tensor(image)
            H0, W0 = chw.shape[-2], chw.shape[-1]
            best_grid = self._get_sliced_grid((H0, W0)) if self.slice_mode else None

            allow_upscale_src = best_grid is None
            src_h, src_w = self._find_best_resize(
                (H0, W0), allow_upscale=allow_upscale_src
            )
            source = _resize(chw, src_h, src_w)

            patches: List[torch.Tensor] = [source]
            patch_h = patch_w = 0
            if best_grid is not None:
                refine_h, refine_w = self._get_refine_size(
                    (H0, W0), best_grid, allow_upscale=True
                )
                refined = _resize(chw, refine_h, refine_w)
                grid_y, grid_x = best_grid
                patch_h = refine_h // grid_y
                patch_w = refine_w // grid_x
                patches.extend(_divide_to_patches(refined, patch_h, patch_w))

            patches = [self._normalize(p) for p in patches]

            pv = [_reshape_by_patch(patches[0], self.patch_size)]
            ts = [[src_h // self.patch_size, src_w // self.patch_size]]
            for p in patches[1:]:
                pv.append(_reshape_by_patch(p, self.patch_size))
                ts.append([patch_h // self.patch_size, patch_w // self.patch_size])

            per_image_pv.append(pv)
            per_image_ts.append(ts)
            all_grids.append(list(best_grid) if best_grid is not None else [0, 0])
            num_patches_per_image.append(len(pv))

        return {
            "pixel_values": per_image_pv,
            "tgt_sizes": per_image_ts,
            "grids": all_grids,
            "num_patches_per_image": num_patches_per_image,
        }


class MiniCPMV4_6MultimodalProcessor(BaseMultimodalProcessor):
    """4.6-only mm processor.

    The legacy ``MiniCPMMultimodalProcessor`` stays for 2.6/4.0/4.5 because its
    ``_processor.tokenizer`` shape and ``(<image>./</image>)`` placeholder
    format don't fit 4.6.
    """

    models = [MiniCPMV4_6ForConditionalGeneration]
    support_dynamic_frame_expansion = False
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # ``_processor`` is either the bare tokenizer (current state — no
        # ``MiniCPMV4_6Processor`` shipped) or a real processor whose
        # ``.tokenizer`` exposes the same.
        self.tokenizer = getattr(_processor, "tokenizer", _processor)

        vision_cfg = getattr(hf_config, "vision_config", None)
        patch_size = (
            getattr(vision_cfg, "patch_size", 14) if vision_cfg is not None else 14
        )
        downsample_mode = getattr(hf_config, "downsample_mode", "16x")
        # Per-image preprocessor; reused for video frames (HF ref's
        # video slicing geometry matches image slicing exactly).
        self.image_processor = MiniCPMV4_6ImageProcessor(
            max_slice_nums=9,
            scale_resolution=448,
            patch_size=patch_size,
            slice_mode=True,
            downsample_mode=downsample_mode,
            use_image_id=True,
        )

        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.image_token_id = getattr(hf_config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = self._token_id(self.image_token)
        self.video_token_id = getattr(hf_config, "video_token_id", None)
        if self.video_token_id is None:
            self.video_token_id = self._token_id(self.video_token)

        # ``<image>``/``<slice>`` wrap the expanded regions for both images and
        # video frames; only the inner per-feature pad token differs.
        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
        self.slice_start_token = "<slice>"
        self.slice_end_token = "</slice>"
        self.image_id_start_token = "<image_id>"
        self.image_id_end_token = "</image_id>"

        self.image_start_id = self._token_id(self.image_start_token)
        self.image_end_id = self._token_id(self.image_end_token)
        self.slice_start_id = self._token_id(self.slice_start_token)
        self.slice_end_id = self._token_id(self.slice_end_token)

        self.pad_divisor = 16 if downsample_mode != "4x" else 4

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.image_token_id,
            video_token=self.video_token,
            video_token_id=self.video_token_id,
        ).build(_processor)

    def _token_id(self, token: str):
        try:
            ids = self.tokenizer.convert_tokens_to_ids([token])
            if ids and ids[0] is not None:
                return int(ids[0])
        except Exception:
            pass
        return None

    def _expand_frame(
        self,
        tgt_sizes: List[List[int]],
        grid: List[int],
    ) -> str:
        """``<image>...</image>`` (+ optional ``<slice>...</slice>`` rows) for
        one image or video frame; inner pads are ``_PAD_PLACEHOLDER`` (caller
        swaps back after splicing).
        """
        h0, w0 = tgt_sizes[0]
        n_src = (h0 * w0) // self.pad_divisor
        out = self.image_start_token + _PAD_PLACEHOLDER * n_src + self.image_end_token

        if len(tgt_sizes) > 1 and grid and grid[0] > 0 and grid[1] > 0:
            grid_y, grid_x = int(grid[0]), int(grid[1])
            h_s, w_s = tgt_sizes[1]
            n_slice = (h_s * w_s) // self.pad_divisor
            slice_chunk = (
                self.slice_start_token
                + _PAD_PLACEHOLDER * n_slice
                + self.slice_end_token
            )
            row_chunks = [slice_chunk * grid_x for _ in range(grid_y)]
            out += "\n".join(row_chunks)
        return out

    def _expand_media(
        self,
        index: int,
        frames: Sequence[Tuple[List[List[int]], List[int]]],
    ) -> str:
        """One image or one video. Image is a single-frame video."""
        body = "".join(self._expand_frame(ts, grid) for ts, grid in frames)
        return f"{self.image_id_start_token}{index}{self.image_id_end_token}" + body

    async def process_mm_data_async(
        self,
        image_data: Sequence[Union[str, bytes]],
        audio_data: Sequence[Union[str, bytes]],
        input_text,
        request_obj,
        **kwargs: Any,
    ):
        # ``TokenizerManager`` does not pass ``video_data`` through the
        # processor signature; read it off the request the way qwen_vl does.
        video_data = getattr(request_obj, "video_data", None) or kwargs.get(
            "video_data"
        )
        base = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            image_data=image_data,
            video_data=video_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base is None:
            return None

        prompt: str = base.input_text or ""
        images = base.images or []
        videos = base.videos or []

        # Image: one "frame" per image. Video: per-frame nesting kept so each
        # frame becomes its own ``<image>...</image>`` block in the expansion.
        img_per_pv, img_per_ts, img_grids = self._preprocess_images(images)
        vid_per_pv, vid_per_ts, vid_grids = self._preprocess_videos(videos)

        prompt = self._splice_expansions(
            prompt,
            (
                self._expand_media(i, [(ts, gd)])
                for i, (ts, gd) in enumerate(zip(img_per_ts, img_grids))
            ),
            (
                self._expand_media(i, list(zip(fts, fgd)))
                for i, (fts, fgd) in enumerate(zip(vid_per_ts, vid_grids))
            ),
        )

        input_ids: List[int] = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)

        # Each patch's pad tokens are guaranteed contiguous (the expansion
        # functions wrap them in ``<image>...</image>`` / ``<slice>...</slice>``
        # with nothing else in between), so a per-token-id contiguous-run scan
        # — base's ``get_mm_items_offset`` — gives one (start, end) per patch.
        mm_items: List[MultimodalDataItem] = []
        mm_items.extend(
            self._build_items(
                input_ids_tensor,
                self.image_token_id,
                _flatten_patches(img_per_pv, img_per_ts),
                Modality.IMAGE,
            )
        )
        # Video: extra ``per-frame -> per-patch`` nesting; pre-flatten one
        # level so ``_flatten_patches`` sees the same shape as image.
        vid_pv_flat = [list(chain.from_iterable(v)) for v in vid_per_pv]
        vid_ts_flat = [list(chain.from_iterable(v)) for v in vid_per_ts]
        mm_items.extend(
            self._build_items(
                input_ids_tensor,
                self.video_token_id,
                _flatten_patches(vid_pv_flat, vid_ts_flat),
                Modality.VIDEO,
            )
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids,
            im_token_id=self.image_token_id,
            im_start_id=self.image_start_id,
            im_end_id=self.image_end_id,
            slice_start_id=self.slice_start_id,
            slice_end_id=self.slice_end_id,
        )

    def _preprocess_images(self, images):
        if not images:
            return [], [], []
        out = self.image_processor.preprocess(images)
        return out["pixel_values"], out["tgt_sizes"], out["grids"]

    def _preprocess_videos(self, videos):
        per_video_pv: List[List[List[torch.Tensor]]] = []
        per_video_ts: List[List[List[List[int]]]] = []
        per_video_grids: List[List[List[int]]] = []
        for frames in videos:
            out = self.image_processor.preprocess(list(frames))
            per_video_pv.append(out["pixel_values"])
            per_video_ts.append(out["tgt_sizes"])
            per_video_grids.append(out["grids"])
        return per_video_pv, per_video_ts, per_video_grids

    def _splice_expansions(self, prompt, image_expansions, video_expansions):
        # The chat template emits exactly one marker per media item; a
        # sequential ``replace(..., n=1)`` walk lines them up by left-to-right
        # order. Expansions carry ``_PAD_PLACEHOLDER`` for inner pads so the
        # next replace doesn't trip on a previous expansion's pads — we swap
        # placeholders back to the real pad token in one pass per modality.
        for token, expansions in (
            (self.image_token, image_expansions),
            (self.video_token, video_expansions),
        ):
            for expansion in expansions:
                if token not in prompt:
                    break
                prompt = prompt.replace(token, expansion, 1)
            prompt = prompt.replace(_PAD_PLACEHOLDER, token)
        return prompt

    def _build_items(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int,
        flat: Tuple[List[torch.Tensor], List[torch.Tensor]],
        modality: Modality,
    ) -> List[MultimodalDataItem]:
        flat_pv, flat_ts = flat
        runs = self.get_mm_items_offset(input_ids, pad_token_id)
        if len(runs) != len(flat_pv):
            raise RuntimeError(
                f"[minicpmv4_6] {modality} pad run / feature count mismatch: "
                f"{len(runs)} runs vs {len(flat_pv)} patches"
            )
        return [
            MultimodalDataItem(
                feature=[pv],
                offsets=[run],
                model_specific_data={"tgt_size": [ts]},
                modality=modality,
            )
            for run, pv, ts in zip(runs, flat_pv, flat_ts)
        ]
