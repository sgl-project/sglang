# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""
Multimodal processor for pi0.5 VLA model.

This processor converts robot state into prompt text tokens:
    state -> normalize -> discretize -> serialize into prompt

Prompt format:
    Task: <task text>, State: <bin ids...>;
    Action:
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.pi05 import Pi05ForActionPrediction
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.common import load_image

logger = logging.getLogger(__name__)

PI05_IMAGE_SIZE = 224
PI05_NUM_IMAGE_TOKENS = 256
PI05_IMAGE_TOKEN_INDEX = 257152
PI05_MAX_CAMERAS = 3
PI05_MAX_TOKEN_LEN = 200
PI05_NUM_BINS = 256


def resize_with_pad(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D (B,C,H,W), got {images.ndim}-D")
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_width, cur_h / target_height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    ac = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=ac)
    resized = resized.clamp(-1.0, 1.0)
    ph0, rem_h = divmod(target_height - rh, 2)
    pw0, rem_w = divmod(target_width - rw, 2)
    return F.pad(resized, (pw0, pw0 + rem_w, ph0, ph0 + rem_h), value=-1.0)


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class Pi05ImageProcessor:
    def __init__(self, image_size: int = PI05_IMAGE_SIZE):
        self.image_size = image_size

    def preprocess_single(self, image: Image.Image) -> torch.Tensor:
        t = pil_image_to_tensor(image)
        if t.shape[2] != self.image_size or t.shape[3] != self.image_size:
            t = resize_with_pad(t, self.image_size, self.image_size)
        return t

    def make_empty_image(self) -> torch.Tensor:
        return torch.full((1, 3, self.image_size, self.image_size), -1.0)


def _get_tokenizer(processor_or_tokenizer):
    if isinstance(processor_or_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        return processor_or_tokenizer
    if hasattr(processor_or_tokenizer, "tokenizer"):
        return processor_or_tokenizer.tokenizer
    return processor_or_tokenizer


class Pi05Processor(BaseMultimodalProcessor):
    models = [Pi05ForActionPrediction]

    def __init__(self, hf_config, server_args, _processor, transport_mode):
        super().__init__(hf_config, server_args, _processor, transport_mode)

        self.image_size = getattr(hf_config, "image_size", PI05_IMAGE_SIZE)
        if hasattr(hf_config, "image_resolution"):
            res = hf_config.image_resolution
            if isinstance(res, (list, tuple)) and len(res) == 2:
                self.image_size = res[0]

        self.image_processor = Pi05ImageProcessor(image_size=self.image_size)
        self.image_token_id = getattr(hf_config, "image_token_index", PI05_IMAGE_TOKEN_INDEX)
        self.num_image_tokens = PI05_NUM_IMAGE_TOKENS
        self.max_cameras = PI05_MAX_CAMERAS
        self.max_token_len = getattr(
            hf_config,
            "tokenizer_max_length",
            getattr(hf_config, "max_token_len", PI05_MAX_TOKEN_LEN),
        )
        self.max_state_dim = getattr(hf_config, "max_state_dim", 32)

        self._tokenizer = _get_tokenizer(_processor)
        self.tokenizer = self._tokenizer
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=self.image_token_id,
        ).build(self)

        # Optional global normalization stats from config
        self.global_state_norm_stats = getattr(hf_config, "state_norm_stats", None)

    async def process_mm_data_async(
        self,
        image_data: Optional[List] = None,
        audio_data: Optional[List] = None,
        input_text: str = "",
        request_obj: Any = None,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        if image_data is None:
            image_data = []

        if isinstance(image_data, list) and image_data and isinstance(image_data[0], dict):
            return self._process_precomputed(image_data)

        loaded_images: List[Image.Image] = []
        for image_file in image_data:
            image, _ = load_image(image_file, gpu_image_decode=False)
            if image.mode != "RGB":
                image = image.convert("RGB")
            loaded_images.append(image)
        if len(loaded_images) > self.max_cameras:
            raise ValueError(
                f"pi0.5 supports at most {self.max_cameras} camera images per request; got {len(loaded_images)}."
            )

        pixel_values_list = [
            self.image_processor.preprocess_single(img) for img in loaded_images
        ]
        image_masks_list = [True] * len(pixel_values_list)

        for _ in range(max(0, self.max_cameras - len(pixel_values_list))):
            pixel_values_list.append(self.image_processor.make_empty_image())
            image_masks_list.append(False)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_masks = torch.tensor(image_masks_list, dtype=torch.bool)

        state = self._extract_state(request_obj)
        num_steps = self._extract_num_steps(request_obj)
        state_norm_stats = self._extract_state_norm_stats(request_obj)

        final_prompt = self._build_pi05_prompt(
            input_text=input_text,
            state=state,
            state_norm_stats=state_norm_stats,
        )

        num_cams = len(pixel_values_list)
        image_placeholder_ids = [self.image_token_id] * (self.num_image_tokens * num_cams)
        lang_token_ids, lang_attention_mask = self._tokenize_prompt(final_prompt)

        model_specific_data = {
            "image_masks": image_masks,
            "num_real_cameras": sum(image_masks_list),
            "lang_tokens": lang_token_ids,
            "lang_attention_mask": lang_attention_mask,
        }
        if num_steps is not None:
            model_specific_data["num_inference_steps"] = num_steps

        mm_items = [
            MultimodalDataItem(
                feature=pixel_values,
                modality=Modality.IMAGE,
                model_specific_data=model_specific_data,
                offsets=[(0, len(image_placeholder_ids) - 1)] if image_placeholder_ids else [],
            )
        ]

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=image_placeholder_ids + lang_token_ids,
            im_token_id=self.image_token_id,
        )

    def _tokenize_prompt(self, text: str) -> tuple[List[int], List[int]]:
        if not text or not text.strip():
            return [], []
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_token_len,
            padding=False,
            return_tensors=None,
        )
        input_ids = list(enc["input_ids"])
        attention_mask = list(enc.get("attention_mask", [1] * len(input_ids)))
        return input_ids, attention_mask

    def _extract_state(self, request_obj: Any) -> Optional[List[float]]:
        if request_obj is None:
            return None
        if hasattr(request_obj, "extra_body") and request_obj.extra_body:
            extra = request_obj.extra_body
            if isinstance(extra, dict):
                return extra.get("state")
            if hasattr(extra, "state"):
                return extra.state
        return None

    def _extract_num_steps(self, request_obj: Any) -> Optional[int]:
        if request_obj is None:
            return None
        if hasattr(request_obj, "extra_body") and request_obj.extra_body:
            extra = request_obj.extra_body
            v = (
                extra.get("num_inference_steps")
                if isinstance(extra, dict)
                else getattr(extra, "num_inference_steps", None)
            )
            return int(v) if v is not None else None
        return None

    def _extract_state_norm_stats(self, request_obj: Any) -> Optional[Dict[str, Any]]:
        if request_obj is None:
            return self.global_state_norm_stats
        if hasattr(request_obj, "extra_body") and request_obj.extra_body:
            extra = request_obj.extra_body
            if isinstance(extra, dict) and "state_norm_stats" in extra:
                return extra["state_norm_stats"]
            if hasattr(extra, "state_norm_stats"):
                return extra.state_norm_stats
        return self.global_state_norm_stats

    def _pad_or_truncate_state(self, state: np.ndarray) -> np.ndarray:
        if state.ndim != 1:
            raise ValueError(f"Expected 1-D state, got shape {state.shape}")

        if state.shape[0] < self.max_state_dim:
            pad_width = self.max_state_dim - state.shape[0]
            state = np.pad(state, (0, pad_width), mode="constant")
        elif state.shape[0] > self.max_state_dim:
            state = state[: self.max_state_dim]

        return state.astype(np.float32)

    def _normalize_state(
        self,
        state: np.ndarray,
        state_norm_stats: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Final runnable normalization logic.

        Supported stats formats:
        1. {"mode": "mean_std", "mean": [...], "std": [...]}
        2. {"mode": "min_max", "min": [...], "max": [...]}
        3. {"mode": "quantile", "low": [...], "high": [...]}
        4. {"mean": [...], "std": [...]}  # mode inferred as mean_std
        5. {"min": [...], "max": [...]}   # mode inferred as min_max
        6. {"low": [...], "high": [...]}  # mode inferred as quantile

        Output is always clipped to [-1, 1].

        If no stats are provided, fallback is direct clip(state, -1, 1).
        """
        if state_norm_stats is None:
            return np.clip(state, -1.0, 1.0)

        stats = state_norm_stats
        mode = stats.get("mode")

        if mode is None:
            if "mean" in stats and "std" in stats:
                mode = "mean_std"
            elif "min" in stats and "max" in stats:
                mode = "min_max"
            elif "q01" in stats and "q99" in stats:
                mode = "quantile"
            elif "low" in stats and "high" in stats:
                mode = "quantile"

        state = state.astype(np.float32)

        if mode == "mean_std":
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            mean = self._pad_or_truncate_state(mean)
            std = self._pad_or_truncate_state(std)
            std = np.where(np.abs(std) < 1e-6, 1.0, std)
            normed = (state - mean) / std
            return np.clip(normed, -1.0, 1.0)

        if mode == "min_max":
            vmin = np.asarray(stats["min"], dtype=np.float32)
            vmax = np.asarray(stats["max"], dtype=np.float32)
            vmin = self._pad_or_truncate_state(vmin)
            vmax = self._pad_or_truncate_state(vmax)
            denom = np.where(np.abs(vmax - vmin) < 1e-6, 1.0, vmax - vmin)
            normed = 2.0 * (state - vmin) / denom - 1.0
            return np.clip(normed, -1.0, 1.0)

        if mode == "quantile":
            low_key = "q01" if "q01" in stats else "low"
            high_key = "q99" if "q99" in stats else "high"
            low = np.asarray(stats[low_key], dtype=np.float32)
            high = np.asarray(stats[high_key], dtype=np.float32)
            low = self._pad_or_truncate_state(low)
            high = self._pad_or_truncate_state(high)
            denom = np.where(np.abs(high - low) < 1e-6, 1.0, high - low)
            normed = 2.0 * (state - low) / denom - 1.0
            return np.clip(normed, -1.0, 1.0)

        logger.warning("Unknown state_norm_stats mode=%s, fallback to clip[-1,1]", mode)
        return np.clip(state, -1.0, 1.0)

    def _discretize_state(self, state: np.ndarray) -> np.ndarray:
        bins = np.linspace(-1.0, 1.0, PI05_NUM_BINS + 1)[:-1]
        return np.digitize(state, bins=bins) - 1

    def _build_pi05_prompt(
        self,
        input_text: str,
        state: Optional[List[float]],
        state_norm_stats: Optional[Dict[str, Any]],
    ) -> str:
        cleaned_text = (input_text or "").strip().replace("_", " ").replace("\n", " ")

        if state is None:
            logger.warning("No state provided for pi0.5 request; using zero state.")
            state_arr = np.zeros((self.max_state_dim,), dtype=np.float32)
        else:
            state_arr = np.asarray(state, dtype=np.float32)
            state_arr = self._pad_or_truncate_state(state_arr)

        normed_state = self._normalize_state(state_arr, state_norm_stats)
        discretized_state = self._discretize_state(normed_state)
        state_str = " ".join(map(str, discretized_state.tolist()))

        return f"Task: {cleaned_text}, State: {state_str};\nAction: "

    def _process_precomputed(self, image_data: List[Dict]) -> MultimodalProcessorOutput:
        mm_items = []
        input_ids = None
        for item in image_data:
            fmt = item.get("format")
            if fmt not in ("processor_output", "precomputed_embedding"):
                continue
            feature = item["feature"] if "feature" in item else item.get("pixel_values")
            meta = item.get("model_specific_data", item)
            missing = [k for k in ("lang_tokens", "lang_attention_mask") if k not in meta]
            if missing:
                raise ValueError(f"pi0.5 precomputed input is missing required metadata fields: {missing}.")
            num_cams = int(feature.shape[0]) if hasattr(feature, "shape") and len(feature.shape) >= 1 else 0
            if input_ids is None:
                input_ids = item.get("input_ids")
                if input_ids is None:
                    input_ids = [self.image_token_id] * (self.num_image_tokens * num_cams) + list(meta["lang_tokens"])
            mm_items.append(
                MultimodalDataItem(
                    feature=feature,
                    modality=Modality.IMAGE,
                    model_specific_data=meta,
                    offsets=[(0, self.num_image_tokens * num_cams - 1)] if num_cams > 0 else [],
                )
            )
        return MultimodalProcessorOutput(mm_items=mm_items, input_ids=input_ids or [], im_token_id=self.image_token_id)

    def get_estimated_frames_list(self, image_data):
        return [1] * len(image_data) if image_data else []
