# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.runtime.utils.vla_observation import (
    VLAObservationBatch,
    stable_tensor_sha256,
)


def _tensor_from_image(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError("Pi05 v1 expects one observation per request")
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims, got {tensor.shape}")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3]
        elif tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        else:
            raise ValueError(
                f"Could not infer image channels from shape {tensor.shape}"
            )
        tensor = tensor.to(dtype=torch.float32)
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        return tensor

    if isinstance(value, Image.Image):
        image = value.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    if isinstance(value, (np.ndarray, list)):
        arr = np.asarray(value)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image array, got shape {arr.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(arr))
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor[:3]
        elif tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        else:
            raise ValueError(
                f"Could not infer image channels from shape {tensor.shape}"
            )
        tensor = tensor.to(dtype=torch.float32)
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        return tensor

    raise TypeError(f"Unsupported Pi05 image type: {type(value)}")


def _resize_with_pad_image_tensor(
    tensor: torch.Tensor, size: tuple[int, int]
) -> torch.Tensor:
    height, width = size
    if tensor.shape[-2:] == (height, width):
        return tensor
    _, cur_height, cur_width = tensor.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    tensor = F.interpolate(
        tensor[None],
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    pad_h0, rem_h = divmod(height - resized_height, 2)
    pad_w0, rem_w = divmod(width - resized_width, 2)
    return F.pad(
        tensor,
        (pad_w0, pad_w0 + rem_w, pad_h0, pad_h0 + rem_h),
        mode="constant",
        value=0.0,
    )


class Pi05Preprocessor:
    def __init__(self, config: Pi05PipelineConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.padding_side = "right"

    def _tokenize(self, prompt: list[str], state: torch.Tensor | None):
        if state is None:
            state_for_prompt = torch.zeros(1, 0, dtype=torch.float32)
        else:
            state_for_prompt = state.detach().cpu().to(torch.float32)
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        state_np = state_for_prompt.numpy()
        discretized = np.digitize(state_np, bins=bins) - 1
        full_prompts = []
        for idx, task in enumerate(prompt):
            cleaned = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized[idx]))
            full_prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")

        encoded = self.tokenizer(
            full_prompts,
            max_length=self.config.max_token_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(torch.long), encoded["attention_mask"].to(
            torch.bool
        )

    def __call__(self, raw_observation: dict[str, Any]) -> VLAObservationBatch:
        prompt_value = raw_observation.get("prompt", "")
        if isinstance(prompt_value, list):
            prompt = [str(x) for x in prompt_value]
        else:
            prompt = [str(prompt_value)]
        if len(prompt) != 1:
            raise ValueError("Pi05 v1 expects one prompt per action request")

        raw_images = raw_observation.get("images") or {}
        image_masks_in = raw_observation.get("image_masks") or {}
        camera_order = tuple(
            raw_observation.get("camera_order") or self.config.image_keys
        )

        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}
        image_hashes: dict[str, str] = {}
        for key in camera_order:
            value = raw_images.get(key)
            is_present = value is not None and bool(image_masks_in.get(key, True))
            if is_present:
                tensor = _tensor_from_image(value)
                tensor = _resize_with_pad_image_tensor(tensor, self.config.image_size)
                tensor = tensor * 2.0 - 1.0
            else:
                channels = 3
                height, width = self.config.image_size
                tensor = torch.ones(channels, height, width, dtype=torch.float32) * -1.0

            images[key] = tensor.unsqueeze(0)
            image_masks[key] = torch.tensor([is_present], dtype=torch.bool)
            image_hashes[key] = stable_tensor_sha256(tensor)

        state = raw_observation.get("state")
        state_tensor = None
        if state is not None:
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if state_tensor.shape[0] != 1:
                raise ValueError("Pi05 v1 expects one state vector per request")
            if state_tensor.shape[-1] > self.config.state_dim:
                raise ValueError(
                    f"Pi05 state dim must be <= {self.config.state_dim}, "
                    f"got {state_tensor.shape[-1]}"
                )

        noise = raw_observation.get("noise")
        noise_tensor = None
        if noise is not None:
            noise_tensor = torch.as_tensor(noise, dtype=torch.float32)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor.unsqueeze(0)
            expected = (1, self.config.action_horizon, self.config.action_dim)
            if tuple(noise_tensor.shape) != expected:
                raise ValueError(
                    f"Pi05 noise must have shape {expected}, "
                    f"got {tuple(noise_tensor.shape)}"
                )

        tokens = raw_observation.get("tokens")
        if tokens is None:
            tokens = raw_observation.get("tokenized_prompt")
        token_masks = raw_observation.get("token_masks")
        if token_masks is None:
            token_masks = raw_observation.get("tokenized_prompt_mask")
        if tokens is not None:
            tokens_tensor = torch.as_tensor(tokens, dtype=torch.long)
            if tokens_tensor.ndim == 1:
                tokens_tensor = tokens_tensor.unsqueeze(0)
            if token_masks is None:
                token_masks_tensor = tokens_tensor != self.tokenizer.pad_token_id
            else:
                token_masks_tensor = torch.as_tensor(token_masks, dtype=torch.bool)
                if token_masks_tensor.ndim == 1:
                    token_masks_tensor = token_masks_tensor.unsqueeze(0)
        else:
            tokens_tensor, token_masks_tensor = self._tokenize(prompt, state_tensor)

        return VLAObservationBatch(
            prompt=prompt,
            images=images,
            image_masks=image_masks,
            state=state_tensor,
            noise=noise_tensor,
            tokens=tokens_tensor,
            token_masks=token_masks_tensor,
            batch_size=1,
            metadata={
                "camera_order": camera_order,
                "image_hashes": image_hashes,
            },
        )
