# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint-bound raw observation and action transforms for LingBot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.functional import resize
from transformers import AutoImageProcessor, AutoTokenizer

from sglang.multimodal_gen.configs.models.lingbot_vla_v2 import LingBotVLAV2Config


def validate_camera_image(
    image: Image.Image | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Check a camera frame and return its CHW view."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    try:
        value = torch.as_tensor(
            np.array(image, copy=True) if isinstance(image, np.ndarray) else image
        )
    except (TypeError, RuntimeError, ValueError, OverflowError) as exc:
        raise ValueError("A camera frame must contain numeric RGB pixels") from exc
    if value.ndim != 3 or value.numel() == 0:
        raise ValueError("A camera frame must be nonempty HWC or CHW RGB")
    if value.shape[-1] == 3 and not (torch.is_tensor(image) and value.shape[0] == 3):
        value = value.permute(2, 0, 1)
    if value.shape[0] != 3:
        raise ValueError("Camera frame must have three RGB channels")
    if value.dtype == torch.uint8:
        return value
    if not value.is_floating_point():
        raise ValueError("Camera frames must be uint8 or floating point in [0, 1]")
    value = value.float()
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("Camera frames must be finite and in [0, 1]")
    return value


def image_tensor(image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert one RGB camera frame to a CHW float tensor in [0, 1]."""
    value = validate_camera_image(image)
    if value.dtype == torch.uint8:
        value = value.float() / 255
    return value.cpu()


class LingBotVLAV2Processor:
    """Maps raw OpenPI observations to model inputs and canonical actions back to robot units."""

    def __init__(
        self,
        config: LingBotVLAV2Config,
        model_dir: str | Path,
        device: torch.device | str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)
        root = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            root / "tokenizer", padding_side="right", local_files_only=True
        )
        self.chat_tokenizer = AutoTokenizer.from_pretrained(
            root / "chat_template_tokenizer", local_files_only=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            root / "image_processor",
            min_pixels=config.image_min_pixels,
            max_pixels=config.image_max_pixels,
            local_files_only=True,
        )
        self.stats = {}
        self.eps = {}
        for filename, step_name, feature in (
            ("policy_preprocessor.json", "normalizer_processor", "observation.state"),
            ("policy_postprocessor.json", "unnormalizer_processor", "action"),
        ):
            data = json.loads((root / filename).read_text())
            step = next(s for s in data["steps"] if s["registry_name"] == step_name)
            self.eps[feature] = step["config"].get("eps", 1e-8)
            if step.get("state_file"):
                tensors = load_file(root / step["state_file"])
                self.stats[feature] = {
                    k.removeprefix(feature + "."): v
                    for k, v in tensors.items()
                    if k.startswith(feature + ".")
                }
        for key, kind, width in (
            ("observation.state", "STATE", config.state_dim),
            ("action", "ACTION", config.action_dim),
        ):
            mode = config.normalization_mapping[kind]
            needed = {
                "IDENTITY": (),
                "MEAN_STD": ("mean", "std"),
                "QUANTILES": ("q01", "q99"),
                "MIN_MAX": ("min", "max"),
            }
            if mode not in needed:
                raise ValueError(f"Unsupported normalization mode: {mode}")
            if config.inputs_normalized:
                continue
            for name in needed[mode]:
                value = self.stats.get(key, {}).get(name)
                if (
                    value is None
                    or tuple(value.shape) != (width,)
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        f"Missing or incompatible {key}.{name} statistics; normalized-space base use must be explicit"
                    )

    def normalize(
        self, value: torch.Tensor, key: str, inverse: bool = False
    ) -> torch.Tensor:
        """Apply (or invert) the checkpoint normalization for state or action values."""
        mode = self.config.normalization_mapping[
            "STATE" if key == "observation.state" else "ACTION"
        ]
        if self.config.inputs_normalized or mode == "IDENTITY":
            return value
        stats = {k: v.to(value.device, value.dtype) for k, v in self.stats[key].items()}
        eps = self.eps[key]
        if mode == "MEAN_STD":
            scale = stats["std"] + eps
            return (
                value * scale + stats["mean"]
                if inverse
                else (value - stats["mean"]) / scale
            )
        low, high = (
            (stats["q01"], stats["q99"])
            if mode == "QUANTILES"
            else (stats["min"], stats["max"])
        )
        scale = high - low
        scale = torch.where(scale == 0, torch.full_like(scale, eps), scale)
        return (
            (value + 1.0) * scale / 2.0 + low
            if inverse
            else 2.0 * (value - low) / scale - 1.0
        )

    def build_model_inputs(self, observation: dict) -> dict[str, torch.Tensor]:
        """Build one-observation model inputs; configured cameras that are absent are masked."""
        cfg = self.config
        obs = {cfg.image_key_map.get(k, k): v for k, v in observation.items()}
        state = obs.get("state", obs.get("observation.state"))
        if state is None:
            raise ValueError("LingBot observation requires state")
        state = torch.as_tensor(state, dtype=torch.float32)
        if tuple(state.shape) != (cfg.state_dim,) or not torch.isfinite(state).all():
            raise ValueError(
                f"Expected a finite state vector of length {cfg.state_dim}"
            )
        state = self.normalize(state, "observation.state")
        src, dst = cfg.slot_indices("observation.state")
        canonical = torch.zeros(1, cfg.max_state_dim)
        canonical[:, dst] = state[src]
        present = [key for key in cfg.camera_keys if key in obs]
        if not present:
            raise ValueError(
                f"Observation requires at least one configured camera: {cfg.camera_keys}"
            )
        frames = [
            resize(
                image_tensor(obs[key]),
                list(cfg.resize_imgs_with_padding),
                antialias=True,
            )
            * 255.0
            for key in present
        ]
        processed = self.image_processor(frames)
        pixels = torch.as_tensor(processed["pixel_values"]).reshape(
            len(present), -1, processed["pixel_values"].shape[-1]
        )
        grids = torch.as_tensor(processed["image_grid_thw"]).reshape(len(present), 3)
        lookup = {key: i for i, key in enumerate(present)}
        images, all_grids = [], []
        for key in cfg.camera_keys:
            i = lookup.get(key)
            images.append(
                pixels[i] if i is not None else torch.full_like(pixels[0], -1)
            )
            all_grids.append(grids[i if i is not None else 0])
        prompt = obs.get("prompt", obs.get("task"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("LingBot requires a nonempty prompt or task string")
        text = self.chat_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = self.tokenizer(
            text,
            max_length=cfg.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        values = {
            "images": torch.stack(images).unsqueeze(0),
            "img_masks": torch.tensor([[key in lookup for key in cfg.camera_keys]]),
            "lang_tokens": tokens["input_ids"],
            "lang_masks": tokens["attention_mask"].bool(),
            "state": canonical,
            "image_grid_thw": torch.stack(all_grids).unsqueeze(0),
        }
        return {k: v.to(self.device) for k, v in values.items()}

    def build_model_outputs(self, actions: torch.Tensor) -> np.ndarray:
        """Map a canonical action chunk back to calibrated robot actions."""
        cfg = self.config
        if tuple(actions.shape) != (1, cfg.chunk_size, cfg.max_action_dim):
            raise ValueError("LingBot returned an unexpected action chunk shape")
        src, dst = cfg.slot_indices("action")
        raw = actions.new_zeros(1, cfg.chunk_size, cfg.action_dim)
        raw[..., src] = actions[..., dst]
        raw = self.normalize(raw, "action", inverse=True)
        if not torch.isfinite(raw).all():
            raise ValueError("LingBot returned non-finite actions")
        return raw[0].detach().float().cpu().numpy().copy()
