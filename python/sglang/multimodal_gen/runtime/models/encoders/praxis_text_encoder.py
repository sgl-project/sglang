# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-2B as the frozen text encoder of Praxis-1.

The prompt gets no chat template. The tokens pad to a multiple of 32, the encoder returns the output of
decoder layer 21 of 24 (``hidden_states[-3]``, before the final norm) and the padding is trimmed again.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from sglang.multimodal_gen.configs.models.encoders.qwen3_5 import Qwen3_5EncoderConfig
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.qwen3_5 import Qwen3_5Model

HIDDEN_LAYER = -3
PAD_TO_MULTIPLE_OF = 32
_LANGUAGE_PREFIX = "model.language_model."


def _language_weights(path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """The text tower of the checkpoint under the names of ``Qwen3_5Model``; vision and MTP weights are skipped."""
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), "pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(_LANGUAGE_PREFIX):
                    yield key.removeprefix("model."), handle.get_tensor(key)


class PraxisTextEncoder:
    """Encodes a prompt to ``(1, S, 2048)`` states; the result of a prompt is cached."""

    def __init__(
        self,
        path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        path = Path(path)
        config = Qwen3_5EncoderConfig()
        config.update_model_arch(json.loads((path / "config.json").read_text()))
        config.post_diffusers_config_update()
        with set_default_torch_dtype(dtype), torch.device(device):
            self.model = Qwen3_5Model(config).eval().requires_grad_(False)
        loaded = self.model.load_weights(_language_weights(path))
        if loaded != set(dict(self.model.named_parameters())):
            raise ValueError(f"text encoder weights incomplete in {path}")
        self.model.to(device)
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True
        )
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        """States of the prompt tokens ``(1, S, 2048)``; a blank prompt is one space."""
        text = prompt if prompt.strip() else " "
        if text not in self._cache:
            self._cache[text] = self._encode(text)
        return self._cache[text]

    def _encode(self, text: str) -> torch.Tensor:
        batch = self._tokenizer(
            text=[text],
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
            return_tensors="pt",
        ).to(self.device)
        mask = batch["attention_mask"]
        with set_forward_context(0, None):
            out = self.model(
                batch["input_ids"], attention_mask=mask, output_hidden_states=True
            )
        return out.hidden_states[HIDDEN_LAYER][:, : int(mask.sum())]
