# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 text context: stacked Qwen3-VL-4B hidden states.

The DiT context is ``hidden_states[k]`` for ``k`` in ``output_layers`` (eight
layers of width 2560) concatenated along channels (20480). Each prompt is
wrapped in the chat template, right-padded to the next multiple of
``pad_multiple`` tokens and encoded with an attention mask; the padded
positions stay in the context (the DiT attends to them unmasked, as in the
reference implementation).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn


def parse_weight_spec(spec: str) -> tuple[str, str | None, str | None]:
    """``repo_id[:subfolder_or_file][@revision]`` -> ``(repo_id, subpath, revision)``."""
    body, _, revision = spec.partition("@")
    repo_id, _, subpath = body.partition(":")
    if not repo_id:
        raise ValueError(f"invalid weight spec {spec!r}")
    return repo_id, subpath or None, revision or None


class Flux3TextEncoder(nn.Module):
    def __init__(
        self,
        spec: str,
        *,
        output_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32),
        pad_multiple: int = 80,
        max_length: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        hub_kwargs = {}
        if not os.path.exists(spec):
            spec, subfolder, revision = parse_weight_spec(spec)
            hub_kwargs = {"subfolder": subfolder or "", "revision": revision}
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            spec, torch_dtype=dtype, **hub_kwargs
        )
        # Text-only use of the multimodal backbone (keeps its M-RoPE position
        # handling for padded prompts); the LM head is dropped.
        self.model = model.model
        # hidden_states[k] is the input of layer k, so layers >= max(output_layers)
        # and the final norm never reach the context.
        language_model = self.model.language_model
        del language_model.layers[max(output_layers) :]
        language_model.norm = nn.Identity()
        self.embed_dtype = dtype
        self.processor = AutoProcessor.from_pretrained(spec, **hub_kwargs)
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError("the FLUX 3 text encoder needs a right-padding tokenizer")
        self.output_layers = tuple(output_layers)
        self.pad_multiple = pad_multiple
        self.max_length = max_length

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _bucket(self, prompt: str) -> int:
        length = self.processor.tokenizer(
            prompt, padding=False, truncation=True, max_length=self.max_length
        )["input_ids"]
        return min(
            math.ceil(len(length) / self.pad_multiple) * self.pad_multiple,
            self.max_length,
        )

    @torch.no_grad()
    def encode(self, texts: list[str]) -> list[torch.Tensor]:
        """Prompts -> contexts ``(1, L_i, len(output_layers) * hidden)``; one forward per length bucket."""
        formatted = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in texts
        ]
        buckets: dict[int, list[int]] = {}
        for i, prompt in enumerate(formatted):
            buckets.setdefault(self._bucket(prompt), []).append(i)
        results: list[torch.Tensor | None] = [None] * len(texts)
        tokenizer = self.processor.tokenizer
        for length, members in sorted(buckets.items()):
            toks = tokenizer(
                [formatted[i] for i in members],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=length,
                padding_side="right",
            )
            out = self.model(
                input_ids=toks["input_ids"].to(self.device),
                attention_mask=toks["attention_mask"].to(self.device),
                output_hidden_states=True,
                use_cache=False,
            )
            stacked = torch.cat(
                [out.hidden_states[k] for k in self.output_layers], dim=-1
            )
            stacked = stacked.to(self.embed_dtype)
            for j, i in enumerate(members):
                results[i] = stacked[j : j + 1]
        return results
