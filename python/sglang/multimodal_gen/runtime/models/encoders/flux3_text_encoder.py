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

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    torch_default_sdpa_backends,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)


def parse_weight_spec(spec: str) -> tuple[str, str | None, str | None]:
    """``repo_id[:subfolder_or_file][@revision]`` -> ``(repo_id, subpath, revision)``."""
    body, _, revision = spec.partition("@")
    repo_id, _, subpath = body.partition(":")
    if not repo_id:
        raise ValueError(f"invalid weight spec {spec!r}")
    return repo_id, subpath or None, revision or None


def _quantize_linears_to_weight_only_fp8(module: nn.Module) -> None:
    """Store every ``nn.Linear`` as E4M3 + one scale per row, dequantized per call.

    The weights stay FP8-resident: the point is the smaller footprint.
    """
    from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
        FP8_WEIGHT_DTYPE,
        WeightOnlyFP8Linear,
    )

    fp8_max = torch.finfo(FP8_WEIGHT_DTYPE).max
    for name, child in list(module.named_children()):
        if not isinstance(child, nn.Linear):
            _quantize_linears_to_weight_only_fp8(child)
            continue
        weight = child.weight.detach().float()
        scale = (weight.abs().amax(dim=1) / fp8_max).clamp(min=1e-12)
        replacement = WeightOnlyFP8Linear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            compute_dtype=child.weight.dtype,
            enable_fused_w8a8=False,
        ).to(child.weight.device)
        replacement.weight.data.copy_(
            (weight / scale[:, None]).clamp(-fp8_max, fp8_max).to(FP8_WEIGHT_DTYPE)
        )
        replacement.weight_scale.data.copy_(scale)
        if child.bias is not None:
            replacement.bias.data.copy_(child.bias.detach())
        replacement._fp8_dequant_decided = True
        setattr(module, name, replacement)


class Flux3TextEncoder(nn.Module, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = ["model.language_model.layers"]

    def __init__(
        self,
        spec: str,
        *,
        output_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32),
        pad_multiple: int = 80,
        max_length: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
        tokenizer_spec: str | None = None,
        weight_only_fp8: bool = False,
        compile_language_model: bool = False,
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
        # handling for padded prompts); the LM head and vision tower are dropped.
        self.model = model.model
        del self.model.visual
        # hidden_states[k] is the input of layer k, so layers >= max(output_layers)
        # and the final norm never reach the context.
        language_model = self.model.language_model
        del language_model.layers[max(output_layers) :]
        language_model.norm = nn.Identity()
        self.embed_dtype = dtype
        if tokenizer_spec is None:
            self.processor = AutoProcessor.from_pretrained(spec, **hub_kwargs)
        else:
            self.processor = AutoProcessor.from_pretrained(tokenizer_spec)
        if self.processor.tokenizer.padding_side != "right":
            raise ValueError("the FLUX 3 text encoder needs a right-padding tokenizer")
        self.output_layers = tuple(output_layers)
        self.pad_multiple = pad_multiple
        self.max_length = max_length
        if weight_only_fp8:
            _quantize_linears_to_weight_only_fp8(self.model)
        elif compile_language_model:
            # One region over mask construction, rotary embedding and the
            # decoder loop; the length varies by pad bucket.
            language_model.forward = torch.compile(language_model.forward, dynamic=True)

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

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Prompts -> tokens and IDs, with one forward per unique length bucket.

        Matches the FLUX.3 image reference encoder: each prompt is padded to its
        own bucket with explicit position ids (real tokens
        count from 0, pads sit at 1), so the pad rows the DiT attends to do not
        depend on the batch or on the transformers default position ids.
        ``ids`` are ``(0, 0, 0, position)``.
        """
        unique = list(dict.fromkeys(texts))
        formatted = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in unique
        ]
        buckets = {}
        for index, prompt in enumerate(formatted):
            buckets.setdefault(self._bucket(prompt), []).append(index)
        results = {}
        tokenizer = self.processor.tokenizer
        for length, members in buckets.items():
            toks = tokenizer(
                [formatted[index] for index in members],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=length,
                padding_side="right",
            )
            mask = toks["attention_mask"].to(self.device, non_blocking=True)
            positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
            # Reference numerics: PyTorch's default SDPA backend choice.
            with torch_default_sdpa_backends():
                out = self.model(
                    input_ids=toks["input_ids"].to(self.device, non_blocking=True),
                    attention_mask=mask,
                    # Qwen3-VL M-RoPE: the text position plus three vision axes.
                    position_ids=positions[None].expand(4, -1, -1),
                    output_hidden_states=True,
                    use_cache=False,
                )
            tokens = torch.cat(
                [out.hidden_states[k] for k in self.output_layers], dim=-1
            ).to(self.embed_dtype)
            positions = torch.arange(length, device=tokens.device)
            zeros = torch.zeros_like(positions)
            ids = torch.stack([zeros, zeros, zeros, positions], -1)
            for row, index in enumerate(members):
                results[unique[index]] = (tokens[row], ids)
        return [results[text] for text in texts]
