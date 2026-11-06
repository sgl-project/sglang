# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ActiveVLN actor adapter with native SRT weight export."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ActiveVLNModel(nn.Module):
    """Train selected language layers; keep vision frozen."""

    def __init__(self, qwen, *, train_last_n_layers=-1, gradient_checkpointing=False):
        super().__init__()
        self.qwen = qwen
        self.gradient_checkpointing = gradient_checkpointing
        self.config = qwen.config
        self.text_config = getattr(self.config, "text_config", self.config)
        if self.config.model_type != "qwen2_5_vl":
            raise ValueError("ActiveVLN requires a Qwen2.5-VL checkpoint")
        if getattr(self.text_config, "use_sliding_window", False):
            raise ValueError("ActiveVLN training requires full attention")
        count = len(self.text.layers)
        if train_last_n_layers != -1 and not 1 <= train_last_n_layers <= count:
            raise ValueError(f"train_last_n_layers must be -1 or in [1, {count}]")
        self.train_full_text = train_last_n_layers == -1
        qwen.requires_grad_(False)
        if train_last_n_layers == -1:
            self.text.requires_grad_(True)
            qwen.lm_head.requires_grad_(True)
        else:
            self.text.layers[-train_last_n_layers:].requires_grad_(True)
            self.text.norm.requires_grad_(True)

    @property
    def text(self):
        return getattr(self.qwen.model, "language_model", self.qwen.model)

    @property
    def visual(self):
        return (
            self.qwen.visual if hasattr(self.qwen, "visual") else self.qwen.model.visual
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        *,
        dtype=torch.bfloat16,
        train_last_n_layers=-1,
        gradient_checkpointing=False,
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration

        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
        return cls(
            qwen,
            train_last_n_layers=train_last_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )

    @torch.no_grad()
    def encode_images(self, pixel_values, image_grid_thw):
        self.visual.eval()
        features = self.visual(
            pixel_values.to(next(self.visual.parameters()).dtype),
            grid_thw=image_grid_thw,
        )
        features = getattr(features, "pooler_output", features)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.detach()

    def embed(self, input_ids, image_features=None, image_token_mask=None):
        # Keep frozen embeddings outside autograd.
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_full_text):
            hidden = self.text.embed_tokens(input_ids)
        mask = (
            input_ids == self.config.image_token_id
            if image_token_mask is None
            else image_token_mask
        )
        if image_features is not None:
            if int(mask.sum()) != image_features.shape[0]:
                raise ValueError(
                    "Image feature count must equal the number of image tokens"
                )
            hidden = hidden.masked_scatter(
                mask.unsqueeze(-1).expand_as(hidden), image_features.to(hidden)
            )
        elif mask.any():
            raise ValueError("Image tokens require image_features")
        return hidden

    def project(self, hidden):
        weight = self.qwen.lm_head.weight
        return F.linear(hidden, weight if self.train_full_text else weight.detach())

    def forward(
        self,
        input_ids,
        position_ids,
        *,
        image_features=None,
        image_token_mask=None,
        attention_mask=None,
    ):
        """Return normalized hidden states for a complete teacher-forced sequence."""
        if position_ids.shape != (3, *input_ids.shape):
            raise ValueError("position_ids must have shape [3, batch, sequence]")
        hidden = self.embed(input_ids, image_features, image_token_mask)

        def text_forward(embeddings):
            return self.text(
                inputs_embeds=embeddings,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(text_forward, hidden, use_reentrant=False)
        return text_forward(hidden)

    def response_logprobs(
        self,
        input_ids,
        position_ids,
        response_start,
        *,
        image_features=None,
        compute_entropy=False,
        temperature=1.0,
    ):
        """Score response tokens including EOS, excluding all prompt/history tokens."""
        if not 0 < response_start < input_ids.shape[1]:
            raise ValueError(
                "response_start must select a nonempty response after a prompt"
            )
        image_mask = (input_ids == self.config.image_token_id) & (
            torch.arange(input_ids.shape[1], device=input_ids.device)[None]
            < response_start
        )
        hidden = self(
            input_ids,
            position_ids,
            image_features=image_features,
            image_token_mask=image_mask,
        )
        if temperature <= 0:
            raise ValueError("Training temperature must be positive")
        logits = self.project(hidden[:, response_start - 1 : -1]).float() / temperature
        logs = logits.log_softmax(-1)
        response = input_ids[:, response_start:]
        logprobs = logs.gather(-1, response.unsqueeze(-1)).squeeze(-1)
        entropy = -(logs.exp() * logs).sum(-1) if compute_entropy else None
        return logprobs, entropy

    def named_srt_weights(self, *, trainable_only=False):
        """Export checkpoint names; SRT handles QKV and gate/up fusion."""
        for name, parameter in self.qwen.named_parameters(remove_duplicate=False):
            if trainable_only and not parameter.requires_grad:
                continue
            if name.startswith("model.language_model."):
                name = "model." + name.removeprefix("model.language_model.")
            elif name.startswith("model.visual."):
                name = name.removeprefix("model.")
            yield name, parameter.detach()
