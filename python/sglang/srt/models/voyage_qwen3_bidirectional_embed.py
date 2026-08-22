# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

WeightItem = Tuple[str, torch.Tensor]

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# Configuration for weight fusion: maps HF weight suffixes to fusion groups
_FUSION_CONFIG = {
    # QKV fusion: q_proj, k_proj, v_proj -> qkv_proj
    "qkv": {
        "suffixes": {
            "self_attn.q_proj.weight": "q",
            "self_attn.k_proj.weight": "k",
            "self_attn.v_proj.weight": "v",
        },
        "parts_order": ("q", "k", "v"),
        "output_template": "layers.{}.self_attn.qkv_proj.weight",
    },
    # MLP fusion: gate_proj, up_proj -> gate_up_proj
    "mlp": {
        "suffixes": {
            "mlp.gate_proj.weight": "gate",
            "mlp.up_proj.weight": "up",
        },
        "parts_order": ("gate", "up"),
        "output_template": "layers.{}.mlp.gate_up_proj.weight",
    },
}


class VoyageQwen3BidirectionalEmbedModel(nn.Module):
    """
    Qwen3Model + Voyage embedding head + bidirectional attention.

    Checkpoint conventions (HF):
      - MLP: gate_proj + up_proj (unfused)
      - Attn: q_proj + k_proj + v_proj (unfused)
      - Linear head: linear.weight
      - Weights prefixed with "model." (e.g., model.layers.0...)

    SGLang Qwen3Model expects:
      - mlp.gate_up_proj (fused)
      - self_attn.qkv_proj (fused)
      - No "model." prefix

    We remap/fuse weights and load directly (bypassing parent's stacked_params_mapping
    which would cause double-transformation like qkv_proj -> qkqkv_proj).
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Enable bidirectional (encoder-only) attention
        config.is_causal = False

        # Base Qwen3 model
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # Embedding head (hidden_size -> num_labels, bias=False)
        self.linear = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=False,
        )

    def _mean_pooling(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Apply mean pooling over sequence tokens for each request."""
        seq_lens = forward_batch.extend_seq_lens
        pooled_outputs = []
        offset = 0
        for seq_len in seq_lens:
            # Get hidden states for this sequence
            seq_hidden = hidden_states[offset : offset + seq_len]
            # Mean pool over sequence length
            pooled = seq_hidden.mean(dim=0)
            pooled_outputs.append(pooled)
            offset += seq_len
        return torch.stack(pooled_outputs)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        """Forward pass for embedding generation."""
        assert (
            get_embedding
        ), "VoyageQwen3BidirectionalEmbedModel is only used for embedding"
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        # Apply the linear head to all hidden states
        hidden_states = self.linear(hidden_states)
        # Apply MEAN pooling over sequence tokens
        pooled_data = self._mean_pooling(hidden_states, forward_batch)
        # Normalize embeddings
        pooled_data = nn.functional.normalize(pooled_data, p=2, dim=-1)
        return EmbeddingPoolerOutput(embeddings=pooled_data)

    def load_weights(self, weights: Iterable[WeightItem]):
        """Remap, fuse, and load weights directly
        (bypass parent's stacked_params_mapping)."""
        out_w: dict[str, torch.Tensor] = {}
        # Initialize fusion buffers from config
        fusion_buffers: dict[str, dict[int, dict[str, torch.Tensor]]] = {
            name: defaultdict(dict) for name in _FUSION_CONFIG
        }
        # Build reverse lookup: suffix -> (fusion_name, part_key)
        suffix_to_fusion: dict[str, tuple[str, str]] = {}
        for fusion_name, cfg in _FUSION_CONFIG.items():
            for suffix, part_key in cfg["suffixes"].items():
                suffix_to_fusion[suffix] = (fusion_name, part_key)

        for name, tensor in weights:
            m = _LAYER_RE.match(name)
            if not m:
                # Non-layer weights: strip "model." prefix if present
                new_name = name[len("model.") :] if name.startswith("model.") else name
                out_w[new_name] = tensor
                continue

            layer_idx = int(m.group(1))
            suffix = m.group(2)

            # Check if this weight needs fusion
            if suffix in suffix_to_fusion:
                fusion_name, part_key = suffix_to_fusion[suffix]
                fusion_buffers[fusion_name][layer_idx][part_key] = tensor
                continue

            # Other layer weights: output with stripped prefix
            out_w[f"layers.{layer_idx}.{suffix}"] = tensor

        # Fuse accumulated weights using config
        for fusion_name, cfg in _FUSION_CONFIG.items():
            buffer = fusion_buffers[fusion_name]
            parts_order = cfg["parts_order"]
            out_template = cfg["output_template"]
            for layer_idx, parts in buffer.items():
                if all(p in parts for p in parts_order):
                    fused = torch.cat([parts[p] for p in parts_order], dim=0)
                    out_w[out_template.format(layer_idx)] = fused
                elif parts:
                    missing = sorted([p for p in parts_order if p not in parts])
                    raise ValueError(
                        f"Layer {layer_idx} missing {fusion_name.upper()} parts: {missing}"
                    )

        # Load weights directly into model parameters
        # bypass parent's stacked_params_mapping
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in out_w.items():
            # Handle linear head weight
            if name == "linear.weight":
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                continue

            # Map base model weights to model.* prefix
            model_name = f"model.{name}" if not name.startswith("model.") else name
            if model_name not in params_dict:
                continue
            param = params_dict[model_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = VoyageQwen3BidirectionalEmbedModel
