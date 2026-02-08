# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

WeightItem = Tuple[str, torch.Tensor]

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


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

        # Pooler for embedding output (MEAN pooling for sentence embeddings)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

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
        assert get_embedding, (
            "VoyageQwen3BidirectionalEmbedModel is only used for embedding"
        )
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        # Apply the linear head
        hidden_states = self.linear(hidden_states)
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[WeightItem]):
        """Remap, fuse, and load weights directly
        (bypass parent's stacked_params_mapping)."""
        out_w: dict[str, torch.Tensor] = {}
        qkv_buf: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        mlp_buf: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)

        for name, tensor in weights:
            m = _LAYER_RE.match(name)
            if not m:
                # Non-layer weights: strip "model." prefix if present
                new_name = name[len("model.") :] if name.startswith("model.") else name
                out_w[new_name] = tensor
                continue

            layer_idx = int(m.group(1))
            suffix = m.group(2)

            # Accumulate Q/K/V for fusion
            if suffix == "self_attn.q_proj.weight":
                qkv_buf[layer_idx]["q"] = tensor
                continue
            if suffix == "self_attn.k_proj.weight":
                qkv_buf[layer_idx]["k"] = tensor
                continue
            if suffix == "self_attn.v_proj.weight":
                qkv_buf[layer_idx]["v"] = tensor
                continue

            # Accumulate gate/up for fusion
            if suffix == "mlp.gate_proj.weight":
                mlp_buf[layer_idx]["gate"] = tensor
                continue
            if suffix == "mlp.up_proj.weight":
                mlp_buf[layer_idx]["up"] = tensor
                continue

            # Other layer weights: output with stripped prefix
            out_w[f"layers.{layer_idx}.{suffix}"] = tensor

        def _fuse_parts(buffer, parts_to_fuse, out_name_template, part_type):
            for layer_idx, parts in buffer.items():
                if all(p in parts for p in parts_to_fuse):
                    fused = torch.cat([parts[p] for p in parts_to_fuse], dim=0)
                    out_w[out_name_template.format(layer_idx)] = fused
                elif parts:
                    missing = sorted([p for p in parts_to_fuse if p not in parts])
                    raise ValueError(
                        f"Layer {layer_idx} is missing {part_type} parts: {missing}"
                    )

        _fuse_parts(
            qkv_buf, ("q", "k", "v"), "layers.{}.self_attn.qkv_proj.weight", "QKV"
        )
        _fuse_parts(mlp_buf, ("gate", "up"), "layers.{}.mlp.gate_up_proj.weight", "MLP")

        # Load weights directly into model parameters
        # bypass parent's stacked_params_mapping
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in out_w.items():
            # Handle linear head weight
            if name == "linear.weight":
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
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
