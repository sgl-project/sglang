"""Qwen3 DSpark draft model for SGLang (DFLASH backbone + Markov/confidence heads)."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix


class DSparkMarkovHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size,
            markov_rank,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("markov_w1", prefix),
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size,
            markov_rank,
            quant_config=quant_config,
            prefix=add_prefix("markov_w2", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def project_bias(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(embeddings, self.markov_w2.weight)


class Qwen3DSparkConfidenceHead(nn.Module):
    """Match DeepSpec AcceptRatePredictor (Linear with bias)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1, bias=True, dtype=torch.float32)

    def forward(
        self, hidden: torch.Tensor, markov_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if markov_embed is not None:
            features = torch.cat([hidden, markov_embed], dim=-1)
        else:
            features = hidden
        return self.proj(features.float()).squeeze(-1)


class Qwen3DSparkDraftModel(DFlashDraftModel):
    """DFlash-style Qwen3 draft backbone with DSpark Markov/confidence heads."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)

        self.vocab_size = int(config.vocab_size)
        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.mask_token_id = int(getattr(config, "mask_token_id", 0))
        self.noise_token_id = self.mask_token_id

        hidden_size = int(config.hidden_size)
        enable_confidence = bool(getattr(config, "enable_confidence_head", False))
        confidence_with_markov = bool(
            getattr(config, "confidence_head_with_markov", False)
        )

        self.lm_head = ParallelLMHead(
            self.vocab_size,
            hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

        if self.markov_rank > 0:
            self.markov_head = DSparkMarkovHead(
                self.vocab_size,
                self.markov_rank,
                quant_config=quant_config,
                prefix=add_prefix("markov_head", prefix),
            )
        else:
            self.markov_head = None

        self.confidence_head = None
        if enable_confidence:
            input_dim = hidden_size
            if confidence_with_markov:
                assert self.markov_head is not None
                input_dim += self.markov_rank
            self.confidence_head = Qwen3DSparkConfidenceHead(input_dim)

    @property
    def num_dspark_layers(self) -> int:
        return len(self.layers)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        markov_weights = []
        confidence_weights = []
        lm_head_weights = []
        backbone_weights = []

        for name, tensor in weights:
            if name.startswith("markov_head."):
                markov_weights.append((name, tensor))
            elif name.startswith("confidence_head."):
                confidence_weights.append((name, tensor))
            elif name.startswith("lm_head."):
                lm_head_weights.append((name, tensor))
            elif name.startswith("embed_tokens."):
                continue
            else:
                backbone_weights.append((name, tensor))

        super().load_weights(backbone_weights)

        params = dict(self.named_parameters())
        for name, loaded in markov_weights + confidence_weights + lm_head_weights:
            if name not in params:
                continue
            param = params[name]
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, loaded)
            else:
                param.data.copy_(loaded)


EntryClass = Qwen3DSparkDraftModel
