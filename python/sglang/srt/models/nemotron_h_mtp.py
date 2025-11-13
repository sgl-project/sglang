from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.configs import NemotronHConfig
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from .nemotron_h import NemotronHAttentionDecoderLayer


class NemotronHMultiTokenPredictorLayer(nn.Module):
    """MTP layer using NemotronH attention architecture."""

    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: QuantizationConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        # Normalization layers for fusion
        self.mtp_emb_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mtp_hidden_norm = RMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        # Fusion layer to combine embeddings with target hidden states
        self.mtp_linear_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        # Use NemotronH's native attention layer (compatible with NemotronHConfig)
        self.mtp_block = NemotronHAttentionDecoderLayer(
            config=config,  # type: ignore[arg-type]
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.mtp_block",
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        """Forward pass: fuse embeddings + hidden states, then apply attention."""
        assert inputs_embeds is not None

        # Normalize both inputs before fusion
        inputs_embeds_normed = self.mtp_emb_norm(inputs_embeds)
        previous_hidden_states_normed = self.mtp_hidden_norm(previous_hidden_states)

        # Fuse via concatenation and linear projection
        hidden_states = self.mtp_linear_proj(
            torch.cat([inputs_embeds_normed, previous_hidden_states_normed], dim=-1)
        )

        # Apply attention block with residual connection
        hidden_states, residual = self.mtp_block(
            forward_batch=forward_batch, hidden_states=hidden_states, residual=None
        )

        # Final residual addition
        hidden_states = residual + hidden_states
        return hidden_states


class NemotronHMultiTokenPredictor(nn.Module):
    """MTP predictor with attention layers (similar to Qwen3NextMTP)."""

    def __init__(
        self,
        *,
        config: NemotronHConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Create MTP layers with attention (using ModuleDict for layer indexing)
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): NemotronHMultiTokenPredictorLayer(
                    config=config,
                    layer_idx=i,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for i, idx in enumerate(
                    range(
                        self.mtp_start_layer_idx - 1,
                        self.mtp_start_layer_idx + self.num_mtp_layers - 1,
                    )
                )
            }
        )

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Forward with PP support and proper residual handling."""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        # Use the MTP layer (cycling for multi-step)
        layer_idx_str = str(
            self.mtp_start_layer_idx + (spec_step_idx % self.num_mtp_layers) - 1
        )
        hidden_states = self.layers[layer_idx_str](
            inputs_embeds,
            forward_batch,
            hidden_states,  # previous hidden states from target
            spec_step_idx,
        )

        # Final normalization
        hidden_states = self.norm(hidden_states)
        return hidden_states


class NemotronHMTPForCausalLM(nn.Module):
    """NemotronH MTP model with attention layers."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(
        self,
        *,
        config: NemotronHConfig,
        quant_config: QuantizationConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # MTP predictor with attention layers
        self.model = NemotronHMultiTokenPredictor(
            config=config, quant_config=quant_config, prefix=f"{prefix}.model"
        )

        # LM head for generating logits
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.lm_head",
        )

        self.logits_processor = LogitsProcessor(self.config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward - applies attention-based MTP."""
        hidden_states = forward_batch.spec_info.hidden_states
        hidden_states = self.model(
            input_ids, forward_batch, hidden_states, inputs_embeds
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """Compute logits for DRAFT token generation."""
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load MTP weights with proper name remapping."""
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Skip tied embeddings if configured
            if self.config.tie_word_embeddings and name.endswith("lm_head.weight"):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            # Remap MTP layer names
            if "mtp" in name or "backbone.norm.weight" in name or "embeddings" in name:
                name = self._rewrite_spec_layer_name(name)

            # Handle stacked parameters (qkv_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "model.layers" not in name:  # Only MTP layers
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle non-stacked parameters
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue

                # Only load MTP-specific weights and shared embeddings/lm_head
                if (
                    "model.layers" not in name
                    and "embed_tokens" not in name
                    and "lm_head" not in name
                    and "model.norm" not in name
                ):
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, name: str) -> str:
        """
        Rewrite the weight name to match the internal model structure.
        """
        if "embeddings" in name:
            name = name.replace("embeddings", "embed_tokens")
        if name.startswith("backbone."):
            name = name.replace("backbone.", "model.")
        return name

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [NemotronHMTPForCausalLM]
