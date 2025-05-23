# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM ModernBERT implementation

import logging
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import ModernBertConfig

# SGLang Imports
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# Custom Pooler replicating vLLM's ModernBertPooler logic for SGLang
class SGLangModernBertPooler(nn.Module):
    """
    Custom pooler for ModernBERT implementing mean -> dense -> act -> norm.
    """

    def __init__(self, config: ModernBertConfig, prefix: str = ""):
        super().__init__()
        # Using ReplicatedLinear as pooling is typically done on rank 0
        # or the result is gathered. If TP is needed, replace with parallel layers.
        # NOTE: Assumes TP size = 1 for the pooling layer for simplicity now.
        # If TP > 1, the mean calculation and linear layers need careful parallelization.
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.classifier_bias
        )
        self.act = get_act_fn("gelu")
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        # Calculate mean pooling based on sequence lengths
        # Note: This requires hidden_states to be the flat tensor of tokens
        # and relies on forward_batch.seq_lens
        # TODO: Verify mean pooling implementation matches desired behavior (e.g., exclude padding)
        # Simple mean for now, might need masking if padding exists in hidden_states
        seq_lens = forward_batch.seq_lens
        pooled_output = []
        start_idx = 0
        for length in seq_lens:
            pooled_output.append(
                hidden_states[start_idx : start_idx + length].mean(dim=0)
            )
            start_idx += length
        pooled_output_tensor = torch.stack(pooled_output, dim=0)

        # Apply dense -> act -> norm
        pooled_output_tensor = self.norm(self.act(self.dense(pooled_output_tensor)))
        return pooled_output_tensor


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig, prefix: str = ""):
        super().__init__()
        self.config = config
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("tok_embeddings", prefix),
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=config.norm_bias
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            # If input embeds are provided, just normalize them
            # Note: SGLang typically works with input_ids in forward batching
            embeddings = self.norm(inputs_embeds)
        else:
            inputs_embeds = self.tok_embeddings(input_ids)
            embeddings = self.norm(inputs_embeds)
        return embeddings


class ModernBertAttention(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        # self.deterministic_flash_attn = config.deterministic_flash_attn # SGLang might handle determinism globally
        self.num_heads = config.num_attention_heads
        assert self.num_heads % tp_size == 0, "num_heads must be divisible by TP size"
        self.tp_num_heads = self.num_heads // tp_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.tp_num_heads * self.head_dim  # Per TP rank
        self.scaling = self.head_dim**-0.5

        # Assuming num_kv_heads = num_heads for ModernBERT
        self.tp_num_kv_heads = self.tp_num_heads

        self.Wqkv = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,  # Assuming square attention
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("Wqkv", prefix),
        )

        # Determine RoPE theta based on local/global logic
        rope_theta = config.global_rope_theta
        # NOTE: Logic determining local_attention window size needs verification
        #       based on how SGLang's RadixAttention consumes this information.
        if layer_id % config.global_attn_every_n_layers != 0:
            # SGLang RadixAttention might take sliding_window_size directly
            self.sliding_window_size = config.local_attention
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta
        else:
            self.sliding_window_size = -1  # Indicates global attention

        # Use SGLang's get_rope
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,  # ModernBERT uses full rotary dim
            max_position=config.max_position_embeddings,
            base=int(rope_theta),  # Ensure base is int
            is_neox_style=True,  # ModernBERT uses Neox style
            # rope_scaling=config.rope_scaling if hasattr(config, 'rope_scaling') else None # Pass if ModernBERTConfig has it
        )

        # NOTE ON SLIDING WINDOW:
        # The handling of local attention windows (sliding_window_size) needs
        # to be verified with SGLang's RadixAttention implementation.
        # It might be passed during initialization (as below), set via
        # ForwardBatch, or configured elsewhere. Check SGLang documentation.
        self.attn = RadixAttention(
            num_heads=self.tp_num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.tp_num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,  # Pass window size here (needs verification)
            attn_type=AttentionType.ENCODER_ONLY,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self.Wo = RowParallelLinear(
            input_size=config.hidden_size,  # Wo input is full hidden size
            output_size=config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("Wo", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.Wqkv(hidden_states)
        # q, k, v shapes: [num_tokens, num_heads * head_dim] per rank
        q, k, v = qkv.split(
            [self.all_head_size, self.all_head_size, self.all_head_size], dim=-1
        )

        # Apply Rotary Embeddings
        q, k = self.rotary_emb(position_ids, q, k)

        # Pass to RadixAttention
        attn_outputs = self.attn(q, k, v, forward_batch=forward_batch)

        # Output projection
        hidden_states, _ = self.Wo(attn_outputs)
        return hidden_states


class ModernBertMLP(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        intermediate_size_half = config.intermediate_size  # Original intermediate size

        # Use two separate ColumnParallelLinear layers for the up-projection
        self.Wi_input = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size_half,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("Wi_input", prefix),  # Unique prefix for weight loading
        )
        self.Wi_gate = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size_half,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("Wi_gate", prefix),  # Unique prefix for weight loading
        )
        self.act = get_act_fn("gelu")  # ModernBERT uses GELU
        self.Wo = RowParallelLinear(
            input_size=intermediate_size_half,  # Input to Wo is half
            output_size=config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=add_prefix("Wo", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_proj, _ = self.Wi_input(hidden_states)
        gate_proj, _ = self.Wi_gate(hidden_states)
        # Apply activation and gate
        intermediate_act = self.act(input_proj) * gate_proj
        # Apply output projection
        output, _ = self.Wo(intermediate_act)
        return output


class ModernBertLayer(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        # LayerNorm before attention (except for layer 0)
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        self.attn = ModernBertAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        # LayerNorm before MLP
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        forward_batch: ForwardBatch,
    ):
        # Attention block with pre-norm (residual added after)
        attn_input = self.attn_norm(hidden_states)
        attn_outputs = self.attn(
            attn_input, position_ids=position_ids, forward_batch=forward_batch
        )
        # Residual connection for attention
        hidden_states = hidden_states + attn_outputs

        # MLP block with pre-norm (residual added after)
        mlp_input = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(mlp_input)
        # Residual connection for MLP
        hidden_states = hidden_states + mlp_output
        return hidden_states


class ModernBertEncoderLayer(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ModernBertLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, forward_batch)
        return hidden_states


class ModernBertModel(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        # Control whether to add the custom pooler for embedding tasks
        add_custom_pooler: bool = True,
    ):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(
            config, prefix=add_prefix("embeddings", prefix)
        )
        self.encoder_layer = ModernBertEncoderLayer(
            config,
            quant_config=quant_config,
            prefix=add_prefix("encoder_layer", prefix),
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.add_custom_pooler = add_custom_pooler
        if add_custom_pooler:
            # Using the custom pooler defined earlier
            self._pooler = SGLangModernBertPooler(
                config, prefix=add_prefix("pooler", prefix)
            )
        else:
            self._pooler = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Handle potential prefix difference (e.g., HF "layers." vs internal "encoder_layer.layers.")
        # This simplified mapper assumes HF checkpoint uses "layers." directly under the main model scope
        hf_prefix = "layers."
        internal_prefix = "encoder_layer.layers."
        mapped_weights = []
        mlp_weights = {}  # Store MLP Wi weights temporarily for splitting

        for name, loaded_weight in weights:
            # Map HF prefix to internal prefix
            if name.startswith(hf_prefix):
                name = internal_prefix + name[len(hf_prefix) :]

            # Identify MLP Wi weights/biases for splitting
            is_mlp_wi_weight = False
            is_mlp_wi_bias = False
            if ".mlp.Wi." in name:
                if name.endswith(".weight"):
                    is_mlp_wi_weight = True
                    base_name = name.rsplit(".weight", 1)[0]
                elif name.endswith(".bias"):
                    is_mlp_wi_bias = True
                    base_name = name.rsplit(".bias", 1)[0]

                if is_mlp_wi_weight or is_mlp_wi_bias:
                    mlp_weights.setdefault(base_name, {})[
                        "weight" if is_mlp_wi_weight else "bias"
                    ] = loaded_weight
                    continue  # Skip adding to mapped_weights for now

            # Handle custom pooler weights (map "head.*" to "_pooler.*")
            if self.add_custom_pooler and name.startswith("head."):
                if hasattr(self, "_pooler"):
                    mapped_name = "_pooler." + name[len("head.") :]
                    mapped_weights.append((mapped_name, loaded_weight))
                else:
                    logger.warning(
                        f"Skipping pooler weight {name} as pooler is disabled."
                    )
                continue  # Skip default loading

            # Add other weights directly
            mapped_weights.append((name, loaded_weight))

        params_dict = dict(self.named_parameters())

        # Load non-MLP-Wi weights
        for name, loaded_weight in mapped_weights:
            if name not in params_dict:
                # This might happen for rotary_emb.inv_freq or unexpected weights
                logger.warning(
                    f"Skipping weight {name} as it's not found in the model parameters."
                )
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                weight_loader(param, loaded_weight)
            except Exception as e:
                logger.error(f"Error loading weight {name}: {e}")

        # Load and split MLP Wi weights
        for base_name, parts in mlp_weights.items():
            if "weight" not in parts:
                logger.warning(f"MLP weight part missing for {base_name}, skipping.")
                continue

            full_weight = parts["weight"]
            input_weight, gate_weight = torch.chunk(full_weight, 2, dim=0)

            target_input_weight_name = base_name.replace(".Wi", ".Wi_input") + ".weight"
            target_gate_weight_name = base_name.replace(".Wi", ".Wi_gate") + ".weight"

            if target_input_weight_name in params_dict:
                param_input = params_dict[target_input_weight_name]
                loader_input = getattr(
                    param_input, "weight_loader", default_weight_loader
                )
                loader_input(param_input, input_weight)
            else:
                logger.warning(f"Parameter {target_input_weight_name} not found.")

            if target_gate_weight_name in params_dict:
                param_gate = params_dict[target_gate_weight_name]
                loader_gate = getattr(
                    param_gate, "weight_loader", default_weight_loader
                )
                loader_gate(param_gate, gate_weight)
            else:
                logger.warning(f"Parameter {target_gate_weight_name} not found.")

            # Handle bias if present
            if "bias" in parts:
                full_bias = parts["bias"]
                input_bias, gate_bias = torch.chunk(full_bias, 2, dim=0)
                target_input_bias_name = base_name.replace(".Wi", ".Wi_input") + ".bias"
                target_gate_bias_name = base_name.replace(".Wi", ".Wi_gate") + ".bias"

                if target_input_bias_name in params_dict:
                    param_input_bias = params_dict[target_input_bias_name]
                    loader_input_bias = getattr(
                        param_input_bias, "weight_loader", default_weight_loader
                    )
                    loader_input_bias(param_input_bias, input_bias)
                else:
                    logger.warning(f"Parameter {target_input_bias_name} not found.")

                if target_gate_bias_name in params_dict:
                    param_gate_bias = params_dict[target_gate_bias_name]
                    loader_gate_bias = getattr(
                        param_gate_bias, "weight_loader", default_weight_loader
                    )
                    loader_gate_bias(param_gate_bias, gate_bias)
                else:
                    logger.warning(f"Parameter {target_gate_bias_name} not found.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        positions: torch.LongTensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,  # Control whether to return pooled embedding
    ) -> Union[torch.Tensor, EmbeddingPoolerOutput]:
        if inputs_embeds is not None:
            hidden_states = self.embeddings.norm(
                inputs_embeds
            )  # Normalize if embeds provided
        elif input_ids is not None:
            hidden_states = self.embeddings(input_ids=input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        encoder_outputs = self.encoder_layer(
            hidden_states=hidden_states,
            position_ids=positions,
            forward_batch=forward_batch,
        )
        norm_outputs = self.final_norm(encoder_outputs)

        if get_embedding:
            if not self.add_custom_pooler or self._pooler is None:
                raise RuntimeError(
                    "Pooling requested (get_embedding=True) but pooler is not configured."
                )
            # Use the custom pooler
            pooled_output = self._pooler(norm_outputs, forward_batch)
            return EmbeddingPoolerOutput(embeddings=pooled_output)
        else:
            # Return the sequence output if not pooling
            return norm_outputs


class ModernBertForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        # Instantiate the base model *without* its internal pooler
        # because classification uses its own pooling logic before the head.
        self.model = ModernBertModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
            add_custom_pooler=False,  # Base model should return sequence output
        )
        # The specific pooler logic for classification (mean -> dense -> act -> norm)
        self._pooler = SGLangModernBertPooler(
            config, prefix=add_prefix("pooler", prefix)
        )
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        model_weights = []
        classifier_weights = []
        pooler_weights = []

        for name, loaded_weight in weights:
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], loaded_weight))
            elif name.startswith("classifier."):
                classifier_weights.append((name, loaded_weight))
            elif name.startswith("head."):  # Map HF 'head.' to internal '_pooler.'
                pooler_weights.append(
                    ("_pooler." + name[len("head.") :], loaded_weight)
                )
            else:
                logger.warning(
                    f"Unexpected weight prefix in classification model: {name}"
                )

        # Load base model weights
        self.model.load_weights(model_weights)

        # Load classifier and custom pooler weights
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in classifier_weights + pooler_weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                logger.warning(
                    f"Parameter {name} not found during classification head/pooler loading."
                )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        positions: torch.LongTensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get sequence output from the base model
        sequence_output = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=inputs_embeds,
            get_embedding=False,  # Ensure base model returns sequence output
        )

        # Apply the specific pooling logic (mean -> dense -> act -> norm)
        pooled_output = self._pooler(sequence_output, forward_batch)

        # Pass through classifier
        logits = self.classifier(pooled_output)
        return logits


# SGLang Entry Point Registration
# Include both the base model (for embedding tasks) and the classification model
EntryClass = [ModernBertModel, ModernBertForSequenceClassification]
