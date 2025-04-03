from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

__all__ = [
    "MoeModelOutputWithPast",
    "MoeCausalLMOutputWithPast",
]

try:
    from transformers.modeling_outputs import (
        MoeCausalLMOutputWithPast,
        MoeModelOutputWithPast,
    )
except:

    @dataclass
    class MoeModelOutputWithPast(ModelOutput):
        """
        Base class for model's outputs, with potential hidden states and attentions.

        Args:
            last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
                `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
                encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
                `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
                input) to speed up sequential decoding.
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
                one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
            router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

                Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
                loss for Mixture of Experts models.
        """

        last_hidden_state: torch.FloatTensor = None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
        router_logits: Optional[Tuple[torch.FloatTensor]] = None

    @dataclass
    class MoeCausalLMOutputWithPast(ModelOutput):
        """
        Base class for causal language model (or autoregressive) with mixture of experts outputs.

        Args:
            loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
                Language modeling loss (for next-token prediction).

            logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
                aux_loss for the sparse modules.

            router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

                Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
                loss for Mixture of Experts models.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

                Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
                `past_key_values` input) to speed up sequential decoding.
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
                one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        loss: Optional[torch.FloatTensor] = None
        aux_loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
        router_logits: Optional[Tuple[torch.FloatTensor]] = None
