# Modified for SGLang; see this directory's README.md for upstream source.

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_neo_chat import NEOMoELLMConfig
from .modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RMSNorm,
    create_block_causal_mask,
)
from .transformers_compat import (
    causal_mask_kwargs,
    model_input_compat,
    tied_weights_keys,
)


class Qwen3MoeMLP(nn.Module):
    """Single expert FFN. Same structure as :class:`Qwen3MLP` but the
    intermediate size is parameterised so it can be ``moe_intermediate_size``
    (per-expert) for experts and ``intermediate_size`` for any dense fallback.
    """

    def __init__(self, config, intermediate_size: Optional[int] = None):
        super().__init__()
        from transformers.activations import ACT2FN

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k softmax-routed MoE block matching HuggingFace's Qwen3-MoE layout.

    Parameter names (``gate.weight``, ``experts.{i}.gate_proj/up_proj/down_proj``)
    are kept identical so converted A3B checkpoints load directly via the
    ``mlp.*`` / ``mlp_mot_gen.*`` keys. The block is parameterised explicitly
    so the same class can serve both the understanding branch (``num_experts``
    experts, top-k = ``num_experts_per_tok``, width ``moe_intermediate_size``)
    and the image-generation branch (``gen_num_experts`` etc.).
    """

    def __init__(
        self,
        config: NEOMoELLMConfig,
        num_experts: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = (
            int(num_experts) if num_experts is not None else int(config.num_experts)
        )
        self.top_k = int(
            num_experts_per_tok
            if num_experts_per_tok is not None
            else config.num_experts_per_tok
        )
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.hidden_size = config.hidden_size

        expert_intermediate_size = int(
            moe_intermediate_size
            if moe_intermediate_size is not None
            else config.moe_intermediate_size
        )

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=expert_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat = hidden_states.view(-1, hidden_dim)
        n_tokens = flat.shape[0]

        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(flat.dtype)

        output = torch.zeros(
            (n_tokens, hidden_dim), dtype=flat.dtype, device=flat.device
        )
        # (num_experts, top_k, num_tokens)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            expert_layer = self.experts[expert_idx]
            current_state = flat.index_select(0, top_x)
            current_out = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            output.index_add_(0, top_x, current_out.to(flat.dtype))

        return output.view(*orig_shape)


class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    """A Qwen3-MoE decoder block with the NEO-Unify two-branch structure.

    Mirrors ``Qwen3DecoderLayer`` from :mod:`modeling_qwen3` but uses sparse
    MoE blocks on *both* branches:

      * ``self.mlp``         - understanding-path MoE
                               (``num_experts`` / ``num_experts_per_tok`` /
                               ``moe_intermediate_size``)
      * ``self.mlp_mot_gen`` - image-generation-path MoE
                               (``gen_num_experts`` / ``gen_num_experts_per_tok`` /
                               ``gen_moe_intermediate_size``)

    Layers listed in ``mlp_only_layers`` or those not aligned with
    ``decoder_sparse_step`` fall back to a dense :class:`Qwen3MoeMLP` on the
    understanding branch (matching upstream Qwen3-MoE), while the
    generation branch still uses a sparse MoE.
    """

    def __init__(self, config: NEOMoELLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        mlp_only_layers = list(getattr(config, "mlp_only_layers", []) or [])
        decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1) or 1)
        is_sparse = (
            int(config.num_experts) > 0
            and layer_idx not in mlp_only_layers
            and (layer_idx + 1) % decoder_sparse_step == 0
        )

        if is_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(
                config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                moe_intermediate_size=config.moe_intermediate_size,
            )
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        # Image-generation branch: in the A3B checkpoint this is *also* a sparse
        # MoE block (``gen_num_experts`` experts, typically smaller than the und
        # branch's ``num_experts``). ``NEOMoELLMConfig`` defaults the gen-path
        # knobs to their und-path counterparts so legacy single-pool configs
        # keep working.
        self.mlp_mot_gen = Qwen3MoeSparseMoeBlock(
            config,
            num_experts=getattr(config, "gen_num_experts", config.num_experts),
            num_experts_per_tok=getattr(
                config, "gen_num_experts_per_tok", config.num_experts_per_tok
            ),
            moe_intermediate_size=getattr(
                config, "gen_moe_intermediate_size", config.moe_intermediate_size
            ),
        )

        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_gen(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                **kwargs,
            )
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                **kwargs,
            )

        # Mixed und/gen path — see the NOTE in Qwen3Attention.forward (modeling_qwen3.py).
        raise NotImplementedError(
            "Mixed und/gen decoder-layer forward is not yet validated (issue #207). "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        # Mixed batch: dispatch tokens per branch then merge back. Matches the
        # dense ``Qwen3DecoderLayer.forward`` mixed-path implementation.
        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.input_layernorm(
                hidden_states[~image_gen_indicators]
            )
        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.input_layernorm_mot_gen(
                hidden_states[image_gen_indicators]
            )
        hidden_states = _hidden_states

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            und_hidden = self.post_attention_layernorm(
                hidden_states[~image_gen_indicators]
            )
            # MoE expects a 3D input (batch, seq, hidden); promote then squeeze.
            if und_hidden.dim() == 2:
                und_hidden = und_hidden.unsqueeze(0)
                _hidden_states[~image_gen_indicators] = self.mlp(und_hidden).squeeze(0)
            else:
                _hidden_states[~image_gen_indicators] = self.mlp(und_hidden)
        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.mlp_mot_gen(
                self.post_attention_layernorm_mot_gen(
                    hidden_states[image_gen_indicators]
                )
            )

        hidden_states = _hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoePreTrainedModel(PreTrainedModel):
    config: NEOMoELLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False  # MoE routing has data-dependent control flow.
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    def __init__(self, config: NEOMoELLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.current_index = -1

        self.post_init()

    @model_input_compat
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_gen_indicators: Optional[torch.Tensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            # Convert the CUDA reductions once before the decoder loop. If the
            # scalar tensors reach every layer, each Python branch can force a
            # host-device synchronization and collapse async weight prefetch.
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if input_ids is not None:
                mask_kwargs = causal_mask_kwargs(
                    create_causal_mask,
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(
                    input_ids.device
                )
            else:
                causal_mask_mapping = {
                    "full_attention": create_block_causal_mask(indexes[0]),
                }
                self.current_index = indexes[0].max()
        else:
            self.current_index = indexes[0].max()

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        if not exist_image_gen_tokens:
            hidden_states = self.norm(hidden_states)
        elif not exist_non_image_gen_tokens:
            hidden_states = self.norm_mot_gen(hidden_states)
        else:
            _hidden_states = hidden_states.new_zeros(hidden_states.shape)
            _hidden_states[~image_gen_indicators] = self.norm(
                hidden_states[~image_gen_indicators]
            )
            _hidden_states[image_gen_indicators] = self.norm_mot_gen(
                hidden_states[image_gen_indicators]
            )
            hidden_states = _hidden_states

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = tied_weights_keys(
        "lm_head.weight", "model.embed_tokens.weight"
    )
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: NEOMoELLMConfig):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeMLP",
]
