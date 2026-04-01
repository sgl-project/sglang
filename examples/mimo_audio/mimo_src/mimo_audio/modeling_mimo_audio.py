# Copyright 2025 Xiaomi Corporation.
import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Union, cast

import torch
import torch.distributed as dist
from torch import nn
from transformers import StoppingCriteria
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateOutput,
    GenerationConfig,
    StoppingCriteriaList,
    is_deepspeed_zero3_enabled,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model,
    Qwen2PreTrainedModel,
)
from transformers.utils import is_torchdynamo_compiling


logger = logging.getLogger(__name__)


class MiMoStopper(StoppingCriteria):
    def __init__(
        self,
        group_size: int,
        audio_channels: int,
        stop_tokens: list[int] | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.audio_channels = audio_channels
        self.step = (audio_channels + 1) * group_size

        self.stop_token_ids = set(stop_tokens or [])

        self.max_length = max_length
        self.min_length = min_length or 0

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor):
        is_done = False
        cur_len = input_ids.shape[-1] // self.step
        
        if self.max_length:
            is_done |= cur_len >= self.max_length
            
        if (self.stop_token_ids and 
            input_ids.shape[1] >= self.step and 
            cur_len >= self.min_length):
            last_token = input_ids[0, -self.step].item()
            is_done |= last_token in self.stop_token_ids

        return torch.full(
            (input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool
        )


@dataclass
class MiMoSampler:
    do_sample: bool | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    def process(self, scores: torch.Tensor):
        if self.temperature is not None:
            scores = scores / self.temperature

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, scores.shape[-1])
            indices_to_remove = scores < torch.topk(scores, top_k)[0][:, -1]
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        if self.top_p is not None and 0.0 < self.top_p <= 1.0:
            top_p = self.top_p if 0.0 < self.top_p <= 1.0 else 1.0
            sorted_logits, sorted_indices = torch.sort(scores)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[:, -1] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        return scores

    def sample(self, scores: torch.Tensor, removed_tokens: list[int] | None = None):
        scores = self.process(scores)
        for t in removed_tokens or []:
            scores[:, t] = float("-inf")

        if self.do_sample:
            probs = scores.softmax(dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        return torch.argmax(scores, dim=-1)


@dataclass
class MiMoAudioOutput(ModelOutput):
    text_logits: torch.FloatTensor | None = None
    local_hidden_states: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    """Downcast hidden states for local transformer generation"""


@dataclass
class MiMoAudioConfig(Qwen2Config):
    def __init__(
        self,
        *,
        speech_vocab_size: str | int = "1025-1025-129-129-129-129-129-129",
        speech_zeroemb_idx: str | int = "1024-1024-128-128-128-128-128-128",
        delay_pattern: str = "0-1-2-3-4-5-6-7",
        head_dim: int = 128,
        group_size: int = 4,
        audio_channels: int = 8,
        local_dim: int = 1024,
        local_layers: int = 16,
        local_attn_heads: int = 64,
        local_ffn_dim: int = 4096,
        local_attn_dropout: float = 0.1,
        input_local_layers: int = 6,
        input_local_dim: int | None = None,
        input_full_attention: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.speech_vocab_size = speech_vocab_size
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.delay_pattern = delay_pattern

        self.head_dim = head_dim

        self.group_size = group_size
        self.audio_channels = audio_channels

        self.local_dim = local_dim
        self.local_layers = local_layers
        self.local_attn_heads = local_attn_heads
        self.local_ffn_dim = local_ffn_dim
        self.local_attn_dropout = local_attn_dropout

        self.input_local_layers = input_local_layers
        self.input_local_dim = input_local_dim or local_dim

        self.input_full_attention = input_full_attention

    def _parse_maybe_list(self, value: str | int, length: int) -> List[int]:
        if isinstance(value, str) and "-" in value:
            return [int(s) for s in value.split("-")]
        return [int(value)] * length

    def parsed_speech_empty_ids(self):
        return self._parse_maybe_list(self.speech_zeroemb_idx, self.audio_channels)

    def parsed_speech_vocab_sizes(self):
        return self._parse_maybe_list(self.speech_vocab_size, self.audio_channels)

    def parsed_delay_pattern(self):
        return self._parse_maybe_list(self.delay_pattern, self.audio_channels)

    def local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.local_dim
        config.num_hidden_layers = self.local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = self.local_ffn_dim
        config.attention_dropout = self.local_attn_dropout

        return config

    def input_local_config(self):
        config = copy.deepcopy(self)

        config.hidden_size = self.input_local_dim
        config.num_hidden_layers = self.input_local_layers
        config.num_attention_heads = self.local_attn_heads
        config.num_key_value_heads = self.local_attn_heads
        config.head_dim = config.hidden_size // self.local_attn_heads
        config.intermediate_size = config.hidden_size * 4
        config.attention_dropout = self.local_attn_dropout

        return config


@dataclass
class MiMoAudioArguments:
    model_name_or_path: str
    sosp_idx: int
    eosp_idx: int
    sostm_idx: int
    eostm_idx: int
    eot_idx: int
    empty_idx: int

    def to_dict(self):
        return {
            "model_name_or_path": self.model_name_or_path,
            "sosp_idx": self.sosp_idx,
            "eosp_idx": self.eosp_idx,
            "sostm_idx": self.sostm_idx,
            "eostm_idx": self.eostm_idx,
            "eot_idx": self.eot_idx,
            "empty_idx": self.empty_idx,
        }


class MiMoAudioForCausalLM(Qwen2PreTrainedModel):
    def __init__(
        self,
        config: MiMoAudioConfig | Qwen2Config,
        args: MiMoAudioArguments | dict,
    ):
        super().__init__(config)
        config = (
            MiMoAudioConfig(**vars(config))
            if isinstance(config, Qwen2Config)
            else config
        )
        args = MiMoAudioArguments(**args) if isinstance(args, dict) else args
        self.config = config
        self.args = args

        self.model = Qwen2Model(config)

        self.speech_vocab_sizes = config.parsed_speech_vocab_sizes()
        self.speech_empty_ids = config.parsed_speech_empty_ids()
        self.delay_pattern = config.parsed_delay_pattern()

        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Construct local transformer
        self.local_config = config.local_config()
        self.local_transformer = Qwen2Model(self.local_config)
        self.local_transformer.embed_tokens = None

        # Add input local transformer if configured
        self.input_local_config = config.input_local_config()
        self.input_local_transformer = Qwen2Model(self.input_local_config)
        self.input_local_transformer.embed_tokens = None

        self.local_transformer_lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    self.local_config.hidden_size,
                    self.speech_vocab_sizes[i],
                    bias=False,
                )
                for i in range(self.audio_channels)
            ]
        )

        self.speech_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    self.speech_vocab_sizes[i],
                    self.input_local_config.hidden_size,
                    padding_idx=self.speech_empty_ids[i],
                )
                for i in range(self.audio_channels)
            ]
        )

        if self.input_local_config.hidden_size != self.local_config.hidden_size:
            self.speech_embeddings_to_local = nn.Linear(
                self.input_local_config.hidden_size,
                self.local_config.hidden_size,
                bias=False,
            )
        else:
            self.speech_embeddings_to_local = None

        # Create speech_group_downcast_first for group_first_in_global_context
        self.speech_group_downcast = nn.Linear(
            self.input_local_config.hidden_size * config.group_size,
            config.hidden_size,
            bias=False,
        )

        self.hidden_states_downcast = nn.Linear(
            config.hidden_size,
            self.local_config.hidden_size,
            bias=False,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor):
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Process each group independently: [B*T//group_size, group_size, hidden_size]
        input_embeddings = speech_embeddings.reshape(
            B * T_groups, group_size, hidden_size
        )

        output: BaseModelOutputWithPast = self.input_local_transformer(
            inputs_embeds=input_embeddings,
            return_dict=True,
            is_causal=not self.config.input_full_attention,  # for SDPA
        )
        encoded_embeddings = output.last_hidden_state

        # Reshape back to original format
        # [B*T//group_size, group_size, hidden_size] -> [B, T//group_size, group_size, hidden_size]
        encoded_embeddings = encoded_embeddings.reshape(
            B, T_groups, group_size, hidden_size
        )

        return encoded_embeddings

    def _prepare_input_embeds(
        self,
        input_ids: torch.LongTensor,  # [B, audio_channels + 1, new_T]
    ):
        B = input_ids.shape[0]

        input_ids = input_ids.int()
        group_size = self.config.group_size

        text_input_ids = input_ids[:, 0, ::group_size]
        speech_input_ids = (
            input_ids[:, 1:, :]
            .view(B, self.audio_channels, -1, group_size)
            .transpose(1, 2)
        )  # [B, T//group_size, audio_channels, group_size]

        is_speech = text_input_ids == self.args.empty_idx  # [B, T//group_size]

        speech_embeds = torch.zeros(
            (
                B,
                is_speech.shape[1],
                group_size,
                self.input_local_config.hidden_size,
            ),
            device=input_ids.device,
            dtype=torch.bfloat16,
        )

        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]
            cur_speech_embeds: torch.Tensor = cur_embed(cur_speech_ids)
            # [B, T_groups, group_size, hidden_size]

            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)

            speech_embeds += cur_speech_embeds

        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        # Apply input local transformer if configured
        speech_embeds = self.apply_input_local_transformer(speech_embeds)
        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        T_groups = speech_embeds.shape[1]
        speech_grouped_embeds: torch.Tensor = self.speech_group_downcast(
            speech_embeds.view(B, T_groups, -1)
        )  # [B, T_groups, hidden_size]

        text_embeds: torch.Tensor = self.model.embed_tokens(text_input_ids)
        text_zero_mask = text_input_ids == self.args.empty_idx
        text_embeds.masked_fill_(text_zero_mask.unsqueeze(-1), 0.0)

        return text_embeds + speech_grouped_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,  # [B, audio_channels + 1, new_T]
        attention_mask: torch.Tensor,  # [B, T_group]
        position_ids: torch.LongTensor,  # [B, new_T_group]
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,  # [new_T_group]
        **_kwargs,
    ):
        # import pdb;pdb.set_trace()
        inputs_embeds = self._prepare_input_embeds(input_ids)

        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state  # [B, new_T_group, hidden_size]

        text_logits: torch.Tensor = self.lm_head(
            hidden_states[:, -1:, :]
        )  # [B, 1, vocab_size]
        shift_hidden_states: torch.Tensor = self.hidden_states_downcast(
            hidden_states[:, -1:, :]
        )  # [B, 1, hidden_size]

        return MiMoAudioOutput(
            text_logits=text_logits,
            local_hidden_states=shift_hidden_states,
            past_key_values=outputs.past_key_values,
        )

    def local_forward(
        self,
        local_embeds: torch.FloatTensor,  # [B, 1, hidden_size]
        tokens_dtype: torch.dtype,
        tokens_device: torch.device,
        local_sampler: MiMoSampler | None = None,
    ):
        B = local_embeds.shape[0]
        delay_iters = self.group_size + max(self.delay_pattern)
        past_key_values = DynamicCache()
        local_tokens = torch.zeros(
            (B, self.group_size, self.audio_channels),
            dtype=tokens_dtype,
            device=tokens_device,
        )
        if local_sampler is None:
            local_sampler = MiMoSampler()

        for t in range(delay_iters):
            output: BaseModelOutputWithPast = self.local_transformer(
                inputs_embeds=local_embeds,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            hidden_state = output.last_hidden_state
            past_key_values = output.past_key_values

            local_embeds = torch.zeros_like(local_embeds)
            for idx in range(self.audio_channels):
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size
                cur_empty = self.speech_empty_ids[idx]
                if cur_start <= t < cur_end:
                    cur_lm_head = self.local_transformer_lm_heads[idx]
                    cur_scores: torch.Tensor = cur_lm_head(hidden_state)[:, -1, :]
                    # [B, vocab_size]
                    cur_token = local_sampler.sample(
                        cur_scores,
                        [cur_empty],
                    )

                    local_tokens[:, t - cur_start, idx] = cur_token
                    cur_input_embed = self.speech_embeddings[idx](
                        cur_token.unsqueeze(1)
                    )
                    if self.speech_embeddings_to_local is not None:
                        cur_input_embed = self.speech_embeddings_to_local(
                            cur_input_embed
                        )
                    local_embeds += cur_input_embed

        return local_tokens  # [B, group_size, audio_channels]

    def _prepare_attention_mask(
        self, inputs: torch.Tensor, input_ids_length: int
    ) -> torch.Tensor:
        # No information for attention mask inference -> return default attention mask
        return torch.ones(
            (inputs.shape[0], input_ids_length),
            dtype=torch.bool,
            device=inputs.device,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        input_ids = input_ids.reshape(
            input_ids.shape[0], -1, (self.audio_channels + 1) * self.config.group_size
        ).transpose(1, 2)  # [B, audio_channels*group_size, T]
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = (
                past_key_values[0][0].shape[2] if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_length,
                input_ids.shape[2],
                dtype=torch.long,
                device=input_ids.device,
            )

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if (
                inputs_embeds is not None or cache_position[-1] >= input_ids.shape[2]
            ):  # Exception 1 or Exception 3
                input_ids = input_ids[:, :, -cache_position.shape[0] :]
            elif (
                input_ids.shape[2] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, :, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = (
            "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        )
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(
                    memory_format=torch.contiguous_format
                )
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(
                memory_format=torch.contiguous_format
            )

        # 4. Create missing `position_ids` on the fly
        if attention_mask is not None and kwargs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = (
                position_ids  # placed in kwargs for further processing (see below)
            )

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input: torch.Tensor = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[2] :]
                    model_input = model_input.clone(
                        memory_format=torch.contiguous_format
                    )
                model_inputs[model_input_name] = model_input

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        if model_inputs[input_ids_key] is not None:
            model_inputs[input_ids_key] = (
                cast(torch.Tensor, model_inputs[input_ids_key])
                .transpose(1, 2)
                .reshape(input_ids.shape[0], -1, (self.audio_channels + 1))
                .transpose(1, 2)
            )  # [B, audio_channels, T*group_size]

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

    def _get_initial_cache_position(self, input_ids: torch.Tensor, model_kwargs: dict):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            cache_position = (
                torch.ones_like(
                    model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64
                ).cumsum(0)
                - 1
            )
        else:
            cache_position = (
                torch.ones(
                    (
                        input_ids.shape[1]
                        // (self.audio_channels + 1)
                        // self.config.group_size,
                    ),
                    dtype=torch.int64,
                    device=input_ids.device,
                ).cumsum(0)
                - 1
            )

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif (
                hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None
            ):
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position

        return model_kwargs

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        stopping_criteria: StoppingCriteriaList | list | None = None,
        streamer: BaseStreamer | None = None,
        synced_gpus: bool | None = None,
        global_sampler: MiMoSampler | None = None,
        local_sampler: MiMoSampler | None = None,
        warmup_run: bool | None = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )

        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        # 3. Define model inputs
        input_ids, _model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        input_ids_length = input_ids.shape[-1]
        input_ids_length //= self.group_size * (self.audio_channels + 1)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        if "attention_mask" not in model_kwargs:
            model_kwargs["attention_mask"] = self._prepare_attention_mask(
                inputs, input_ids_length
            )

        device = input_ids.device
        self._prepare_special_tokens(generation_config, True, device=device)

        model_kwargs["use_cache"] = True
        model_kwargs["past_key_values"] = DynamicCache()

        prepared_stopping_criteria = StoppingCriteriaList(
            stopping_criteria if stopping_criteria is not None else []
        )
        prepared_stopping_criteria.append(
            MiMoStopper(
                self.group_size,
                self.audio_channels,
                max_length=generation_config.max_length,
            )
        )
        stance = "default" if warmup_run else "eager_on_recompile"
        with torch.compiler.set_stance(stance):
            return self.slm_sample(
                input_ids,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                global_sampler=global_sampler,
                local_sampler=local_sampler,
                **model_kwargs,
            )

    def slm_sample(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: BaseStreamer | None,
        global_sampler: MiMoSampler | None = None,
        local_sampler: MiMoSampler | None = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        max_length = generation_config.max_length

        B, cur_len = input_ids.shape
        cur_len //= self.group_size * (self.audio_channels + 1)
        initial_len = cur_len
        this_peer_finished = False
        unfinished_sequences = torch.ones(B, dtype=torch.long, device=input_ids.device)
        
        min_length = 0
        stop_token_ids = set()
        for criterion in stopping_criteria:
            if isinstance(criterion, MiMoStopper):
                if criterion.min_length is not None:
                    min_length = max(min_length, criterion.min_length)
                stop_token_ids.update(criterion.stop_token_ids)

        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            if (
                cast(torch.Tensor, model_inputs["input_ids"]).shape[2]
                != self.group_size
            ):
                # prefill run
                with torch.compiler.set_stance("force_eager"):
                    outputs: MiMoAudioOutput = self(**model_inputs)
            else:
                outputs: MiMoAudioOutput = self(**model_inputs)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            text_logits: torch.Tensor = outputs.text_logits[:, -1, :].clone()
            # [B, vocab_size]
            
            removed_tokens = None
            if cur_len < min_length:
                removed_tokens = list(stop_token_ids)
            
            next_text_tokens = global_sampler.sample(text_logits, removed_tokens=removed_tokens)
            # [B]

            local_hidden_states = outputs.local_hidden_states

            # Only Supports batch_size=1 here
            if next_text_tokens[0] != self.args.empty_idx:
                zero_embed_tensor = torch.tensor(
                    self.speech_empty_ids,
                    device=next_text_tokens.device,
                    dtype=input_ids.dtype,
                )
                next_speech_tokens = zero_embed_tensor.view(
                    1, 1, self.audio_channels
                ).expand(B, self.config.group_size, -1)
            else:
                next_speech_tokens = self.local_forward(
                    local_embeds=local_hidden_states,
                    tokens_dtype=next_text_tokens.dtype,
                    tokens_device=next_text_tokens.device,
                    local_sampler=local_sampler,
                )

            next_text_tokens = next_text_tokens.reshape(B, 1, 1).expand(
                -1, self.group_size, -1
            )  # [B, group_size, 1]

            # generate speech tokens
            next_tokens = torch.cat(
                (next_text_tokens, next_speech_tokens), dim=-1
            ).reshape(B, -1)  # [B, group_size * (audio_channels + 1)]

            input_ids = torch.cat(
                [input_ids, next_tokens], dim=-1
            )  # [B, T*group_size*vq]

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, None
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        input_ids = input_ids[:B]

        return input_ids
