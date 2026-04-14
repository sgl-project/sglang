# Adapted from qwen2.py and mimo.py for MiMoAudio compatibility
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as Qwen2ModelHF

from sglang.srt.configs.mimo_audio import MiMoAudioConfig
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model as Qwen2ModelSGLang
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


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


class MiMoAudioForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: MiMoAudioConfig | Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = (
            MiMoAudioConfig(**vars(config))
            if isinstance(config, Qwen2Config)
            else config
        )
        self.config = config
        self.quant_config = quant_config
        self.args = MiMoAudioArguments(
            model_name_or_path="",
            sosp_idx=151665,
            eosp_idx=151666,
            empty_idx=151667,
            sostm_idx=151670,
            eostm_idx=151671,
            eot_idx=151672,
        )

        # 1. Main Language Model (using SGLang's efficient Qwen2)
        self.model = Qwen2ModelSGLang(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        self.speech_vocab_sizes = config.parsed_speech_vocab_sizes()
        self.speech_empty_ids = config.parsed_speech_empty_ids()
        self.delay_pattern = config.parsed_delay_pattern()
        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

        # Construct local transformer
        self.local_config = config.local_config()
        # self.local_transformer = Qwen2ModelHF(self.local_config, prefix="local_transformer")
        self.local_transformer = Qwen2ModelHF(self.local_config)
        self.local_transformer.embed_tokens = None

        # Add input local transformer if configured
        self.input_local_config = config.input_local_config()
        # self.input_local_transformer = Qwen2ModelHF(self.input_local_config, prefix="input_local_transformer")
        self.input_local_transformer = Qwen2ModelHF(self.input_local_config)
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

        self.speech_group_downcast = nn.Linear(
            self.input_local_config.hidden_size * self.group_size,
            config.hidden_size,
            bias=False,
        )

        self.hidden_states_downcast = nn.Linear(
            config.hidden_size,
            self.local_config.hidden_size,
            bias=False,
        )

        # Processors
        self.logits_processor = LogitsProcessor(config)
        # self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor):
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Process each group independently: [B*T//group_size, group_size, hidden_size]
        input_embeddings = speech_embeddings.reshape(
            B * T_groups, group_size, hidden_size
        )

        output = self.input_local_transformer(
            inputs_embeds=input_embeddings,
            return_dict=True,
            is_causal=False,
            # is_causal=not self.config.input_full_attention,  # for SDPA
        )
        encoded_embeddings = output.last_hidden_state
        # encoded_embeddings = output[0]      # hidden_state

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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        # forward_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        input_ids = input_ids.view(1, -1, 9).transpose(
            -1, -2
        )  # [B, audio_channels + 1, new_T]
        input_embeds = self._prepare_input_embeds(input_ids)
        # forward_batch.num_token_non_padded_cpu=input_embeds.shape[1]
        input_ids = None
        # positions = None
        input_embeds = input_embeds.squeeze(0)  # 去掉 第一个 B 维度
        # positions = torch.arange(input_embeds.shape[0]).to("cuda") # 暂时添加 positions
        outputs = self.model(input_ids, positions, forward_batch, input_embeds)

        hidden_states = (
            outputs  # outputs is last_hidden_state shape [T_group, hidden_dim]
        )

        # text_logits: torch.Tensor = self.lm_head(
        #     hidden_states[-1:, :]
        # )  # [1, vocab_size]
        text_logits = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )  # text_logits.next_token_logits shape [1, vocab_size]
        shift_hidden_states: torch.Tensor = self.hidden_states_downcast(
            hidden_states[-1:, :]
        )  # [1, hidden_size]
        forward_batch.shift_hidden_states = shift_hidden_states

        return text_logits

    def sample(
        self,
        forward_batch: ForwardBatch,
        sample_func: Callable,
        text_logits_output: LogitsProcessorOutput,
    ):
        global_sampler = MiMoSampler()
        text_logits: torch.Tensor = (
            text_logits_output.next_token_logits
        )  # [B. vocab_size]
        next_text_tokens = global_sampler.sample(text_logits)  # [B]

        local_hidden_states = forward_batch.shift_hidden_states  # [B, hidden_size]

        B = 1
        # Only Support batch_size=1 here
        if next_text_tokens[0] != self.args.empty_idx:
            # text token
            zero_embed_tensor = torch.tensor(
                self.speech_empty_ids,
                device=next_text_tokens.device,
                dtype=forward_batch.input_ids.dtype,
            )
            next_speech_tokens = zero_embed_tensor.view(
                1, 1, self.audio_channels
            ).expand(1, self.config.group_size, -1)
        else:
            # audio token
            next_speech_tokens = self.local_forward(
                local_embeds=local_hidden_states,
                tokens_dtype=next_text_tokens.dtype,
                tokens_device=next_text_tokens.device,
            )
        next_text_tokens = next_text_tokens.reshape(B, 1, 1).expand(
            -1, self.group_size, -1
        )  # [B, group_size, 1]

        # generate speech tokens
        next_tokens = torch.cat((next_text_tokens, next_speech_tokens), dim=-1).reshape(
            B, -1
        )  # [B, group_size * (audio_channels + 1)]

        # input_ids = torch.cat(
        #     [input_ids, next_tokens], dim=-1
        # )  # [B, T*group_size*vq]
        return next_tokens  # [B, group_size * (audio_channels + 1)]

    def local_forward(
        self,
        local_embeds: torch.FloatTensor,  # [B, 1, hidden_size]
        tokens_dtype: torch.dtype,
        tokens_device: torch.device,
        local_sampler: MiMoSampler | None = None,
    ):
        local_embeds = local_embeds.unsqueeze(0)  # [B, 1, hidden_size]
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(
            self.named_parameters()
        )  # params_dict 是 当前模型所有参数字典
        for (
            name,
            loaded_weight,
        ) in (
            weights
        ):  # name 是权重文件中参数名 loaded_weight 是权重文件中参数对应权重值
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            # Route mapping correctly
            is_mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                mapped_name = name.replace(weight_name, param_name)

                # Check mapping for nested transformers
                if mapped_name not in params_dict:
                    continue

                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                # Apply Sharding only to main SGLang Qwen2 Model
                # 不需要这样做 local_transformer 名字已经统一了
                # if "local_transformer" in mapped_name:
                #     # HF models in SGLang context expect full weights or standard dict copy
                #     param.data.copy_(loaded_weight)
                # else:
                weight_loader(param, loaded_weight, shard_id)
                is_mapped = True
                break

            if not is_mapped:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                if (
                    "local_transformer" in name
                    or "speech_embeddings" in name
                    or "hidden_states_downcast" in name
                ):
                    param.data.copy_(loaded_weight)
                else:
                    weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


class MiMoAudioModel(MiMoAudioForCausalLM):
    def __init__(
        self,
        config: MiMoAudioConfig | Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)


EntryClass = MiMoAudioModel
