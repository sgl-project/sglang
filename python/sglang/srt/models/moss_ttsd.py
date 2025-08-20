import logging
from typing import Iterable, Optional, Tuple

import torch
from transformers.models.qwen3 import Qwen3Config

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix, remove_prefix

logger = logging.getLogger(__name__)


class MOSSTTSDConfig(Qwen3Config):
    def __init__(
        self,
        channels=8,
        speech_pad_token=1024,
        speech_token_range=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.speech_pad_token = speech_pad_token
        self.speech_token_range = speech_token_range


class MOSSTTSD(torch.nn.Module):
    # BitandBytes specific attributes
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(
        self,
        config: MOSSTTSDConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "model",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        # 创建多通道嵌入层
        self.embedding_list = torch.nn.ModuleList([])
        if self.pp_group.is_first_rank or (
            self.config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            for i in range(self.config.channels):
                self.embedding_list.append(
                    VocabParallelEmbedding(
                        self.config.vocab_size_list[i],
                        self.config.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"embedding_list.{i}", prefix),
                    )
                )
        else:
            for _ in range(self.config.channels):
                self.embedding_list.append(PPMissingLayer())

        # 核心语言模型
        self.language_model = Qwen3Model(
            config=self.config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        # 多通道语言模型头
        self.lm_heads = torch.nn.ModuleList([])
        if self.pp_group.is_last_rank:
            if self.config.tie_word_embeddings:
                self.lm_heads = self.embedding_list
            else:
                for i in range(self.config.channels):
                    self.lm_heads.append(
                        ParallelLMHead(
                            num_embeddings=self.config.vocab_size_list[i],
                            embedding_dim=self.config.hidden_size,
                            prefix=add_prefix(prefix, f"lm_heads.{i}"),
                        )
                    )
        else:
            for _ in range(self.config.channels):
                self.lm_heads.append(PPMissingLayer())

        # 多通道logits处理器
        for i in range(self.config.channels):
            self.logits_processors.append(
                LogitsProcessor(self.config, self.config.vocab_size_list[i], channel=i)
            )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_embed = torch.sum(
            torch.stack(
                [
                    embed_layer(input_ids[..., i])
                    for i, embed_layer in enumerate(self.embedding_list)
                ]
            ),
            dim=0,
        )
        return input_embed

    def _prepare_multi_modal_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Prepares multi-modal embeddings from input_ids.
        Input can be either:
        - 1D tensor: flattened multi-channel input (needs reshaping)
        - 2D tensor: (seq_length, channels) format
        For channel 0: text + speech tokens, for channels 1 to channels-1: speech tokens padded with speech_pad_token.
        """

        # Handle different input shapes
        if input_ids.dim() == 1:
            # Flattened input - need to reshape to (seq_length, channels)
            total_tokens = input_ids.shape[0]
            channels = self.config.channels

            if total_tokens % channels != 0:
                # For non-multi-channel inputs (like health checks), treat as single channel
                # and pad to create multi-channel format
                seq_length = total_tokens
                # Create multi-channel format: first channel = input, others = pad tokens
                pad_token_id = getattr(self.config, "speech_pad_token_id", 1024)
                input_ids_2d = torch.full(
                    (seq_length, channels),
                    pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids_2d[:, 0] = input_ids  # First channel gets the actual input
            else:
                # Normal multi-channel input
                seq_length = total_tokens // channels
                input_ids_2d = input_ids.view(seq_length, channels)

        elif input_ids.dim() == 2:
            # Already in correct shape
            seq_length, channels = input_ids.shape
            input_ids_2d = input_ids
        else:
            raise ValueError(
                f"Expected input_ids to be 1D or 2D tensor, got {input_ids.dim()}D tensor with shape {input_ids.shape}"
            )

        # Update input_ids to use the processed version
        input_ids = input_ids_2d
        seq_length, channels = input_ids.shape
        if channels != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} channels, got {channels}"
            )

        # Get the weight tensor's dtype safely
        if hasattr(self.embedding_list[0], "weight") and isinstance(
            self.embedding_list[0].weight, torch.Tensor
        ):
            dtype = self.embedding_list[0].weight.dtype
        else:
            dtype = torch.float32  # Default fallback dtype

        inputs_embeds = torch.zeros(
            seq_length,
            self.config.hidden_size,
            device=input_ids.device,
            dtype=dtype,
        )
        for i in range(channels):
            embed_layer = self.embedding_list[i]
            channel_input = input_ids[..., i]
            inputs_embeds += embed_layer(channel_input)

        return inputs_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_ids is not None:
            input_embeds = self._prepare_multi_modal_inputs(input_ids)

        # 步骤1: 获取输入嵌入
        if input_embeds is not None:
            # 直接使用提供的嵌入
            hidden_states = input_embeds
        elif self.pp_group.is_first_rank:
            # 第一个rank：从input_ids计算嵌入
            hidden_states = self.get_input_embeddings(input_ids)
        else:
            # 非第一个rank：嵌入将通过pp_proxy_tensors从上一个rank传递
            # 这里设置为None，让language_model处理
            hidden_states = None

        # 步骤2: 通过语言模型进行前向传播
        hidden_states = self.language_model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=hidden_states,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # 步骤3: 在最后一个rank上计算多通道输出
        if self.pp_group.is_last_rank:
            # 多通道输出处理
            channel_outputs = []
            for i in range(self.config.channels):
                # 为每个通道计算logits
                channel_logits = self.logits_processors[i](
                    None,
                    hidden_states=hidden_states,
                    lm_head=self.lm_heads[i],
                    logits_metadata=forward_batch,
                )
                channel_outputs.append(channel_logits)

            return channel_outputs
        else:
            # 非最后一个rank，返回隐藏状态供下一个rank使用
            return hidden_states

    @property
    def start_layer(self):
        return self.language_model.start_layer

    @property
    def end_layer(self):
        return self.language_model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = remove_prefix(name, "model.")
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.language_model, "start_layer")
                and (
                    layer_id < self.language_model.start_layer
                    or layer_id >= self.language_model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # Handle pp weight tying here
                    # find the embed_tokens.weight in the weights
                    embed_token_weights = next(
                        filter(
                            lambda x: x[0] == "language_model.embed_tokens.weight",
                            weights,
                        )
                    )[1]
                    loaded_weight = embed_token_weights
                else:
                    continue
            if (
                name.startswith("language_model.vision_tower")
                and name not in params_dict
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        # 返回所有通道的嵌入层和头部权重
        embed_weights = []
        head_weights = []

        # 获取所有嵌入层权重
        if self.pp_group.is_first_rank or (
            self.config.tie_word_embeddings and self.pp_group.is_last_rank
        ):
            for i in range(self.config.channels):
                if hasattr(self.embedding_list[i], "weight"):
                    embed_weights.append(self.embedding_list[i].weight)
                else:
                    embed_weights.append(None)
        else:
            embed_weights = [None] * self.config.channels

        # 获取所有头部权重
        if self.pp_group.is_last_rank:
            for i in range(self.config.channels):
                if hasattr(self.lm_heads[i], "weight"):
                    head_weights.append(self.lm_heads[i].weight)
                else:
                    head_weights.append(None)
        else:
            head_weights = [None] * self.config.channels

        return embed_weights, head_weights

    def set_embed_and_head(self, embed_list, head_list):
        # 设置所有通道的嵌入层和头部权重
        if embed_list is not None and len(embed_list) == self.config.channels:
            if self.pp_group.is_first_rank or (
                self.config.tie_word_embeddings and self.pp_group.is_last_rank
            ):
                for i, embed in enumerate(embed_list):
                    if embed is not None and hasattr(self.embedding_list[i], "weight"):
                        if hasattr(self.embedding_list[i], "weight"):
                            del self.embedding_list[i].weight
                        self.embedding_list[i].weight = embed

        if head_list is not None and len(head_list) == self.config.channels:
            if self.pp_group.is_last_rank:
                for i, head in enumerate(head_list):
                    if head is not None and hasattr(self.lm_heads[i], "weight"):
                        if hasattr(self.lm_heads[i], "weight"):
                            del self.lm_heads[i].weight
                        self.lm_heads[i].weight = head

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.language_model.load_kv_cache_scales(quantization_param_path)


EntryClass = MOSSTTSD
