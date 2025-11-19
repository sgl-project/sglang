# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Inference-only Qwen3-VL model compatible with HuggingFace weights."""
import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput

from sglang.srt.configs.qwen3_omni import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from sglang.srt.configs.qwen3_vl import Qwen3VLMoeConfig
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.models.qwen3_vl_moe import (
    Qwen3MoeLLMModel,
    Qwen3VLMoeForConditionalGeneration,
    load_fused_expert_weights,
)
from sglang.srt.utils import add_prefix, logger


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3OmniMoeAudioEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        embed_dim = config.d_model
        self.embed_dim = config.d_model
        self.self_attn = VisionAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            projection_size=embed_dim,
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            x=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = (
            torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        )
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


def _get_feat_extract_output_lengths(input_lengths):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class Qwen3OmniMoeAudioEncoder(PreTrainedModel):
    config: Qwen3OmniMoeAudioEncoderConfig

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = nn.ModuleList(
            [
                Qwen3OmniMoeAudioEncoderLayer(config)
                for _ in range(config.encoder_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            2,
            padding=1,
        )
        self.conv_out = nn.Linear(
            config.downsample_hidden_size
            * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(length, dtype=torch.bool, device=padded_feature.device)
                for length in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (
            self.n_window_infer // (self.n_window * 2)
        )
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(
            -1, dtype=torch.int32
        )

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class Qwen3OmniMoeVisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_postshuffle_norm=False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = RMSNorm(
            self.hidden_size if use_postshuffle_norm else context_dim, eps=1e-6
        )
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.0", prefix),
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=quant_config,
                    prefix=add_prefix("mlp.2", prefix),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            x.view(-1, self.hidden_size)
            if self.use_postshuffle_norm
            else x.view(-1, x.shape[-1])
        )
        hidden = self.ln_q(x).view(-1, self.hidden_size)
        for layer in self.mlp:
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = layer(hidden)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        return hidden


class Qwen3OmniMoeVisionEncoder(Qwen3VLMoeVisionModel):
    config: Qwen3OmniMoeVisionEncoderConfig

    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = None,
        **kwargs,
    ):
        super().__init__(
            vision_config=config,
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        self.merger = Qwen3OmniMoeVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            quant_config=quant_config,
            use_postshuffle_norm=False,
            prefix=add_prefix("merger", prefix),
        )
        self.merger_list = nn.ModuleList(
            [
                Qwen3OmniMoeVisionPatchMerger(
                    dim=config.out_hidden_size,
                    context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                    use_postshuffle_norm=True,
                    quant_config=quant_config,
                    prefix=add_prefix("merger_list", prefix),
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )
        del self.deepstack_merger_list

    @property
    def deepstack_merger_list(self):
        return self.merger_list

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device


class Qwen3OmniMoeThinkerForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    config: Qwen3OmniMoeThinkerConfig

    def __init__(
        self,
        config: Qwen3OmniMoeThinkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            config, quant_config, prefix, language_model_cls=Qwen3MoeLLMModel
        )
        self.audio_tower = Qwen3OmniMoeAudioEncoder(config.audio_config)
        self.visual = Qwen3OmniMoeVisionEncoder(
            config.vision_config,
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("visual", prefix),
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def get_audio_feature(self, items: List[MultimodalDataItem]):
        feature_attention_mask = torch.cat(
            [item.feature_attention_mask for item in items], dim=0
        ).type(torch.long)
        input_features = (
            torch.cat([item.feature for item in items])
            .type(self.audio_tower.dtype)
            .to(next(self.audio_tower.parameters()).device)
        )
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[
                feature_attention_mask.bool()
            ].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = (
            audio_feature_lengths
            if audio_feature_lengths is not None
            else feature_attention_mask.sum(-1)
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
        )
        audio_features = audio_outputs.last_hidden_state

        return audio_features


class Qwen3OmniMoeForConditionalGeneration(PreTrainedModel):
    def __init__(
        self,
        config: Qwen3VLMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config)
        self.config = config

        self.thinker = Qwen3OmniMoeThinkerForConditionalGeneration(
            config.thinker_config, quant_config=quant_config, prefix=prefix
        )
        self.enable_talker = False
        self.pad_input_ids = self.thinker.pad_input_ids
        self.forward = self.thinker.forward

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict

        for name, loaded_weight in weights:
            name = name.replace(r"model.language_model.", r"model.")

            if ("talker" in name or "code2wav" in name) and not self.enable_talker:
                continue

            name = name.replace(".self_attn.out_proj", ".self_attn.proj")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue

                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                # [TODO] Skip layers that are on other devices (check if sglang has a similar function)
                # if is_pp_missing_parameter(name, self):
                #     continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or "audio_tower" in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        loaded_weight = loaded_weight.transpose(-1, -2)  # no bias
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue
                    if "visual" in name or "audio_tower" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")
                        name = name.replace(r"attn.out_proj.", r"attn.proj.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            f"Loaded weight with {name=} not found in params_dict"
                        )


EntryClass = Qwen3OmniMoeForConditionalGeneration
