# Copyright 2023-2025 SGLang Team
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

"""Inference-only FireRedAudio model compatible with HuggingFace weights."""

import logging
import os
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.fireredasr import (
    FireRedAsrConfig,
    FireRedAsrAudioConfig,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, residual_dropout: float = 0.1, dropout_rate: float = 0.1, kernel_size: int = 33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model, residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size, dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        slf_attn_mask: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)
        self.subsampling = 4
        left_context = right_context = 3
        self.context = left_context + 1 + right_context

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x = self.out(x.transpose(1, 2).reshape(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(nn.Module):
    """Positional encoding loaded from weights (nn.Parameter)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros([1, max_len * 2 - 1, d_model])
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T]
        return pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.linear_expand = nn.Linear(d_model, d_model * 4)
        self.nonlinear = nn.SiLU()
        self.dropout_pre = nn.Dropout(dropout_rate)
        self.linear_project = nn.Linear(d_model * 4, d_model)
        self.dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(
            self.pre_layer_norm,
            self.linear_expand,
            self.nonlinear,
            self.dropout_pre,
            self.linear_project,
            self.dropout_post,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        output = self.net(x)
        return output + residual


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33, dropout_rate: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 4, kernel_size=1, bias=False
        )
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model * 2,
            bias=False,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            d_model * 2, d_model, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        out = self.pre_layer_norm(x)
        w1 = self.pointwise_conv1.weight.squeeze(-1)
        out = F.linear(out, w1)
        out = F.glu(out, dim=-1)
        out = out.transpose(1, 2)
        out = self.depthwise_conv(out)
        out = out.transpose(1, 2)
        out = self.batch_norm(out)
        out = self.swish(out)
        w2 = self.pointwise_conv2.weight.squeeze(-1)
        out = self.dropout(F.linear(out, w2))
        if mask is not None:
            out.masked_fill_(mask.transpose(1, 2).ne(1), 0.0)
        return out + residual


class RelPosMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, residual_dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.scale = 1.0 / (self.d_k**0.5)
        self.linear_pos = nn.Linear(d_model, n_head * self.d_k, bias=False)
        pos_bias_u = torch.zeros([n_head, self.d_k])
        pos_bias_v = torch.zeros([n_head, self.d_k])
        self.pos_bias_u = nn.Parameter(pos_bias_u, requires_grad=False)
        self.pos_bias_v = nn.Parameter(pos_bias_v, requires_grad=False)

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(residual_dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        B, H, T1, T2 = x.shape
        x = torch.nn.functional.pad(x, (1, 0))
        x = x.view(B, H, T2 + 1, T1)
        x = x[:, :, 1:, :].reshape(B, H, T1, T2)
        return x[:, :, :, :T1]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sz_b, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        residual = q

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v).transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.permute(0, 2, 3, 1)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p) * self.scale
        matrix_bd = self._rel_shift(matrix_bd)

        attn_mask = matrix_bd
        if mask is not None:
            attn_mask = attn_mask.masked_fill(mask.unsqueeze(1).eq(0), float('-inf'))
        attn_output = F.scaled_dot_product_attention(
            q_with_bias_u, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        
        output = attn_output.permute(0, 2, 1, 3).reshape(sz_b, len_q, -1)
        fc_out = self.fc(output)
        output = self.dropout(fc_out)
        output = output + residual
        return output


class FireRedASREncoder(nn.Module):
    def __init__(self, config: FireRedAsrAudioConfig):
        super().__init__()
        self.encoder_dim = config.odim
        self.input_preprocessor = Conv2dSubsampling(config.idim, config.d_model)
        self.positional_encoding = RelPositionalEncoding(
            config.d_model, config.pe_maxlen
        )
        self.dropout = nn.Dropout(config.residual_dropout)

        self.layer_stack = nn.ModuleList(
            [
                RelPosEmbConformerBlock(
                    config.d_model,
                    config.encoder_attention_heads,
                    config.residual_dropout,
                    config.dropout_rate,
                    config.kernel_size,
                )
                for _ in range(config.encoder_layers)
            ]
        )

    def forward(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor, pad: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pad:
            padded_input = F.pad(
                padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1),
                "constant",
                0.0,
            )
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(
            padded_input, src_mask
        )
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(
        self, padded_input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        N, T = padded_input.size()[:2]
        positions = torch.arange(T, device=padded_input.device)
        positions = positions.unsqueeze(0).expand(N, T)
        lengths = input_lengths.unsqueeze(1).expand(N, T)
        mask = (positions < lengths).unsqueeze(1).to(torch.uint8)
        return mask


class FireRedASRMultiModalProjector(nn.Module):
    def __init__(self, audio_hidden_size: int, text_hidden_size: int, downsample_rate: int = 2):
        super().__init__()
        self.linear1 = nn.Linear(audio_hidden_size * downsample_rate, text_hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(text_hidden_size, text_hidden_size)
        self.ds = downsample_rate

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.reshape(batch_size, seq_len // self.ds, feat_dim * self.ds)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens


class FireRedASRForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "kv_proj": ["k_proj", "v_proj"],
    }
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "llm.": "model.decoder.",
            "encoder.": "model.encoder.audio_encoder.",
            "encoder_projector.": "model.encoder_projector.",
            "net.0": "pre_layer_norm",
            "net.1": "linear_expand",
            "net.4": "linear_project",
        },
    )

    def __init__(
        self,
        config: FireRedAsrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.audio_tower = FireRedASREncoder(config.audio_config)
        self.encoder_projector = FireRedASRMultiModalProjector(
            config.audio_config.d_model,
            config.text_config.hidden_size,
            config.encoder_downsample_rate,
        )
        self.language_model = Qwen2ForCausalLM(
            config.text_config, quant_config, prefix=add_prefix("language_model", prefix)
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Extract audio features from input items
        input_features = torch.cat([item.feature for item in items], dim=0).type(
            self.audio_tower.dtype
        )

        audio_lengths = []
        for item in items:
            if hasattr(item, "audio_length") and item.audio_length is not None:
                audio_lengths.append(item.audio_length)
            else:
                audio_lengths.append(item.feature.shape[1])
        audio_length = torch.tensor(audio_lengths, device=input_features.device)
        encoder_outputs, output_lengths, _ = self.audio_tower(input_features, audio_length)
        audio_embeds, projected_lens = self.encoder_projector(encoder_outputs, output_lengths)

        has_feature_lens = all(
            hasattr(item, "audio_feature_lens") and item.audio_feature_lens is not None
            for item in items
        )
        if has_feature_lens:
            audio_feature_lens = torch.cat([item.audio_feature_lens for item in items])
        else:
            audio_feature_lens = projected_lens

        new_embeds = []
        for length, embed in zip(audio_feature_lens, audio_embeds):
            new_embeds.append(embed[: length.item()])

        return torch.cat(new_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "audio_tower" in name:
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
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = FireRedASRForConditionalGeneration
