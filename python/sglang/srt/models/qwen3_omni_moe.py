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

import dataclasses
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
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.models.qwen3_vl_moe import (
    Qwen3MoeLLMModel,
    Qwen3VLMoeForConditionalGeneration,
    load_fused_expert_weights,
)
from sglang.srt.multimodal.audio_encoder_cuda_graph_runner import (
    GRAPH_ATTENTION_BACKEND,
    GRAPH_SAFE_ATTENTION_BACKENDS,
    AudioEncoderCudaGraphRunner,
)
from sglang.srt.multimodal.mrope_rope_index import (
    _get_feat_extract_output_lengths,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_capability, is_cpu, is_cuda, is_npu, logger

_is_cpu = is_cpu()


def get_head_dim_and_projection_size(
    embed_dim: int,
    num_heads: int,
    original_num_heads: Optional[int] = None,
) -> Tuple[Optional[int], int]:
    if (not _is_cpu) or original_num_heads is None:
        return None, embed_dim

    # On CPU, TP may pad num_heads (e.g. for tp=3/6). In that case we keep the
    # original per-head width (from original_num_heads) and recompute projection_size
    # with padded num_heads, so attention tensor shapes stay TP-friendly while
    # preserving checkpoint semantics.
    head_dim = embed_dim // original_num_heads
    projection_size = num_heads * head_dim
    return head_dim, projection_size


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
        head_dim, projection_size = get_head_dim_and_projection_size(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            original_num_heads=getattr(
                config, "original_encoder_attention_heads", None
            ),
        )
        self.self_attn = VisionAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            head_dim=head_dim,
            projection_size=projection_size,
            # Keep the measured Triton graph default. An explicit multimodal
            # backend override still wins; Hopper FA3 also supports replay.
            qkv_backend=(
                GRAPH_ATTENTION_BACKEND
                if is_cuda()
                and envs.SGLANG_AUDIO_ENCODER_ENABLE_CUDA_GRAPH.get()
                and get_parallel().tp_size == 1
                else None
            ),
            use_qkv_parallel=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        tp_size = get_parallel().tp_size
        use_replicated = config.encoder_ffn_dim % tp_size != 0
        fc1_cls = ReplicatedLinear if use_replicated else ColumnParallelLinear
        fc2_cls = ReplicatedLinear if use_replicated else RowParallelLinear
        self.fc1 = fc1_cls(
            self.embed_dim,
            config.encoder_ffn_dim,
            quant_config=quant_config,
            bias=True,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = fc2_cls(
            config.encoder_ffn_dim,
            self.embed_dim,
            quant_config=quant_config,
            bias=True,
            prefix=f"{prefix}.fc2",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            cu_seqlens: attention window offsets.
            forward_metadata: window metadata prepared once per encoder call,
                so the layers do not read ``cu_seqlens`` back to the host.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            x=hidden_states,
            cu_seqlens=cu_seqlens,
            forward_metadata=forward_metadata,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
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


def _pack_audio_features(
    items: List[MultimodalDataItem],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Pack request-local padded audio features into one ragged batch.

    Audio preprocessing pads each request independently, so items collected
    from different requests may have different time dimensions. Compact every
    sample with its own attention mask before concatenating the valid frames.

    Returns the packed ``[num_mel_bins, frames]`` features, the per-sample
    lengths on ``device`` and the same lengths on the host. Right-padded host
    masks are sliced with host lengths, so packing launches no device-to-host
    copy.
    """
    if not items:
        raise ValueError("Expected at least one audio item")

    parts = []
    lengths: List[int] = []
    num_mel_bins = None

    for item_idx, item in enumerate(items):
        features = item.feature
        attention_mask = getattr(item, "feature_attention_mask", None)

        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            shape = getattr(features, "shape", None)
            raise ValueError(
                f"Audio item {item_idx} features must have shape "
                f"[batch, mel, time], got {shape}"
            )
        if num_mel_bins is None:
            num_mel_bins = features.shape[1]
        elif features.shape[1] != num_mel_bins:
            raise ValueError(
                f"Audio item {item_idx} mel dimension mismatch: expected "
                f"{num_mel_bins}, got {features.shape[1]}"
            )

        if attention_mask is None:
            # every frame is valid
            lengths.extend([features.shape[2]] * features.shape[0])
            parts.append(features.permute(1, 0, 2).reshape(num_mel_bins, -1))
            continue

        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            shape = getattr(attention_mask, "shape", None)
            raise ValueError(
                f"Audio item {item_idx} attention mask must have shape "
                f"[batch, time], got {shape}"
            )
        if features.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                f"Audio item {item_idx} batch size mismatch: features have "
                f"{features.shape[0]} samples, mask has {attention_mask.shape[0]}"
            )
        if features.shape[2] != attention_mask.shape[1]:
            raise ValueError(
                f"Audio item {item_idx} time dimension mismatch: features have "
                f"{features.shape[2]} frames, mask has {attention_mask.shape[1]}"
            )

        mask = attention_mask.to(torch.bool)
        item_lengths = mask.sum(dim=-1).tolist()
        lengths.extend(item_lengths)
        right_padded = not mask.is_cuda and all(
            bool(row[:n].all()) for row, n in zip(mask, item_lengths)
        )
        if right_padded:
            parts.extend(sample[:, :n] for sample, n in zip(features, item_lengths))
        else:
            selected = features.to(device=device, dtype=dtype).permute(0, 2, 1)
            parts.append(selected[mask.to(device)].permute(1, 0))

    if all(part.device.type == "cpu" for part in parts):
        packed = torch.cat(parts, dim=1).to(device=device, dtype=dtype)
    else:
        packed = torch.cat(
            [part.to(device=device, dtype=dtype) for part in parts], dim=1
        )
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    if torch.device(device).type == "cuda":
        lengths_tensor = lengths_tensor.pin_memory()
    return packed, lengths_tensor.to(device, non_blocking=True), lengths


@dataclasses.dataclass
class _AudioWindowPlan:
    chunk_lengths: List[int]
    chunk_out_lens: List[int]
    max_len_after_cnn: int
    cu_seqlens: List[int]
    max_seqlen: int


class Qwen3OmniMoeAudioEncoder(PreTrainedModel):
    config: Qwen3OmniMoeAudioEncoderConfig

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig, quant_config=None):
        super().__init__(config)

        embed_dim = config.d_model
        self.max_source_positions = config.max_source_positions
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
        conv_out_dim = config.downsample_hidden_size * (
            (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        )
        self.conv_out = ReplicatedLinear(
            conv_out_dim,
            config.d_model,
            bias=False,
            quant_config=quant_config,
        )
        self.proj1 = ReplicatedLinear(
            config.d_model, config.d_model, quant_config=quant_config
        )
        self.act = ACT2FN[config.activation_function]
        self.proj2 = ReplicatedLinear(
            config.d_model, config.output_dim, quant_config=quant_config
        )
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize
        self._cuda_graph_runner: Optional[AudioEncoderCudaGraphRunner] = None
        self._cuda_graph_disabled = False

    def plan_windows(self, feature_lens: List[int]) -> "_AudioWindowPlan":
        """Plan conv chunks and attention windows from host-side mel lengths.

        Matches the device-side math this encoder used before, without
        reading any length back from the GPU.
        """
        chunk = self.n_window * 2
        chunk_lengths: List[int] = []
        for length in feature_lens:
            num_chunks = -(-length // chunk)
            tail = length % chunk or chunk
            chunk_lengths.extend([chunk] * (num_chunks - 1) + [tail])
        chunk_out_lens = [_get_feat_extract_output_lengths(c) for c in chunk_lengths]
        max_len_after_cnn = max(chunk_out_lens, default=0)
        window = max_len_after_cnn * (self.n_window_infer // chunk)
        cu_seqlens = [0]
        max_seqlen = 0
        for length in feature_lens:
            cnn_len = _get_feat_extract_output_lengths(length)
            num_full, remainder = divmod(cnn_len, window)
            for size in [window] * num_full + ([remainder] if remainder else []):
                cu_seqlens.append(cu_seqlens[-1] + size)
                max_seqlen = max(max_seqlen, size)
        return _AudioWindowPlan(
            chunk_lengths=chunk_lengths,
            chunk_out_lens=chunk_out_lens,
            max_len_after_cnn=max_len_after_cnn,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    def embed_audio_features(
        self, input_features: torch.Tensor, plan: "_AudioWindowPlan"
    ) -> torch.Tensor:
        """Conv frontend + positional embedding, packed to valid tokens."""
        chunk_list = input_features.T.split(plan.chunk_lengths, dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)
        padded_feature = padded_feature.unsqueeze(1)

        # Add fast path + chunk normal path
        if padded_feature.size(0) <= self.conv_chunksize:
            padded_embed = F.gelu(self.conv2d1(padded_feature))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
        else:
            padded_embeds = []
            for chunk in padded_feature.split(self.conv_chunksize, dim=0):
                x = F.gelu(self.conv2d1(chunk))
                x = F.gelu(self.conv2d2(x))
                x = F.gelu(self.conv2d3(x))
                padded_embeds.append(x)
            padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        )[0]

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        # Valid rows of each chunk, gathered with host-planned indices instead
        # of a boolean mask (which would sync to size its output).
        row_stride = padded_embed.shape[1]
        valid_rows = torch.from_numpy(
            np.concatenate(
                [
                    np.arange(n, dtype=np.int64) + i * row_stride
                    for i, n in enumerate(plan.chunk_out_lens)
                ]
            )
        )
        if padded_embed.is_cuda:
            valid_rows = valid_rows.pin_memory()
        valid_rows = valid_rows.to(padded_embed.device, non_blocking=True)
        return padded_embed.reshape(b * row_stride, -1).index_select(0, valid_rows)

    def forward_transformer(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: Optional[VisionAttentionMetadata],
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encoder layers and output projection."""
        if cu_seqlens is None:
            cu_seqlens = forward_metadata.cu_seqlens
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                forward_metadata=forward_metadata,
            )[0]
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)[0]
        return hidden_states

    def _get_cuda_graph_runner(
        self, hidden_states: torch.Tensor
    ) -> Optional[AudioEncoderCudaGraphRunner]:
        if self._cuda_graph_runner is not None or self._cuda_graph_disabled:
            return self._cuda_graph_runner
        backend = getattr(self.layers[0].self_attn, "qkv_backend_name", None)
        if (
            not envs.SGLANG_AUDIO_ENCODER_ENABLE_CUDA_GRAPH.get()
            or not is_cuda()
            or not hidden_states.is_cuda
            or backend not in GRAPH_SAFE_ATTENTION_BACKENDS
            or (backend == "fa3" and get_device_capability()[0] != 9)
            or get_parallel().tp_size != 1
        ):
            self._cuda_graph_disabled = True
            return None
        chunk = self.n_window * 2
        window = _get_feat_extract_output_lengths(chunk) * (
            self.n_window_infer // chunk
        )
        self._cuda_graph_runner = AudioEncoderCudaGraphRunner(
            lambda x, metadata: self.forward_transformer(x, metadata),
            hidden_size=hidden_states.shape[-1],
            window_size=window,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return self._cuda_graph_runner

    def forward(
        self,
        input_features,
        feature_lens=None,
        feature_lens_cpu: Optional[List[int]] = None,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        feature_lens_cpu (`List[int]`, *optional*):
            host copy of ``feature_lens``; avoids reading it back from the GPU
        """
        if feature_lens_cpu is None:
            feature_lens_cpu = feature_lens.tolist()
        plan = self.plan_windows(feature_lens_cpu)
        hidden_states = self.embed_audio_features(input_features, plan)

        runner = self._get_cuda_graph_runner(hidden_states)
        if runner is not None:
            budget = runner.select_budget(
                hidden_states.shape[0], len(plan.cu_seqlens) - 1
            )
            if budget is not None:
                output = runner.run(hidden_states, plan.cu_seqlens, budget)
                if output is not None:
                    return BaseModelOutput(last_hidden_state=output)

        cu_seqlens = torch.tensor(plan.cu_seqlens, dtype=torch.int32)
        # cu_seqlens must be on cpu because of npu_flash_attention_unpad operator restriction
        if is_npu():
            forward_metadata = None
        else:
            forward_metadata = prepare_vision_attention_metadata(
                cu_seqlens.pin_memory() if hidden_states.is_cuda else cu_seqlens,
                hidden_states.device,
                max_seqlen=plan.max_seqlen,
            )
            cu_seqlens = forward_metadata.cu_seqlens
        hidden_states = self.forward_transformer(
            hidden_states, forward_metadata, cu_seqlens=cu_seqlens
        )
        return BaseModelOutput(last_hidden_state=hidden_states)


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
        self.ln_q = nn.LayerNorm(
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
        self.audio_tower = Qwen3OmniMoeAudioEncoder(config.audio_config, quant_config)
        self.visual = Qwen3OmniMoeVisionEncoder(
            config.vision_config,
            quant_config=quant_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            prefix=add_prefix("visual", prefix),
        )

    def get_audio_feature(self, items: List[MultimodalDataItem]):
        device = next(self.audio_tower.parameters()).device
        input_features, feature_lens, feature_lens_cpu = _pack_audio_features(
            items,
            device=device,
            dtype=self.audio_tower.dtype,
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=feature_lens,
            feature_lens_cpu=feature_lens_cpu,
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

        # Pre-define `params_dict` to avoid repeated expensive traversal of model parameters.
        params_dict = dict(self.named_parameters())

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
                        if name_mapped in params_dict.keys():
                            param = params_dict[name_mapped]
                        else:
                            continue
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
