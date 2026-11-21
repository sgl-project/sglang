# Copyright 2026 SGLang Team
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
"""GigaChat3.5-Omni: the GigaChat3.5 LM with InternViT vision and a conformer audio encoder."""

import math
from array import array
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.gigachat35_omni import (
    GigaChat35OmniAudioConfig,
    GigaChat35OmniConfig,
)
from sglang.srt.layers.attention.vision import (
    QKV_BACKEND_IMPL,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.managers.mm_utils import general_mm_embed_routine, pad_mm_tokens
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gigachat35 import GigaChat35ForCausalLM
from sglang.srt.models.internvl import InternVisionModel
from sglang.srt.multimodal.gigachat35_omni import GigaChat35OmniVideoEmbedding
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_mm
from sglang.srt.utils import get_device_capability, is_cuda


class GigaChat35OmniConformerSubsampling(nn.Module):
    """Stride-2 convolutions over mel frames, `subsampling_factor` times shorter output."""

    def __init__(self, config: GigaChat35OmniAudioConfig) -> None:
        super().__init__()
        num_stages = int(math.log2(config.subsampling_factor))
        layers = []
        in_channels = config.num_mel_bins

        for stage in range(num_stages):
            out_channels = (
                config.hidden_size
                if stage == num_stages - 1
                else config.subsampling_conv_channels
            )
            layers.append(nn.Conv1d(in_channels, out_channels, 5, stride=2, padding=2))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Subsamples one clip.

        Args:
            mel: Mel frames, `[num_mel_frames, num_mel_bins]`.

        Returns:
            Frames of shape `[num_frames, hidden_size]`.
        """
        return self.conv(mel.t().unsqueeze(0)).squeeze(0).t()


class GigaChat35OmniConformerFeedForward(nn.Module):
    """Conformer feed-forward block."""

    def __init__(self, config: GigaChat35OmniAudioConfig) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block.

        Args:
            x: Frames, `[num_frames, hidden_size]`.

        Returns:
            Frames of the same shape.
        """
        return self.linear2(F.silu(self.linear1(x)))


class GigaChat35OmniConformerConvolution(nn.Module):
    """Conformer convolution block: GLU, depthwise conv, LayerNorm over channels, SiLU."""

    def __init__(self, config: GigaChat35OmniAudioConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1)
        self.depthwise_conv = nn.Conv1d(
            d,
            d,
            config.conv_kernel_size,
            padding=(config.conv_kernel_size - 1) // 2,
            groups=d,
        )
        self.norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the block.

        Args:
            x: Frames, `[num_frames, hidden_size]`.

        Returns:
            Frames of the same shape.
        """
        x = F.glu(self.pointwise_conv1(x.t().unsqueeze(0)), dim=1)
        x = self.depthwise_conv(x)
        x = F.silu(self.norm(x.transpose(1, 2)))
        return self.pointwise_conv2(x.transpose(1, 2)).squeeze(0).t()


class GigaChat35OmniConformerAttention(nn.Module):
    """Self-attention inside `attention_chunk_size`-frame blocks.

    RoPE rotates the block's frames before the q / k projections, not after.
    """

    def __init__(self, config: GigaChat35OmniAudioConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = d // self.num_heads
        self.linear_q = nn.Linear(d, d)
        self.linear_k = nn.Linear(d, d)
        self.linear_v = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.attn = QKV_BACKEND_IMPL[self._select_backend()](
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            dropout=0.0,
            flatten_batch=True,
            softmax_in_single_precision=False,
            softmax_scale=self.head_dim**-0.5,
        )

    @staticmethod
    def _select_backend() -> str:
        """Picks the multimodal attention backend like the other encoders do.

        Returns:
            A `QKV_BACKEND_IMPL` key.
        """
        override = get_mm().mm_attention_backend

        if override is not None:
            return override

        if is_cuda():
            major, _ = get_device_capability()
            return "fa3" if major == 9 else "triton_attn"

        return "sdpa"

    def _rotate(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Applies RoPE (rotate-half) per head in fp32.

        Args:
            x: Frames, `[num_frames, hidden_size]`.
            cos: Cosines, `[num_frames, head_dim // 2]`.
            sin: Sines, `[num_frames, head_dim // 2]`.

        Returns:
            Rotated frames in the input dtype.
        """
        x1, x2 = x.view(-1, self.num_heads, self.head_dim).float().chunk(2, dim=-1)
        cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(1).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        """Applies the block.

        Args:
            x: Frames, `[num_frames, hidden_size]`.
            cos: Cosines in the frames' dtype, `[num_frames, head_dim // 2]`.
            sin: Sines in the frames' dtype, `[num_frames, head_dim // 2]`.
            metadata: Attention block boundaries of the clip.

        Returns:
            Frames of the same shape.
        """
        rotated = self._rotate(x, cos, sin)
        q = self.linear_q(rotated).view(-1, self.num_heads, self.head_dim)
        k = self.linear_k(rotated).view(-1, self.num_heads, self.head_dim)
        v = self.linear_v(x).view(-1, self.num_heads, self.head_dim)
        out = self.attn.forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_seqlens,
            bsz=1,
            seq_len=x.shape[0],
            max_seqlen=metadata.max_seqlen,
            softmax_scale=self.head_dim**-0.5,
            forward_metadata=metadata,
        )
        return self.linear_out(out.reshape(x.shape))


class GigaChat35OmniConformerLayer(nn.Module):
    """Conformer block: half FF, attention, convolution, half FF, LayerNorm."""

    def __init__(self, config: GigaChat35OmniAudioConfig) -> None:
        super().__init__()
        d, eps = config.hidden_size, config.layer_norm_eps
        self.norm_feed_forward1 = nn.LayerNorm(d, eps=eps)
        self.feed_forward1 = GigaChat35OmniConformerFeedForward(config)
        self.norm_self_att = nn.LayerNorm(d, eps=eps)
        self.self_attn = GigaChat35OmniConformerAttention(config)
        self.norm_conv = nn.LayerNorm(d, eps=eps)
        self.conv = GigaChat35OmniConformerConvolution(config)
        self.norm_feed_forward2 = nn.LayerNorm(d, eps=eps)
        self.feed_forward2 = GigaChat35OmniConformerFeedForward(config)
        self.norm_out = nn.LayerNorm(d, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        """Applies the block to one clip.

        Args:
            x: Frames, `[num_frames, hidden_size]`.
            cos: Cosines, `[num_frames, head_dim // 2]`.
            sin: Sines, `[num_frames, head_dim // 2]`.
            metadata: Attention block boundaries of the clip.

        Returns:
            Frames of the same shape.
        """
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), cos, sin, metadata)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class GigaChat35OmniConformerEncoder(nn.Module):
    """Conformer audio encoder, one clip per call.

    Clips longer than `_MAX_WINDOW_FRAMES` frames run in windows, so the
    activation memory of a multi-hour clip stays bounded.
    """

    # 32 min at 25 frames/s, a whole number of attention blocks.
    _MAX_WINDOW_FRAMES = 48000

    def __init__(self, config: GigaChat35OmniAudioConfig) -> None:
        super().__init__()
        self.chunk_size = config.attention_chunk_size
        assert self._MAX_WINDOW_FRAMES % self.chunk_size == 0
        # Attention stays inside a block and the depthwise conv reaches only
        # into the neighbouring block, so each layer widens a frame's context
        # by at most one block per side.
        self.halo = config.num_hidden_layers * self.chunk_size
        self.pre_encode = GigaChat35OmniConformerSubsampling(config)
        self.layers = nn.ModuleList(
            GigaChat35OmniConformerLayer(config)
            for _ in range(config.num_hidden_layers)
        )
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        freqs = torch.outer(
            torch.arange(self.chunk_size, dtype=torch.float32), inv_freq
        )
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encodes one clip.

        Args:
            mel: Mel frames, `[num_mel_frames, num_mel_bins]`.

        Returns:
            Frames of shape `[num_frames, hidden_size]`.
        """
        x = self.pre_encode(mel)
        num_frames = x.shape[0]
        outputs = []

        for start in range(0, num_frames, self._MAX_WINDOW_FRAMES):
            end = min(start + self._MAX_WINDOW_FRAMES, num_frames)
            lo, hi = max(0, start - self.halo), min(num_frames, end + self.halo)
            outputs.append(self.encode_frames(x[lo:hi])[start - lo : end - lo])

        return torch.cat(outputs)

    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the conformer layers on frames that start an attention block.

        Args:
            x: Subsampled frames, `[num_frames, hidden_size]`.

        Returns:
            Frames of the same shape.
        """
        num_frames = x.shape[0]
        positions = torch.arange(num_frames, device=x.device) % self.chunk_size
        # The checkpoint was trained with bf16 RoPE tables.
        cos = self.rope_cos.to(x.device)[positions].to(x.dtype)
        sin = self.rope_sin.to(x.device)[positions].to(x.dtype)
        cu_seqlens = torch.tensor(
            [*range(0, num_frames, self.chunk_size), num_frames], dtype=torch.int32
        )
        metadata = prepare_vision_attention_metadata(
            cu_seqlens, x.device, max_seqlen=min(num_frames, self.chunk_size)
        )

        for layer in self.layers:
            x = layer(x, cos, sin, metadata)

        return x


class GigaChat35OmniForConditionalGeneration(nn.Module):
    """GigaChat3.5-Omni: InternViT and conformer encoders around the GigaChat3.5 LM.

    Takes images, videos (with or without soundtrack) and audio.
    """

    _VISION_BATCH_SIZE = 128

    def __init__(
        self,
        config: GigaChat35OmniConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = GigaChat35ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.vision_tower = InternVisionModel(
            config.vision_config,
            use_data_parallel=get_mm().mm_enable_dp_encoder,
        )
        vit_dim = (
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2
        )
        lm_dim = config.text_config.hidden_size
        self.vision_projector = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )
        self.audio_tower = GigaChat35OmniConformerEncoder(config.audio_config)
        self.audio_projector = nn.Linear(
            config.audio_config.hidden_size * config.audio_stack_factor, lm_dim
        )
        # Only the checkpoint load must hold every encoder weight; online
        # weight updates may send any subset.
        self.check_missing_weights = True

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Merges every 2x2 patch neighbourhood into one token (InternVL layout).

        Args:
            x: ViT patches, `[n, h, w, c]`.

        Returns:
            Tokens of shape `[n, h * ratio, w * ratio, c / ratio ** 2]`.
        """
        ratio = self.config.downsample_ratio
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * ratio), int(c / ratio))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * ratio), int(w * ratio), int(c / (ratio * ratio)))
        return x.permute(0, 2, 1, 3).contiguous()

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encodes 448x448 crops into LM embeddings.

        Args:
            pixel_values: Crops, `[n, 3, 448, 448]`.

        Returns:
            Embeddings of shape `[n * 256, lm_hidden_size]`.
        """
        outputs = []

        for batch in pixel_values.split(self._VISION_BATCH_SIZE):
            vit_embeds = self.vision_tower(pixel_values=batch).last_hidden_state[:, 1:]
            side = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], side, side, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds).flatten(1, 2)
            outputs.append(self.vision_projector(vit_embeds))

        return torch.cat(outputs).flatten(0, 1)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encodes the crops of every image.

        Args:
            items: Image items.

        Returns:
            Embeddings of shape `[sum of tokens, lm_hidden_size]`.
        """
        return self.extract_feature(torch.cat([item.feature for item in items]))

    def get_video_feature(
        self, items: List[MultimodalDataItem]
    ) -> torch.Tensor | GigaChat35OmniVideoEmbedding:
        """Encodes every frame of a request's videos and keeps their reduced tokens.

        Args:
            items: The request's video items.

        Returns:
            The kept embeddings with their per-frame counts, or every embedding
            when every video is a single frame, which keeps all its tokens.
        """
        if all(item.thw_grids[0][0] == 1 for item in items):
            return self.extract_feature(torch.cat([item.feature for item in items]))

        kept, counts = [], []

        for item in items:
            embeds, frame_counts = GigaChat35OmniVideoEmbedding.reduce(
                self.extract_feature(item.feature),
                item.thw_grids[0],
                self.config.video_tokens_per_frame,
            )
            kept.append(embeds)
            counts.append(frame_counts)

        return GigaChat35OmniVideoEmbedding(
            embedding=torch.cat(kept), frame_counts=counts
        )

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encodes audio clips and soundtracks.

        Args:
            items: Audio items, mel frames `[num_mel_frames, num_mel_bins]` each.

        Returns:
            Embeddings of shape `[sum of tokens, lm_hidden_size]`.
        """
        stack = self.config.audio_stack_factor
        device = self.audio_projector.weight.device
        dtype = self.audio_projector.weight.dtype
        outputs = []

        for item in items:
            frames = self.audio_tower(item.feature.to(device=device, dtype=dtype))
            frames = F.pad(frames, (0, 0, 0, -frames.shape[0] % stack))
            embeds = self.audio_projector(frames.reshape(-1, stack * frames.shape[1]))
            num_tokens = sum(end - start + 1 for start, end in item.offsets)
            assert embeds.shape[0] == num_tokens, (
                f"Audio clip gave {embeds.shape[0]} embeddings for {num_tokens} placeholders"
            )
            outputs.append(embeds)

        return torch.cat(outputs)

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        """Replaces placeholders with item pad values and saves each video's layout.

        A video's frame counts are known only after encoding, so every prefill
        chunk rebuilds the video's span from the ids saved here and moves its
        soundtrack's runs from the offsets saved here.

        Args:
            input_ids: Prompt token ids.
            mm_inputs: The request's multimodal items.

        Returns:
            The padded token ids.
        """
        input_ids = pad_mm_tokens(input_ids, mm_inputs)
        videos = [item for item in mm_inputs.mm_items if item.is_video()]

        keys = GigaChat35OmniVideoEmbedding

        for item in videos:
            item.set(keys.REWRITES_LAYOUT_KEY, True)
            item.set(
                keys.SPAN_IDS_KEY,
                input_ids[item.offsets[0][0] : item.offsets[-1][1] + 1],
            )

        for item in mm_inputs.mm_items:
            video_index = item.model_specific_data.get(keys.SOUNDTRACK_OF_KEY)

            if video_index is not None:
                videos[video_index].set(keys.SOUNDTRACK_KEY, item)
                videos[video_index].set(keys.SOUNDTRACK_OFFSETS_KEY, list(item.offsets))

        return input_ids

    def get_input_embeddings(self) -> nn.Embedding:
        """Returns the LM token embedding.

        Returns:
            The vocabulary embedding.
        """
        return self.language_model.get_input_embeddings()

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the weights MTP draft heads share with the LM.

        Returns:
            The embedding and LM head weights.
        """
        return self.language_model.get_embed_and_head()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Runs the LM on token and multimodal embeddings.

        Args:
            input_ids: Token ids with multimodal pad values.
            positions: Token positions.
            forward_batch: The batch.

        Returns:
            The LM output.
        """
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_video_feature,
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Loads the LM through its own loader and the encoders and projectors directly.

        Args:
            weights: Checkpoint `(name, tensor)` pairs.
        """
        params = dict(self.named_parameters())
        loaded = set()

        def language_model_weights():
            for name, weight in weights:
                if name.startswith("language_model."):
                    yield name.removeprefix("language_model."), weight
                    continue

                if name.startswith("vision_tower."):
                    name = name.replace(".attn.qkv.", ".attn.attn.qkv_proj.")
                    name = name.replace(".attn.proj.", ".attn.attn.proj.")

                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)

        self.language_model.load_weights(language_model_weights())

        if self.check_missing_weights:
            missing = {
                name for name in params if not name.startswith("language_model.")
            } - loaded
            assert not missing, f"Missing encoder weights: {sorted(missing)}"

        self.check_missing_weights = False

    def post_load_weights(self) -> None:
        """Finishes the LM weights after a loader that bypasses `load_weights`."""
        self.language_model.post_load_weights()
        self.check_missing_weights = False


EntryClass = [GigaChat35OmniForConditionalGeneration]
