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
"""DeepSeek-V4-Flash-Vision: the DeepSeek-V4 language model plus a vision tower.

The checkpoint (``deepseek-ai/DeepSeek-V4-Flash-Vision-Exp``) is
DeepSeek-V4-Flash with three additions, all of which live here:

*   a 32-layer ViT with 2D RoPE and an aligner MLP that folds each
    ``downsample_ratio``-squared block of ViT patches into one LLM token,
    plus four learned framing embeddings (``image_start``, ``image_pad``,
    ``image_newline``, ``image_end``);
*   a second routed-expert bias, ``e_score_correction_bias_vl``, used for image
    tokens in place of the text bias -- and, on the hash-routed layers, in place
    of hash routing entirely; and
*   bidirectional attention inside each image's token block.

The vision tower is replicated on every rank: the checkpoint does not shard it,
and at ~0.4B parameters it is negligible next to the 284B language model.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    compute_mm_input_embeds,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v4_vl_image_processing import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PAD,
    IMAGE_START,
    NUM_IMAGE_SLOT_KINDS,
)
from sglang.srt.runtime_context import get_forward

logger = logging.getLogger(__name__)


def vision_rope_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE tables for one ``n_h x n_w`` patch grid, row-major.

    Each head's ``2 * dim`` rotary channels split in half: the first ``dim / 2``
    frequencies encode the patch row, the second ``dim / 2`` the patch column.
    Shapes are ``[n_h * n_w, 1, dim]`` so they broadcast over heads.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device).div_(dim))
    )
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_vision_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4VisionRMSNorm(nn.Module):
    """The tower's own RMSNorm: fp32 weight and fp32 math, eps 1e-6.

    Deliberately not :class:`sglang.srt.layers.layernorm.RMSNorm`; the tower's
    eps is 1e-6 while the language model's ``rms_norm_eps`` is 1e-20, and the
    checkpoint's reference implementation keeps this weight in fp32.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4VisionPatchEmbed(nn.Module):
    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.proj = nn.Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlens: List[int],
    ) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = _apply_vision_rope(q, cos, sin)
        k = _apply_vision_rope(k, cos, sin)
        # Attention is unmasked within an image and must not cross images, so it
        # runs per image while every projection above and below stays batched
        # across the whole request (that is where the tower's FLOPs are).
        if len(seqlens) == 1:
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            ).transpose(0, 1)
        else:
            outs = []
            offset = 0
            for seqlen in seqlens:
                sl = slice(offset, offset + seqlen)
                outs.append(
                    F.scaled_dot_product_attention(
                        q[sl].transpose(0, 1),
                        k[sl].transpose(0, 1),
                        v[sl].transpose(0, 1),
                    ).transpose(0, 1)
                )
                offset += seqlen
            o = torch.cat(outs)
        return self.wo(o.reshape(n, -1))


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        # w1 emits the gate and up halves in one matmul.
        self.w1 = nn.Linear(
            config.vision_dim,
            2 * config.vision_inter_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            config.vision_inter_dim,
            config.vision_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.norm1 = DeepseekV4VisionRMSNorm(config.vision_dim)
        self.attn = DeepseekV4VisionAttention(config)
        self.norm2 = DeepseekV4VisionRMSNorm(config.vision_dim)
        self.mlp = DeepseekV4VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlens: List[int],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, seqlens)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """Full bidirectional attention over one image's patches, with 2D RoPE."""

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4VisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [DeepseekV4VisionBlock(config) for _ in range(config.vision_n_layers)]
        )
        self.norm = DeepseekV4VisionRMSNorm(config.vision_dim)

    def forward(
        self, patches: torch.Tensor, grids: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """``patches`` is every image's patches concatenated; ``grids`` their dims."""
        device = patches.device
        tables = [
            vision_rope_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, device)
            for n_h, n_w in grids
        ]
        if len(tables) == 1:
            cos, sin = tables[0]
        else:
            cos = torch.cat([t[0] for t in tables])
            sin = torch.cat([t[1] for t in tables])
        seqlens = [n_h * n_w for n_h, n_w in grids]

        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, cos, sin, seqlens)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """Folds a ``r x r`` block of ViT patch features into one LLM-width token."""

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        # Pad right/bottom so the grid divides evenly; unfold then walks the
        # r x r blocks row-major, matching the aligner's training layout.
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class DeepseekV4VisionModel(nn.Module):
    """The tower, the aligner, and the four learned block-framing embeddings."""

    def __init__(self, config: DeepSeekV4Config):
        super().__init__()
        self.config = config
        self.vision = DeepseekV4ViT(config)
        self.aligner = DeepseekV4Aligner(config)
        self.image_start = nn.Parameter(torch.empty(config.hidden_size))
        self.image_end = nn.Parameter(torch.empty(config.hidden_size))
        self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
        self.image_pad = nn.Parameter(torch.empty(config.hidden_size))

    def _slot_embeddings(self) -> torch.Tensor:
        """The learned embedding per slot kind, indexable by slot type id.

        The ``IMAGE`` row is filler -- every ``IMAGE`` slot is overwritten with
        an aligner output -- but keeping it in the table lets the block be built
        with a single gather.
        """
        table = [None] * NUM_IMAGE_SLOT_KINDS
        table[IMAGE_START] = self.image_start
        table[IMAGE_PAD] = self.image_pad
        table[IMAGE] = self.image_pad
        table[IMAGE_NEW_LINE] = self.image_newline
        table[IMAGE_END] = self.image_end
        return torch.stack(table)

    def forward(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Embeddings for every slot of every item's image block, concatenated.

        The returned rows cover the *whole* block -- framing, newline and
        alignment-padding slots included -- because all of them are placeholder
        positions in ``input_ids`` and must be filled by the scatter.
        """
        device = self.image_start.device
        patches, grids = [], []
        for item in items:
            feature = item.feature
            if not isinstance(feature, torch.Tensor):
                feature = torch.as_tensor(feature)
            patches.append(feature.to(device=device))
            grids.append((int(item.n_vit_h), int(item.n_vit_w)))

        patch_features = self.vision(
            torch.cat(patches) if len(patches) > 1 else patches[0], grids
        )

        table = self._slot_embeddings()
        blocks = []
        offset = 0
        for item, (n_vit_h, n_vit_w) in zip(items, grids):
            num_patches = n_vit_h * n_vit_w
            embeds = self.aligner(
                patch_features[offset : offset + num_patches], n_vit_h, n_vit_w
            )
            offset += num_patches
            types = torch.as_tensor(item.slot_types, device=device)
            perm = torch.as_tensor(item.aligner_perm, device=device)
            block = table[types]
            block[types == IMAGE] = embeds[perm]
            blocks.append(block)
        return torch.cat(blocks) if len(blocks) > 1 else blocks[0]


class DeepseekV4VLForCausalLM(DeepseekV4ForCausalLM):
    """DeepSeek-V4 with the vision tower wired into the embedding path."""

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        # The tower only lives on the rank that owns the embedding table, since
        # that is the only rank whose input embeddings it contributes to.
        if self.pp_group.is_first_rank:
            self.vision_model = DeepseekV4VisionModel(config)
        else:
            self.vision_model = None

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self.vision_model(items)

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """Stamp each image block's per-image pad value over its placeholders.

        The block is one contiguous placeholder run, so the whole span --
        framing and alignment padding included -- takes the item's pad value.
        Both the MoE's image-token mask and the attention backend's image spans
        then read those out-of-vocabulary sentinels straight out of input_ids,
        the same test the checkpoint's reference implementation makes.
        """
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(
            input_ids, mm_inputs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # Read before embedding: compute_mm_input_embeds clears mm_inputs.
        vision_routing = forward_batch.contains_mm_inputs()
        if input_embeds is None and self.pp_group.is_first_rank:
            # Embed here rather than through general_mm_embed_routine: the
            # hash-routed MoE layers route on input_ids, so the language model
            # cannot be called with input_ids=None. Feed the embedder a copy --
            # it clamps multimodal pad sentinels in place, and both the MoE
            # image mask and the attention backend's image spans read them.
            input_embeds, _ = compute_mm_input_embeds(
                input_ids=input_ids.clone(),
                forward_batch=forward_batch,
                embed_tokens=self.get_input_embeddings(),
                multimodal_model=self,
            )
        with get_forward().scoped(vision_expert_routing=vision_routing):
            return super().forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn: bool = False
    ):
        if self.vision_model is None:
            # Only the first pipeline rank owns the tower.
            weights = (
                (name, weight)
                for name, weight in weights
                if not name.startswith(("vision.", "aligner.", "image_"))
            )
        return super().load_weights(weights, is_nextn=is_nextn)

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str,
        is_nextn: bool = False,
        num_hidden_layers: Optional[int] = None,
    ) -> str:
        if name.startswith(("vision.", "aligner.", "image_")):
            # Keep the checkpoint's own names for the tower; the base remapper
            # would rewrite .w1./.w2. into MLP projection names.
            return "vision_model." + name
        return DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
            name, is_nextn=is_nextn, num_hidden_layers=num_hidden_layers
        )


EntryClass = [DeepseekV4VLForCausalLM]
