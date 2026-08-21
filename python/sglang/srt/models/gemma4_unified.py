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
"""Gemma4 *Unified* (encoder-free) multimodal model for SGLang.

The unified Gemma4 family (e.g. ``google/gemma-4-12B-it``, arch
``Gemma4UnifiedForConditionalGeneration``, ``model_type="gemma4_unified"``)
shares the Gemma4 *text* decoder verbatim but replaces both modality towers
with light "encoder-free" projection pipelines:

* **Vision** — raw merged pixel patches are projected directly into LM space:
  ``LN -> Dense -> LN -> +factorized_posemb -> LN`` (``Gemma4UnifiedVisionEmbedder``)
  followed by ``RMSNorm -> Linear`` (``Gemma4UnifiedMultimodalEmbedder``).
  There is **no** SigLIP attention tower.
* **Audio** — raw 16 kHz waveform is chunked into fixed ``audio_samples_per_token``
  frames and projected straight through ``RMSNorm -> Linear``. There is **no**
  conformer/USM encoder and no mel spectrogram.

Because the text path is identical to ``gemma4``, we reuse ``Gemma4TextModel``
and subclass ``Gemma4ForConditionalGeneration`` (reusing its ``forward``,
bidirectional-image ``prepare_attn_masks`` and PP/embed plumbing), overriding
only construction, per-modality feature extraction and weight loading.
"""

import logging
import re
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    flatten_nested_list,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma4_causal import Gemma4TextModel, pp_filter_load_weight
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free vision embedder.

    Projects raw merged pixel patches ``(..., model_patch_size**2 * 3)`` into
    ``mm_embed_dim`` via ``LN1 -> Dense -> LN2``, adds factorized 2D positional
    embeddings, and applies a final ``LN``.  Mirrors HF
    ``Gemma4UnifiedVisionEmbedder``; runs on the first PP rank only, so it uses
    plain (un-sharded) ``nn`` modules.
    """

    def __init__(self, config):
        super().__init__()
        patch_dim = config.model_patch_size**2 * 3  # 48*48*3 = 6912
        mm_embed_dim = config.mm_embed_dim

        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(mm_embed_dim)

        # Factorized 2D positional embedding table: (mm_posemb_size, 2, mm_embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(config.mm_posemb_size, 2, mm_embed_dim)
        )
        self.pos_norm = nn.LayerNorm(mm_embed_dim)

    def forward(
        self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor
    ) -> torch.Tensor:
        # pixel_values: (B, num_patches, patch_dim); image_position_ids: (B, num_patches, 2)
        hidden_states = self.patch_ln1(pixel_values.to(self.patch_dense.weight.dtype))
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=image_position_ids.device)
        pos_embs = (self.pos_embedding[clamped, axes] * valid).sum(-2)
        hidden_states = hidden_states + pos_embs
        hidden_states = self.pos_norm(hidden_states)
        return hidden_states


class Gemma4UnifiedMultimodalEmbedder(nn.Module):
    """Shared vision/audio projection: ``RMSNorm(no scale) -> Linear`` to LM space.

    Both the vision and audio configs expose ``output_proj_dims`` (the projection
    input dim) and ``rms_norm_eps``.  ``embedding_pre_projection_norm`` has no
    learnable scale, so the only checkpoint tensor is ``embedding_projection.weight``.
    """

    def __init__(self, multimodal_config, text_config):
        super().__init__()
        self.multimodal_hidden_size = multimodal_config.output_proj_dims
        self.text_hidden_size = text_config.hidden_size
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            self.multimodal_hidden_size,
            eps=multimodal_config.rms_norm_eps,
            with_scale=False,
        )
        self.embedding_projection = nn.Linear(
            self.multimodal_hidden_size, self.text_hidden_size, bias=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        inputs_embeds = inputs_embeds.to(self.embedding_projection.weight.dtype)
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


class Gemma4UnifiedForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Encoder-free unified Gemma4 (text + vision + audio).

    Reuses the Gemma4 text decoder and the multimodal ``forward`` / attention
    plumbing from :class:`Gemma4ForConditionalGeneration`, swapping the SigLIP
    vision tower and conformer audio tower for the encoder-free embedders.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Skip Gemma4ForConditionalGeneration.__init__ (it builds the SigLIP /
        # conformer towers we do not have) and initialise the HF base directly.
        PreTrainedModel.__init__(self, config=config)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        text_config = config.text_config

        # Encoder-free embedders are consumed only at the input-embedding stage,
        # so they live on the first PP rank only.
        if self.pp_group.is_first_rank:
            self.vision_embedder = (
                Gemma4UnifiedVisionEmbedder(config.vision_config)
                if getattr(config, "vision_config", None) is not None
                else None
            )
            self.embed_vision = (
                Gemma4UnifiedMultimodalEmbedder(config.vision_config, text_config)
                if getattr(config, "vision_config", None) is not None
                else None
            )
            self.embed_audio = (
                Gemma4UnifiedMultimodalEmbedder(config.audio_config, text_config)
                if getattr(config, "audio_config", None) is not None
                else None
            )
        else:
            self.vision_embedder = None
            self.embed_vision = None
            self.embed_audio = None

        # Placeholders so methods inherited from the tower-based parent that
        # reference these attributes never AttributeError.
        self.vision_tower = None
        self.audio_tower = None

        self.vocab_size = text_config.vocab_size
        self.vocab_size_per_layer_input = getattr(
            text_config, "vocab_size_per_layer_input", text_config.vocab_size
        )

        self.language_model = Gemma4TextModel(
            text_config,
            quant_config,
            prefix=add_prefix("language_model", add_prefix("model", prefix)),
        )

        text_tie = getattr(text_config, "tie_word_embeddings", True)
        if self.pp_group.world_size == 1 and text_tie:
            self.lm_head = self.language_model.embed_tokens
        elif self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                text_config.vocab_size,
                text_config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(text_config)
        self.capture_aux_hidden_states = False

        # The unified checkpoint folds mm-projection vectors into the eoi/eoa
        # rows of the (tied) embed_tokens, which inflates their lm-head logits.
        # These are input-only markers that must never be sampled — HF applies a
        # SuppressTokensLogitsProcessor for exactly these ids
        # (generation_config.suppress_tokens). We reproduce that by masking only
        # the next-token logits (input-logprob scoring of real eoi/eoa input
        # tokens is left untouched).
        suppress = []
        for attr in ("eoi_token_id", "eoa_token_id", "eoa_token_index"):
            tok_id = getattr(config, attr, None)
            if isinstance(tok_id, int):
                suppress.append(tok_id)
        self.suppress_token_ids = sorted(set(suppress))
        # Pre-materialize the index as a (non-persistent) buffer so it lives on
        # the model's device — indexing with a Python list builds a CPU index
        # tensor, which fails CUDA-graph capture ("cannot copy CPU<->CUDA").
        self.register_buffer(
            "_suppress_idx",
            torch.tensor(self.suppress_token_ids, dtype=torch.long),
            persistent=False,
        )

        self.post_init()

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        if (
            self._suppress_idx.numel() > 0
            and isinstance(out, LogitsProcessorOutput)
            and out.next_token_logits is not None
        ):
            out.next_token_logits.index_fill_(
                1, self._suppress_idx, torch.finfo(out.next_token_logits.dtype).min
            )
        return out

    # ------------------------------------------------------------------
    # Per-modality feature extraction (encoder-free)
    # ------------------------------------------------------------------
    def _empty_embeds(self) -> torch.Tensor:
        return torch.empty(
            0,
            self.language_model.config.hidden_size,
            device=next(self.parameters()).device,
            dtype=self.language_model.dtype(),
        )

    def _embed_patches(
        self, items: List[MultimodalDataItem], position_attr: str
    ) -> torch.Tensor:
        all_embeds = []
        for item in items:
            all_pixel_values = flatten_nested_list([item.feature])
            all_position_ids = flatten_nested_list([getattr(item, position_attr, None)])
            for pv_idx, pv in enumerate(all_pixel_values):
                # Pre-embedded passthrough (already at text hidden size).
                if (
                    pv.dim() in (2, 3)
                    and pv.shape[-1] == self.config.text_config.hidden_size
                ):
                    all_embeds.append(pv.to(self.language_model.device))
                    continue

                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:
                    raise ValueError(
                        f"pixel_values[{pv_idx}] has no matching {position_attr}. "
                        "The HF image/video processor likely renamed this output — "
                        "update ATTR_NAME_TO_MODALITY in the Gemma4Unified processor."
                    )
                pp = all_position_ids[pv_idx]

                # Collapse video (num_videos, num_frames, P, ...) -> (frames, P, ...)
                if pv.dim() == 4:
                    pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])
                if pp.dim() == 4:
                    pp = pp.reshape(-1, pp.shape[-2], pp.shape[-1])
                if pv.dim() == 2:
                    pv = pv.unsqueeze(0)
                if pp.dim() == 2:
                    pp = pp.unsqueeze(0)

                pv = pv.to(
                    device=self.language_model.device, dtype=self.language_model.dtype()
                )
                pp = pp.to(device=self.language_model.device)

                embedded = self.vision_embedder(pv, pp)  # (B, P, mm_embed_dim)
                projected = self.embed_vision(embedded)  # (B, P, hidden)

                # Drop padding patches (position_ids == -1 on both axes).
                padding_mask = (pp == -1).all(dim=-1)  # (B, P)
                all_embeds.append(projected[~padding_mask])

        return torch.cat(all_embeds, dim=0) if all_embeds else self._empty_embeds()

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self._embed_patches(items, "image_position_ids")

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self._embed_patches(items, "video_position_ids")

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if self.embed_audio is None:
            raise ValueError(
                "Audio inputs provided but the model was built without an audio_config."
            )
        all_input_features = flatten_nested_list([item.feature for item in items])
        # input_features_mask convention: True = valid token.
        all_masks = flatten_nested_list([item.input_features_mask for item in items])

        all_embeds = []
        for input_features, mask in zip(all_input_features, all_masks):
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            input_features = input_features.to(
                device=self.language_model.device, dtype=self.language_model.dtype()
            )
            mask = mask.to(device=input_features.device)

            # Raw waveform frames -> RMSNorm -> Linear (no conformer/mel).
            projected = self.embed_audio(inputs_embeds=input_features)  # (B, T, hidden)
            for enc, m in zip(projected, mask):
                all_embeds.append(enc[m])  # keep valid frames only

        return torch.cat(all_embeds, dim=0) if all_embeds else self._empty_embeds()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        k_eq_v_layers = self._get_k_eq_v_layers()

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        non_persistent_buffers: Set[str] = set()
        for mod_name, mod in self.named_modules():
            for buf_name in getattr(mod, "_non_persistent_buffers_set", set()):
                full = f"{mod_name}.{buf_name}" if mod_name else buf_name
                non_persistent_buffers.add(full)

        text_tie = getattr(self.config.text_config, "tie_word_embeddings", True)
        start_layer = self.language_model.start_layer
        end_layer = self.language_model.end_layer

        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            name = re.sub(r"^model\.", "", name)

            if pp_filter_load_weight(
                name,
                loaded_weight,
                pp_group=self.pp_group,
                start_layer=start_layer,
                end_layer=end_layer,
                params_dict=params_dict,
                loaded_params=loaded_params,
                tie_word_embeddings=text_tie,
                embed_weight_name="language_model.embed_tokens.weight",
                first_rank_only_patterns=(
                    "language_model.embed_tokens",
                    "language_model.per_layer_model_projection",
                    "language_model.per_layer_projection_norm",
                    "vision_embedder.",
                    "embed_vision.",
                    "embed_audio.",
                ),
                last_rank_only_prefixes=("language_model.norm.", "lm_head."),
            ):
                continue

            # attention_k_eq_v: full-attention layers ship only k_proj (V == K).
            # Load k_proj into both the "k" and "v" shards of the fused QKV.
            should_dup_k_to_v = (
                ".k_proj." in name
                and k_eq_v_layers
                and "language_model." in name
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None
                and int(m.group(1)) in k_eq_v_layers
            )

            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                if should_dup_k_to_v:
                    param.weight_loader(param, loaded_weight, "v")
                loaded_params.add(mapped)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            param_names = set(dict(self.named_parameters()).keys())
            buckets = {
                logging.WARNING: (
                    "Some weights are not initialized from checkpoints",
                    lambda p: p in param_names,
                ),
                logging.INFO: (
                    "Persistent buffers not in checkpoint (using default init)",
                    lambda p: p not in param_names and p not in non_persistent_buffers,
                ),
                logging.DEBUG: (
                    "Non-persistent buffers not in checkpoint (expected)",
                    lambda p: p in non_persistent_buffers,
                ),
            }
            for level, (msg, pred) in buckets.items():
                names = sorted(p for p in unloaded_params if pred(p))
                if names:
                    logger.log(level, "%s: %s", msg, names)
        return loaded_params


EntryClass = Gemma4UnifiedForConditionalGeneration
