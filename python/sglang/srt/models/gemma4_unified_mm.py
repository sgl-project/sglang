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
"""Gemma 4 Unified (12B) — encoder-free multimodal model.

Unlike the 26B-A4B / 31B models (heavy ViT ``vision_tower`` + Conformer
``audio_tower``), the 12B Unified projects raw image patches and raw audio
waveforms directly into the decoder backbone via lightweight linear embedders.
The dense text decoder (``Gemma4TextModel``) is reused verbatim.
"""

import logging
import re
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel

from sglang.srt.configs.gemma4_unified import (
    Gemma4UnifiedAudioConfig,
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVisionConfig,
)
from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.models.gemma4_causal import Gemma4TextModel, pp_filter_load_weight
from sglang.srt.models.gemma4_mm import Gemma4MultimodalEmbedder
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free vision path.

    Raw ``model_patch_size`` x ``model_patch_size`` pixel patches are projected
    to ``mm_embed_dim`` by a single matmul, given factorized X/Y coordinate
    position embeddings, pooled (kernel ``pooling_kernel_size``) down to
    ``num_soft_tokens`` per image, then projected into the text hidden size via
    the reused ``Gemma4MultimodalEmbedder`` (pre-norm + projection).
    """

    def __init__(
        self,
        vision_config: Gemma4UnifiedVisionConfig,
        text_config: Gemma4UnifiedTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vision_config = vision_config
        # Channels assumed 3 (RGB). NOTE(verify): confirm against the HF image
        # processor output (patch flattening order + channel count).
        patch_pixels = vision_config.model_patch_size * vision_config.model_patch_size * 3
        self.patch_projection = ReplicatedLinear(
            patch_pixels,
            vision_config.mm_embed_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("patch_projection", prefix),
        )
        # Factorized coordinate lookup tables (X and Y).
        self.pos_emb_x = nn.Parameter(
            torch.zeros(vision_config.mm_posemb_size, vision_config.mm_embed_dim)
        )
        self.pos_emb_y = nn.Parameter(
            torch.zeros(vision_config.mm_posemb_size, vision_config.mm_embed_dim)
        )
        self.pooling_kernel_size = vision_config.pooling_kernel_size
        # Output projection back to the text hidden size. Reuses the existing
        # embedder (reads output_proj_dims first, falls back to hidden_size).
        self.embedding = Gemma4MultimodalEmbedder(
            vision_config,
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("embedding", prefix),
        )

    def forward(
        self, patches: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """patches: (num_patches, patch_pixels); coords: (num_patches, 2)."""
        x, _ = self.patch_projection(patches)
        x = x + self.pos_emb_x[coords[:, 0]] + self.pos_emb_y[coords[:, 1]]
        x = self._pool(x)
        return self.embedding(inputs_embeds=x)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Average-pool patch tokens down to soft tokens.

        NOTE(verify): exact pooling layout (spatial 2D vs flattened) must match
        the reference once num_soft_tokens=280 / pooling_kernel_size=3 are
        reconciled with the HF processor. Kept as a 1D fold for the skeleton.
        """
        k = self.pooling_kernel_size
        if k <= 1 or x.shape[0] % k != 0:
            return x
        n = (x.shape[0] // k) * k
        pooled = x[:n].reshape(n // k, k, x.shape[-1]).mean(dim=1)
        return pooled


class Gemma4UnifiedAudioEmbedder(nn.Module):
    """Encoder-free audio path.

    Raw 16 kHz audio is sliced into ``audio_samples_per_token``-sample (40 ms)
    frames upstream by the processor; each frame is linearly projected, then
    projected into the text hidden size via the reused ``Gemma4MultimodalEmbedder``.
    """

    def __init__(
        self,
        audio_config: Gemma4UnifiedAudioConfig,
        text_config: Gemma4UnifiedTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.audio_config = audio_config
        self.wave_projection = ReplicatedLinear(
            audio_config.audio_samples_per_token,
            audio_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wave_projection", prefix),
        )
        self.embedding = Gemma4MultimodalEmbedder(
            audio_config,
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("embedding", prefix),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (num_frames, audio_samples_per_token)."""
        x, _ = self.wave_projection(frames)
        return self.embedding(inputs_embeds=x)


class Gemma4UnifiedForConditionalGeneration(PreTrainedModel):
    config_class = Gemma4UnifiedConfig
    """Gemma 4 Unified (12B) encoder-free multimodal model."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True
    supported_lora_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".up_proj", 1),
        (".gate_up_proj", ".gate_proj", 0),
    ]

    def __init__(
        self,
        config: Gemma4UnifiedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        text_config = config.text_config
        prefix = add_prefix("model", prefix)

        # Encoder-free embedders consume input-embedding-stage activations only,
        # so they live on the first PP rank.
        if self.pp_group.is_first_rank:
            self.embed_vision = Gemma4UnifiedVisionEmbedder(
                config.vision_config,
                text_config,
                quant_config=quant_config,
                prefix=add_prefix("embed_vision", prefix),
            )
            self.embed_audio = Gemma4UnifiedAudioEmbedder(
                config.audio_config,
                text_config,
                quant_config=quant_config,
                prefix=add_prefix("embed_audio", prefix),
            )
        else:
            self.embed_vision = PPMissingLayer()
            self.embed_audio = PPMissingLayer()

        self.vocab_size = text_config.vocab_size

        # Dense decoder reused verbatim. PLE auto-disables when
        # hidden_size_per_layer_input == 0; MoE auto-disables when
        # enable_moe_block is False.
        self.language_model = Gemma4TextModel(
            text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
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
        self.post_init()

    @property
    def model(self):
        return self.language_model

    def __setattr__(self, name, value):
        if name == "model":
            return
        super().__setattr__(name, value)

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        embed = self.language_model.embed_tokens.weight
        return embed, embed

    def get_attention_sliding_window_size(self):
        return getattr(self.config.text_config, "sliding_window", -1) - 1

    # ------------------------------------------------------------------ #
    # Multimodal feature extraction (encoder-free)
    # ------------------------------------------------------------------ #
    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        all_embeds = []
        for item in items:
            patches_list = flatten_nested_list([item.feature])
            coords_list = flatten_nested_list(
                [getattr(item, "image_position_ids", None)]
            )
            for idx, patches in enumerate(patches_list):
                # Precomputed embeddings pass straight through.
                if (
                    patches.dim() in (2, 3)
                    and patches.shape[-1] == self.config.text_config.hidden_size
                ):
                    all_embeds.append(patches.to(self.language_model.device))
                    continue
                if idx >= len(coords_list) or coords_list[idx] is None:
                    raise ValueError(
                        f"patches[{idx}] has no matching image_position_ids; "
                        "check the Gemma4Unified processor output."
                    )
                coords = coords_list[idx]
                dev = self.embed_vision.patch_projection.weight.device
                patches = patches.to(device=dev, dtype=self.language_model.dtype())
                coords = coords.to(device=dev)
                all_embeds.append(self.embed_vision(patches, coords))

        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        return torch.empty(
            0,
            self.language_model.config.hidden_size,
            device=next(self.parameters()).device,
            dtype=self.language_model.dtype(),
        )

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        all_frames = flatten_nested_list([item.feature for item in items])
        all_embeds = []
        for frames in all_frames:
            dev = self.embed_audio.wave_projection.weight.device
            frames = frames.to(device=dev, dtype=self.language_model.dtype())
            all_embeds.append(self.embed_audio(frames))

        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        return torch.empty(
            0,
            self.language_model.config.hidden_size,
            device=next(self.parameters()).device,
            dtype=self.language_model.dtype(),
        )

    # ------------------------------------------------------------------ #
    # Bidirectional image attention (reused, identical to gemma4_mm)
    # ------------------------------------------------------------------ #
    def prepare_attn_masks(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        mask_dtype: torch.dtype,
    ):
        if not isinstance(get_attn_backend(), TritonAttnBackend):
            logger.warning_once(
                "Bidirectional attention for image tokens requires "
                "TritonAttnBackend. Falling back to causal attention."
            )
            return
        assert forward_batch.forward_mode == ForwardMode.EXTEND

        masks_list = []
        mask_indptr = torch.zeros(
            forward_batch.batch_size + 1, dtype=torch.int32, device=input_ids.device
        )
        split_images = []
        for i in range(forward_batch.batch_size):
            extend_seq_len = forward_batch.extend_seq_lens[i]
            prefix_len = forward_batch.extend_prefix_lens[i]
            mask = torch.zeros(
                extend_seq_len,
                extend_seq_len + prefix_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            mask.fill_(1)
            mask = mask.tril(diagonal=prefix_len)
            mm_inputs = forward_batch.mm_inputs[i]
            if mm_inputs is not None:
                for mm_item in mm_inputs.mm_items:
                    if mm_item.is_image():
                        for im_begin, im_end in mm_item.offsets:
                            if (
                                im_begin >= prefix_len
                                and im_end < prefix_len + extend_seq_len
                            ):
                                mask[
                                    im_begin - prefix_len : im_end + 1 - prefix_len,
                                    im_begin : im_end + 1,
                                ] = 1
                            elif (
                                im_end >= prefix_len
                                and im_begin < prefix_len + extend_seq_len
                            ):
                                split_images.append((i, im_begin, im_end))
            masks_list.append(mask.flatten())
            mask_indptr[i + 1] = mask_indptr[i] + mask.nelement()
        if split_images:
            logger.warning_once(
                f"{len(split_images)} images split across chunk boundaries will "
                "receive causal attention. Disable chunked prefill for full "
                "bidirectional attention."
            )
        if masks_list:
            get_attn_backend().forward_metadata.mask_indptr = mask_indptr
            get_attn_backend().forward_metadata.custom_mask = torch.cat(
                masks_list, dim=0
            )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        is_first_rank = self.pp_group.is_first_rank
        is_last_rank = self.pp_group.is_last_rank

        if is_first_rank and (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if envs.SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION.get():
            positions = positions + 1
        else:
            positions += 1

        # No PLE for the 12B (hidden_size_per_layer_input == 0).

        if (
            forward_batch.forward_mode == ForwardMode.EXTEND
            and forward_batch.contains_image_inputs()
        ):
            self.prepare_attn_masks(forward_batch, input_ids, mask_dtype=torch.bool)

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
            **kwargs,
        )

        if not is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        head = (
            self.language_model.embed_tokens
            if self.pp_group.world_size == 1
            and getattr(self.config.text_config, "tie_word_embeddings", True)
            else self.lm_head
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            head,
            forward_batch,
            aux_hidden_states,
        )

    def tie_weights(self, recompute_mapping=False):
        if self.pp_group.world_size > 1:
            return
        return self.language_model.tie_weights()

    # ------------------------------------------------------------------ #
    # Weight loading (no towers, no MoE)
    # ------------------------------------------------------------------ #
    def _get_k_eq_v_layers(self) -> Set[int]:
        text_config = self.config.text_config
        if not getattr(text_config, "attention_k_eq_v", False):
            return set()
        return {
            i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"
        }

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
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
                    "embed_vision.",
                    "embed_audio.",
                ),
                last_rank_only_prefixes=("language_model.norm.", "lm_head."),
            ):
                continue

            # attention_k_eq_v: full-attention layers store only k_proj; load it
            # into both the k and v shards of the fused QKV.
            should_dup_k_to_v = (
                ".k_proj." in name
                and k_eq_v_layers
                and "language_model." in name
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None
                and int(m.group(1)) in k_eq_v_layers
            )

            orig_name = name
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                name = orig_name
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                if should_dup_k_to_v:
                    param.weight_loader(param, loaded_weight, "v")
                loaded_params.add(name)
                break
            else:
                name = orig_name
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
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
