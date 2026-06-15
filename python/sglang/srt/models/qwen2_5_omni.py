# coding=utf-8
"""Inference-only Qwen2.5-Omni thinker model for text generation.

This implementation serves text generation only.  It loads the Qwen2.5-Omni
``thinker`` submodule and intentionally skips the speech ``talker`` and
``token2wav`` modules.
"""

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniThinkerConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder,
)

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen2_5OmniThinkerForConditionalGeneration(nn.Module):
    default_bitsandbytes_target_modules = [
        ".gate_up_proj.",
        ".down_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Qwen2_5OmniThinkerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.text_config = config.text_config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        self.model = Qwen2Model(
            self.text_config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.text_config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.text_config.vocab_size,
                    self.text_config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.audio_tower = Qwen2_5OmniAudioEncoder(config.audio_config)
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(self.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
            max_context_len=self.text_config.max_position_embeddings,
        )

        rope_scaling = getattr(self.text_config, "rope_scaling", None) or {}
        self.is_mrope_enabled = "mrope_section" in rope_scaling
        self.logits_processor = LogitsProcessor(self.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
            )
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)

        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, video_grid_thw.tolist(), rope_type="rope_3d"
            )
        return self.visual(pixel_values, grid_thw=video_grid_thw)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        feature_attention_mask = torch.cat(
            [item.feature_attention_mask for item in items], dim=0
        ).type(torch.long)
        audio_device = next(self.audio_tower.parameters()).device
        feature_attention_mask = feature_attention_mask.to(audio_device)

        input_features = torch.cat([item.feature for item in items], dim=0).type(
            self.audio_tower.dtype
        )
        input_features = input_features.to(audio_device)

        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[
            feature_attention_mask.bool()
        ].permute(1, 0)

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths)
        )
        audio_outputs = self.audio_tower(
            input_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,
            return_dict=True,
        )
        audio_features = audio_outputs.last_hidden_state
        if audio_features.shape[0] != int(audio_output_lengths.sum().item()):
            raise ValueError(
                "Length of Qwen2.5-Omni audio features does not match "
                "audio output lengths."
            )
        return audio_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            return self.pooler(hidden_states, forward_batch)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".attn.qkv_proj", ".attn.q", "q"),
            (".attn.qkv_proj", ".attn.k", "k"),
            (".attn.qkv_proj", ".attn.v", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("thinker."):
                name = name[len("thinker.") :]
            if name.startswith(("talker.", "token2wav.")):
                continue

            if (
                self.text_config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and name == "model.embed_tokens.weight"
                and "lm_head.weight" in params_dict
            ):
                lm_head_param = params_dict["lm_head.weight"]
                weight_loader = getattr(
                    lm_head_param, "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_param, loaded_weight)

            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "audio_tower" in name:
                    continue
                if (
                    "visual" in name
                    and weight_name in (".q_proj", ".k_proj", ".v_proj")
                ):
                    continue
                if "visual" not in name and weight_name in (
                    ".attn.q",
                    ".attn.k",
                    ".attn.v",
                ):
                    continue

                mapped_name = name.replace(weight_name, param_name)
                layer_id = get_layer_id(mapped_name)
                if (
                    layer_id is not None
                    and hasattr(self.model, "start_layer")
                    and (
                        layer_id < self.model.start_layer
                        or layer_id >= self.model.end_layer
                    )
                ):
                    loaded = True
                    break

                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    loaded = True
                    break
                if mapped_name not in params_dict:
                    continue

                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded = True
                break

            if loaded:
                continue

            if "visual" in name:
                name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                logger.warning("Loaded weight %s not found in Qwen2.5-Omni", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.text_config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


class Qwen2_5OmniForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Qwen2_5OmniConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(
            config.thinker_config, quant_config=quant_config, prefix=prefix
        )
        self.enable_talker = False
        self.pad_input_ids = self.thinker.pad_input_ids
        self.forward = self.thinker.forward

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return self.thinker.load_weights(weights)

    def get_embed_and_head(self):
        return self.thinker.get_embed_and_head()


class Qwen2_5OmniModel(Qwen2_5OmniForConditionalGeneration):
    """Alias for checkpoints whose config.architectures is Qwen2_5OmniModel."""


EntryClass = [Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniModel]
