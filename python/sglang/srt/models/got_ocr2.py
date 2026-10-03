# Adapted from https://huggingface.co/stepfun-ai/GOT-OCR2_0

from array import array
from functools import partial
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_ocr import ImageEncoderViT
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

# The tower emits a 16x16 grid of 1024-dim features, i.e. config.image_token_len.
VISION_FEATURE_DIM = 1024


def build_got_vision_tower() -> ImageEncoderViT:
    """SAM ViTDet-B tower, matching GOT's `build_GOT_vit_b`.

    Same geometry as DeepSeek-OCR's SAM encoder (which was derived from
    GOT/Vary), so `ImageEncoderViT` is reused verbatim. Input is always
    1024x1024, so its `get_abs_pos_sam` interpolation is a no-op here.
    """
    return ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
        net_3_out_channels=VISION_FEATURE_DIM,
    )


class GOTQwenForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.language_model = Qwen2ForCausalLM(config=config, quant_config=quant_config)
        self.vision_tower_high = build_got_vision_tower()
        self.mm_projector_vary = nn.Linear(VISION_FEATURE_DIM, config.hidden_size)

        self.external_mm_data_embedding_funcs = {
            Modality.IMAGE: self.get_image_feature,
        }

        self.model = self.language_model.model

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        projector_weight = self.mm_projector_vary.weight
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=projector_weight.device, dtype=projector_weight.dtype
        )
        cnn_feature = self.vision_tower_high(pixel_values)
        # [N, 1024, 16, 16] -> [N, 256, 1024]
        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)
        return self.mm_projector_vary(cnn_feature)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs=self.external_mm_data_embedding_funcs,
            positions=positions,
        )

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        media_token_pairs = [(mm_inputs.im_start_id, mm_inputs.im_end_id)]
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return helper.pad_input_tokens(input_ids, mm_inputs)

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
            # Tying makes lm_head the embedding parameter, so it has no separate
            # entry in params_dict. The checkpoint ships both and they are
            # byte-identical, so dropping this copy loses nothing.
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                continue

            # The checkpoint nests the tower and projector under the text model;
            # here they are siblings of it.
            if name.startswith(
                ("model.vision_tower_high.", "model.mm_projector_vary.")
            ):
                name = name[len("model.") :]
            elif name.startswith(("model.", "lm_head.")):
                name = "language_model." + name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = GOTQwenForCausalLM
