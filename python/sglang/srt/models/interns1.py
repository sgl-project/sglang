from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.attention import vision_utils
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
from sglang.srt.model_loader.auto_loader import AutoWeightsLoader, WeightsMapper
from sglang.srt.models.internvl import InternVisionModel
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.utils import logger


class InternS1ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        use_flash_attn=True,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        vision_utils.update_vit_attn_dummy_heads_config(self.config)
        image_size = (
            getattr(config, "force_image_size", None) or config.vision_config.image_size
        )
        patch_size = config.vision_config.patch_size
        if isinstance(image_size, list):
            image_size = image_size[0]
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.select_layer = config.vision_feature_layer
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio

        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.text_config._attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")

        self.vision_model = InternVisionModel(config.vision_config)
        if config.text_config.architectures[0] == "Qwen2ForCausalLM":
            self.language_model = Qwen2ForCausalLM(
                config=config.text_config, quant_config=quant_config
            )
        elif config.text_config.architectures[0] == "Qwen3MoeForCausalLM":
            self.language_model = Qwen3MoeForCausalLM(
                config=config.text_config, quant_config=quant_config
            )
        elif config.text_config.architectures[0] == "Qwen3ForCausalLM":
            self.language_model = Qwen3ForCausalLM(
                config=config.text_config, quant_config=quant_config
            )
        else:
            raise NotImplementedError(
                f"{config.text_config.architectures[0]} is not implemented."
            )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        pixel_values = torch.cat([item.feature for item in items])
        image_features = self.extract_feature(pixel_values)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        hs = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hs

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = mm_inputs.im_start_id
        im_end_id: int = mm_inputs.im_end_id

        media_token_pairs = [(im_start_id, im_end_id)]
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return helper.pad_input_tokens(input_ids, mm_inputs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Map HF checkpoint names onto the runtime module tree before the
        # walker sees them. Order matters: substr rewrites apply first, then
        # prefixes. The vision tower's HF naming uses ``encoder.layer.*``,
        # ``attention``, ``projection_layer``, ``lambda_{1,2}``,
        # ``layernorm_{before,after}``; the runtime ``InternVisionModel``
        # exposes ``encoder.layers.*``, ``attn.attn``, ``proj``, ``ls{1,2}``,
        # ``norm{1,2}``. The HF projector lives under
        # ``model.multi_modal_projector.{layer_norm,linear_1,linear_2}``;
        # runtime exposes ``mlp1.0`` (LayerNorm), ``mlp1.1`` (Linear),
        # ``mlp1.3`` (Linear).
        mapper = WeightsMapper(
            orig_to_new_substr={
                ".attention.": ".attn.attn.",
                ".projection_layer.": ".proj.",
                ".lambda_1": ".ls1",
                ".lambda_2": ".ls2",
                ".layernorm_before.": ".norm1.",
                ".layernorm_after.": ".norm2.",
                ".embeddings.patch_embeddings.projection.": ".embeddings.patch_embedding.",
                ".embeddings.position_embeddings": ".embeddings.position_embedding",
                ".embeddings.cls_token": ".embeddings.class_embedding",
                # Vision tower encoder layer collection rename: only the
                # ``encoder.layer.`` segment, not generic ``layer.``.
                "encoder.layer.": "encoder.layers.",
            },
            orig_to_new_prefix={
                "model.multi_modal_projector.layer_norm.": "mlp1.0.",
                "model.multi_modal_projector.linear_1.": "mlp1.1.",
                "model.multi_modal_projector.linear_2.": "mlp1.3.",
                "model.language_model.": "language_model.model.",
                "model.vision_tower.": "vision_model.",
                "lm_head.": "language_model.lm_head.",
            },
        )

        # Apply mapping first so ``pad_vit_attn_dummy_heads`` sees the
        # runtime names it inspects (``attn.qkv_proj``, ``attn.proj`` etc.).
        def _pad_vision(stream):
            for name, w in mapper.apply(stream):
                if "vision_model" in name:
                    w = vision_utils.pad_vit_attn_dummy_heads(self.config, name, w)
                yield name, w

        # Mapper has already been applied; pass an empty mapper to the loader.
        return AutoWeightsLoader(self).load_weights(_pad_vision(weights))


EntryClass = InternS1ForConditionalGeneration
