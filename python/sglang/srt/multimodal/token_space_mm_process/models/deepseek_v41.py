"""DeepSeek-V4.1 token-space strategy: one ``image_token_id`` placeholder per image.

An image expands to its whole span, ``image_token_id`` at every position (start, rows, newlines, end), as the legacy
processor (sglang.srt.multimodal.processors.deepseek_v41) expands it; the model tells the positions apart by token
type. Patches come from the release's PIL patchify, the legacy processor's CPU path, so serving and a trainer that
rebuilds the inputs get the same BF16 patches.
"""

from types import MappingProxyType

import torch

from sglang.srt.multimodal.deepseek_v41_image_processing import (
    image_token_types,
    patchify_image,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)

# The checkpoint names DeepseekV41ForCausalLM; SGLang's V4.1 config maps it to the V4 class.
_MODELS = ("DeepseekV41ForCausalLM", "DeepseekV4ForCausalLM")
# Per-image token counts the expansion reads; not model inputs.
_EXPANSION_FIELD = "num_image_tokens"


def _has_vision(hf_config) -> bool:
    return (
        hf_config.model_type == "deepseek_v41"
        and hf_config.vision_n_layers > 0
        and not getattr(hf_config, "language_model_only", False)
    )


class DeepseekV41TokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    """[n_vit_h * n_vit_w, 3, p, p] BF16 patches and the [1, 2] ViT grid of each image."""

    models = _MODELS
    mm_item_fields = MappingProxyType(
        {"pixel_values": Modality.IMAGE, "image_grid_hw": Modality.IMAGE}
    )
    mm_feature_fields = frozenset({"pixel_values"})
    # As the legacy processor: PIL decodes, and patchify converts to RGB itself.
    gpu_image_decode = False

    def __init__(self, hf_config, processor, *, mm_process_config=None):
        super().__init__(hf_config, processor, mm_process_config=mm_process_config)
        self.mm_token_expansion_params = {
            "image_token_id": hf_config.image_token_id
            if _has_vision(hf_config)
            else None
        }

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        if self.mm_token_expansion_params["image_token_id"] is None:
            raise ValueError(
                "This checkpoint has no vision tower, so image input is not supported"
            )
        if isinstance(image, torch.Tensor):
            raise ValueError("DeepSeek-V4.1 patchifies PIL images only")
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = patchify_image(
            image, self.hf_config
        )
        return {
            "pixel_values": patches,
            "image_grid_hw": torch.tensor([[n_vit_h, n_vit_w]], dtype=torch.long),
            _EXPANSION_FIELD: [len(image_token_types(n_llm_h, n_llm_w))],
        }

    def build_processor_output(self, media_features, input_ids, *, processor):
        output = super().build_processor_output(
            media_features, input_ids, processor=processor
        )
        output.pop(_EXPANSION_FIELD, None)
        return output

    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        token_id = self.mm_token_expansion_params["image_token_id"]
        if token_id is None:
            return {}
        return {
            token_id: [
                [token_id] * count for count in media_features.get(_EXPANSION_FIELD, [])
            ]
        }


class DeepseekV41TokenSpaceServingSpec(TokenSpaceServingSpec):
    models = _MODELS

    def build_mm_tokens(self, processor):
        # Serving-only: the trainer imports this module without the serving base.
        from sglang.srt.multimodal.processors.base_processor import (
            MultimodalSpecialTokens,
        )

        if not _has_vision(self.hf_config):
            return None
        image_token_id = self.hf_config.image_token_id
        return MultimodalSpecialTokens(
            image_token=processor.convert_ids_to_tokens(image_token_id),
            image_token_id=image_token_id,
        ).build(processor)

    def split_mm_items(self, mm_items):
        """One item per image, carrying the n_vit_h / n_vit_w the model reads, as the legacy processor's items."""
        from sglang.srt.managers.schedule_batch import MultimodalDataItem

        split = []
        for item in mm_items:
            grids = torch.as_tensor(item.model_specific_data["image_grid_hw"])
            patches = item.feature.split(grids.prod(dim=-1).tolist())
            if len(item.offsets) != len(patches):
                raise ValueError(
                    f"{len(patches)} images for {len(item.offsets)} image spans"
                )
            for offsets, feature, grid in zip(item.offsets, patches, grids):
                n_vit_h, n_vit_w = (int(v) for v in grid)
                split.append(
                    MultimodalDataItem(
                        modality=item.modality,
                        feature=feature,
                        offsets=[offsets],
                        model_specific_data={
                            "image_grid_hw": grid[None],
                            "n_vit_h": n_vit_h,
                            "n_vit_w": n_vit_w,
                        },
                    )
                )
        return split
