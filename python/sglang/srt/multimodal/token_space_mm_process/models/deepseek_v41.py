"""DeepSeek-V4.1 token-space strategy: one ``image_token_id`` placeholder per image.

The placeholder becomes a run of itself; the model tells start, image, newline and end
apart by position (``image_token_types``). The caller's other IDs, which Engram hashes,
pass through unchanged.
"""

from types import MappingProxyType

import torch

from sglang.srt.multimodal.deepseek_v41_image_processing import (
    num_image_tokens,
    patchify_image,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)

# Per-image token counts the expansion reads; not a model input.
_NUM_IMAGE_TOKENS = "num_image_tokens"


class DeepseekV41TokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    """ViT patches and grids, from the legacy processor's PIL backend."""

    models = ("DeepseekV4ForCausalLM",)
    mm_item_fields = MappingProxyType(
        {
            "pixel_values": Modality.IMAGE,
            "n_vit_h": Modality.IMAGE,
            "n_vit_w": Modality.IMAGE,
        }
    )
    mm_feature_fields = frozenset({"pixel_values"})
    # As legacy: patchify_image reads a PIL image.
    gpu_image_decode = False

    # Adapted from sglang.srt.multimodal.processors.deepseek_v41.DeepseekV41ImageProcessor.__init__,
    # without the Rust and GPU image processor switch.
    def __init__(self, hf_config, processor, *, mm_process_config=None):
        super().__init__(hf_config, processor, mm_process_config=mm_process_config)
        if self.gpu_image_decode:
            raise ValueError(
                "mm_process_config image_decoder must be 'pil' for DeepSeek-V4.1, "
                "which patchifies PIL images"
            )
        self.mm_token_expansion_params = {"image_token_id": hf_config.image_token_id}

    # Legacy loads images only; reject other media before decoding it.
    def load_video(self, source, *, video_decode_device):
        raise ValueError("DeepSeek-V4.1 takes image input only, not video")

    def load_audio(self, source):
        raise ValueError("DeepSeek-V4.1 takes image input only, not audio")

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = patchify_image(
            image=image, args=self.hf_config
        )
        return {
            "pixel_values": patches,
            "n_vit_h": torch.tensor([n_vit_h]),
            "n_vit_w": torch.tensor([n_vit_w]),
            _NUM_IMAGE_TOKENS: [num_image_tokens(n_llm_h=n_llm_h, n_llm_w=n_llm_w)],
        }

    def build_processor_output(self, media_features, input_ids, *, processor):
        output = super().build_processor_output(
            media_features, input_ids, processor=processor
        )
        output.pop(_NUM_IMAGE_TOKENS, None)
        return output

    # Adapted from DeepseekV41ImageProcessor.process_mm_data_async;
    # each placeholder becomes a run of itself, one token per position of the image's span.
    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        image_token_id = self.mm_token_expansion_params["image_token_id"]
        return {
            image_token_id: [
                [image_token_id] * count
                for count in media_features.get(_NUM_IMAGE_TOKENS, [])
            ]
        }


class DeepseekV41TokenSpaceServingSpec(TokenSpaceServingSpec):
    models = DeepseekV41TokenSpaceProcessStrategy.models

    def build_mm_tokens(self, processor):
        # Serving-only: the trainer imports this module without the serving base.
        from sglang.srt.multimodal.processors.base_processor import (
            MultimodalSpecialTokens,
        )

        image_token_id = self.hf_config.image_token_id
        return MultimodalSpecialTokens(
            image_token=processor.convert_ids_to_tokens(image_token_id),
            image_token_id=image_token_id,
        ).build(processor)

    def split_mm_items(self, mm_items):
        """One item per image with its own grid, as legacy: the model encodes item by item."""
        from sglang.srt.managers.schedule_batch import MultimodalDataItem

        split = []
        for item in mm_items:
            grids = list(zip(item.n_vit_h.tolist(), item.n_vit_w.tolist(), strict=True))
            features = item.feature.split([h * w for h, w in grids])
            for offsets, feature, (h, w) in zip(
                item.offsets, features, grids, strict=True
            ):
                split.append(
                    MultimodalDataItem(
                        modality=item.modality,
                        # Own storage, as legacy: a view would pickle the whole batch.
                        feature=feature.clone(),
                        offsets=[offsets],
                        model_specific_data={"n_vit_h": h, "n_vit_w": w},
                    )
                )
        return split
