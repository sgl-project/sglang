from types import MappingProxyType

import torch
from transformers import BatchFeature

from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.mrope_rope_index import get_rope_index_glm4v
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)

_GLM5_NEXT_MODELS = ("Glm5NextForConditionalGeneration",)


# Adapted from sglang.srt.multimodal.processors.glm4v.Glm4vImageProcessor.__init__.
class Glm5NextTokenSpaceServingSpec(TokenSpaceServingSpec):
    models = _GLM5_NEXT_MODELS

    def build_mm_tokens(self, processor):
        # Serving-only: the trainer imports this module without the serving base.
        from sglang.srt.multimodal.processors.base_processor import (
            MultimodalSpecialTokens,
        )

        return MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=self.hf_config.image_token_id,
            video_token="<|video|>",
            # As legacy: GLM video frames hold the image token once expanded.
            video_token_id=self.hf_config.image_token_id,
        ).build(processor)


class Glm5NextTokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    """GLM-5.3 images: one <|image|> per image between <|begin_of_image|> and
    <|end_of_image|>, expanded to one <|image|> per merged patch.

    Videos stay on the legacy processor, which samples and resizes frames itself.
    """

    models = _GLM5_NEXT_MODELS
    mm_item_fields = MappingProxyType(
        {"pixel_values": Modality.IMAGE, "image_grid_thw": Modality.IMAGE}
    )
    mm_feature_fields = frozenset({"pixel_values"})
    # As legacy Glm4vImageProcessor.
    smart_rgb_conversion = True
    video_preprocessing_device = "cpu"

    def __init__(self, hf_config, processor, *, mm_process_config=None):
        super().__init__(hf_config, processor, mm_process_config=mm_process_config)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise ValueError(
                "GLM-5.3 token-space processing needs the HF Glm5NextProcessor "
                f"(transformers>=5.19), got {type(processor).__name__}"
            )
        self.mm_token_expansion_params = {
            "image_token_id": hf_config.image_token_id,
            # Legacy reports the image token as the video token too.
            "video_token_id": hf_config.image_token_id,
            # HF Glm5NextProcessor.replace_image_token reads the image processor's.
            "merge_size": image_processor.merge_size,
        }

    def resolve_media_options(
        self, processor, *, image_device=None, video_device=None
    ) -> dict:
        """Image options as the HF processor call merges them, plus the config."""
        image_kwargs = processor._merge_kwargs(
            processor.valid_processor_kwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
            return_tensors="pt",
            padding=True,
        )["images_kwargs"]
        image_kwargs.update(self.image_config)
        if image_device is not None:
            image_kwargs["device"] = image_device
        return {
            Modality.IMAGE: image_kwargs,
            Modality.VIDEO: dict(self.video_config),
            Modality.AUDIO: dict(self.audio_config),
        }

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        return dict(
            processor.image_processor([image], **{**kwargs, **preprocess_kwargs})
        )

    def process_video(self, video, processor, preprocess_kwargs, **kwargs):
        raise ValueError("GLM-5.3 token-space processing does not support video")

    def build_processor_output(
        self,
        media_features: BatchFeature,
        input_ids: list[int],
        *,
        processor,
    ) -> BatchFeature:
        """Return what processor(text, images) returns, from expanded input_ids.

        Glm5NextProcessor returns mm_token_type_ids by default. A single unpadded
        request needs no attention_mask.
        """
        text_kwargs = processor._merge_kwargs(
            processor.valid_processor_kwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        )["text_kwargs"]
        output = {"input_ids": torch.tensor([input_ids], dtype=torch.long)}
        if text_kwargs.get("return_mm_token_type_ids", False):
            output["mm_token_type_ids"] = torch.tensor(
                processor.create_mm_token_type_ids([input_ids]), dtype=torch.long
            )
        output.update(media_features)
        return BatchFeature(output)

    # Adapted from transformers' Glm5NextProcessor.replace_image_token.
    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        image_token_id = self.mm_token_expansion_params["image_token_id"]
        merge_length = self.mm_token_expansion_params["merge_size"] ** 2
        return {
            image_token_id: [
                [image_token_id] * (int(grid.prod()) // merge_length)
                for grid in media_features.get("image_grid_thw", ())
            ]
        }

    # Adapted from the M-RoPE call in sglang.srt.multimodal.processors.glm4v.Glm4vImageProcessor.process_mm_data_async.
    def compute_mrope_positions(self, output):
        """Legacy's 3-D positions; a model without mrope_section reads 1-D ones."""
        input_ids = torch.as_tensor(output["input_ids"], dtype=torch.long).flatten()
        # No attention mask: every caller ID is a real token.
        mrope_positions, mrope_position_delta = get_rope_index_glm4v(
            input_ids=input_ids.unsqueeze(0),
            hf_config=self.hf_config,
            image_grid_thw=output.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=None,
        )
        return mrope_positions.squeeze(1), mrope_position_delta
