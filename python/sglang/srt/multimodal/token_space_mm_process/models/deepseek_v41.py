"""DeepSeek-V4.1 token-space strategy: one ``image_token_id`` placeholder per image.

The placeholder becomes a run of itself; the model tells start, image, newline and end
apart by position (``image_token_types``). The caller's other IDs, which Engram hashes,
pass through unchanged.
"""

from types import MappingProxyType

import msgspec
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
_NO_VISION_TOWER = (
    "This checkpoint has no vision tower, so image input is not supported"
)


class _VisionSettings(msgspec.Struct, frozen=True):
    """The config fields patchify_image reads, fixed when the strategy is built."""

    vision_patch_size: int
    vision_downsample_ratio: int
    vision_max_n_token: int
    vision_min_pixels: int
    vision_max_wh_ratio: int | None


def _check_image_device(device, *, source):
    # Patches are made on the CPU, so another device would be ignored without this.
    if device is not None and torch.device(device).type != "cpu":
        raise ValueError(
            f"DeepSeek-V4.1 patchifies on the CPU, so {source} must be 'cpu' "
            f"or unset, not {device!r}"
        )


class DeepseekV41TokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    """ViT patches and grids, from the legacy processor's PIL backend."""

    # As DeepseekV41Config normalizes it; a checkpoint's own config names DeepseekV41ForCausalLM.
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
        _check_image_device(
            self.image_config.get("device"),
            source="mm_process_config['image']['device']",
        )
        # A vision tower as the model decides; text-only checkpoints share the architecture name.
        # Serving never builds this without one; a trainer that does gets text, and an error for an image.
        has_vision_tower = (
            hf_config.model_type == "deepseek_v41"
            and hf_config.vision_n_layers > 0
            and not hf_config.language_model_only
        )
        # Patches come from this snapshot, whatever happens to hf_config afterwards.
        self._vision = (
            _VisionSettings(
                vision_patch_size=hf_config.vision_patch_size,
                vision_downsample_ratio=hf_config.vision_downsample_ratio,
                vision_max_n_token=hf_config.vision_max_n_token,
                vision_min_pixels=hf_config.vision_min_pixels,
                vision_max_wh_ratio=hf_config.vision_max_wh_ratio,
            )
            if has_vision_tower
            else None
        )
        self.mm_token_expansion_params = {
            "image_token_id": hf_config.image_token_id if has_vision_tower else None
        }

    def resolve_media_options(self, processor, *, image_device=None, video_device=None):
        options = super().resolve_media_options(
            processor, image_device=image_device, video_device=video_device
        )
        # A trainer can pass the device per call, past the check in __init__.
        # Dropping it gives a call naming "cpu" and one naming nothing one cache key.
        _check_image_device(
            options[Modality.IMAGE].pop("device", None), source="image_device"
        )
        return options

    def load_image(self, source):
        if self._vision is None:
            raise ValueError(_NO_VISION_TOWER)
        return super().load_image(source)

    # Legacy loads images only; reject other media before decoding it.
    def load_video(self, source, *, video_decode_device):
        raise ValueError("DeepSeek-V4.1 takes image input only, not video")

    def load_audio(self, source):
        raise ValueError("DeepSeek-V4.1 takes image input only, not audio")

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = patchify_image(
            image=image, args=self._vision
        )
        return {
            "pixel_values": patches,
            # On the CPU with the patches, even under a trainer's default device.
            "n_vit_h": torch.tensor([n_vit_h], device="cpu"),
            "n_vit_w": torch.tensor([n_vit_w], device="cpu"),
            # A tensor, not a list, so a sampled cache check compares it.
            _NUM_IMAGE_TOKENS: torch.tensor(
                [num_image_tokens(n_llm_h=n_llm_h, n_llm_w=n_llm_w)], device="cpu"
            ),
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
        counts = media_features.get(_NUM_IMAGE_TOKENS)
        counts = [] if counts is None else counts.tolist()
        if image_token_id is None:
            # Cache hits skip load_image, and a processor with a vision tower may have filled them.
            if counts:
                raise ValueError(_NO_VISION_TOWER)
            return {}
        return {image_token_id: [[image_token_id] * count for count in counts]}


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
                        # Own storage, as legacy: a view would pickle the whole batch,
                        # and a lone image would share its tensor with the cache.
                        feature=feature.clone(),
                        offsets=[offsets],
                        model_specific_data={"n_vit_h": h, "n_vit_w": w},
                    )
                )
        return split
