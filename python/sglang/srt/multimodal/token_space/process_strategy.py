import logging
from typing import Optional

import torch
from transformers import BatchFeature

from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders
from sglang.srt.multimodal.modality import Modality
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

logger = logging.getLogger(__name__)


def list_media_items(
    *,
    images=None,
    videos=None,
    audios=None,
    image_source_configs=None,
    video_source_configs=None,
    audio_source_configs=None,
    video_metadata=None,
):
    """Flatten loaded media into (modality, item, source_config, item_kwargs) in modality order."""
    media_items = []
    for modality, loaded_media, source_configs in (
        (Modality.IMAGE, images, image_source_configs),
        (Modality.VIDEO, videos, video_source_configs),
        (Modality.AUDIO, audios, audio_source_configs),
    ):
        loaded_media = loaded_media or []
        if source_configs is not None and len(source_configs) != len(loaded_media):
            raise ValueError("Source configs must align with the loaded media list")
        for index, item in enumerate(loaded_media):
            item_kwargs = {}
            if modality == Modality.VIDEO and video_metadata is not None:
                item_kwargs["video_metadata"] = video_metadata[index]
            source_config = source_configs[index] if source_configs else {}
            media_items.append((modality, item, source_config, item_kwargs))
    return media_items


def close_loaded_media(media_items):
    """Release decoders opened by loading once every item has been processed."""
    for _, item, _, _ in media_items:
        if isinstance(item, VideoDecoderWrapper):
            # One failed close must not leak the rest or hide the request's error.
            try:
                item.close()
            except Exception:
                logger.warning("Failed to close a loaded video decoder", exc_info=True)


def concat_media_features(features):
    """Concatenate tensors along dim 0 and chain lists, field by field, in item order."""
    return {
        name: (
            torch.cat([feature[name] for feature in features])
            if isinstance(features[0][name], torch.Tensor)
            else [value for feature in features for value in feature[name]]
        )
        for name in features[0]
    }


class TokenSpaceProcessStrategy:
    """Prepare native media fields one item at a time and expand placeholders in token space."""

    media_processor_kwargs_type = None

    def __init__(self, hf_config, processor, *, mm_process_config=None):
        self.hf_config = hf_config
        self._processor = processor
        mm_process_config = mm_process_config or {}
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

    def resolve_media_options(
        self, processor, *, image_device=None, video_device=None, **kwargs
    ) -> dict:
        """Resolve each modality's processor options once per request."""
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", True)
        kwargs_type = (
            self.media_processor_kwargs_type or processor.valid_processor_kwargs
        )
        processor_kwargs = processor._merge_kwargs(
            kwargs_type,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
            **kwargs,
        )
        image_kwargs = processor_kwargs["images_kwargs"]
        video_kwargs = processor_kwargs["videos_kwargs"]
        audio_kwargs = processor_kwargs["audio_kwargs"]
        image_kwargs.update(self.image_config)
        video_kwargs.update(self.video_config)
        audio_kwargs.update(self.audio_config)
        if image_device is not None:
            image_kwargs["device"] = image_device
            video_kwargs.setdefault("device", image_device)
            # The legacy call's common `device` reaches audio only where HF accepts it.
            if "device" in kwargs_type.__annotations__["audio_kwargs"].__annotations__:
                audio_kwargs.setdefault("device", image_device)
        if video_device is not None:
            video_kwargs["device"] = video_device
        return {
            Modality.IMAGE: image_kwargs,
            Modality.VIDEO: video_kwargs,
            Modality.AUDIO: audio_kwargs,
        }

    def process_item(
        self, modality, item, processor=None, source_config=None, **options
    ) -> dict:
        """Process one loaded media item into its own native fields."""
        processor = self._processor if processor is None else processor
        source_config = source_config or {}
        process_modality = {
            Modality.IMAGE: self.process_image,
            Modality.VIDEO: self.process_video,
            Modality.AUDIO: self.process_audio,
        }[modality]
        return process_modality(item, processor, source_config, **options)

    def merge_media(self, media_items, item_features) -> BatchFeature:
        """Assemble per-item fields, in item order, into one native BatchFeature.

        A modality with one item keeps that item's fields as they are, so the
        merge_*_features hooks run only for two or more items and must leave a
        single item unchanged.
        """
        features_by_modality = {}
        for (modality, *_), features in zip(media_items, item_features):
            features_by_modality.setdefault(modality, []).append(features)
        media_features = BatchFeature()
        for modality, features in features_by_modality.items():
            merge_modality = {
                Modality.IMAGE: self.merge_image_features,
                Modality.VIDEO: self.merge_video_features,
                Modality.AUDIO: self.merge_audio_features,
            }[modality]
            # A single item needs no assembly; returning it avoids a copy.
            modality_features = (
                features[0] if len(features) == 1 else merge_modality(features)
            )
            duplicate_keys = media_features.keys() & modality_features.keys()
            if duplicate_keys:
                raise ValueError(
                    f"Conflicting encoder input fields: {sorted(duplicate_keys)}"
                )
            media_features.update(modality_features)
        return media_features

    def process_image(self, image, processor, source_config, **kwargs):
        return dict(processor.image_processor([image], **{**kwargs, **source_config}))

    def process_video(self, video, processor, source_config, **kwargs):
        raise NotImplementedError

    def process_audio(self, audio, processor, source_config, **kwargs):
        raise NotImplementedError

    def merge_image_features(self, features: list[dict]) -> dict:
        """Batch ordered per-image fields; defaults to concatenation."""
        return concat_media_features(features)

    def merge_video_features(self, features: list[dict]) -> dict:
        """Batch ordered per-video fields; defaults to concatenation."""
        return concat_media_features(features)

    def merge_audio_features(self, features: list[dict]) -> dict:
        """Batch ordered per-audio fields; padding is model-specific."""
        raise NotImplementedError

    def expand_media_tokens(
        self,
        media_features: BatchFeature,
        *,
        input_ids: Optional[list[int]] = None,
        input_text: Optional[str] = None,
        mm_token_expansion_start_len: int = 0,
        add_special_tokens: bool = True,
        processor=None,
    ) -> list[int]:
        """Tokenize when no input_ids are given, then expand media placeholders."""
        processor = self._processor if processor is None else processor
        if input_ids is None:
            if mm_token_expansion_start_len:
                raise ValueError("Partial expansion requires input_ids.")
            tokenizer = processor.tokenizer
            if tokenizer.bos_token and input_text.startswith(tokenizer.bos_token):
                add_special_tokens = False
            input_ids = tokenizer.encode(
                input_text, add_special_tokens=add_special_tokens
            )
        mm_token_expansion_spec = self.get_mm_token_expansion_spec(
            processor, media_features
        )
        return self.mm_token_expansion(
            input_ids, mm_token_expansion_spec, mm_token_expansion_start_len
        )

    def get_mm_token_expansion_spec(self, processor, media_features):
        raise NotImplementedError

    mm_token_expansion = staticmethod(expand_token_placeholders)
