from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from transformers import BatchFeature

from sglang.srt.multimodal.modality import Modality


def concat_media_features(item_features):
    """Concatenate tensors along dim 0 and chain lists, field by field, in item order."""
    if len(item_features) == 1:
        return dict(item_features[0])
    return {
        name: (
            torch.cat([features[name] for features in item_features])
            if isinstance(item_features[0][name], torch.Tensor)
            else [value for features in item_features for value in features[name]]
        )
        for name in item_features[0]
    }


def _to_contiguous_spans(positions: torch.Tensor) -> list[tuple[int, int]]:
    """[start, end) of each stretch of consecutive positions."""
    breaks = (positions[1:] - positions[:-1] != 1).nonzero().squeeze(1) + 1
    starts = torch.cat([positions[:1], positions[breaks]])
    ends = torch.cat([positions[breaks - 1], positions[-1:]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


class TokenSpaceProcessStrategy:
    """Per-item media processing and token-space expansion for one model family.

    Subclasses declare the class attributes below and implement `process_*` and
    `get_mm_token_expansion_spec`; the processors that run them are shared.

    `processor` is what serving or the trainer loads with `get_processor`:
    an HF processor or only a tokenizer. A strategy uses the parts its model
    has, or none.

    One instance serves every worker thread, and HF processors and Rust
    tokenizers are not thread-safe, so each worker passes its own copy as
    `processor` and binds `encode_expansion_text` to that copy's tokenizer.
    Hence `process_*`, `build_processor_output`, `get_mm_token_expansion_spec`
    and `locate_media` are static: they read only their arguments and `__init__`
    state, and never write request-level values onto `self` or `processor`.
    """

    # hf_config.architectures names this strategy serves.
    models: tuple[str, ...] = ()
    # Every output field that belongs to a media item, mapped to its modality.
    mm_item_fields: Mapping[str, Modality] = MappingProxyType({})
    # The fields of mm_item_fields that hold an item's primary feature tensor.
    mm_feature_fields: frozenset[str] = frozenset()
    # Processor class attributes for media loading, e.g. {"gpu_image_decode": False}.
    processor_class_attributes: Mapping[str, Any] = MappingProxyType({})
    # Media loading options; None keeps the loader's 16 kHz audio default.
    audio_sample_rate: int | None = None
    discard_alpha_channel: bool = True

    def __init__(self, hf_config, processor, *, mm_process_config=None):
        # `processor` is the shared original; read config from it here, never keep it.
        self.hf_config = hf_config
        mm_process_config = mm_process_config or {}
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})
        # Token IDs and sizes for expansion, e.g. image_token_id, spatial_merge_size.
        self.mm_token_expansion_params = {}

    def resolve_media_options(
        self, processor, *, image_device=None, video_device=None, **kwargs
    ) -> dict:
        """Per-modality options for one request: configured options plus devices.

        Other request kwargs are dropped unless a strategy routes them itself.
        """
        image_options = dict(self.image_config)
        video_options = dict(self.video_config)
        if image_device is not None:
            image_options["device"] = image_device
            video_options.setdefault("device", image_device)
        if video_device is not None:
            video_options["device"] = video_device
        return {
            Modality.IMAGE: image_options,
            Modality.VIDEO: video_options,
            Modality.AUDIO: dict(self.audio_config),
        }

    def process_item(
        self, modality, item, processor, preprocess_kwargs, **options
    ) -> dict:
        """Process one loaded media item into its own native fields."""
        process_modality_item = {
            Modality.IMAGE: self.process_image,
            Modality.VIDEO: self.process_video,
            Modality.AUDIO: self.process_audio,
        }[modality]
        return process_modality_item(item, processor, preprocess_kwargs, **options)

    def merge_media(self, media_items, item_features) -> BatchFeature:
        """Merge per-item fields, in item order, into one BatchFeature.

        Every modality goes through its merge_*_features hook, even with one item.
        """
        item_features_by_modality = {}
        for (modality, *_), features in zip(media_items, item_features):
            item_features_by_modality.setdefault(modality, []).append(features)
        media_features = BatchFeature()
        for modality, modality_item_features in item_features_by_modality.items():
            merge_item_features = {
                Modality.IMAGE: self.merge_image_features,
                Modality.VIDEO: self.merge_video_features,
                Modality.AUDIO: self.merge_audio_features,
            }[modality]
            modality_features = merge_item_features(modality_item_features)
            duplicate_keys = media_features.keys() & modality_features.keys()
            if duplicate_keys:
                raise ValueError(
                    f"Conflicting encoder input fields: {sorted(duplicate_keys)}"
                )
            media_features.update(modality_features)
        return media_features

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        raise NotImplementedError

    def process_video(self, video, processor, preprocess_kwargs, **kwargs):
        raise NotImplementedError

    def process_audio(self, audio, processor, preprocess_kwargs, **kwargs):
        raise NotImplementedError

    def merge_image_features(self, item_features: list[dict]) -> dict:
        """Batch ordered per-image fields; defaults to concatenation."""
        return concat_media_features(item_features)

    def merge_video_features(self, item_features: list[dict]) -> dict:
        """Batch ordered per-video fields; defaults to concatenation."""
        return concat_media_features(item_features)

    def merge_audio_features(self, item_features: list[dict]) -> dict:
        """Batch ordered per-audio fields; padding is model-specific."""
        raise NotImplementedError

    def build_processor_output(
        self,
        media_features: BatchFeature,
        input_ids: list[int],
        *,
        processor,
        **kwargs,
    ) -> BatchFeature:
        """Return the batched expanded input_ids with the media fields."""
        return BatchFeature(
            {"input_ids": torch.tensor([input_ids], dtype=torch.long), **media_features}
        )

    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        """Return {placeholder token ID: expansions of every such media, in order}.

        Each media is one placeholder token, and only it is replaced; neighbors such
        as Qwen's vision_start/end stay. `encode_expansion_text` encodes text an
        expansion inserts, e.g. Qwen3-VL timestamps, on its own, and expansions are
        spliced in as IDs: text HF would tokenize together with a neighbor is not
        merged, e.g. Gemma3's blank lines around an image. A strategy's tests
        compare its expansions with its HF processor's whole-prompt encoding.
        """
        raise NotImplementedError

    def locate_media(
        self, input_ids: list[int], media_items, mm_token_expansion_spec
    ) -> list[list[tuple[int, int]]]:
        """Return each media's tower output positions in the expanded input_ids.

        One list of [start, end) spans per entry of `media_items`, in its order.
        Within a modality, media take the tokens of their placeholder,
        mm_token_expansion_params "<modality>_token_id", in list order, as many as
        their expansions hold, so a partial expansion's prefix needs its media too.
        """
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        indices_by_modality = {}
        for index, (modality, *_) in enumerate(media_items):
            indices_by_modality.setdefault(modality, []).append(index)
        spans = [None] * len(media_items)
        for modality, indices in indices_by_modality.items():
            token_id = self.mm_token_expansion_params[
                f"{modality.name.lower()}_token_id"
            ]
            counts = [
                expansion.count(token_id)
                for expansion in mm_token_expansion_spec[token_id]
            ]
            positions = (ids == token_id).nonzero().squeeze(1)
            if positions.numel() != sum(counts):
                raise ValueError(
                    f"{modality.name}: {positions.numel()} placeholder tokens in "
                    f"input_ids for {len(indices)} media expanding to {sum(counts)}; "
                    "a partial expansion needs every media of its prefix"
                )
            for index, media_positions in zip(indices, torch.split(positions, counts)):
                spans[index] = _to_contiguous_spans(media_positions)
        return spans

    def compute_mrope_positions(self, input_ids: list[int], media_features: Mapping):
        """Return (mrope_positions, mrope_position_delta), or (None, None) without M-RoPE.

        `media_features` holds the merged media fields, e.g. grids.
        """
        return None, None
