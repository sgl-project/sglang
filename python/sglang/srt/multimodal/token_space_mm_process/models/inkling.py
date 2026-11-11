"""Inkling token-space strategy: one in-vocabulary placeholder per image or audio.

The chat template renders an image as ``<|content_image|><|unused_200054|><|end_message|>``
and an audio as ``<|content_audio_input|><|unused_200053|><|audio_end|><|end_message|>``.
The engine's own renderer puts the out-of-vocabulary sentinels -101 / -102 at the same
spots; those go through the legacy processor, not this strategy.
"""

import base64
import os
from types import MappingProxyType

import torch

from sglang.srt.multimodal.inkling.feature_extraction import (
    InklingAudioFeatureExtractor,
)
from sglang.srt.multimodal.inkling.image_processing import InklingImageProcessor
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.multimodal_processor_mixin import _tokenizer_of
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
    concat_media_features,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)
from sglang.srt.parser.inkling_tokenizer import AUDIO_END
from sglang.srt.utils import ImageData
from sglang.srt.utils.common import download_remote_media

IMAGE_PLACEHOLDER = "<|unused_200054|>"
AUDIO_PLACEHOLDER = "<|unused_200053|>"
# Per-media token counts the expansion reads; they are not model inputs.
_EXPANSION_FIELDS = ("num_image_tokens", "num_audio_tokens")


def _token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"The tokenizer does not define {token!r}")
    return token_id


# Adapted from sglang.srt.multimodal.processors.inkling._resolve_media_item; paths are read
# here too, so the Inkling processors always get the file's bytes.
def _media_bytes(source) -> bytes:
    if isinstance(source, ImageData):
        source = source.url
    elif isinstance(source, dict):
        source = source["url"]
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if not isinstance(source, str):
        raise ValueError(f"Cannot read media bytes from {type(source).__name__}")
    if not source.strip():
        raise ValueError("empty media URL")
    if source.startswith("data:"):
        header, _, payload = source.partition(",")
        return base64.b64decode(payload) if ";base64" in header else payload.encode()
    if source.startswith(("http://", "https://")):
        # The same env the shared loaders read.
        return download_remote_media(
            source, timeout=int(os.getenv("REQUEST_TIMEOUT", "5"))
        )
    with open(source.removeprefix("file://"), "rb") as f:
        return f.read()


class InklingTokenSpaceProcessStrategy(TokenSpaceProcessStrategy):
    """hMLP image patches and dMel audio frames, decoded as the legacy processor decodes them."""

    models = ("InklingForConditionalGeneration",)
    mm_item_fields = MappingProxyType(
        {"vision_patches_bthwc": Modality.IMAGE, "dmel_bins": Modality.AUDIO}
    )
    mm_feature_fields = frozenset({"vision_patches_bthwc", "dmel_bins"})

    # Adapted from sglang.srt.multimodal.processors.inkling.InklingMultimodalProcessor.__init__,
    # without the Rust image processor switch.
    def __init__(self, hf_config, processor, *, mm_process_config=None):
        super().__init__(hf_config, processor, mm_process_config=mm_process_config)
        vision_config = hf_config.vision_config
        audio_config = hf_config.audio_config
        # A text-only checkpoint keeps both sub-configs; a tower exists only with its width.
        self._image_processor = (
            InklingImageProcessor(patch_size=vision_config.patch_size)
            if vision_config.decoder_dmodel is not None
            else None
        )
        self._audio_feature_extractor = (
            InklingAudioFeatureExtractor(
                params={
                    "n_mels": audio_config.n_mel_bins,
                    "num_dmel_bins": audio_config.mel_vocab_size,
                    "dmel_min_value": audio_config.dmel_min_value,
                    "dmel_max_value": audio_config.dmel_max_value,
                }
            )
            if audio_config.decoder_dmodel is not None
            else None
        )
        tokenizer = _tokenizer_of(processor)
        has_audio = self._audio_feature_extractor is not None
        self.mm_token_expansion_params = {
            "image_token_id": (
                _token_id(tokenizer, IMAGE_PLACEHOLDER)
                if self._image_processor is not None
                else None
            ),
            "audio_token_id": (
                _token_id(tokenizer, AUDIO_PLACEHOLDER) if has_audio else None
            ),
            "audio_end_token_id": (
                _token_id(tokenizer, AUDIO_END) if has_audio else None
            ),
        }

    # Both Inkling processors decode the media themselves, from the bytes.
    def load_image(self, source):
        if self._image_processor is None:
            raise ValueError(
                "This checkpoint has no vision tower, so image input is not supported"
            )
        return _media_bytes(source)

    def load_audio(self, source):
        if self._audio_feature_extractor is None:
            raise ValueError(
                "This checkpoint has no audio tower, so audio input is not supported"
            )
        return self._audio_feature_extractor.decode(_media_bytes(source))

    def process_image(self, image, processor, preprocess_kwargs, **kwargs):
        features = self._image_processor.preprocess([image])
        return {
            "vision_patches_bthwc": features["vision_patches_bthwc"],
            "num_image_tokens": features["num_tokens"],
        }

    def process_audio(self, audio, processor, preprocess_kwargs, **kwargs):
        dmel_bins = self._audio_feature_extractor.waveform_dmel_bins(audio)
        # As the extractor's __call__: float32 bins, one count per clip.
        return {
            "dmel_bins": dmel_bins.to(torch.float32),
            "num_audio_tokens": [int(dmel_bins.shape[0])],
        }

    def merge_audio_features(self, item_features):
        return concat_media_features(item_features)

    def build_processor_output(self, media_features, input_ids, *, processor):
        output = super().build_processor_output(
            media_features, input_ids, processor=processor
        )
        for name in _EXPANSION_FIELDS:
            output.pop(name, None)
        return output

    # Adapted from InklingMultimodalProcessor.assemble: each placeholder becomes a run of
    # itself, one token per hMLP patch or per dMel frame.
    def get_mm_token_expansion_spec(self, media_features, *, encode_expansion_text):
        params = self.mm_token_expansion_params
        spec = {}
        for token_id, counts_field in (
            (params["image_token_id"], "num_image_tokens"),
            (params["audio_token_id"], "num_audio_tokens"),
        ):
            if token_id is not None:
                spec[token_id] = [
                    [token_id] * count for count in media_features.get(counts_field, [])
                ]
        return spec


class InklingTokenSpaceServingSpec(TokenSpaceServingSpec):
    models = InklingTokenSpaceProcessStrategy.models

    def split_mm_items(self, mm_items):
        """One item per media, as legacy: a media's rows are the tokens of its own span."""
        from sglang.srt.managers.schedule_batch import MultimodalDataItem

        split = []
        for item in mm_items:
            lengths = [end - start + 1 for start, end in item.offsets]
            for offsets, feature in zip(item.offsets, item.feature.split(lengths)):
                split.append(
                    MultimodalDataItem(
                        modality=item.modality, feature=feature, offsets=[offsets]
                    )
                )
        return split
