# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SGLang multimodal processor for Inkling models.

Adapts the HF-convention feature extraction in ``sglang.srt.multimodal.inkling`` to
SGLang's ``BaseMultimodalProcessor``: expands the MM placeholder tokens in a
pre-rendered ``input_ids`` (the chat renderer is a separate workstream) and attaches
per-item features as ``MultimodalDataItem``s. A modality is enabled only when its
``*_config.decoder_dmodel`` is set, so text-only checkpoints disable both towers.
"""

from __future__ import annotations

import base64
import logging
import urllib.request
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.inkling import InklingForConditionalGeneration
from sglang.srt.multimodal.inkling import (
    InklingAudioFeatureExtractor,
    InklingImageProcessor,
    InklingProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.parser.inkling_tokenizer import (
    AUDIO_END,
)
from sglang.srt.parser.inkling_tokenizer import AUDIO_TOKEN_ID as INKLING_AUDIO_TOKEN_ID
from sglang.srt.parser.inkling_tokenizer import IMAGE_TOKEN_ID as INKLING_IMAGE_TOKEN_ID
from sglang.srt.parser.inkling_tokenizer import (
    INKLING_SPECIAL_TOKEN_IDS,
)

logger = logging.getLogger(__name__)


def _cfg(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _resolve_media_item(item):
    """Resolve a request media item to raw bytes for Inkling preprocessing.

    The OpenAI chat path hands ImageData/AudioData objects or data:/http(s) URLs; the
    /generate path hands raw bytes or local file paths. Resolve URLs to bytes here
    (data: base64 -> exact original bytes; http(s) -> download) and pass bytes / file
    paths / PIL through unchanged for the per-modality byte loader to handle.
    """
    url = None
    if isinstance(item, str):
        url = item
    elif isinstance(item, Mapping):
        url = item.get("url")
    elif hasattr(item, "url"):
        url = getattr(item, "url")
    if not isinstance(url, str):
        return item
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        return base64.b64decode(payload) if ";base64" in header else payload.encode()
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    return url  # plain path / file:// -> handled by the per-modality byte loader


_MISSING = object()


def _require(obj, name, *, where):
    """Read a config field that MUST be present — no silent default.

    A wrong-but-silent fallback here corrupts model inputs (e.g. encoding with a
    different dmel grid than the model de-bins with yields garbage audio embeddings
    and no error), so fail loudly instead.
    """
    val = getattr(obj, name, _MISSING) if obj is not None else _MISSING
    if val is _MISSING or val is None:
        raise ValueError(
            f"InklingMultimodalProcessor: required config field {where}.{name!r} is "
            f"missing. It must be set in the model config so preprocessing matches "
            f"the model. Add it to config.json."
        )
    return val


class InklingMultimodalProcessor(SGLangBaseProcessor):
    # import_processors() registers this for the Inkling arch. Text-only checkpoints leave
    # both towers disabled (gated on *_config.decoder_dmodel), so it is a no-op there.
    models: List[Type] = [InklingForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        vision_config = _cfg(hf_config, "vision_config")
        audio_config = _cfg(hf_config, "audio_config")
        # InklingMMConfig always builds default vision/audio sub-configs (decoder_dmodel=None)
        # even for the text-only model, so gate "enabled" on decoder_dmodel being set, not
        # on the sub-config existing — else a text-only checkpoint trips the checks below.
        vision_enabled = _cfg(vision_config, "decoder_dmodel") is not None
        audio_enabled = _cfg(audio_config, "decoder_dmodel") is not None

        patch_size = _cfg(vision_config, "patch_size", 40)
        if envs.SGLANG_INKLING_RS_MM_PREPROCESS.get():
            try:
                from sglang.srt.multimodal.inkling.image_processing_rust import (
                    InklingRustImageProcessor,
                )

                image_processor = InklingRustImageProcessor(patch_size=patch_size)
                logger.info("Using Rust-accelerated Inkling image processor")
            except ImportError:
                logger.warning(
                    "SGLANG_INKLING_RS_MM_PREPROCESS=1 but sglang.srt.multimodal._core is not available; "
                    "falling back to the default image processor."
                )
                image_processor = InklingImageProcessor(patch_size=patch_size)
        else:
            image_processor = InklingImageProcessor(patch_size=patch_size)
        # The dmel grid used here at encode time must equal what the model de-bins with
        # at decode (InklingAudio reads it from audio_config); require it when audio
        # is enabled rather than guessing a default that would silently corrupt embeds.
        if audio_enabled:
            audio_params = {
                "n_mels": _require(audio_config, "n_mel_bins", where="audio_config"),
                "num_dmel_bins": _require(
                    audio_config, "mel_vocab_size", where="audio_config"
                ),
                "dmel_min_value": _require(
                    audio_config, "dmel_min_value", where="audio_config"
                ),
                "dmel_max_value": _require(
                    audio_config, "dmel_max_value", where="audio_config"
                ),
            }
        else:
            audio_params = {}
        audio_extractor = InklingAudioFeatureExtractor(params=audio_params)

        # Inkling's placeholder ids are protocol constants. Older tensor-IO checkpoints may
        # omit them from config.json, so fall back to the local renderer constants when
        # the corresponding tower is enabled.
        self.IMAGE_TOKEN_ID = _cfg(hf_config, "image_token_id")
        self.AUDIO_TOKEN_ID = _cfg(hf_config, "audio_token_id")
        self.AUDIO_END_TOKEN_ID = _cfg(hf_config, "audio_end_token_id")
        if vision_enabled and self.IMAGE_TOKEN_ID is None:
            self.IMAGE_TOKEN_ID = INKLING_IMAGE_TOKEN_ID
        if audio_enabled and self.AUDIO_TOKEN_ID is None:
            self.AUDIO_TOKEN_ID = INKLING_AUDIO_TOKEN_ID
        if audio_enabled and self.AUDIO_END_TOKEN_ID is None:
            self.AUDIO_END_TOKEN_ID = INKLING_SPECIAL_TOKEN_IDS[AUDIO_END]

        self.inkling_processor = InklingProcessor(
            image_processor=image_processor,
            audio_feature_extractor=audio_extractor,
            tokenizer=self._tokenizer,
        )

    # ---- core (pure, testable) ------------------------------------------

    def assemble(
        self,
        input_ids: List[int],
        image_data: Optional[List] = None,
        audio_data: Optional[List] = None,
    ) -> MultimodalProcessorOutput:
        """Expand single MM placeholders in ``input_ids`` into per-item token
        blocks and attach features. Walks ``input_ids`` left-to-right, consuming
        ``image_data`` / ``audio_data`` in encounter order.
        """
        image_data = image_data or []
        audio_data = audio_data or []

        # One placeholder per media item (expanded below); a count mismatch (incl. a
        # None token id absent from config) must fail loudly, not drop media silently.
        n_img_ph = (
            sum(1 for t in input_ids if t == self.IMAGE_TOKEN_ID)
            if self.IMAGE_TOKEN_ID is not None
            else 0
        )
        n_aud_ph = (
            sum(1 for t in input_ids if t == self.AUDIO_TOKEN_ID)
            if self.AUDIO_TOKEN_ID is not None
            else 0
        )
        if n_img_ph != len(image_data):
            raise ValueError(
                f"InklingMultimodalProcessor: {n_img_ph} image placeholder token(s) in "
                f"input_ids but {len(image_data)} image(s) provided; counts must match."
            )
        if n_aud_ph != len(audio_data):
            raise ValueError(
                f"InklingMultimodalProcessor: {n_aud_ph} audio placeholder token(s) in "
                f"input_ids but {len(audio_data)} audio(s) provided; counts must match."
            )

        img_feat = (
            self.inkling_processor.process_images(image_data) if image_data else None
        )
        aud_feat = (
            self.inkling_processor.process_audios(audio_data) if audio_data else None
        )

        # Rust processor returns content_hashes; original processor does not.
        img_hashes = img_feat.get("content_hashes") if img_feat else None

        out_ids: List[int] = []
        image_items: List[Tuple[int, int, torch.Tensor]] = []  # (start, end, feature)
        audio_items: List[Tuple[int, int, torch.Tensor]] = []
        i_img = i_aud = 0

        for tok in input_ids:
            if self.IMAGE_TOKEN_ID is not None and tok == self.IMAGE_TOKEN_ID:
                # hMLP folds each patch's interior to channel depth -> one token per
                # patch, so num_tokens == num_patches. Assert in case a fold is ever added.
                n_tokens = img_feat["num_tokens"][i_img]
                n_patches = img_feat["num_patches"][i_img]
                assert n_tokens == n_patches, (
                    f"num_tokens ({n_tokens}) != num_patches ({n_patches}); the hMLP "
                    f"emits one token per patch. Drive the placeholder count and the "
                    f"feature row count from a single source if a fold is added."
                )
                start = len(out_ids)
                out_ids.extend([self.IMAGE_TOKEN_ID] * n_tokens)
                # patches for this image are the i_img-th contiguous slice
                base = sum(img_feat["num_patches"][:i_img])
                feat = img_feat["vision_patches_bthwc"][base : base + n_patches]
                image_items.append((start, start + n_tokens - 1, feat))
                i_img += 1
            elif self.AUDIO_TOKEN_ID is not None and tok == self.AUDIO_TOKEN_ID:
                n_tokens = aud_feat["num_audio_tokens"][i_aud]
                start = len(out_ids)
                out_ids.extend([self.AUDIO_TOKEN_ID] * n_tokens)
                feat = aud_feat["dmel_bins"][i_aud]
                audio_items.append((start, start + n_tokens - 1, feat))
                i_aud += 1
            else:
                out_ids.append(tok)

        mm_items: List[MultimodalDataItem] = []
        for idx, (start, end, feat) in enumerate(image_items):
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=feat,
                    offsets=[(start, end)],
                    hash=img_hashes[idx] if img_hashes else None,
                )
            )
        for start, end, feat in audio_items:
            mm_items.append(
                MultimodalDataItem(
                    modality=Modality.AUDIO, feature=feat, offsets=[(start, end)]
                )
            )

        return MultimodalProcessorOutput(
            input_ids=out_ids,
            mm_items=mm_items,
            im_token_id=self.IMAGE_TOKEN_ID,
            audio_token_id=self.AUDIO_TOKEN_ID,
            audio_end_id=self.AUDIO_END_TOKEN_ID,
        )

    # ---- SGLang entrypoint ----------------------------------------------

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj: Any = None,
        *args,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        input_ids = getattr(request_obj, "input_ids", None)
        if input_ids is None:
            if self._tokenizer is None:
                raise ValueError(
                    "InklingMultimodalProcessor v1 requires pre-rendered input_ids "
                    "(request_obj.input_ids); the custom Inkling chat renderer is a "
                    "separate workstream. No tokenizer available to render text."
                )
            input_ids = self._tokenizer(input_text).input_ids
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.flatten().tolist()

        # Resolve request media (data:/http URLs, ImageData objects) to bytes so the
        # Inkling preprocessors can consume them; bytes / paths pass through unchanged.
        if image_data:
            image_data = [_resolve_media_item(it) for it in image_data]
        if audio_data:
            audio_data = [_resolve_media_item(it) for it in audio_data]
        return self.assemble(list(input_ids), image_data, audio_data)
