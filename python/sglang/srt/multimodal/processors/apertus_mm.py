# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The SwissAI Initiative
"""Multimodal request processing for Apertus 1.5."""

from typing import Dict, List, Optional, Union

import torch

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.models.apertus_mm import Apertus1p5ForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)

_DEFAULT_IMAGE_TOKEN_ID = 131079
_DEFAULT_AUDIO_TOKEN_ID = 131085
_DEFAULT_IMAGE_START_TOKEN_ID = 131073
_DEFAULT_IMAGE_END_TOKEN_ID = 131074
_DEFAULT_AUDIO_START_TOKEN_ID = 131080
_DEFAULT_AUDIO_END_TOKEN_ID = 131081
_DEFAULT_AUDIO_SAMPLE_RATE = 24_000


def _as_payloads(payload, *, split_batched_tensor: bool = False):
    """Return source payloads without splitting a single rank-2 embedding."""
    if isinstance(payload, (list, tuple)):
        return list(payload)
    if (
        split_batched_tensor
        and isinstance(payload, torch.Tensor)
        and payload.dim() > 2
    ):
        return list(payload.unbind(dim=0))
    return [payload]


class Apertus1p5SGLangProcessor(SGLangBaseProcessor):
    """Use the HF Apertus processor's exact discrete-token prompt expansion."""

    models = [Apertus1p5ForConditionalGeneration]
    # Apertus's HF processor batches images itself. Disabling generic GPU JPEG
    # decoding prevents mixed CUDA/CPU image tensors when a request has several images.
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=getattr(
                hf_config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID
            ),
            audio_token_id=getattr(
                hf_config, "audio_token_id", _DEFAULT_AUDIO_TOKEN_ID
            ),
        ).build(_processor)
        tokenizer = _processor.tokenizer
        self.image_start_token_id = getattr(
            tokenizer, "boi_token_id", _DEFAULT_IMAGE_START_TOKEN_ID
        )
        self.image_end_token_id = getattr(
            tokenizer, "eoi_token_id", _DEFAULT_IMAGE_END_TOKEN_ID
        )
        self.audio_start_token_id = getattr(
            tokenizer, "audio_start_token_id", _DEFAULT_AUDIO_START_TOKEN_ID
        )
        self.audio_end_token_id = getattr(
            tokenizer, "audio_end_token_id", _DEFAULT_AUDIO_END_TOKEN_ID
        )
        self.audio_sample_rate = getattr(
            getattr(_processor, "feature_extractor", None),
            "sampling_rate",
            _DEFAULT_AUDIO_SAMPLE_RATE,
        )

    def process_mm_data(
        self,
        input_text: str,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        del videos
        processor = processor or self._processor
        outputs = processor(
            text=input_text,
            images=images,
            audio=audios,
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        result = {"input_ids": outputs["input_ids"]}
        if images:
            result.update(self._normalize_image_outputs([outputs]))
        if audios:
            result.update(self._normalize_audio_outputs([outputs]))
        return result

    @staticmethod
    def _normalize_image_outputs(outputs: List[Dict]) -> Dict[str, List[torch.Tensor]]:
        """Crop one or more image-scoped HF processor outputs."""
        pixel_values = []

        for output in outputs:
            pixel_values.extend(
                image[:, : int(height), : int(width)].contiguous()
                for image, (height, width) in zip(
                    output["pixel_values"], output["image_sizes"]
                )
            )
        return {"pixel_values": pixel_values}

    @staticmethod
    def _normalize_audio_outputs(outputs: List[Dict]) -> Dict[str, List[torch.Tensor]]:
        """Trim one or more audio-scoped HF processor outputs."""
        input_features = []

        for output in outputs:
            input_features.extend(
                audio[0, : int(mask.sum())].contiguous()
                for audio, mask in zip(
                    output["input_features"], output["feature_attention_mask"]
                )
            )

        return {"input_features": input_features}

    def _get_delimited_mm_offsets(
        self,
        input_ids: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        mm_token_id: int,
    ) -> List[List[tuple[int, int]]]:
        """Find each delimited media item and its replaceable token fragments."""
        # Assumption: canonical input IDs contain ordered, non-nested Apertus
        # start/end delimiters around every source media item.
        item_regions = self.get_mm_items_offset_by_pair(
            input_ids, start_token_id, end_token_id
        )
        token_offsets = self.get_mm_items_offset(input_ids, mm_token_id)
        item_offsets = []
        offset_index = 0
        for region_start, region_end in item_regions:
            while (
                offset_index < len(token_offsets)
                and token_offsets[offset_index][1] < region_start
            ):
                offset_index += 1

            offsets = []
            while (
                offset_index < len(token_offsets)
                and token_offsets[offset_index][0] <= region_end
            ):
                if token_offsets[offset_index][1] <= region_end:
                    offsets.append(token_offsets[offset_index])
                offset_index += 1
            item_offsets.append(offsets)

        return item_offsets

    def _get_mm_item_offsets(self, input_ids: torch.Tensor, modality: Modality):
        """Derive per-media embedding spans from Apertus's expanded prompt layout."""
        if modality == Modality.IMAGE:
            start_token_id = self.image_start_token_id
            end_token_id = self.image_end_token_id
        else:
            start_token_id = self.audio_start_token_id
            end_token_id = self.audio_end_token_id

        # The boundaries assign fragments to one image/audio item; selecting only
        # the modality token excludes Apertus layout tokens such as row separators.
        return self._get_delimited_mm_offsets(
            input_ids,
            start_token_id,
            end_token_id,
            self.mm_tokens.get_token_id_by_modality(modality),
        )

    def _expand_mm_items_with_apertus_offsets(
        self, mm_items: List[MultimodalDataItem], input_ids: torch.Tensor
    ) -> None:
        """Normalize and split media items by Apertus delimiter regions."""
        for modality in (Modality.IMAGE, Modality.AUDIO):
            modality_items = [item for item in mm_items if item.modality == modality]
            if not modality_items:
                continue
            offsets_per_item = self._get_mm_item_offsets(input_ids, modality)

            expanded_items = []
            for bundled_item in modality_items:
                if bundled_item.format == MultimodalInputFormat.PROCESSOR_OUTPUT:
                    if modality == Modality.IMAGE:
                        image_sizes = bundled_item.model_specific_data.get(
                            "image_sizes"
                        )
                        payloads = self._normalize_image_outputs(
                            [
                                {
                                    "pixel_values": bundled_item.feature,
                                    "image_sizes": image_sizes,
                                }
                            ]
                        )["pixel_values"]
                    else:
                        feature_attention_mask = bundled_item.model_specific_data.get(
                            "feature_attention_mask"
                        )
                        payloads = self._normalize_audio_outputs(
                            [
                                {
                                    "input_features": bundled_item.feature,
                                    "feature_attention_mask": feature_attention_mask,
                                }
                            ]
                        )["input_features"]
                elif (
                    bundled_item.format
                    == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
                ):
                    embeddings = bundled_item.precomputed_embeddings
                    if embeddings is None:
                        # The shared collector keeps this payload in ``feature``.
                        embeddings = bundled_item.feature
                    payloads = _as_payloads(
                        embeddings,
                        split_batched_tensor=True,
                    )
                else:
                    payloads = _as_payloads(bundled_item.feature)

                for payload in payloads:
                    model_specific_data = dict(bundled_item.model_specific_data)
                    if bundled_item.format == MultimodalInputFormat.PROCESSOR_OUTPUT:
                        model_specific_data.pop("image_sizes", None)
                        model_specific_data.pop("feature_attention_mask", None)
                    is_precomputed = (
                        bundled_item.format
                        == MultimodalInputFormat.PRECOMPUTED_EMBEDDING
                    )
                    item = MultimodalDataItem(
                        modality=bundled_item.modality,
                        format=bundled_item.format,
                        feature=None if is_precomputed else payload,
                        precomputed_embeddings=payload if is_precomputed else None,
                        model_specific_data=model_specific_data,
                    )
                    expanded_items.append(item)

            if len(expanded_items) != len(offsets_per_item):
                raise ValueError(
                    "Apertus multimodal feature count does not match the number of "
                    f"{modality.name.lower()} prompt spans."
                )

            for item, offsets in zip(expanded_items, offsets_per_item):
                item.offsets = offsets
                item.hash = None
                item.pad_value = None
                if item.format != MultimodalInputFormat.NORMAL:
                    item.set_pad_value()

            mm_items[:] = [item for item in mm_items if item.modality != modality]
            mm_items.extend(expanded_items)

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.audio_sample_rate,
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        self._expand_mm_items_with_apertus_offsets(mm_items, input_ids)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )
