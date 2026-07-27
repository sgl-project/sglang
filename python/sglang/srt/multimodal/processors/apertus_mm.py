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

    @staticmethod
    def _has_special_format(image_data, audio_data):
        """Return whether any input uses an offline multimodal format."""
        for data in (*(image_data or ()), *(audio_data or ())):
            if SGLangBaseProcessor._is_preprocessed_input(data):
                return True
        return False

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

    def _build_mm_items_from_outputs(self, outputs):
        input_ids = outputs["input_ids"].flatten()
        mm_items = []

        if "pixel_values" in outputs:
            image_offsets = self._get_mm_item_offsets(input_ids, Modality.IMAGE)
            if len(outputs["pixel_values"]) != len(image_offsets):
                raise ValueError(
                    "Apertus image features and placeholder regions must have the same length."
                )
            mm_items.extend(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=image,
                    offsets=offsets,
                )
                for image, offsets in zip(outputs["pixel_values"], image_offsets)
            )

        if "input_features" in outputs:
            audio_offsets = self._get_mm_item_offsets(input_ids, Modality.AUDIO)
            if len(outputs["input_features"]) != len(audio_offsets):
                raise ValueError(
                    "Apertus audio features and placeholder regions must have the same length."
                )
            mm_items.extend(
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=audio,
                    offsets=offsets,
                )
                for audio, offsets in zip(outputs["input_features"], audio_offsets)
            )

        return mm_items, input_ids

    @staticmethod
    def _split_precomputed_embeddings(artifact: Dict) -> List[torch.Tensor]:
        """Return one embedding tensor for each prompt media item."""
        embedding = artifact["feature"]
        if embedding.dim() == 2:
            return [embedding]
        # Batched embeddings must already be unpadded and ordered by prompt media.
        return list(embedding.unbind(dim=0))

    def _build_precomputed_mm_items(
        self,
        image_artifacts,
        audio_artifacts,
        input_ids: torch.Tensor,
    ):
        """Build encoder-bypassing items by consuming prompt spans in artifact order."""
        mm_items = []
        for modality, artifacts in (
            (Modality.IMAGE, image_artifacts),
            (Modality.AUDIO, audio_artifacts),
        ):
            if artifacts is None:
                continue
            expected_item_offsets = self._get_mm_item_offsets(input_ids, modality)
            offset_index = 0
            for artifact in artifacts:
                for embedding in self._split_precomputed_embeddings(artifact):
                    # Assumption: canonical input IDs already contain one
                    # Apertus start/end-delimited region per source media item.
                    item_offsets = expected_item_offsets[offset_index]
                    offset_tokens = sum(end - start + 1 for start, end in item_offsets)
                    # Assumption: each precomputed embedding is for exactly one
                    # source media item, in the same order as prompt regions.
                    if embedding.shape[0] != offset_tokens:
                        raise ValueError(
                            "Apertus precomputed embedding length does not match its "
                            "expanded prompt media item."
                        )
                    mm_items.append(
                        MultimodalDataItem(
                            modality=modality,
                            offsets=item_offsets,
                            format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                            precomputed_embeddings=embedding,
                        )
                    )
                    offset_index += 1

            if offset_index != len(expected_item_offsets):
                raise ValueError(
                    "Apertus precomputed embeddings do not cover every "
                    f"{modality.name.lower()} prompt span."
                )

        return mm_items, input_ids

    def _get_special_input_ids(self, input_text) -> torch.Tensor:
        """Tokenize an already-expanded prompt or preserve caller-supplied IDs."""
        if isinstance(input_text, list):
            return self._ensure_input_ids_is_tensor(input_text)

        # The caller has already expanded all multimodal markers in this string.
        # Only tokenize it here; do not call the HF multimodal processor again.
        add_special_tokens = True
        bos = getattr(self._tokenizer, "bos_token", None)
        if self._tokenizer_auto_adds_specials and bos and input_text.startswith(bos):
            add_special_tokens = False
        return self._tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        ).input_ids.flatten()

    def _validate_special_artifacts(self, modality: Modality, artifacts):
        """Validate one modality's artifacts without using the base one-item limit."""
        artifacts = list(artifacts or [])
        formats = {
            self._get_preprocessed_input_format(artifact) for artifact in artifacts
        }
        if None in formats:
            raise ValueError(
                f"[apertus] {modality.name.lower()} special input must contain only "
                "processor_output or precomputed_embedding artifacts."
            )
        if len(formats) > 1:
            raise ValueError(
                f"[apertus] Cannot mix processor_output and precomputed_embedding "
                f"within {modality.name.lower()} input."
            )
        return artifacts, next(iter(formats), None)

    async def _process_special_format(
        self,
        image_data,
        audio_data,
        input_text,
    ):
        """Build MM items from already-preprocessed, modality-scoped artifacts."""
        input_ids = self._get_special_input_ids(input_text)
        images, image_format = self._validate_special_artifacts(
            Modality.IMAGE, image_data
        )
        audios, audio_format = self._validate_special_artifacts(
            Modality.AUDIO, audio_data
        )
        if not images and not audios:
            return [], input_ids

        mm_items = []
        if image_format == MultimodalInputFormat.PROCESSOR_OUTPUT:
            # List order is ownership order for processor outputs. Apertus's
            # start/end markers assign each image to its expanded marker fragments.
            image_outputs = {"input_ids": input_ids}
            image_outputs.update(self._normalize_image_outputs(images))
            image_items, _ = self._build_mm_items_from_outputs(image_outputs)
            mm_items.extend(image_items)
        if audio_format == MultimodalInputFormat.PROCESSOR_OUTPUT:
            # Audio artifacts follow the same list-order contract as images.
            audio_outputs = {"input_ids": input_ids}
            audio_outputs.update(self._normalize_audio_outputs(audios))
            audio_items, _ = self._build_mm_items_from_outputs(audio_outputs)
            mm_items.extend(audio_items)

        if image_format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            image_items, _ = self._build_precomputed_mm_items(images, None, input_ids)
            mm_items.extend(image_items)
        if audio_format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
            audio_items, _ = self._build_precomputed_mm_items(None, audios, input_ids)
            mm_items.extend(audio_items)

        if self.use_cuda_ipc:
            # The special path bypasses the base processor's normal collection
            # flow, so prepare its tensor payloads for CUDA-IPC here as well.
            for item in mm_items:
                if isinstance(item.feature, torch.Tensor):
                    item.feature = self._wrap_tensor_for_cuda_ipc(item.feature)
                if isinstance(item.precomputed_embeddings, torch.Tensor):
                    item.precomputed_embeddings = self._wrap_tensor_for_cuda_ipc(
                        item.precomputed_embeddings
                    )

        return mm_items, input_ids

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        if self._has_special_format(image_data, audio_data):
            # Offline/preprocessed media already supplies processor artifacts.
            mm_items, input_ids = await self._process_special_format(
                image_data=image_data,
                audio_data=audio_data,
                input_text=input_text,
            )
        else:
            base_output = await self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                audio_data=audio_data,
                multimodal_tokens=self.mm_tokens,
            )
            images = base_output.images or []
            audios = base_output.audios or []
            if not images and not audios:
                # Text-only request: keep the generic tokenization path.
                mm_items, input_ids, _ = self.process_and_combine_mm_data(
                    base_output, self.mm_tokens
                )
            else:
                # Raw media: preserve Apertus's expanded token layout. Do not use
                # process_and_combine_mm_data here: one source image can occupy
                # non-contiguous placeholder fragments inside its delimiters.
                outputs = self.process_mm_data(
                    input_text=base_output.input_text,
                    images=base_output.images,
                    audios=base_output.audios,
                )
                mm_items, input_ids = self._build_mm_items_from_outputs(outputs)

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
        )
