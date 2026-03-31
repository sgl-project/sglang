from typing import List, Union

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
)
from sglang.srt.models.minicpmo import MiniCPMO
from sglang.srt.models.minicpmv import MiniCPMV
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)


# Compatible with both 'O' and 'V'
class MiniCPMMultimodalProcessor(BaseMultimodalProcessor):
    models = [MiniCPMV, MiniCPMO]
    support_dynamic_frame_expansion = True
    gpu_image_decode = False  # MiniCPM HF processor does not support tensor inputs

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.slice_start_id = getattr(tokenizer, "slice_start_id", None)
        self.slice_end_id = getattr(tokenizer, "slice_end_id", None)
        self.audio_start_id = getattr(tokenizer, "audio_start_id", None)
        self.audio_end_id = getattr(tokenizer, "audio_end_id", None)
        self.im_start_id = getattr(tokenizer, "im_start_id", None)
        self.im_end_id = getattr(tokenizer, "im_end_id", None)
        self.im_token_id = getattr(tokenizer, "unk_id", None)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="(<image>./</image>)",
            audio_token="(<audio>./</audio>)",
            video_token="(<video>./</video>)",
            image_token_id=self.im_token_id,
        ).build(_processor)

    @staticmethod
    def _has_special_format(image_data, audio_data):
        """Check if any input items use processor_output or precomputed_embedding format."""
        for data in list(image_data or []) + list(audio_data or []):
            if isinstance(data, dict) and data.get("format") in (
                "processor_output",
                "precomputed_embedding",
            ):
                return True
        return False

    async def _process_special_format(
        self, image_data, audio_data, input_text, request_obj, **kwargs
    ):
        """Handle processor_output and precomputed_embedding input formats.

        Delegates to the base class process_and_combine_mm_data which has
        built-in support for these formats.
        """
        if isinstance(input_text, list):
            user_input_ids = input_text
            prompt = ""
        else:
            user_input_ids = None
            prompt = input_text or ""

        # Normalize dicts: the HF MiniCPM processor returns "tgt_sizes" (plural)
        # but the base class ATTR_NAME_TO_MODALITY maps "tgt_size" (singular).
        # Also flatten the nested batch dimension so the structure matches
        # what the NORMAL path produces (flat list of per-patch tensors).
        normalized_images = []
        for d in image_data or []:
            if isinstance(d, dict):
                d = dict(d)
                if "tgt_sizes" in d and "tgt_size" not in d:
                    d["tgt_size"] = d.pop("tgt_sizes")
                if d.get("format") == "processor_output":
                    pixel_values = d.get("pixel_values")
                    tgt_size = d.get("tgt_size")
                    if pixel_values is not None and tgt_size is not None:
                        pv_flat, ts_flat = [], []
                        for pixel_b, tgt_b in zip(pixel_values, tgt_size):
                            if isinstance(pixel_b, (list, tuple)):
                                for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                                    pv_flat.append(pixel_n)
                                    ts_flat.append(tgt_n)
                            else:
                                pv_flat.append(pixel_b)
                                ts_flat.append(tgt_b)
                        d["pixel_values"] = pv_flat
                        d["tgt_size"] = ts_flat
                normalized_images.append(d)
            else:
                normalized_images.append(d)

        normalized_audios = list(audio_data or [])

        if not prompt and (normalized_images or normalized_audios):
            images = [d for d in normalized_images if isinstance(d, dict)]
            audios = [d for d in normalized_audios if isinstance(d, dict)]

            raw_img_dropped = len(normalized_images) - len(images)
            raw_aud_dropped = len(normalized_audios) - len(audios)
            if raw_img_dropped > 0 or raw_aud_dropped > 0:
                raise ValueError(
                    f"[minicpm] Cannot process raw media with pre-tokenized "
                    f"input_ids. Provide multimodal data in 'processor_output' or "
                    f"'precomputed_embedding' format, or use a text prompt instead. "
                    f"(raw images dropped: {raw_img_dropped}, "
                    f"raw audios dropped: {raw_aud_dropped})"
                )

            base_output = BaseMultiModalProcessorOutput(
                input_text=prompt,
                images=images,
                audios=audios,
            )
        else:
            base_output = self.load_mm_data(
                prompt=prompt,
                image_data=normalized_images,
                audio_data=audio_data,
                multimodal_tokens=self.mm_tokens,
            )

        if base_output is None:
            return None

        mm_items, input_ids_tensor, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        if user_input_ids is not None:
            input_ids_tensor = torch.tensor(user_input_ids, dtype=torch.long)
            for mm_item in mm_items:
                if mm_item.modality == Modality.IMAGE:
                    image_offsets = self.get_mm_items_offset_by_pair(
                        input_ids=input_ids_tensor,
                        mm_start_id=self.im_start_id,
                        mm_end_id=self.im_end_id,
                    )
                    slice_offsets = self.get_mm_items_offset_by_pair(
                        input_ids=input_ids_tensor,
                        mm_start_id=self.slice_start_id,
                        mm_end_id=self.slice_end_id,
                    )
                    image_offsets.extend(slice_offsets)
                    mm_item.offsets = sorted(image_offsets)
                elif mm_item.modality == Modality.AUDIO:
                    if (
                        self.audio_start_id is not None
                        and self.audio_end_id is not None
                    ):
                        mm_item.offsets = self.get_mm_items_offset_by_pair(
                            input_ids=input_ids_tensor,
                            mm_start_id=self.audio_start_id,
                            mm_end_id=self.audio_end_id,
                        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids_tensor.flatten().tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_end_id": self.audio_end_id,
            "im_token_id": self.im_token_id,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "slice_start_id": self.slice_start_id,
            "slice_end_id": self.slice_end_id,
        }

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        **kwargs,
    ):
        if isinstance(input_text, list) or self._has_special_format(
            image_data, audio_data
        ):
            return await self._process_special_format(
                image_data=image_data,
                audio_data=audio_data,
                input_text=input_text,
                request_obj=request_obj,
                **kwargs,
            )

        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        res = self.process_mm_data(
            input_text=base_output.input_text,
            images=base_output.images,
            audios=base_output.audios,
        )

        pixel_values = res["pixel_values"]
        tgt_sizes = res["tgt_sizes"]

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
            )

        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of target sizes. " f"Got type: {type(tgt_sizes)}"
            )

        if len(pixel_values) != len(tgt_sizes):
            raise ValueError(
                "Inconsistent batch lengths, found: "
                f"{len(pixel_values)} vs. {len(tgt_sizes)}"
            )

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
            # per image
            if len(pixel_b) != len(tgt_b):
                raise ValueError(
                    "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                )
            for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                pixel_values_flat += [pixel_n]
                tgt_sizes_flat += [tgt_n]

        pixel_values = pixel_values_flat

        items = []
        input_ids = res["input_ids"].flatten()
        image_offsets = self.get_mm_items_offset_by_pair(
            input_ids=input_ids, mm_start_id=self.im_start_id, mm_end_id=self.im_end_id
        )
        slice_offsets = self.get_mm_items_offset_by_pair(
            input_ids=input_ids,
            mm_start_id=self.slice_start_id,
            mm_end_id=self.slice_end_id,
        )
        image_offsets.extend(slice_offsets)
        image_offsets = sorted(image_offsets)

        if len(pixel_values) != 0:
            item = MultimodalDataItem(
                feature=pixel_values,
                offsets=image_offsets,
                model_specific_data={"tgt_size": tgt_sizes_flat},
                modality=Modality.IMAGE,
            )
            items += [item]

        if (
            "audio_features" in res
            and res["audio_features"] is not None
            and len(res["audio_features"]) != 0
        ):
            if self.audio_start_id is not None and self.audio_end_id is not None:
                audio_offsets = self.get_mm_items_offset_by_pair(
                    input_ids=input_ids,
                    mm_start_id=self.audio_start_id,
                    mm_end_id=self.audio_end_id,
                )
            else:
                audio_offsets = None
            item = MultimodalDataItem(
                feature=[res["audio_features"]],
                model_specific_data={"audio_feature_lens": res["audio_feature_lens"]},
                offsets=audio_offsets,
                modality=Modality.AUDIO,
            )
            items += [item]
        return {
            "mm_items": items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_end_id": self.audio_end_id,
            "im_token_id": self.im_token_id,
            "im_start_id": self.im_start_id,
            "im_end_id": self.im_end_id,
            "slice_start_id": self.slice_start_id,
            "slice_end_id": self.slice_end_id,
        }
