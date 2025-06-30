import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessorFast

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.utils import encode_video, load_audio, load_image


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image and video, in given order
    images: Optional[list[Union[Image.Image, dict]]] = None

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = None

    def normalize(self):
        for field_name in ["images", "audios"]:
            field = getattr(self, field_name, None)
            if field is not None and isinstance(field, list) and len(field) == 0:
                setattr(self, field_name, None)


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[int, str, List[str]]] = None
    video_token: Optional[Union[int, str, List[str]]] = None
    audio_token: Optional[Union[int, str, List[str]]] = None

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        self.image_token = self.convert_to_str(self.image_token, processor)
        self.video_token = self.convert_to_str(self.video_token, processor)
        self.audio_token = self.convert_to_str(self.audio_token, processor)

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    def __post_init__(self):
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def collect(self) -> re.Pattern:
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
        return re.compile(combined, flags)


class BaseMultimodalProcessor(ABC):
    models = []

    def __init__(self, hf_config, server_args, _processor):
        self.hf_config = hf_config
        self._processor = _processor
        self.arch = hf_config.architectures[0]
        self.server_args = server_args

        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

        # Mapping from attribute names to modality types
        self.ATTR_NAME_TO_MODALITY = {
            # Image-related attributes
            "pixel_values": Modality.IMAGE,
            "image_sizes": Modality.IMAGE,
            "image_grid_thw": Modality.IMAGE,
            "image_emb_mask": Modality.IMAGE,
            "image_spatial_crop": Modality.IMAGE,
            "tgt_size": Modality.IMAGE,
            "image_grid_hws": Modality.IMAGE,
            "aspect_ratio_id": Modality.IMAGE,
            "aspect_ratio_mask": Modality.IMAGE,
            "second_per_grid_ts": Modality.IMAGE,
            # Audio-related attributes
            "audio_features": Modality.AUDIO,
            "audio_feature_lens": Modality.AUDIO,
            "input_features": Modality.AUDIO,
            "input_features_mask": Modality.AUDIO,
            # Video-related attributes
            "video_grid_thws": Modality.VIDEO,
            # Generic attributes that could apply to multiple modalities
            # "precomputed_features" - handled specially as it can be any modality
        }

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        """
        process multimodal data with transformers AutoProcessor
        """
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        if audios:
            kwargs["audios"] = audios
            if self.__class__.__name__ == "Gemma3nSGLangProcessor":
                # Note(Xinyuan): for gemma3n, ref: https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/processing_gemma3n.py#L107
                kwargs["audio"] = audios

        processor = self._processor
        if hasattr(processor, "image_processor") and isinstance(
            processor.image_processor, BaseImageProcessorFast
        ):
            kwargs["device"] = "cuda"
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        if "pixel_values" in result and isinstance(
            result["pixel_values"], torch.Tensor
        ):
            result["pixel_values"] = result["pixel_values"].to("cpu")
        return result

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        pass

    def get_estimated_frames_list(self, image_data):
        """
        estimate the total frame count from all visual input
        """
        # Lazy import because decord is not available on some arm platforms.
        from decord import VideoReader, cpu

        # Before processing inputs
        if not image_data or len(image_data) == 0:
            return []
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                # Estimate frames for the video
                vr = VideoReader(path, ctx=cpu(0))
                num_frames = len(vr)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    @staticmethod
    def _load_single_item(
        data, is_video, is_audio, frame_count_limit=None, discard_alpha_channel=True
    ):
        """Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            if is_audio:
                return load_audio(data)
            elif is_video:
                path = data[len("video:") :]
                return encode_video(path, frame_count_limit)
            else:
                img, _ = load_image(data)
                return img.convert("RGB") if discard_alpha_channel else img
        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        discard_alpha_channel: bool = True,
    ):
        """
        load multimodal data parallelly
        """

        # TODO(mick): load from server_args, env, or sampling_params
        MAX_NUM_FRAMES = 30
        estimated_frames_list = self.get_estimated_frames_list(image_data=image_data)
        total_frame_count = sum(estimated_frames_list)
        # a heuristic value, suggesting the maximum fraction of frames to embed from all visual inputs.
        # e.g., 0.1 suggests that 1 frame out of 10 input frames should be used
        scaling_factor = min(1.0, MAX_NUM_FRAMES / max(1, total_frame_count))

        assert len(image_data) == len(estimated_frames_list)
        # Submit all tasks
        futures = []
        task_info = []
        image_index, audio_index = 0, 0

        for text_part in text_parts:
            if (
                multimodal_tokens.image_token_regex
                and multimodal_tokens.image_token_regex.match(text_part)
            ):
                data = image_data[image_index]
                is_video = isinstance(data, str) and data.startswith("video:")
                estimated_frames = estimated_frames_list[image_index]
                frame_count_limit = max(1, int(estimated_frames * scaling_factor))
                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        is_video,
                        False,
                        frame_count_limit,
                        discard_alpha_channel,
                    )
                )
                task_info.append((Modality.IMAGE, data, frame_count_limit))
                image_index += 1
            elif (
                multimodal_tokens.audio_token_regex
                and multimodal_tokens.audio_token_regex.match(text_part)
            ):
                data = audio_data[audio_index]
                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        False,
                        True,
                        None,
                        discard_alpha_channel,
                    )
                )
                task_info.append((Modality.AUDIO, data, None))
                audio_index += 1

        return futures, task_info

    def load_mm_data(
        self,
        prompt: str | List[int],
        multimodal_tokens: MultimodalSpecialTokens,
        max_req_input_len: int,
        image_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """
        if not return_text:
            raise NotImplementedError()
        if image_data is None:
            image_data = []

        multimodal_tokens.convert_to_strs(self._processor)
        multimodal_tokens_pattern = multimodal_tokens.collect()

        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._processor.tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)

        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            image_data=image_data,
            audio_data=audio_data,
            discard_alpha_channel=discard_alpha_channel,
        )
        # Process results
        images, audios = [], []
        new_text = ""
        task_ptr = 0

        for text_part in text_parts:
            if multimodal_tokens_pattern.match(text_part):
                task_type, data, frame_limit = task_info[task_ptr]
                result = futures[task_ptr].result()
                task_ptr += 1

                if task_type == Modality.IMAGE:
                    # If data is already processed it will be a
                    # dictionary. In this case we want to keep the
                    # expanded tokens in text_part. Otherwise, we will
                    # call the processor code, so keep only a single image
                    # token.
                    mm_tokens = (
                        text_part
                        if isinstance(data, dict)
                        else multimodal_tokens.image_token
                    )
                    frames = [result] if not isinstance(result, list) else result
                    if frames:
                        images += frames
                        new_text += mm_tokens * len(frames)
                elif task_type == Modality.AUDIO:
                    # audio
                    mm_tokens = (
                        text_part
                        if isinstance(data, dict)
                        else multimodal_tokens.audio_token
                    )
                    audios.append(result)
                    new_text += mm_tokens
                # TODO: handle video
            else:
                new_text += text_part

        out = BaseMultiModalProcessorOutput(
            input_text=new_text,
            images=images,
            audios=audios,
        )
        out.normalize()
        return out

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get a set of range for mm_items from input_ids
        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        mask = input_ids == mm_token_id

        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]

        return list(zip(start_positions.tolist(), end_positions.tolist()))

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1

        return list(zip(indices_start.tolist(), indices_end.tolist()))

    @staticmethod
    def _extract_processor_features(
        items: List[dict], attr_name: str
    ) -> Optional[torch.Tensor]:
        """
        Helper function to concat extracted attributes from processor output.
        """
        values = [value for item in items if (value := item.get(attr_name)) is not None]
        return torch.cat(values) if values else None

    # When we assume that all the items have the same attributes
    def _extract_processor_features_from_all_attributes(
        self, items: List[dict]
    ) -> dict:
        values = {}
        # Verify all items have the same keys
        first_keys = set(items[0].keys())
        for item in items[1:]:
            if set(item.keys()) != first_keys:
                raise ValueError(
                    f"All items must have the same attributes. "
                    f"First item has {first_keys}, but found {set(item.keys())}"
                )

        # Process each attribute
        for k, v in items[0].items():
            if isinstance(v, list):
                values[k] = self._extract_processor_features(items, k)
            else:
                # Verify all items have the same value for non-list attributes
                for item in items[1:]:
                    if item[k] != v:
                        raise ValueError(
                            f"All items must have the same value for attribute {k}. "
                            f"First item has {v}, but found {item[k]}"
                        )
                values[k] = v
        return values

    def collect_mm_items_from_processor_output(
        self, data_dict: dict
    ) -> List[MultimodalDataItem]:
        """Create mm_items directly from processor output."""
        items = {}  # modality -> MultimodalDataItem

        for attr_name, value in data_dict.items():
            if attr_name == "input_ids":
                continue

            # Get modality for this attribute
            modality = self.ATTR_NAME_TO_MODALITY.get(attr_name)

            if not modality and attr_name == "precomputed_features":
                modality_str = data_dict.get("modality")
                try:
                    modality = (
                        Modality.from_str(modality_str)
                        if modality_str
                        else Modality.IMAGE
                    )
                except ValueError:
                    modality = Modality.IMAGE

            if modality:
                # Create item if needed
                if modality not in items:
                    items[modality] = MultimodalDataItem(modality=modality)

                # Set attribute
                if hasattr(items[modality], attr_name):
                    setattr(items[modality], attr_name, value)

        return list(items.values())

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """
        ret = self.process_mm_data(
            input_text=input_text, images=images, audios=audios, videos=videos, **kwargs
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids

    def process_and_combine_mm_data(
        self, base_output: BaseMultiModalProcessorOutput
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor]:
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).

        Returns:
            Tuple of (list of mm_items, input_ids)
        """
        # Collect all items and categorize them
        all_items = (base_output.images or []) + (base_output.audios or [])

        # Handle text-only case
        if not all_items:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return [], input_ids

        dict_items, raw_images, raw_audios = [], [], []
        for item in all_items:
            if isinstance(item, dict):
                dict_items.append(item)
            elif isinstance(item, Image.Image):
                raw_images.append(item)
            elif isinstance(item, np.ndarray):
                raw_audios.append(item)
            else:
                raise ValueError(f"Unknown multimodal item type: {type(item)}")

        # Process items and get input_ids
        all_collected_items = []
        input_ids = None

        # Handle dict items (already processed)
        for dict_item in dict_items:
            all_collected_items.extend(
                self.collect_mm_items_from_processor_output(dict_item)
            )

        # Handle raw items (need processing)
        if raw_images or raw_audios:
            collected_items, input_ids = self._process_and_collect_mm_items(
                input_text=base_output.input_text,
                images=raw_images,
                audios=raw_audios,
            )
            all_collected_items.extend(collected_items)

        # Fallback tokenization if no raw items were processed
        if input_ids is None:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        for mm_item in all_collected_items:
            if mm_item.modality in [Modality.IMAGE, Modality.MULTI_IMAGES]:
                mm_item.image_offsets = self.get_mm_items_offset(
                    input_ids=input_ids,
                    mm_token_id=self.IM_TOKEN_ID,
                )
            elif mm_item.modality == Modality.AUDIO:
                mm_item.audio_offsets = self.get_mm_items_offset(
                    input_ids=input_ids,
                    mm_token_id=self.AUDIO_TOKEN_ID,
                )
            elif mm_item.modality == Modality.VIDEO:
                mm_item.video_offsets = self.get_mm_items_offset(
                    input_ids=input_ids,
                    mm_token_id=self.VIDEO_TOKEN_ID,
                )
            else:
                raise ValueError(f"Unknown modality: {mm_item.modality}")

        return all_collected_items, input_ids
