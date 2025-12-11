import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessorFast

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.utils import envs, is_npu, load_audio, load_image, load_video, logger
from sglang.srt.utils.cuda_ipc_transport_utils import (
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
)

_is_npu = is_npu()

SGL_USE_CUDA_IPC = envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get()


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text with all multimodality placeholder token expanded
    input_text: str

    # frames loaded from image, in given order
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )

    # videos
    videos: Optional[list[Union[torch.Tensor, dict]]] = dataclasses.field(
        default_factory=list
    )

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = dataclasses.field(
        default_factory=list
    )

    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """

        :return: a list of results, with their corresponding modalities
        """
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[str, List[str]]] = None
    video_token: Optional[Union[str, List[str]]] = None
    audio_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    combined_regex: Optional[re.Pattern] = None

    def build(self, processor):
        self.convert_to_strs(processor)
        self.parse_regex()
        self.get_combined_regex()
        return self

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        if not self.image_token:
            self.image_token = self.convert_to_str(self.image_token_id, processor)
        if not self.video_token:
            self.video_token = self.convert_to_str(self.video_token_id, processor)
        if not self.audio_token:
            self.audio_token = self.convert_to_str(self.audio_token_id, processor)

    def get_modality_of_token(self, token: str) -> Optional[Modality]:
        """
        :return: the modality associated with the given token, if the token is a special_token or matches with the multimodal token regex
        """
        modality = {
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }.get(token)
        if modality:
            return modality

        for regex, modality in [
            (self.image_token_regex, Modality.IMAGE),
            (self.video_token_regex, Modality.VIDEO),
            (self.audio_token_regex, Modality.AUDIO),
        ]:
            if regex and regex.match(token):
                return modality

        return None

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.MULTI_IMAGES: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)

    def parse_regex(self):
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def get_combined_regex(self) -> re.Pattern:
        """
        Builds and returns a regex, used to split input str into tokens (with mm special tokens)
        """
        if self.combined_regex:
            return self.combined_regex
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
        self.combined_regex = re.compile(combined, flags)
        return self.combined_regex


class BaseMultimodalProcessor(ABC):
    models = []

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode

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
            "image_attention_mask": Modality.IMAGE,
            "image_emb_mask": Modality.IMAGE,
            "images_spatial_crop": Modality.IMAGE,
            "images_crop": Modality.IMAGE,
            "tgt_size": Modality.IMAGE,
            "image_grid_hws": Modality.IMAGE,
            "aspect_ratio_ids": Modality.IMAGE,
            "aspect_ratio_mask": Modality.IMAGE,
            "num_patches": Modality.IMAGE,
            "patch_pixel_values": Modality.IMAGE,
            "block_sizes": Modality.IMAGE,
            # Audio-related attributes
            "audio_features": Modality.AUDIO,
            "audio_feature_lens": Modality.AUDIO,
            "input_features": Modality.AUDIO,
            "input_features_mask": Modality.AUDIO,
            "audio_attention_mask": Modality.AUDIO,
            "feature_attention_mask": Modality.AUDIO,
            # Video-related attributes
            "pixel_values_videos": Modality.VIDEO,
            "second_per_grid_ts": Modality.VIDEO,
            "video_grid_thw": Modality.VIDEO,
            # Generic attributes that could apply to multiple modalities
            # "precomputed_embeddings" - handled specially as it can be any modality
        }

        # name of the feature filed
        # TODO: pass from processors
        self.FEATURE_NAMES = [
            "pixel_values",
            "pixel_values_videos",
            "audio_features",
            "input_features",
        ]

        if SGL_USE_CUDA_IPC:
            self.cudaipc_mmfeature_pool = MmItemMemoryPool(
                MM_FEATURE_CACHE_SIZE,
                MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
            )

    @property
    def spatial_merge_size(self):
        return self.hf_config.vision_config.spatial_merge_size

    def build_input_ids(self, prompt, img_grid_thw):
        """
        Use prompt and img_grid_thw to build input_ids
        """
        if not isinstance(prompt, list):
            prompt = self._processor.tokenizer.encode(prompt)

        img_token_id = self.IM_TOKEN_ID
        spatial_merge_size = self.spatial_merge_size

        input_ids = []
        offsets = []

        cur_idx = 0

        # Use img_token_id instead of im_start_id, because a dummy im_start_id
        # may be generated by the tokenizer.
        img_start_indices = list(
            filter(lambda i: prompt[i + 1] == img_token_id, range(len(prompt) - 1))
        )

        for cur_img_idx, img_start_idx in enumerate(img_start_indices):
            assert cur_idx <= img_start_idx
            # include img_start_id
            input_ids.extend(prompt[cur_idx : img_start_idx + 1])
            img_offset_start = len(input_ids)
            img_token_num = img_grid_thw[cur_img_idx].prod() // (spatial_merge_size**2)
            input_ids.extend([img_token_id] * img_token_num)
            # jump to img_end_id
            cur_idx = img_start_idx + 2
            offsets.append((img_offset_start, len(input_ids) - 1))
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets

    def get_mm_data(self, prompt, embeddings, img_grid_thw):
        input_ids, offsets = self.build_input_ids(prompt, img_grid_thw)
        mm_items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=offsets,
                precomputed_embeddings=embeddings,
            )
        ]

        return {
            "input_ids": input_ids,
            "mm_items": mm_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.IM_TOKEN_ID,
        }

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor
        """
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        if audios:
            if self._processor.__class__.__name__ in {
                "Gemma3nProcessor",
                "Qwen2AudioProcessor",
                "Qwen3OmniMoeProcessor",
            }:
                # Note(Xinyuan): for gemma3n, ref: https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/processing_gemma3n.py#L107
                kwargs["audio"] = audios
                kwargs["audio_kwargs"] = {}
                kwargs["audio_kwargs"].setdefault("truncation", False)
            else:
                kwargs["audios"] = audios

        processor = self._processor
        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessorFast)
            and not self.server_args.disable_fast_image_processor
        ):
            if not _is_npu:
                kwargs["device"] = "cuda"
            elif processor.__class__.__name__ not in {
                "Qwen2_5_VLProcessor",
                "Qwen3VLProcessor",
                "Glm4vProcessor",
                "Qwen2VLProcessor",
            }:
                # Note: for qwen-vl, processor has some reshape issue because of dims restriction on Ascend.
                kwargs["device"] = "npu"
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        if not self.server_args.keep_mm_feature_on_device:
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:
                if SGL_USE_CUDA_IPC:
                    pass
                else:
                    if feature_name in result and isinstance(
                        result[feature_name], torch.Tensor
                    ):
                        result[feature_name] = result[feature_name].to("cpu")

        return result

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
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
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.

        If data is processor_output or precomputed embedding, return directly.

        Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            data_format = data.get("format")
            if data_format in (
                MultimodalInputFormat.PROCESSOR_OUTPUT.name,
                MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
                "processor_output",
                "precomputed_embedding",
            ):
                return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data)
                if discard_alpha_channel and img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        data_iterators: dict[Modality, Iterator[Any]],
        discard_alpha_channel: bool = True,
        image_estimated_frames_iter: Optional[iter] = None,
        image_scaling_factor: float = 1.0,
        max_image_frames: int = 30,
        audio_sample_rate: Optional[int] = None,
    ) -> Tuple[List, List]:
        """
        load multimodal data parallelly using iterators.
        """
        futures = []
        task_info = []

        for text_part in text_parts:
            modality = multimodal_tokens.get_modality_of_token(text_part)
            if modality is not None:
                data_iterator = data_iterators.get(modality)
                if data_iterator is None:
                    raise ValueError(f"No data iterator found for token: {text_part}")

                try:
                    data = next(data_iterator)
                except StopIteration:
                    logger.warning(
                        f"Mismatch: More '{modality.name}' tokens found than corresponding data provided."
                    )
                    return futures, task_info

                frame_count_limit = None
                if modality == Modality.IMAGE and image_estimated_frames_iter:
                    try:
                        estimated_frames = next(image_estimated_frames_iter)
                        # Use the pre-calculated scaling factor and max frames
                        frame_count_limit = max(
                            1, int(estimated_frames * image_scaling_factor)
                        )
                        # Ensure we don't exceed the absolute max (redundant if scaling_factor handles it)
                        # frame_count_limit = min(frame_count_limit, max_image_frames)
                    except StopIteration:
                        raise ValueError(
                            "Mismatch between image tokens and estimated frame counts."
                        )

                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        modality,
                        frame_count_limit,
                        audio_sample_rate,
                        discard_alpha_channel,
                    )
                )
                task_info.append((modality, data, frame_count_limit))

        for modality, iterator in data_iterators.items():
            try:
                next(iterator)
                logger.warning(
                    f"Warning: More {modality.name.lower()} data items provided than corresponding tokens found in the prompt."
                )
            except StopIteration:
                pass
            except Exception:
                pass

        return futures, task_info

    @staticmethod
    def _validate_one_modality(modality: Modality, data_list: Optional[list]):
        if data_list is None:
            return
        if not isinstance(data_list, list):
            raise TypeError(
                f"{modality.name} must be a list or None, got {type(data_list)}"
            )

        formatted_indices = []
        for idx, item in enumerate(data_list):
            if isinstance(item, dict):
                fmt = item.get("format")
                if fmt in {"processor_output", "precomputed_embedding"}:
                    formatted_indices.append(idx)

        if formatted_indices:
            if len(data_list) != 1:
                raise ValueError(
                    f"For {modality}, when providing a 'processor_output' or "
                    f"'precomputed_embedding', you must pass exactly one item; "
                    f"received {len(data_list)} items (formatted at indices {formatted_indices})."
                )

    @staticmethod
    def validate_mm_data(
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
    ):
        """
        Validate multimodal input lists per modality.

        Rule per modality (image/video/audio):
        - Either the list has exactly one item and that single item is a dict with
          format in {"processor_output", "precomputed_embedding"};
        - Or, the list contains only "normal" items (i.e., does not include any
          item whose format is one of the two above).

        Empty or None lists are considered valid.
        """

        BaseMultimodalProcessor._validate_one_modality(Modality.IMAGE, image_data)
        BaseMultimodalProcessor._validate_one_modality(Modality.VIDEO, video_data)
        BaseMultimodalProcessor._validate_one_modality(Modality.AUDIO, audio_data)

    def _process_loaded_mm_data(self, modality, raw_data, result):
        images, videos, audios = [], [], []

        is_precomputed = isinstance(raw_data, dict) and raw_data.get("format") in [
            MultimodalInputFormat.PROCESSOR_OUTPUT.name,
            MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
            "processor_output",
            "precomputed_embedding",
        ]

        if modality == Modality.IMAGE:
            if is_precomputed:
                images.append(result)
            else:
                if isinstance(result, list):
                    images.extend(result)
                else:
                    images.append(result)
        elif modality == Modality.VIDEO:
            videos.append(result)
        elif modality == Modality.AUDIO:
            audios.append(result)

        return is_precomputed, images, videos, audios

    def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """

        BaseMultimodalProcessor.validate_mm_data(image_data, video_data, audio_data)

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._processor.tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)
        # collect all data
        data_iterators = {}
        if multimodal_tokens.image_token and image_data:
            data_iterators[Modality.IMAGE] = iter(image_data)
        if multimodal_tokens.video_token and video_data:
            data_iterators[Modality.VIDEO] = iter(video_data)
        if multimodal_tokens.audio_token and audio_data:
            data_iterators[Modality.AUDIO] = iter(audio_data)

        # futures: the futures of loaded data
        # task_info: modality, raw_data, and other metadata of each data
        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            data_iterators=data_iterators,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
        )
        task_info_iter = iter(task_info)
        futures_iter = iter(futures)

        # Process results
        images, videos, audios = [], [], []
        new_text_parts = []
        has_precomputed_input = False
        for text_part in text_parts:
            try:
                if multimodal_tokens_pattern.match(text_part):
                    modality, raw_data, frame_limit = next(task_info_iter)
                    result = next(futures_iter).result()

                    is_precomputed, new_imgs, new_vids, new_auds = (
                        self._process_loaded_mm_data(modality, raw_data, result)
                    )

                    has_precomputed_input |= is_precomputed
                    images.extend(new_imgs)
                    videos.extend(new_vids)
                    audios.extend(new_auds)

                    if modality == Modality.IMAGE:
                        if is_precomputed:
                            new_text_parts += [text_part]
                        else:
                            count = len(new_imgs)
                            if count > 0:
                                new_text_parts += [
                                    multimodal_tokens.image_token
                                ] * count
                    elif modality == Modality.VIDEO:
                        # load as video
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.video_token
                        )
                        new_text_parts += mm_tokens
                    elif modality == Modality.AUDIO:
                        # audio
                        mm_tokens = (
                            text_part
                            if is_precomputed
                            else multimodal_tokens.audio_token
                        )
                        new_text_parts += mm_tokens
                else:
                    # normal text
                    new_text_parts += [text_part]

            except StopIteration as e:
                # when precomputed_input is presented with multi-images, StopIteration is expected
                if has_precomputed_input:
                    new_text_parts += [text_part]
                    continue
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )
        return BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text="".join(new_text_parts),
        )

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

    def collect_mm_items_from_processor_output(
        self, data_dict: dict, modality: Modality = None
    ) -> List[MultimodalDataItem]:
        """
        Create mm_items directly from processor output, with one item for each modality

        Note that the data_dict can be passed via offline engine api
        """

        items: dict[Modality, MultimodalDataItem] = {}
        for attr_name, value in data_dict.items():
            if attr_name == "input_ids":
                continue

            # Get modality for this attribute
            current_modality = modality or self.ATTR_NAME_TO_MODALITY.get(attr_name)

            if attr_name == "precomputed_embeddings":
                modality_str = data_dict.get("modality")
                current_modality = Modality.IMAGE
                if modality_str:
                    try:
                        current_modality = Modality.from_str(modality_str)
                    except ValueError:
                        pass

            if current_modality:
                # Create item if needed
                if current_modality not in items:
                    items[current_modality] = MultimodalDataItem(
                        modality=current_modality
                    )

                if attr_name in self.FEATURE_NAMES:
                    attr_name = "feature"

                items[current_modality].set(attr_name, value)

        return list(items.values())

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, audios=None, videos=None, **kwargs
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
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

        return collected_items, input_ids, ret

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).

        Returns:
            Tuple of (list of mm_items, input_ids)
        """
        # Collect all items and categorize them
        all_loaded_data = base_output.organize_results()
        # Handle text-only case
        if not all_loaded_data:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return [], input_ids, {}

        dict_items, raw_images, raw_audios, raw_videos = [], [], [], []
        for modality, item in all_loaded_data:
            if isinstance(item, dict):
                dict_items.append((modality, item))
            elif modality == Modality.IMAGE:
                raw_images.append(item)
            elif modality == Modality.AUDIO:
                raw_audios.append(item)
            elif modality == Modality.VIDEO:
                raw_videos.append(item)
            else:
                raise ValueError(f"Unknown multimodal item type: {type(item)}")
        # Process items and get input_ids
        all_collected_items: list[MultimodalDataItem] = []
        input_ids = None

        # Handle raw items (need processing)
        if raw_images or raw_audios or raw_videos:
            collected_items, input_ids, ret = self._process_and_collect_mm_items(
                input_text=base_output.input_text,
                images=raw_images,
                audios=raw_audios,
                videos=raw_videos,
                **kwargs,
            )
            all_collected_items = collected_items
        else:
            ret = None

        # Handle dict items (processed or precomputed)
        for modality, dict_item in dict_items:
            input_format = dict_item.get("format", None)
            if input_format == "processor_output":
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected_items.extend(items)
            elif input_format == "precomputed_embedding":
                feature = dict_item["feature"]
                del dict_item["feature"]
                all_collected_items.append(
                    MultimodalDataItem(
                        modality=modality,
                        feature=feature,
                        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                        model_specific_data=dict_item,
                    )
                )
        # Fallback tokenization if no raw items were processed
        if input_ids is None:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        for mm_item in all_collected_items:
            mm_token_id = mm_tokens.get_token_id_by_modality(mm_item.modality)
            if mm_token_id is None:
                raise ValueError(f"No token id found for modality: {mm_item.modality}")
            mm_item.offsets = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=mm_token_id,
            )

        """
        solution for cuda-ipc memory-leak:
        1. memory-pool:  each time get a slice from memory-pool and use it as transport-data (with async lock guard)
        2. if can not get a slice , transport normal tensor
        3. copy tensor in scheduler and release it (use position mark)
        4. copy
        """

        if SGL_USE_CUDA_IPC:
            # post-process
            for item in all_collected_items:
                if isinstance(item.feature, torch.Tensor) and item.feature.is_cuda:
                    sync_flag, available_slice = (
                        self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                            item.feature
                        )
                    )
                    if isinstance(available_slice, torch.Tensor):
                        available_slice.copy_(
                            item.feature.view(torch.int8).view(-1), non_blocking=True
                        )
                        item.feature = CudaIpcTensorTransportProxy(
                            data=available_slice,
                            info_data=item.feature,
                            sync_buffer_meta=sync_flag,
                        )
                elif (
                    isinstance(item.precomputed_embeddings, torch.Tensor)
                    and item.precomputed_embeddings.is_cuda
                ):

                    sync_flag, available_slice = (
                        self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(
                            item.precomputed_embeddings
                        )
                    )
                    if isinstance(available_slice, torch.Tensor):
                        available_slice.copy_(
                            item.precomputed_embeddings.view(torch.int8).view(-1),
                            non_blocking=True,
                        )
                        item.precomputed_embeddings = CudaIpcTensorTransportProxy(
                            data=available_slice,
                            info_data=item.precomputed_embeddings,
                            sync_buffer_meta=sync_flag,
                        )

        return all_collected_items, input_ids, ret
