import asyncio
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
from transformers import BaseImageProcessor

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.runtime_context import get_device, get_exec, get_mm, get_serving
from sglang.srt.utils import (
    envs,
    is_cpu,
    is_npu,
    is_xpu,
    load_audio,
    load_image,
    load_video,
    logger,
)
from sglang.srt.utils.cuda_ipc_transport_utils import (
    MM_FEATURE_CACHE_SIZE,
    MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
    CudaIpcTensorTransportProxy,
    MmItemMemoryPool,
    get_mm_feature_pool_size_per_worker,
)

_is_cpu = is_cpu()
_is_npu = is_npu()
_is_xpu = is_xpu()

_IPC_POOL_HANDLE_CACHE = envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get()


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text with all multimodality placeholder token expanded
    input_text: str

    # original pre-tokenized ids, useful for processor_output/precomputed inputs,
    # when they already carry the input ids
    input_ids: Optional[Union[List[int], torch.Tensor]] = None

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
    gpu_image_decode = True  # Enable GPU decoding by default
    auto_mm_processor_worker_num = 1
    auto_mm_io_worker_num = 4
    supports_mm_processor_concurrency = False

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode
        self.keep_mm_feature_on_device = server_args.keep_mm_feature_on_device
        configured_mm_feature_transport = getattr(
            server_args, "mm_feature_transport", "cpu"
        )
        self.mm_feature_transport = (
            configured_mm_feature_transport
            if configured_mm_feature_transport in ("cpu", "cuda_ipc")
            else "cpu"
        )
        self.use_cuda_ipc = self.mm_feature_transport == "cuda_ipc"
        self.disable_fast_image_processor = server_args.disable_fast_image_processor
        self.skip_tokenizer_init = server_args.skip_tokenizer_init

        mm_process_config = get_mm().mm_process_config
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

        # Resolve tokenizer: some processors (e.g. InternVL) pass a tokenizer
        # directly as _processor rather than a processor that wraps a tokenizer.
        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor

        # Same guard as in serving_chat.py against double BOS.
        try:
            self._tokenizer_auto_adds_specials = len(self._tokenizer.encode("")) > 0
        except Exception:
            self._tokenizer_auto_adds_specials = False

        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        requested_mm_io_worker_num = self.server_args.mm_io_worker_num
        env_mm_io_worker_num = os.environ.get("SGLANG_IO_WORKERS")
        if requested_mm_io_worker_num:
            self.mm_io_worker_num = requested_mm_io_worker_num
            io_worker_source = "explicit"
        elif env_mm_io_worker_num is not None:
            self.mm_io_worker_num = int(env_mm_io_worker_num)
            io_worker_source = "environment"
        else:
            self.mm_io_worker_num = self.auto_mm_io_worker_num
            io_worker_source = "auto"
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.mm_io_worker_num,
            thread_name_prefix="sglang-mm-io",
        )
        if self.mm_io_worker_num > 4:
            logger.info(
                "Multimodal data loading enabled with %d worker threads (%s).",
                self.mm_io_worker_num,
                io_worker_source,
            )
        skip_mm_pool = kwargs.get("skip_mm_pool", False)
        requested_mm_processor_worker_num = self.server_args.mm_processor_worker_num
        self.mm_processor_worker_num = (
            1
            if skip_mm_pool
            else requested_mm_processor_worker_num or self.auto_mm_processor_worker_num
        )
        if (
            self.mm_processor_worker_num > 1
            and not self.supports_mm_processor_concurrency
        ):
            logger.warning(
                "Concurrent multimodal processing is not supported by %s; "
                "using synchronous processing.",
                type(self).__name__,
            )
            self.mm_processor_worker_num = 1
        self.mm_processor_executor = None
        if self.mm_processor_worker_num > 1:
            try:
                self.mm_processor_executor = MultimodalProcessorExecutor(
                    self._processor, self.mm_processor_worker_num
                )
            except Exception:
                logger.warning(
                    "Unable to clone the multimodal processor for concurrent "
                    "workers; falling back to synchronous processing.",
                    exc_info=True,
                )
                self.mm_processor_worker_num = 1
        if self.mm_processor_executor is not None:
            logger.info(
                "Multimodal processor concurrency enabled with %d isolated "
                "worker threads (%s).",
                self.mm_processor_worker_num,
                "auto" if requested_mm_processor_worker_num == 0 else "explicit",
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
            "has_local_crops": Modality.IMAGE,
            "has_images": Modality.IMAGE,
            "tgt_size": Modality.IMAGE,
            "image_grid_hws": Modality.IMAGE,
            "aspect_ratio_ids": Modality.IMAGE,
            "aspect_ratio_mask": Modality.IMAGE,
            "num_patches": Modality.IMAGE,
            "patch_pixel_values": Modality.IMAGE,
            "block_sizes": Modality.IMAGE,
            "grid_thws": Modality.IMAGE,  # for kimi k2.5
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

        if self.use_cuda_ipc and not skip_mm_pool:
            # SGLANG_MM_FEATURE_CACHE_MB is the total pool budget across all
            # tokenizer workers. Each worker gets an equal share so that adding
            # workers doesn't multiply the GPU-side footprint.
            worker_num = get_serving().tokenizer_worker_num
            per_worker_pool_size = get_mm_feature_pool_size_per_worker(
                MM_FEATURE_CACHE_SIZE, worker_num
            )
            total_pool_size = per_worker_pool_size * worker_num
            logger.info(
                "CUDA IPC multimodal feature pools reserve %.0f MiB total on "
                "GPU %d (%.0f MiB per tokenizer worker × %d; configured "
                "budget %.0f MiB).",
                total_pool_size / (1024 * 1024),
                get_device().base_gpu_id,
                per_worker_pool_size / (1024 * 1024),
                worker_num,
                MM_FEATURE_CACHE_SIZE / (1024 * 1024),
            )
            self.cudaipc_mmfeature_pool = MmItemMemoryPool(
                per_worker_pool_size,
                MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL,
                get_device().base_gpu_id,
            )

    def compute_mrope_positions(self, input_ids, mm_items):
        """Compute M-RoPE positions from expanded input_ids and multimodal items.

        Returns (mrope_positions, mrope_position_delta) or (None, None) if the
        model does not use M-RoPE.
        """
        return None, None

    @property
    def spatial_merge_size(self):
        return self.hf_config.vision_config.spatial_merge_size

    def build_input_ids(
        self, prompt, img_grid_thw=None, video_grid_thw=None, audio_seq_lens=None
    ):
        """
        Use prompt, img_grid_thw, video_grid_thw, and audio_seq_lens to build input_ids.
        Supports image, video, and audio tokens.
        """
        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        audio_token_id = getattr(self, "audio_token_id", None)
        spatial_merge_size = getattr(self, "spatial_merge_size", 1)

        input_ids = []
        offsets = []

        cur_idx = 0

        # Use img_token_id instead of im_start_id, because a dummy im_start_id
        # may be generated by the tokenizer.
        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))
            elif audio_token_id is not None and prompt[i + 1] == audio_token_id:
                vision_start_indices.append((i, Modality.AUDIO))
        # get modality list with order preserved
        modality_list = [modality for _, modality in vision_start_indices]

        img_idx = 0
        video_idx = 0
        audio_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            if modality == Modality.IMAGE:
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
                mm_token_id = img_token_id
                img_idx += 1
            elif modality == Modality.VIDEO:
                mm_token_num = video_grid_thw[video_idx].prod() // (
                    spatial_merge_size**2
                )
                mm_token_id = video_token_id
                video_idx += 1
            elif modality == Modality.AUDIO:
                mm_token_num = int(audio_seq_lens[audio_idx].item())
                mm_token_id = audio_token_id
                audio_idx += 1
            else:
                raise ValueError(f"Invalid modality: {modality}")
            assert cur_idx <= mm_start_idx

            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            mm_offset_start = len(input_ids)
            input_ids.extend([mm_token_id] * mm_token_num)
            cur_idx = (
                mm_start_idx + 2
            )  # jump to img_end_id, video_end_id, or audio_end_id
            offsets.append((mm_offset_start, len(input_ids) - 1))
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)

        input_ids, offsets, modality_list = self.build_input_ids(
            prompt,
            img_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            audio_seq_lens=audio_feature_lens,
        )
        assert all(isinstance(modality, Modality) for modality in modality_list)

        mm_items = []
        consumed_per_modality = {}

        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            embedding_start = consumed_per_modality.get(modality, 0)
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.IM_TOKEN_ID,
            video_token_id=getattr(self, "VIDEO_TOKEN_ID", None),
        )

    def _resolve_processor(self, processor=None):
        if processor is None:
            return self._processor, self._tokenizer
        return processor, processor.tokenizer

    def process_mm_data(
        self,
        input_text,
        images=None,
        videos=None,
        audios=None,
        processor=None,
        **kwargs,
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor
        """
        processor, tokenizer = self._resolve_processor(processor)

        if images:
            kwargs["images"] = images
            if self.image_config:
                kwargs.setdefault("images_kwargs", {}).update(self.image_config)
        if videos:
            kwargs["videos"] = videos
            if self.video_config:
                kwargs.setdefault("videos_kwargs", {}).update(self.video_config)
        if audios:
            if processor.__class__.__name__ in {
                "Gemma3nProcessor",
                "Gemma4Processor",
                "Gemma4UnifiedProcessor",
                "GlmAsrProcessor",
                "Qwen2AudioProcessor",
                "Qwen3ASRProcessor",
                "Qwen3OmniMoeProcessor",
            }:
                # Note(Xinyuan): for gemma3n, ref: https://github.com/huggingface/transformers/blob/ccf2ca162e33f381e454cdb74bf4b41a51ab976d/src/transformers/models/gemma3n/processing_gemma3n.py#L107
                kwargs["audio"] = audios
                kwargs.setdefault("audio_kwargs", {})
                kwargs["audio_kwargs"].setdefault("truncation", False)
            else:
                kwargs["audios"] = audios
            if self.audio_config:
                kwargs.setdefault("audio_kwargs", {}).update(self.audio_config)

        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessor)
            and not self.disable_fast_image_processor
        ):
            if _is_cpu or get_exec().deterministic.rl_on_policy_target is not None:
                kwargs["device"] = "cpu"
            elif _is_xpu:
                kwargs["device"] = "xpu"
            elif not _is_npu:
                base_gpu_id = get_device().base_gpu_id
                kwargs["device"] = f"cuda:{base_gpu_id}"
            elif processor.__class__.__name__ not in {
                "Glm4vProcessor",
                "Glm46VProcessor",
            }:
                # Note: for qwen-vl, processor has some reshape issue because of dims restriction on Ascend.
                from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
                    npu_apply_qwen_image_preprocess_patch,
                )

                npu_apply_qwen_image_preprocess_patch()
                kwargs["device"] = "npu"
            elif processor.__class__.__name__ == "Glm46VProcessor":
                from sglang.srt.hardware_backend.npu.modules.glm46v_processor import (
                    npu_apply_glm46v_image_preprocess_patch,
                )

                npu_apply_glm46v_image_preprocess_patch()
                kwargs["device"] = "npu"

        # Avoid double BOS when the chat template already wrote one.
        if self._tokenizer_auto_adds_specials and isinstance(input_text, str):
            bos = getattr(tokenizer, "bos_token", None)
            if bos and input_text.startswith(bos):
                kwargs.setdefault("add_special_tokens", False)

        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )
        if not self.keep_mm_feature_on_device:
            # move feature tensors to cpu
            for feature_name in self.FEATURE_NAMES:
                if self.use_cuda_ipc:
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
        from sglang.srt.utils.video_decoder import VideoDecoderWrapper

        # Before processing inputs
        if not image_data or len(image_data) == 0:
            return []
        estimated_frames_list = []
        for image in image_data:
            if isinstance(image, str) and image.startswith("video:"):
                path = image[len("video:") :]
                decoder = VideoDecoderWrapper(path)
                num_frames = len(decoder)
            else:
                # For images, each contributes one frame
                num_frames = 1
            estimated_frames_list.append(num_frames)

        return estimated_frames_list

    @classmethod
    def _load_single_item(
        cls,
        data,
        modality: Modality,
        frame_count_limit=None,
        audio_sample_rate: Optional[int] = None,
        discard_alpha_channel=True,
    ):
        """
        Load a single multimodal data.

        If data is processor_output or precomputed embedding, return directly.

        Class method that can be pickled for multiprocessing
        """
        if cls._is_preprocessed_input(data):
            return data
        try:
            if modality == Modality.IMAGE:
                img, _ = load_image(data, cls.gpu_image_decode)
                if isinstance(img, torch.Tensor):
                    return img  # JPEG already decoded on GPU by nvJPEG
                # PIL decodes lazily; do it here in the io worker so the decode
                # doesn't run later on the event-loop thread.
                if discard_alpha_channel and img.mode != "RGB":
                    return img.convert("RGB")
                img.load()
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, frame_count_limit)
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except ValueError as e:
            # Bad input (e.g. invalid base64) -> 400, not 500.
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise ValueError(f"Error while loading data {data_str}: {e}") from e
        except Exception as e:
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            raise RuntimeError(f"Error while loading data {data_str}: {e}") from e

    @staticmethod
    def _get_preprocessed_input_format(data):
        """returns the detailed format if the provided data is already preprocessed.
        returns none if the provided data is not preprocessed
        """
        if not isinstance(data, dict):
            return None
        data_format = data.get("format")
        if isinstance(data_format, MultimodalInputFormat):
            return data_format
        if data_format in (
            MultimodalInputFormat.PROCESSOR_OUTPUT.name,
            "processor_output",
        ):
            return MultimodalInputFormat.PROCESSOR_OUTPUT
        if data_format in (
            MultimodalInputFormat.PRECOMPUTED_EMBEDDING.name,
            "precomputed_embedding",
        ):
            return MultimodalInputFormat.PRECOMPUTED_EMBEDDING
        return None

    @classmethod
    def _is_preprocessed_input(cls, data):
        """returns if the data is already preprocessed (by the vlm processor)"""
        return cls._get_preprocessed_input_format(data) is not None

    @classmethod
    def _all_mm_data_is_preprocessed(cls, *data_lists):
        has_mm_data = False
        for data_list in data_lists:
            if not data_list:
                continue
            if not isinstance(data_list, list):
                data_list = [data_list]
            for item in data_list:
                if item is None:
                    continue
                has_mm_data = True
                if not cls._is_preprocessed_input(item):
                    return False
        return has_mm_data

    def _submit_mm_data_loading_tasks_simple(
        self,
        data_list: Optional[list],
        modality: Modality,
        audio_sample_rate: Optional[int],
        discard_alpha_channel: bool,
    ) -> List[Tuple[Modality, int, concurrent.futures.Future]]:
        """
        Simple version: For one modal data submit IO load task.
        Return:
            List[(modality, index_in_that_modality, future)]
        """
        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []

        if not data_list:
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] no data for modality=%s",
                modality.name,
            )
            return futures

        for idx, data in enumerate(data_list):
            logger.debug(
                "[_submit_mm_data_loading_tasks_simple] submit load task: "
                "modality=%s, index=%d, data_type=%s",
                modality.name,
                idx,
                type(data),
            )
            future = self.io_executor.submit(
                self.__class__._load_single_item,
                data,
                modality,
                None,  # frame_count_limit: no consider for fast path
                audio_sample_rate,
                discard_alpha_channel,
            )
            futures.append((modality, idx, future))

        return futures

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
                        self.__class__._load_single_item,
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
            if BaseMultimodalProcessor._is_preprocessed_input(item):
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

        is_precomputed = self._is_preprocessed_input(raw_data)

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

    async def load_mm_data(
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

        BaseMultimodalProcessor.validate_mm_data(image_data, video_data, audio_data)

        input_ids = prompt if isinstance(prompt, list) else None
        if input_ids is not None and self._all_mm_data_is_preprocessed(
            image_data, video_data, audio_data
        ):
            # fast path for preprocessed data: early return
            return BaseMultiModalProcessorOutput(
                input_text="",
                input_ids=input_ids,
                images=list(image_data or []),
                videos=list(video_data or []),
                audios=list(audio_data or []),
            )

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)

        cnt = {Modality.IMAGE: 0, Modality.VIDEO: 0, Modality.AUDIO: 0}
        for text_part in text_parts:
            modality = multimodal_tokens.get_modality_of_token(text_part)
            if modality is not None:
                cnt[modality] += 1

        n_image = len(image_data) if image_data else 0
        n_video = len(video_data) if video_data else 0
        n_audio = len(audio_data) if audio_data else 0

        # For MiniCPMO and MiniCPMV or multimodal_tokens not totally align, legacy show path
        if (
            self.skip_tokenizer_init
            or cnt[Modality.IMAGE] != n_image
            or cnt[Modality.VIDEO] != n_video
            or cnt[Modality.AUDIO] != n_audio
            or getattr(self, "support_dynamic_frame_expansion", False)
        ):
            return await self.legacy_load_mm_data(
                prompt=prompt,
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                return_text=return_text,
                discard_alpha_channel=discard_alpha_channel,
                audio_sample_rate=audio_sample_rate,
                input_ids=input_ids,
            )
        # For models other than MiniCPMO and MiniCPMV,
        # totally align multimodal_tokens, fast path
        return await self.fast_load_mm_data(
            prompt=prompt,
            multimodal_tokens=multimodal_tokens,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            return_text=return_text,
            discard_alpha_channel=discard_alpha_channel,
            audio_sample_rate=audio_sample_rate,
            input_ids=input_ids,
        )

    async def fast_load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
        input_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        A fast version of `load_mm_data` that loads multimodal data directly.
        This version does not scan the prompt to recognize tokens. It assumes
        that the caller has already aligned the tokens and data in a 1:1 manner.
        The behavior is as follows:
          1. It runs `_load_single_item` for all input data concurrently.
          2. It returns the loaded images, videos, and audios in their original order.
          3. It returns the input prompt as a string.
        """

        # Convert prompt into str
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt_str = self._tokenizer.decode(prompt)
        else:
            assert isinstance(prompt, str)
            prompt_str = prompt

        futures: List[Tuple[Modality, int, concurrent.futures.Future]] = []

        modalities_data = [
            (image_data, Modality.IMAGE),
            (video_data, Modality.VIDEO),
            (audio_data, Modality.AUDIO),
        ]

        for data_list, modality in modalities_data:
            futures.extend(
                self._submit_mm_data_loading_tasks_simple(
                    data_list, modality, audio_sample_rate, discard_alpha_channel
                )
            )

        logger.debug("[load_mm_data(simple)] total futures submitted: %d", len(futures))

        images: List[Any] = [None] * len(image_data) if image_data else []
        videos: List[Any] = [None] * len(video_data) if video_data else []
        audios: List[Any] = [None] * len(audio_data) if audio_data else []

        for modality, idx, future in futures:
            try:
                result = await asyncio.wrap_future(future)
            except Exception as e:
                logger.exception(
                    "[load_mm_data(simple)] error loading %s data at index=%d",
                    modality.name,
                    idx,
                )
                raise RuntimeError(
                    f"An exception occurred while loading {modality.name} data at index {idx}: {e}"
                )

            if modality == Modality.IMAGE:
                images[idx] = result
            elif modality == Modality.VIDEO:
                videos[idx] = result
            elif modality == Modality.AUDIO:
                audios[idx] = result

        logger.debug(
            "[load_mm_data(simple)] loaded counts: images=%d, videos=%d, audios=%d",
            len(images),
            len(videos),
            len(audios),
        )

        return BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text=prompt_str,
            input_ids=input_ids,
        )

    async def legacy_load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        video_data: Optional[list] = None,
        audio_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
        audio_sample_rate: Optional[int] = None,
        input_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """

        multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._tokenizer.decode(prompt)
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
                    result = await asyncio.wrap_future(next(futures_iter))

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
            input_ids=input_ids,
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
        Create mm_items from processor output.

        Initially creates one item per modality; these are later split into per-image/video items by get_new_expanded_mm_items.

        Note that the data_dict can be hf processor output, or passed via offline engine api

        Args:
            modality: if provided, force the data into a single MultimodalDataItem of that modality
        """

        # universal getter for data_dict
        get_data_value = (
            data_dict.get
            if hasattr(data_dict, "get")
            else lambda name, default=None: getattr(data_dict, name, default)
        )

        # decide explicitly-set modality
        explicit_modality = modality
        modality_value = get_data_value("modality")
        if explicit_modality is None and modality_value is not None:
            explicit_modality = (
                modality_value
                if isinstance(modality_value, Modality)
                else Modality.from_str(str(modality_value))
            )

        items: dict[Modality, MultimodalDataItem] = {}
        for attr_name, value in data_dict.items():
            if attr_name in (
                "input_ids",
                "format",
                "modality",
                "hash",
                "pad_value",
                "offsets",
            ):
                # metadata fields need explicit handling, skip generic item.set
                continue

            # Get modality for this attribute
            current_modality = explicit_modality or self.ATTR_NAME_TO_MODALITY.get(
                attr_name
            )

            if attr_name == "precomputed_embeddings":
                current_modality = current_modality or Modality.IMAGE

            if current_modality:
                # Create item if needed
                if current_modality not in items:
                    items[current_modality] = MultimodalDataItem(
                        modality=current_modality
                    )

                if attr_name in self.FEATURE_NAMES:
                    attr_name = "feature"

                items[current_modality].set(attr_name, value)

        # deal with metadata fields when data_dict is preprocessed input: convert from tensor to expected python types
        # the attribution of the metadata fields is only clear when number of MultimodalDataItem is 1
        if len(items) == 1:
            item = next(iter(items.values()))

            # adjust offset
            offsets = get_data_value("offsets")
            if offsets is not None:
                if isinstance(offsets, torch.Tensor):
                    offsets = offsets.detach().cpu().tolist()
                item.offsets = [(int(start), int(end)) for start, end in offsets]

            # adjust hash_value
            hash_value = get_data_value("hash")
            if hash_value is not None:
                if isinstance(hash_value, torch.Tensor):
                    hash_value = hash_value.item()
                item.hash = int(hash_value)
                pad_value = get_data_value("pad_value")
                if pad_value is not None:
                    if isinstance(pad_value, torch.Tensor):
                        pad_value = pad_value.item()
                    item.pad_value = int(pad_value)

        return list(items.values())

    def _process_and_collect_mm_items(
        self,
        input_text: str,
        images=None,
        audios=None,
        videos=None,
        processor=None,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """
        if processor is not None:
            kwargs["processor"] = processor
        ret = self.process_mm_data(
            input_text=input_text,
            images=images,
            audios=audios,
            videos=videos,
            **kwargs,
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret

    @staticmethod
    def _ensure_input_ids_is_tensor(input_ids) -> Optional[torch.Tensor]:
        """make sure the input_ids is a flattened tensor"""
        if input_ids is None:
            return None
        if isinstance(input_ids, torch.Tensor):
            return input_ids.flatten().to(dtype=torch.long)
        return torch.tensor(input_ids, dtype=torch.long).flatten()

    def _wrap_tensor_for_cuda_ipc(self, tensor: torch.Tensor):
        """helper function to turn a tensor into a cuda-ipc tensor"""
        if not tensor.is_cuda:
            return tensor

        sync_flag, available_slice, byte_offset = (
            self.cudaipc_mmfeature_pool.return_a_slice_tensor_with_flag(tensor)
        )
        if isinstance(available_slice, torch.Tensor):
            available_slice.copy_(tensor.view(torch.int8).view(-1), non_blocking=True)
            return CudaIpcTensorTransportProxy(
                data=available_slice,
                info_data=tensor,
                sync_buffer_meta=sync_flag,
                pool_ipc_handle=(
                    self.cudaipc_mmfeature_pool._pool_ipc_handle
                    if _IPC_POOL_HANDLE_CACHE
                    else None
                ),
                pool_byte_offset=byte_offset,
                pool_device_index=self.cudaipc_mmfeature_pool._pool_device_index,
            )
        if self.keep_mm_feature_on_device:
            return tensor
        return tensor.cpu()

    def resolve_image_token_counts(self, images: List) -> List[int]:
        """Per-image expanded token counts, computed without re-tokenizing.

        Default implementation uses the transformers in-tree convention
        ``_get_num_multimodal_tokens(image_sizes=...)`` (present on the in-tree
        VLM processors, e.g. Qwen-VL, Gemma3, GLM4V). Models whose processor
        does not implement it (e.g. Kimi) override this method.

        """
        assert images is not None
        image_sizes = [(image.height, image.width) for image in images]
        num_image_tokens = self._processor._get_num_multimodal_tokens(
            image_sizes=image_sizes
        ).num_image_tokens
        return [int(count) for count in num_image_tokens]

    @staticmethod
    def _expand_input_ids(
        original_ids: List[int],
        counts: List[int],
        placeholder_token_id: Optional[int],
    ) -> List[int]:
        """Rebuild final input_ids for a pre-tokenized (list[int]) prompt.

        Keep the user's ORIGINAL tokens verbatim and expand the i-th image
        placeholder into ``counts[i]`` copies of ``placeholder_token_id``. The HF
        processor's re-tokenization is discarded, so non-media tokens cannot
        drift.

        """
        if placeholder_token_id is None:
            raise ValueError("placeholder_token_id is not set for this processor")

        num_placeholders = sum(
            1 for token_id in original_ids if token_id == placeholder_token_id
        )
        if num_placeholders != len(counts):
            raise ValueError(
                f"prompt has {num_placeholders} image placeholder token(s) but "
                f"{len(counts)} image(s) were provided"
            )

        rebuilt: List[int] = []
        next_image_idx = 0
        for token_id in original_ids:
            if token_id == placeholder_token_id:
                rebuilt.extend([placeholder_token_id] * counts[next_image_idx])
                next_image_idx += 1
            else:
                rebuilt.append(token_id)
        return rebuilt

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        processor=None,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).

        Returns:
            Tuple of (list of mm_items, input_ids)
        """
        processor_override = processor
        processor, tokenizer = self._resolve_processor(processor)

        # Collect all items and categorize them
        all_loaded_data = base_output.organize_results()
        # Handle text-only case
        if not all_loaded_data:
            input_ids = tokenizer(
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
            if processor_override is not None:
                kwargs["processor"] = processor
            collected_items, input_ids, ret = self._process_and_collect_mm_items(
                input_text=base_output.input_text,
                images=raw_images,
                audios=raw_audios,
                videos=raw_videos,
                **kwargs,
            )
            all_collected_items = collected_items

            # When SGLANG_MM_AVOID_RETOKENIZE is on, keep the user's exact tokens to avoid retokenize drift.
            # Drift happens when Retokenization is not identity: Decode(X) => String => Re-tokenize => Y, X != Y.
            if (
                envs.SGLANG_MM_AVOID_RETOKENIZE.get()
                and base_output.input_ids is not None
                and input_ids is not None
                and raw_images
                and not raw_audios
                and not raw_videos
            ):
                assert isinstance(
                    base_output.input_ids, list
                ), f"expected list[int] input_ids, got {type(base_output.input_ids)}"
                try:
                    counts = self.resolve_image_token_counts(raw_images)
                    image_placeholder_token_id = mm_tokens.image_token_id
                    if image_placeholder_token_id is None:
                        raise ValueError(
                            "image placeholder token id is not set for this processor"
                        )
                    processor_placeholder_count = int(
                        (input_ids == image_placeholder_token_id).sum().item()
                    )
                    if processor_placeholder_count != sum(counts):
                        raise ValueError(
                            "processor image placeholder count mismatch: "
                            f"processor={processor_placeholder_count}, "
                            f"resolved={sum(counts)}"
                        )
                    input_ids = torch.tensor(
                        self._expand_input_ids(
                            base_output.input_ids,
                            counts,
                            image_placeholder_token_id,
                        ),
                        dtype=input_ids.dtype,
                    )
                except Exception as e:
                    logger.warning(
                        f"Due to {e}, falling back to decode+retokenize, which may change prompt length (token drift)."
                    )
        else:
            ret = None

        # Handle dict items (processed or precomputed)
        dict_ret = None
        for modality, dict_item in dict_items:
            input_format = self._get_preprocessed_input_format(dict_item)
            if input_format is not None and dict_ret is None:
                dict_ret = dict_item
            if input_format == MultimodalInputFormat.PROCESSOR_OUTPUT:
                items = self.collect_mm_items_from_processor_output(dict_item)
                for item in items:
                    item.format = MultimodalInputFormat.PROCESSOR_OUTPUT
                all_collected_items.extend(items)
            elif input_format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING:
                dict_item = dict(dict_item)
                feature = dict_item.pop("feature")
                all_collected_items.append(
                    MultimodalDataItem(
                        modality=modality,
                        feature=feature,
                        format=MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
                        model_specific_data=dict_item,
                    )
                )
        # Fallback tokenization if no raw items were processed
        if ret is None and dict_ret is not None:
            ret = dict_ret

        if input_ids is None:
            input_ids = self._ensure_input_ids_is_tensor(base_output.input_ids)

        if input_ids is None:
            for _, dict_item in dict_items:
                input_ids = self._ensure_input_ids_is_tensor(dict_item.get("input_ids"))
                if input_ids is not None:
                    break

        if input_ids is None:
            input_ids = tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        for mm_item in all_collected_items:
            if mm_item.offsets is not None:
                continue
            mm_token_id = mm_tokens.get_token_id_by_modality(mm_item.modality)
            if mm_token_id is None:
                raise ValueError(f"No token id found for modality: {mm_item.modality}")
            mm_item.offsets = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=mm_token_id,
            )

        # Split bundled items into per-image/video items for better cache granularity
        from sglang.srt.managers.mm_utils import get_new_expanded_mm_items

        all_collected_items = get_new_expanded_mm_items(all_collected_items)

        for item in all_collected_items:
            if item.format in (
                MultimodalInputFormat.PROCESSOR_OUTPUT,
                MultimodalInputFormat.PRECOMPUTED_EMBEDDING,
            ):
                item.set_pad_value()

        """
        solution for cuda-ipc memory-leak:
        1. memory-pool:  each time get a slice from memory-pool and use it as transport-data (with async lock guard)
        2. if can not get a slice , transport normal tensor
        3. copy tensor in scheduler and release it (use position mark)
        4. copy
        """

        if self.use_cuda_ipc:
            # post-process, prepare for cuda-ipc transfer
            for item in all_collected_items:
                if isinstance(item.feature, torch.Tensor):
                    item.feature = self._wrap_tensor_for_cuda_ipc(item.feature)
                if isinstance(item.precomputed_embeddings, torch.Tensor):
                    item.precomputed_embeddings = self._wrap_tensor_for_cuda_ipc(
                        item.precomputed_embeddings
                    )

        return all_collected_items, input_ids, ret

    async def process_and_combine_mm_data_async(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        **kwargs,
    ) -> Tuple[List[MultimodalDataItem], torch.Tensor, dict]:
        """Run multimodal preprocessing without blocking the event loop."""
        if self.mm_processor_executor is None:
            return self.process_and_combine_mm_data(base_output, mm_tokens, **kwargs)

        return await self.mm_processor_executor.run(
            self.process_and_combine_mm_data,
            base_output,
            mm_tokens,
            **kwargs,
        )
