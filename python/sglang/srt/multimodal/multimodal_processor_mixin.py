import asyncio
import concurrent.futures
import os
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.utils import (
    CLIENT_MEDIA_EXCEPTIONS,
    configure_media_url_security,
    load_audio,
    load_image,
    load_video,
    logger,
    smart_to_rgb,
)


def _tokenizer_of(processor):
    """The tokenizer reached from an HF processor.

    Some processors (e.g. InternVL) are handed a tokenizer directly as their
    ``_processor`` rather than one that wraps a tokenizer. Every path that
    resolves a tokenizer -- construction and per-worker processor clones alike --
    goes through here, so a clone cannot resolve differently from the original.
    """
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


class MultimodalProcessorMixin:
    """Share media loading and preprocessing resources across training and serving."""

    gpu_image_decode = True  # Enable GPU decoding by default
    smart_rgb_conversion = False
    auto_mm_io_worker_num = 4
    # Processors opt out only when their preprocessing is not thread-safe. The
    # worker pool gives each thread its own `copy.deepcopy` of the HF processor
    # and injects it, and the single function it runs --
    # `process_and_combine_mm_data` -- resolves that clone instead of
    # `self._processor`, so isolation does not depend on the subclass.
    supports_mm_processor_concurrency = True
    video_preprocessing_device = None
    precompute_hash_before_cpu_transfer = False
    keep_mm_features_on_device = False

    def _initialize_processor(
        self, hf_config, _processor, *, processor_config, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor

        configure_media_url_security(
            processor_config.allowed_media_domains,
            processor_config.media_url_max_file_size_mb,
        )

        self.image_processor_backend = processor_config.image_processor_backend
        if processor_config.disable_fast_image_processor:
            self.image_processor_backend = "pil"
        self.disable_fast_image_processor = self.image_processor_backend == "pil"

        mm_process_config = processor_config.mm_process_config
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

        # name of the feature filed
        # TODO: pass from processors
        self.FEATURE_NAMES = [
            "pixel_values",
            "pixel_values_videos",
            "audio_features",
            "input_features",
        ]

        self._tokenizer = _tokenizer_of(self._processor)

        # Same guard as in serving_chat.py against double BOS.
        try:
            self._tokenizer_auto_adds_specials = len(self._tokenizer.encode("")) > 0
        except Exception:
            self._tokenizer_auto_adds_specials = False

        requested_mm_io_worker_num = processor_config.mm_io_worker_num
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
        requested_mm_processor_worker_num = processor_config.mm_processor_worker_num
        self.mm_processor_worker_num = (
            1
            if skip_mm_pool
            else requested_mm_processor_worker_num
            or self._resolve_auto_mm_processor_worker_num()
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
                # A callable, not the object: subclasses finish customizing
                # `_processor` after this returns, and the workers must clone it
                # as the subclass left it.
                self.mm_processor_executor = MultimodalProcessorExecutor(
                    lambda: self._processor, self.mm_processor_worker_num
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

    def shutdown(self) -> None:
        """Stop every processor-side executor."""
        self.io_executor.shutdown(wait=False, cancel_futures=True)
        if self.mm_processor_executor is not None:
            self.mm_processor_executor.shutdown()

    def _resolve_auto_mm_processor_worker_num(self) -> int:
        """Worker count when none is requested; serving overrides this."""
        return 1

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
                if discard_alpha_channel:
                    if cls.smart_rgb_conversion:
                        return smart_to_rgb(img)
                    if img.mode != "RGB":
                        return img.convert("RGB")
                img.load()
                return img
            elif modality == Modality.VIDEO:
                return load_video(data, use_gpu=cls.video_preprocessing_device != "cpu")
            elif modality == Modality.AUDIO:
                return load_audio(data, audio_sample_rate)

        except CLIENT_MEDIA_EXCEPTIONS as e:
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
            if MultimodalProcessorMixin._is_preprocessed_input(item):
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

        MultimodalProcessorMixin._validate_one_modality(Modality.IMAGE, image_data)
        MultimodalProcessorMixin._validate_one_modality(Modality.VIDEO, video_data)
        MultimodalProcessorMixin._validate_one_modality(Modality.AUDIO, audio_data)

    async def _load_media_lists(
        self,
        image_data: Optional[list],
        video_data: Optional[list],
        audio_data: Optional[list],
        *,
        audio_sample_rate: Optional[int],
        discard_alpha_channel: bool,
    ) -> Tuple[List[Any], List[Any], List[Any]]:
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
            except ValueError as e:
                logger.info(
                    "[load_mm_data(simple)] invalid %s data at index=%d: %s",
                    modality.name,
                    idx,
                    e,
                )
                raise ValueError(
                    f"An exception occurred while loading {modality.name} data "
                    f"at index {idx}: {e}"
                ) from e
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
        return images, videos, audios
