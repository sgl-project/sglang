import asyncio
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
import threading
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.multimodal.modality import Modality, MultimodalInputFormat
from sglang.srt.multimodal.processors.executor import MultimodalProcessorExecutor
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.token_space.process_strategy import (
    close_loaded_media,
    list_media_items,
)
from sglang.srt.utils import (
    CLIENT_MEDIA_EXCEPTIONS,
    ImageData,
    VideoData,
    configure_media_url_security,
    load_audio,
    load_image,
    load_video,
    logger,
    smart_to_rgb,
)


def get_media_source_configs(mm_data):
    """Return per-source preprocessing options in the original media order."""
    source_configs = []
    for source in mm_data or []:
        if isinstance(source, (ImageData, VideoData)):
            options = source.preprocess_kwargs
        elif isinstance(source, dict):
            options = source.get("preprocess_kwargs")
        else:
            options = None
        source_configs.append(options or {})
    return source_configs


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
    use_token_space_processor = False

    def _initialize_processor(
        self, hf_config, _processor, *, processor_config, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.cpu_process_start_method = processor_config.cpu_process_start_method
        self.cpu_worker_num = processor_config.cpu_worker_num

        allowed_media_domains = processor_config.allowed_media_domains
        media_url_max_file_size_mb = processor_config.media_url_max_file_size_mb
        if allowed_media_domains is not None or media_url_max_file_size_mb is not None:
            configure_media_url_security(
                allowed_media_domains,
                max_file_size_mb=media_url_max_file_size_mb,
                preserve_allowed_domains=allowed_media_domains is None,
            )

        self.image_processor_backend = processor_config.image_processor_backend
        if processor_config.disable_fast_image_processor:
            self.image_processor_backend = "pil"
        self.disable_fast_image_processor = self.image_processor_backend == "pil"

        mm_process_config = processor_config.mm_process_config
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

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
        self._cpu_executor_lock = threading.Lock()
        self.cpu_executor = self._create_cpu_executor()

    def shutdown(self) -> None:
        """Stop every processor-side executor."""
        self.io_executor.shutdown(wait=False, cancel_futures=True)
        self.cpu_executor.shutdown(wait=False, cancel_futures=True)
        if self.mm_processor_executor is not None:
            self.mm_processor_executor.shutdown()

    def _resolve_auto_mm_processor_worker_num(self) -> int:
        """Worker count when none is requested; serving overrides this."""
        return 1

    def _create_cpu_executor(self) -> concurrent.futures.ProcessPoolExecutor:
        return concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context(self.cpu_process_start_method),
            max_workers=self.cpu_worker_num
            or int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

    def _replace_broken_cpu_executor(
        self, failed_executor: concurrent.futures.ProcessPoolExecutor
    ) -> None:
        """Replace a failed preprocess pool once across concurrent requests."""
        with self._cpu_executor_lock:
            if self.cpu_executor is not failed_executor:
                return
            self.cpu_executor = self._create_cpu_executor()
        logger.warning("Replaced a broken multimodal CPU preprocess pool")
        threading.Thread(
            target=self._shutdown_broken_cpu_executor,
            args=(failed_executor,),
            name="sglang-mm-cpu-pool-cleanup",
            daemon=True,
        ).start()

    @staticmethod
    def _shutdown_broken_cpu_executor(
        failed_executor: concurrent.futures.ProcessPoolExecutor,
    ) -> None:
        try:
            failed_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.warning(
                "Failed to shut down a broken multimodal CPU preprocess pool",
                exc_info=True,
            )

    def _resolve_processor(self, processor=None):
        if processor is None:
            return self._processor, self._tokenizer
        return processor, _tokenizer_of(processor)

    async def _run_mm_processor(self, function, **kwargs):
        if self.mm_processor_executor is None:
            return function(**kwargs)
        return await self.mm_processor_executor.run(function, **kwargs)

    async def process_media_async(
        self,
        *,
        images=None,
        videos=None,
        audios=None,
        image_source_configs=None,
        video_source_configs=None,
        audio_source_configs=None,
        video_metadata=None,
        image_device=None,
        video_device=None,
        **kwargs,
    ):
        """Process loaded media one task per item.

        Tasks run in parallel on clone workers if configured, else sequentially on the calling thread.
        """
        strategy = self.token_space_process_strategy
        media_options = strategy.resolve_media_options(
            self._processor,
            image_device=image_device,
            video_device=video_device,
            **kwargs,
        )
        media_items = list_media_items(
            images=images,
            videos=videos,
            audios=audios,
            image_source_configs=image_source_configs,
            video_source_configs=video_source_configs,
            audio_source_configs=audio_source_configs,
            video_metadata=video_metadata,
        )
        # Wait for every task before closing decoders, even if one item fails.
        item_features = await asyncio.gather(
            *(
                self._run_mm_processor(
                    self._process_media_item,
                    modality=modality,
                    item=item,
                    source_config=source_config,
                    options={**media_options[modality], **item_kwargs},
                )
                for modality, item, source_config, item_kwargs in media_items
            ),
            return_exceptions=True,
        )
        close_loaded_media(media_items)
        failures = [f for f in item_features if isinstance(f, BaseException)]
        if failures:
            for failure in failures[1:]:
                logger.warning("Another media item also failed", exc_info=failure)
            raise failures[0]
        return strategy.merge_media(media_items, item_features)

    def _process_media_item(
        self, *, modality, item, source_config, options, processor=None
    ):
        processor, _ = self._resolve_processor(processor)
        return self.token_space_process_strategy.process_item(
            modality, item, processor, source_config, **options
        )

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
                return load_video(data, frame_count_limit)
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

        images, videos, audios = await self._load_media_lists(
            image_data,
            video_data,
            audio_data,
            audio_sample_rate=audio_sample_rate,
            discard_alpha_channel=discard_alpha_channel,
        )

        return BaseMultiModalProcessorOutput(
            images=images,
            audios=audios,
            videos=videos,
            input_text=prompt_str,
            input_ids=input_ids,
        )

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


class TokenSpaceMMProcessor(MultimodalProcessorMixin):
    """Load media and run a token-space process strategy without serving dependencies."""

    use_token_space_processor = True

    def __init__(
        self,
        hf_config,
        processor,
        token_space_process_strategy_class,
        *,
        processor_config=None,
        **kwargs,
    ):
        if processor_config is None:
            processor_config = MultimodalProcessorConfig()
        self.token_space_process_strategy = token_space_process_strategy_class(
            hf_config, processor, mm_process_config=processor_config.mm_process_config
        )
        self._initialize_processor(
            hf_config, processor, processor_config=processor_config, **kwargs
        )
