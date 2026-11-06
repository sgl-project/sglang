import asyncio
import functools
import itertools
import logging

import torch

from sglang.srt.multimodal.cache.identity import build_cache_key
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.multimodal_processor_mixin import (
    MultimodalProcessorMixin,
    _tokenizer_of,
)
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.token_space_mm_process.expansion import (
    expand_token_placeholders,
)
from sglang.srt.utils import ImageData, VideoData
from sglang.srt.utils.video_decoder import VideoDecoderWrapper

logger = logging.getLogger(__name__)

_MODALITIES = (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)


class TokenSpaceProcessor(MultimodalProcessorMixin):
    """Turn caller input_ids and raw media into model inputs with a token-space strategy.

    `process` runs a whole request; `_process_item` and `_build_output` are the
    parts that run on a worker thread with that worker's processor copy.
    Build one with `TokenSpaceProcessor.for_strategy(strategy_class)`.
    """

    strategy_class = None

    @classmethod
    @functools.cache
    def for_strategy(cls, strategy_class):
        return type(
            f"{cls.__name__}[{strategy_class.__name__}]",
            (cls,),
            {
                "strategy_class": strategy_class,
                "gpu_image_decode": strategy_class.gpu_image_decode,
            },
        )

    def __init__(
        self,
        hf_config,
        processor,
        *,
        processor_config=None,
        worker_thread_device=None,
        item_feature_cache=None,
    ):
        strategy_class = self.strategy_class
        self._worker_thread_device = worker_thread_device
        # Any map with get and __setitem__, e.g. a dict, an LRU or a disk cache.
        self._item_feature_cache = item_feature_cache
        if processor_config is None:
            processor_config = MultimodalProcessorConfig()
        self._initialize_processor(
            hf_config, processor, processor_config=processor_config
        )
        self.FEATURE_NAMES = sorted(strategy_class.mm_feature_fields)
        self.keep_mm_features_on_device = processor_config.keep_mm_features_on_device
        self.strategy = strategy_class(
            hf_config, processor, mm_process_config=self.mm_process_config
        )
        self._video_preprocessing_device = self.mm_process_config.get(
            "video_preprocessing_device", strategy_class.video_preprocessing_device
        )

    def _worker_thread_initializer(self):
        # Media decodes and preprocesses on device "cuda", which a new thread
        # resolves to cuda:0 whatever device the caller's rank selected.
        if self._worker_thread_device is None:
            return None
        return functools.partial(torch.cuda.set_device, self._worker_thread_device)

    async def _run_mm_processor(self, function, **kwargs):
        # Without a worker pool the function runs inline on the caller's thread,
        # where bare "cuda" (e.g. video frame decode) is still cuda:0.
        if (
            self.mm_processor_executor is None
            and self._worker_thread_device is not None
        ):
            with torch.cuda.device(self._worker_thread_device):
                return function(processor=self._processor, **kwargs)
        return await super()._run_mm_processor(function, **kwargs)

    async def process(
        self,
        image_data=None,
        video_data=None,
        audio_data=None,
        *,
        input_ids,
        mm_token_expansion_start_len=0,
        video_metadata=None,
        image_device=None,
        video_device=None,
        compute_mrope_positions=False,
        package_output=None,
    ):
        """Process media and expand the caller's IDs, like processor(text, media).

        Args:
            image_data, video_data, audio_data: Raw media sources, or None.
            input_ids: The caller's IDs, one placeholder token per media.
            mm_token_expansion_start_len: Length of an already expanded prefix.
            video_metadata: Per-video frame metadata, in video order.
            image_device, video_device: Preprocessing devices; None keeps defaults.
                Videos also decode on CPU when `video_device` is "cpu". A video
                request takes both from the video preprocessing device, if any.
            compute_mrope_positions: Also return M-RoPE positions; off for trainers
                such as Megatron Bridge that compute them themselves.
            package_output: Runs on the BatchFeature in the same worker task,
                concurrently with other requests; it may read only state fixed at
                construction and the worker's own `processor` clone.

        Returns:
            The BatchFeature with `mm_modalities` and `mm_spans`, or what
            `package_output` returns for it.
        """
        sources = (image_data, video_data, audio_data)
        self._check_media_sources(*sources)
        # As legacy: a video request preprocesses all its media on the video device.
        if video_data and self._video_preprocessing_device is not None:
            image_device = video_device = self._video_preprocessing_device
        request_options = dict(image_device=image_device, video_device=video_device)
        media_items = self._list_media_items(sources, video_metadata)
        item_features = await self._get_item_features(media_items, request_options)
        return await self._run_mm_processor(
            self._build_output,
            media_items=media_items,
            item_features=item_features,
            input_ids=input_ids,
            mm_token_expansion_start_len=mm_token_expansion_start_len,
            compute_mrope_positions=compute_mrope_positions,
            package_output=package_output,
        )

    def _check_media_sources(self, image_data, video_data, audio_data):
        sources = {
            "image_data": image_data,
            "video_data": video_data,
            "audio_data": audio_data,
        }
        self.validate_mm_data(**sources)
        for name, data_list in sources.items():
            # Their input_ids were already expanded by an HF processor.
            if any(self._is_preprocessed_input(data) for data in data_list or ()):
                raise ValueError(
                    f"{name}: token-space processing does not accept processor_output "
                    "or precomputed_embedding inputs; pass raw media instead"
                )

    @staticmethod
    def _list_media_items(sources, video_metadata):
        """(modality, source, preprocess_kwargs, per-item kwargs), in modality order."""
        media_items = []
        for modality, data_list in zip(_MODALITIES, sources):
            for index, source in enumerate(data_list or ()):
                if isinstance(source, (ImageData, VideoData)):
                    preprocess_kwargs = source.preprocess_kwargs or {}
                elif isinstance(source, dict):
                    preprocess_kwargs = source.get("preprocess_kwargs") or {}
                else:
                    preprocess_kwargs = {}
                item_kwargs = {}
                if modality == Modality.VIDEO and video_metadata is not None:
                    item_kwargs["video_metadata"] = video_metadata[index]
                media_items.append((modality, source, preprocess_kwargs, item_kwargs))
        return media_items

    async def _get_item_features(self, media_items, request_options):
        """Each item's features, from the item feature cache or by loading and processing it.

        A key is the item's source before loading with everything else its features
        depend on, so a hit skips both loading and processing. The checkpoint,
        mm_process_config and image processor backend let a trainer and serving
        share one map.
        """
        item_features = [None] * len(media_items)
        if self._item_feature_cache is not None:
            item_keys = [
                build_cache_key(
                    (
                        self.hf_config.name_or_path,
                        self.mm_process_config,
                        self.image_processor_backend,
                        modality,
                        source,
                        item_kwargs,
                        request_options,
                    )
                )
                for modality, source, _, item_kwargs in media_items
            ]
            item_features = [self._item_feature_cache.get(key) for key in item_keys]
        missed = [
            index for index, features in enumerate(item_features) if features is None
        ]
        missed_items = [media_items[index] for index in missed]
        loaded = await self._load_media_lists(
            *(
                [
                    source
                    for item_modality, source, *_ in missed_items
                    if item_modality == modality
                ]
                or None
                for modality in _MODALITIES
            ),
            audio_sample_rate=self.strategy.audio_sample_rate,
            discard_alpha_channel=self.strategy.discard_alpha_channel,
            video_decode_device=request_options["video_device"],
        )
        # Missed items are in modality order, as the loaded lists are.
        loaded_items = [
            (modality, item, preprocess_kwargs, item_kwargs)
            for (modality, _, preprocess_kwargs, item_kwargs), item in zip(
                missed_items, itertools.chain.from_iterable(loaded)
            )
        ]
        missed_features = await self._process_items(loaded_items, request_options)
        for index, features in zip(missed, missed_features):
            item_features[index] = features
            if self._item_feature_cache is not None:
                self._item_feature_cache[item_keys[index]] = features
        return item_features

    async def _process_items(self, media_items, request_options):
        """One worker task per item; decoders are closed once every task has finished."""
        item_features = await asyncio.gather(
            *(
                self._run_mm_processor(
                    self._process_item,
                    modality=modality,
                    item=item,
                    preprocess_kwargs=preprocess_kwargs,
                    request_options=request_options,
                    item_kwargs=item_kwargs,
                )
                for modality, item, preprocess_kwargs, item_kwargs in media_items
            ),
            return_exceptions=True,
        )
        for _, item, _, _ in media_items:
            if isinstance(item, VideoDecoderWrapper):
                # One failed close must not leak the rest or hide the request's error.
                try:
                    item.close()
                except Exception:
                    logger.warning("Failed to close a video decoder", exc_info=True)
        failures = [f for f in item_features if isinstance(f, BaseException)]
        if failures:
            for failure in failures[1:]:
                logger.warning("Another media item also failed", exc_info=failure)
            raise failures[0]
        return item_features

    def _process_item(
        self,
        *,
        modality: Modality,
        item,
        preprocess_kwargs,
        request_options,
        item_kwargs,
        processor,
    ):
        # On a worker thread `processor` is its own clone; see TokenSpaceProcessStrategy.
        options = {
            **self.strategy.resolve_media_options(processor, **request_options)[
                modality
            ],
            **item_kwargs,
        }
        # The CUDA memory pool is thread-local, so each worker wraps its own item.
        with self._temporary_fast_processor_cuda_pool(options.get("device")):
            features = self.strategy.process_item(
                modality, item, processor, preprocess_kwargs, **options
            )
            self._move_processor_output_to_cpu(features)
        return features

    def _build_output(
        self,
        *,
        media_items,
        item_features,
        input_ids,
        mm_token_expansion_start_len,
        compute_mrope_positions,
        processor,
        package_output=None,
    ):
        # On a worker thread `processor` is its own clone; see TokenSpaceProcessStrategy.
        media_features = self.strategy.merge_media(media_items, item_features)
        tokenizer = _tokenizer_of(processor)
        mm_token_expansion_spec = self.strategy.get_mm_token_expansion_spec(
            media_features,
            encode_expansion_text=functools.partial(
                tokenizer.encode, add_special_tokens=False
            ),
        )
        expanded_input_ids = expand_token_placeholders(
            input_ids, mm_token_expansion_spec, mm_token_expansion_start_len
        )
        mm_spans = self.strategy.locate_media(
            expanded_input_ids, media_items, mm_token_expansion_spec
        )
        output = self.strategy.build_processor_output(
            media_features, expanded_input_ids, processor=processor
        )
        # Where each media's tower output goes, aligned with media_items.
        output["mm_modalities"] = [modality for modality, *_ in media_items]
        output["mm_spans"] = mm_spans
        if compute_mrope_positions:
            mrope_positions, mrope_position_delta = (
                self.strategy.compute_mrope_positions(output)
            )
            if mrope_positions is not None:
                output["mrope_positions"] = mrope_positions
                output["mrope_position_delta"] = mrope_position_delta
        return output if package_output is None else package_output(output)
