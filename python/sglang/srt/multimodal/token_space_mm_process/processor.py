import asyncio
import functools
import logging

import torch

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
        # The media loader reads these attributes from the class, not the instance.
        return type(
            f"{cls.__name__}[{strategy_class.__name__}]",
            (cls,),
            {
                "strategy_class": strategy_class,
                **strategy_class.processor_class_attributes,
            },
        )

    def __init__(
        self,
        hf_config,
        processor,
        *,
        processor_config=None,
        compute_mrope_positions=False,
        pin_io_threads_to_current_device=True,
        **kwargs,
    ):
        strategy_class = self.strategy_class
        self._pin_io_threads_to_current_device = pin_io_threads_to_current_device
        if processor_config is None:
            processor_config = MultimodalProcessorConfig()
        self._initialize_processor(
            hf_config, processor, processor_config=processor_config, **kwargs
        )
        self.FEATURE_NAMES = sorted(strategy_class.mm_feature_fields)
        self.keep_mm_features_on_device = processor_config.keep_mm_features_on_device
        self.strategy = strategy_class(
            hf_config, processor, mm_process_config=self.mm_process_config
        )
        # Off by default: Megatron Bridge and similar trainers compute positions themselves.
        self._compute_mrope_positions = compute_mrope_positions

    def _io_thread_initializer(self):
        # Loaders decode on device "cuda", which a new thread resolves to cuda:0
        # whatever device this trainer rank selected.
        if (
            not self._pin_io_threads_to_current_device
            or not torch.cuda.is_initialized()
        ):
            return None
        return functools.partial(torch.cuda.set_device, torch.cuda.current_device())

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
        package_output=None,
        **kwargs,
    ):
        """Process media and expand the caller's IDs, like processor(text, media).

        Args:
            image_data, video_data, audio_data: Raw media sources, or None.
            input_ids: The caller's IDs, one placeholder token per media.
            mm_token_expansion_start_len: Length of an already expanded prefix.
            video_metadata: Per-video frame metadata, in video order.
            image_device, video_device: Preprocessing devices; None keeps defaults.
            package_output: Runs on the BatchFeature in the same worker task.
            **kwargs: Request kwargs for the strategy and its output builder.

        Returns:
            The BatchFeature with `mm_modalities` and `mm_spans`, or what
            `package_output` returns for it.
        """
        sources = (image_data, video_data, audio_data)
        self._check_media_sources(*sources)
        loaded = await self._load_media_lists(
            *sources,
            audio_sample_rate=self.strategy.audio_sample_rate,
            discard_alpha_channel=self.strategy.discard_alpha_channel,
        )
        _, videos, _ = loaded
        image_device, video_device = self._media_devices(
            videos, image_device, video_device
        )
        request_options = dict(
            image_device=image_device, video_device=video_device, **kwargs
        )
        media_items = self._list_media_items(sources, loaded, video_metadata)
        item_features = await self._process_items(media_items, request_options)
        return await self._run_mm_processor(
            self._build_output,
            media_items=media_items,
            item_features=item_features,
            input_ids=input_ids,
            mm_token_expansion_start_len=mm_token_expansion_start_len,
            request_kwargs=kwargs,
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

    def _media_devices(self, videos, image_device, video_device):
        # A strategy may pin all media of a video request to one device.
        if videos and self.video_preprocessing_device is not None:
            return self.video_preprocessing_device, self.video_preprocessing_device
        return image_device, video_device

    @staticmethod
    def _list_media_items(sources, loaded, video_metadata):
        """(modality, loaded item, preprocess_kwargs, per-item kwargs), in modality order."""
        media_items = []
        modalities = (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
        for modality, data_list, items in zip(modalities, sources, loaded):
            for index, (source, item) in enumerate(zip(data_list or (), items)):
                if isinstance(source, (ImageData, VideoData)):
                    preprocess_kwargs = source.preprocess_kwargs or {}
                elif isinstance(source, dict):
                    preprocess_kwargs = source.get("preprocess_kwargs") or {}
                else:
                    preprocess_kwargs = {}
                item_kwargs = {}
                if modality == Modality.VIDEO and video_metadata is not None:
                    item_kwargs["video_metadata"] = video_metadata[index]
                media_items.append((modality, item, preprocess_kwargs, item_kwargs))
        return media_items

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
        request_kwargs,
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
            media_features, expanded_input_ids, processor=processor, **request_kwargs
        )
        # Where each media's tower output goes, aligned with media_items.
        output["mm_modalities"] = [modality for modality, *_ in media_items]
        output["mm_spans"] = mm_spans
        if self._compute_mrope_positions:
            mrope_positions, mrope_position_delta = (
                self.strategy.compute_mrope_positions(
                    expanded_input_ids, media_features
                )
            )
            if mrope_positions is not None:
                output["mrope_positions"] = mrope_positions
                output["mrope_position_delta"] = mrope_position_delta
        return output if package_output is None else package_output(output)
