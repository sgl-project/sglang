import asyncio
import concurrent.futures
import functools
import itertools
import logging
import random
import threading

import torch
import transformers

from sglang.srt.multimodal.cache import MultimodalPreprocessCache
from sglang.srt.multimodal.cache.digest import structured_digest
from sglang.srt.multimodal.media_io import ImageData, VideoData
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.multimodal_processor_mixin import (
    MultimodalProcessorMixin,
    _tokenizer_of,
)
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.token_space_mm_process.expansion import (
    expand_token_placeholders,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.version import __version__ as sglang_version

logger = logging.getLogger(__name__)

_MODALITIES = (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO)
# What a joined request receives when the item's owner published nothing.
_RELEASED = object()


def _device_type(device):
    return None if device is None else torch.device(device).type


def _log_failed_write(future):
    if future.exception() is not None:
        logger.warning("Item feature cache write failed", exc_info=future.exception())


def _check_cached_features(media_item, cached, computed):
    modality = media_item[0]
    if cached.keys() != computed.keys():
        raise RuntimeError(
            f"The item feature cache holds fields {sorted(cached)} for a {modality.name}"
            f" item that computes to {sorted(computed)}"
        )
    for name, value in computed.items():
        if not isinstance(value, torch.Tensor):
            continue
        other = cached[name]
        if other.dtype != value.dtype or not torch.equal(other.cpu(), value.cpu()):
            raise RuntimeError(
                f"The item feature cache holds other {name} for a {modality.name} item"
                " than computing it now gives: its key leaves out something the"
                " features depend on"
            )


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
        item_feature_cache_check_rate=None,
    ):
        strategy_class = self.strategy_class
        self._worker_thread_device = worker_thread_device
        # Any map with get and __setitem__, e.g. a dict, an LRU or a disk cache.
        self._item_feature_cache = item_feature_cache
        # The share of hits computed again and compared with the cache; 0 or None is off.
        self._item_feature_cache_check_rate = item_feature_cache_check_rate
        self._item_feature_cache_check_rng = random.Random()
        # Single-flight: key -> (future, generation) of the request computing it. Kept
        # here, not in the cache, so a disk cache or a plain dict shares work too.
        self._item_feature_flights = {}
        self._item_feature_flights_lock = threading.Lock()
        self._item_feature_generation = 0
        # Strong references: a computation outlives the request that started it.
        self._item_feature_tasks = set()
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
        depend on, including the type of device its modality runs on and the library
        versions, so a hit skips both loading and processing. A trainer and serving
        can share one map.
        """
        item_features = [None] * len(media_items)
        checked = {}
        if self._item_feature_cache is not None:
            media_options = self.strategy.resolve_media_options(
                self._processor, **request_options
            )
            # Types, not indices: a trainer rank reuses what serving computed on another GPU.
            item_devices = {
                modality: _device_type(options.get("device"))
                for modality, options in media_options.items()
            }
            # The loader decodes a video on the GPU unless its device is "cpu".
            item_devices[Modality.VIDEO] = (
                item_devices[Modality.VIDEO],
                request_options["video_device"] != "cpu",
            )
            item_keys = [
                structured_digest(
                    (
                        "token-space-item-feature-cache-v1",
                        transformers.__version__,
                        sglang_version,
                        self.hf_config.name_or_path,
                        self.mm_process_config,
                        self.image_processor_backend,
                        modality,
                        source,
                        item_kwargs,
                        item_devices[modality],
                    )
                ).hex()
                for modality, source, _, item_kwargs in media_items
            ]
            # A backend may block on IO, e.g. a disk cache reads whole files, so get runs
            # in the default thread pool, off the event loop and the media IO pool.
            item_features = list(
                await asyncio.gather(
                    *(
                        asyncio.to_thread(self._item_feature_cache.get, key)
                        for key in item_keys
                    )
                )
            )
            # A hit is loaded on the CPU; misses stay on their modality's device.
            self._move_to_request_devices(
                media_items, item_features, range(len(media_items)), media_options
            )
            # A key that leaves out something the features depend on fails here, not in training.
            if self._item_feature_cache_check_rate:
                checked = {
                    index: features
                    for index, features in enumerate(item_features)
                    if features is not None
                    and self._item_feature_cache_check_rng.random()
                    < self._item_feature_cache_check_rate
                }
            for index in checked:
                item_features[index] = None
        missed = [
            index for index, features in enumerate(item_features) if features is None
        ]
        if self._item_feature_cache is None:
            # No keys, so nothing to share: every request computes its own items.
            computed = await self._compute_items(media_items, missed, request_options)
            for index, features in zip(missed, computed):
                item_features[index] = features
            return item_features

        copies, pending = {}, missed
        # At least once: a request whose items all hit still goes through the loader.
        while True:
            claim = self._claim_item_features(item_keys, pending, checked)
            compute, owned, joined, round_copies = claim[:4]
            copies.update(round_copies)
            computing = self._compute_and_publish(
                media_items,
                compute,
                owned,
                checked,
                item_keys,
                request_options,
                *claim[4:],
            )
            if owned:
                # Shielded: cancelling this request must not cancel work others joined.
                # Others may join until it settles, so it runs to the end and is cached
                # even if nobody did: unlike before, a cancelled request's items finish.
                computing = asyncio.shield(
                    self._create_item_feature_task(
                        computing,
                        functools.partial(
                            self._release_item_feature_flights,
                            item_keys,
                            owned,
                            claim[4],
                        ),
                    )
                )
            computed, *joined_features = await asyncio.gather(
                computing,
                *(self._wait_for_item_feature_flight(f) for f in joined.values()),
            )
            for index, features in zip(compute, computed):
                item_features[index] = features
            received, pending, checked = [], [], {}
            for index, features in zip(joined, joined_features):
                if features is _RELEASED:
                    # Its owner published nothing (loading failed, or a sampled check
                    # did): claim it again, to join a new owner or compute it here.
                    pending.append(index)
                    continue
                # Each request gets its own dict; tensors are shared, as with a hit.
                item_features[index] = dict(features)
                received.append(index)
            # Another request computed them on its own device: move them like hits.
            self._move_to_request_devices(
                media_items, item_features, received, media_options
            )
            if not pending:
                break
        for index, first in copies.items():
            item_features[index] = dict(item_features[first])
        return item_features

    def _claim_item_features(self, item_keys, missed, checked):
        """Split the missed items into those this request computes and those it waits for.

        Returns the indices to compute (owned and checked), {index: future} owned,
        {index: future} joined from another request, {index: first index} for a key
        repeated in this request, the generation the claims belong to, and the in-memory
        cache's generation (None for other maps) its writes are made against.
        """
        compute, owned, joined, copies, first_index = [], {}, {}, {}, {}
        cache = self._item_feature_cache
        cache_generation = (
            cache.generation if isinstance(cache, MultimodalPreprocessCache) else None
        )
        with self._item_feature_flights_lock:
            generation = self._item_feature_generation
            for index in missed:
                if index in checked:
                    # A sampled hit is computed again to be compared, never shared.
                    compute.append(index)
                    continue
                key = item_keys[index]
                if key in first_index:
                    # Waiting on its own claim would deadlock this request.
                    copies[index] = first_index[key]
                    continue
                first_index[key] = index
                flight = self._item_feature_flights.get(key)
                if flight is not None and flight[1] == generation:
                    joined[index] = flight[0]
                    continue
                future = concurrent.futures.Future()
                self._item_feature_flights[key] = (future, generation)
                owned[index] = future
                compute.append(index)
        return compute, owned, joined, copies, generation, cache_generation

    async def _compute_and_publish(
        self,
        media_items,
        compute,
        owned,
        checked,
        item_keys,
        request_options,
        generation,
        cache_generation,
    ):
        """Compute this request's items, then give each owned one to its joined requests.

        The request fails as before: with its first failed item, else a failed sampled
        check. A joined request gets only what belongs to its item: the features, or
        that item's own error. If loading fails (for the whole request) or a sampled
        check does (the key leaves something out), nothing is published or written:
        the claims are released and joined requests claim the items again. Otherwise
        each item that succeeded is published and written even if another failed:
        unlike before, a failing request still caches its good items, as joined
        requests may need them.
        """
        try:
            results = await self._compute_items(
                media_items, compute, request_options, return_exceptions=True
            )
        except BaseException:
            self._release_item_feature_flights(item_keys, owned, generation)
            raise
        by_index = dict(zip(compute, results))
        failures = [result for result in results if isinstance(result, BaseException)]
        check_error = None
        for index, cached in checked.items():
            if isinstance(by_index[index], BaseException):
                continue
            try:
                _check_cached_features(media_items[index], cached, by_index[index])
            except RuntimeError as error:
                check_error = error
                break
        if check_error is None:
            self._publish_item_features(
                owned, by_index, item_keys, generation, cache_generation
            )
        else:
            self._release_item_feature_flights(item_keys, owned, generation)
        if failures:
            for failure in failures[1:]:
                logger.warning("Another media item also failed", exc_info=failure)
            raise failures[0]
        if check_error is not None:
            raise check_error
        return results

    def _publish_item_features(
        self, owned, by_index, item_keys, generation, cache_generation
    ):
        loop = asyncio.get_running_loop()
        for index, future in owned.items():
            key, result = item_keys[index], by_index[index]
            if isinstance(result, BaseException):
                # The item's own error: every request that joined it asked for it too.
                self._end_item_feature_flight(key, future, generation)
                future.set_exception(result)
                future.exception()  # retrieved, as no request may have joined it
                continue
            future.set_result(result)
            # Written behind the request. The key stays in flight until the write lands,
            # so a request arriving before then joins the result instead of recomputing.
            loop.run_in_executor(
                None,
                self._write_item_features,
                key,
                result,
                cache_generation,
                future,
                generation,
            ).add_done_callback(_log_failed_write)

    def _release_item_feature_flights(self, item_keys, owned, generation, *_):
        """Settle every owned claim not yet settled with nothing to share.

        Also the done callback of the computation's task, so claims are released
        however it ends, even cancelled before its first step.
        """
        for index, future in owned.items():
            if future.done():
                continue
            # Unregistered first, so a woken joiner that claims again cannot rejoin it.
            self._end_item_feature_flight(item_keys[index], future, generation)
            future.set_result(_RELEASED)

    def _write_item_features(self, key, features, cache_generation, future, generation):
        try:
            if cache_generation is None:
                # A disk cache or plain map: a flush never removes its entries and a
                # key fixes its value, so a write begun before a flush is still
                # correct. No lock is held across the write, which may be slow disk IO.
                self._item_feature_cache[key] = features
            else:
                # In memory: the cache drops it, atomically under its own lock, if a
                # flush came after this computation was claimed.
                self._item_feature_cache.put(
                    key, features, _generation=cache_generation
                )
        finally:
            # Here, not in a loop callback, so it happens even if the loop is gone.
            self._end_item_feature_flight(key, future, generation)

    def _end_item_feature_flight(self, key, future, generation):
        with self._item_feature_flights_lock:
            if self._item_feature_flights.get(key) == (future, generation):
                del self._item_feature_flights[key]

    async def _wait_for_item_feature_flight(self, future):
        """Wait for another request's computation; cancelling this request leaves it running."""
        return await asyncio.shield(asyncio.wrap_future(future))

    def _create_item_feature_task(self, coroutine, release):
        task = asyncio.create_task(coroutine)
        self._item_feature_tasks.add(task)
        task.add_done_callback(self._item_feature_task_done)
        task.add_done_callback(release)
        return task

    def _item_feature_task_done(self, task):
        self._item_feature_tasks.discard(task)
        # Retrieved: the request that started it may have been cancelled.
        if not task.cancelled():
            task.exception()

    def reset_item_feature_flights(self):
        """Start a new generation of computations, e.g. on a cache flush.

        Later requests compute again instead of joining earlier computations. Those
        still reach the requests that joined them; an in-memory cache drops their
        writes by its own generation, and a disk cache, which a flush keeps, takes them.
        """
        with self._item_feature_flights_lock:
            self._item_feature_generation += 1
            self._item_feature_flights.clear()

    def _move_to_request_devices(
        self, media_items, item_features, indices, media_options
    ):
        """Move features computed elsewhere to this request's modality devices."""
        if not self.keep_mm_features_on_device:
            return
        for index in indices:
            features = item_features[index]
            device = media_options[media_items[index][0]].get("device")
            if features is None or device is None:
                continue
            for name in self.FEATURE_NAMES:
                if isinstance(features.get(name), torch.Tensor):
                    features[name] = features[name].to(device)

    async def _compute_items(
        self, media_items, indices, request_options, return_exceptions=False
    ):
        """Load and process the items at `indices`, returning their features in that order.

        Loading fails for all of them at once; with `return_exceptions`, a processing
        failure is returned in its item's place instead of raised.
        """
        items = [media_items[index] for index in indices]
        loaded = await self._load_media_lists(
            *(
                [
                    source
                    for item_modality, source, *_ in items
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
                items, itertools.chain.from_iterable(loaded)
            )
        ]
        return await self._process_items(
            loaded_items, request_options, return_exceptions=return_exceptions
        )

    async def _process_items(
        self, media_items, request_options, return_exceptions=False
    ):
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
        if return_exceptions:
            return item_features
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
