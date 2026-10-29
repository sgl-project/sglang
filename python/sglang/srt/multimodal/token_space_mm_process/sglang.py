import contextlib
import functools
import hashlib
import os
from typing import Optional

import torch
from transformers import BaseImageProcessor, BatchFeature

from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.cache import (
    CONTENT_HASH_PREFIX,
    resolve_multimodal_item_hash,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.hash_executor import MultimodalHashExecutor
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.serving_preprocessing import (
    apply_npu_image_preprocess_patches,
    fast_image_processor_device,
    preprocessing_competes_with_the_scheduler,
    resolve_auto_mm_processor_worker_num,
    resolve_hf_image_token_counts,
)
from sglang.srt.multimodal.token_space_mm_process.processor import (
    TokenSpaceProcessor,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)
from sglang.srt.multimodal.transport.feature_transport import MultimodalFeatureTransport
from sglang.srt.runtime_context import get_exec, get_mm
from sglang.srt.true_on_policy import is_true_on_policy_enabled

# MultimodalProcessorOutput token-ID field -> mm_token_expansion_params key.
_OUTPUT_TOKEN_ID_PARAMS = {
    "im_token_id": "image_token_id",
    "im_start_id": "vision_start_token_id",
    "im_end_id": "vision_end_token_id",
    "slice_start_id": "slice_start_token_id",
    "slice_end_id": "slice_end_token_id",
    "video_token_id": "video_token_id",
    "audio_token_id": "audio_token_id",
    "audio_start_id": "audio_start_token_id",
    "audio_end_id": "audio_end_token_id",
}
# MultimodalProcessorOutput fields copied from the processor output when present.
_OUTPUT_FIELDS = (
    "mrope_positions",
    "mrope_position_delta",
    "vision_position_ids",
    "media_nums_per_sample",
    "visible_frame_counts",
    "token_type_ids",
)


class TokenSpaceSGLangProcessor:
    """Serve a model through its token-space strategy.

    `_core` is the same `TokenSpaceProcessor` a trainer uses; this class
    configures it from the server arguments and packages its BatchFeature for the
    scheduler: items, offsets, hashes, and feature transport. Public members are
    only those sglang reads from a multimodal processor; the rest is private.
    """

    prefer_tokenized_input = True
    _strategy_class = None
    _serving_spec_class = TokenSpaceServingSpec

    @classmethod
    @functools.cache
    def for_strategy_and_serving_spec(cls, strategy_class, serving_spec_class, /):
        # Positional-only, so every call caches under one key per pair.
        return type(
            f"{cls.__name__}[{strategy_class.__name__}, {serving_spec_class.__name__}]",
            (cls,),
            {
                "_strategy_class": strategy_class,
                "_serving_spec_class": serving_spec_class,
            },
        )

    def __init__(self, hf_config, server_args, _processor, transport_mode, **kwargs):
        strategy_class = self._strategy_class
        self._serving_spec = self._serving_spec_class(hf_config)
        self._server_args = server_args
        mm = get_mm()
        self._video_preprocessing_device = (mm.mm_process_config or {}).get(
            "video_preprocessing_device", self._serving_spec.video_preprocessing_device
        )
        skip_mm_pool = kwargs.get("skip_mm_pool", False)
        self._feature_transport = MultimodalFeatureTransport(
            mm.mm_feature_transport,
            base_gpu_id=server_args.base_gpu_id,
            skip_mm_pool=skip_mm_pool,
        )
        self._disable_fast_image_processor = (
            mm.disable_fast_image_processor or mm.image_processor_backend == "pil"
        )
        self._rl_on_policy_target = get_exec().deterministic.rl_on_policy_target
        self._true_on_policy_enabled = is_true_on_policy_enabled()
        processor_config = MultimodalProcessorConfig(
            image_processor_backend=mm.image_processor_backend,
            disable_fast_image_processor=mm.disable_fast_image_processor,
            mm_process_config=mm.mm_process_config,
            # As in legacy: a processor built without the feature pool runs synchronously.
            mm_processor_worker_num=1
            if skip_mm_pool
            else mm.mm_processor_worker_num
            or resolve_auto_mm_processor_worker_num(
                competes_with_the_scheduler=preprocessing_competes_with_the_scheduler(
                    _processor,
                    disable_fast_image_processor=self._disable_fast_image_processor,
                    rl_on_policy_target=self._rl_on_policy_target,
                    true_on_policy_enabled=self._true_on_policy_enabled,
                ),
                declared=None,
            ),
            mm_io_worker_num=mm.mm_io_worker_num
            or int(os.environ.get("SGLANG_IO_WORKERS", 0))
            or self._serving_spec.auto_mm_io_worker_num,
            allowed_media_domains=mm.allowed_media_domains,
            media_url_max_file_size_mb=mm.media_url_max_file_size_mb,
            keep_mm_features_on_device=self._feature_transport.keep_mm_features_on_device,
        )
        self._core = TokenSpaceProcessor.for_strategy(strategy_class)(
            hf_config,
            _processor,
            processor_config=processor_config,
        )
        # The tokenizer manager sets pad values through it.
        self.hash_executor = MultimodalHashExecutor()
        # Request validation reads the prompt's media markers through it.
        self.mm_tokens = self._serving_spec.build_mm_tokens(_processor)
        self._image_device = self._serving_image_processor_device()

    @property
    def cudaipc_mmfeature_pool(self):
        # PositionalEmbedsTransport looks this pool up by name.
        return self._feature_transport.cudaipc_mmfeature_pool

    # Adapted from the device choice in sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor.process_mm_data.
    def _serving_image_processor_device(self) -> Optional[str]:
        processor = self._core._processor
        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessor)
            and not self._disable_fast_image_processor
        ):
            device = fast_image_processor_device(
                processor,
                base_gpu_id=self._server_args.base_gpu_id,
                rl_on_policy_target=self._rl_on_policy_target,
                true_on_policy_enabled=self._true_on_policy_enabled,
            )
            if device == "npu":
                apply_npu_image_preprocess_patches(processor)
            return device
        return None

    async def process_mm_data_async(
        self,
        image_data=None,
        input_text="",
        request_obj=None,
        *,
        input_ids=None,
        audio_data=None,
        video_data=None,
        max_req_input_len=None,
        **kwargs,
    ) -> MultimodalProcessorOutput:
        """The processor entry; `input_text` holds the request's input_ids when it has them."""
        if input_ids is None and isinstance(input_text, list):
            input_ids = input_text
        if video_data is None and request_obj is not None:
            video_data = request_obj.video_data
        if audio_data is None and request_obj is not None:
            audio_data = request_obj.audio_data
        image_device, video_device = self._image_device, None
        # As legacy: a video request preprocesses all its media on the video device.
        if video_data and self._video_preprocessing_device is not None:
            image_device = video_device = self._video_preprocessing_device
        return await self._core.process(
            image_data,
            video_data,
            audio_data,
            input_ids=input_ids,
            image_device=image_device,
            video_device=video_device,
            compute_mrope_positions=True,
            package_output=self._package_output,
        )

    def _package_output(self, output: BatchFeature) -> MultimodalProcessorOutput:
        """Package the token-space output for the scheduler.

        Runs on a processor worker thread, concurrently with other requests. Read
        only state fixed at construction, never `self._core._processor`: that is the
        shared original, and each worker holds its own clone. The CUDA IPC pool is
        the only shared mutable state, and it locks its own allocations.
        """
        input_ids = output["input_ids"].flatten()
        # Offsets are where locate_media put each media, inclusive, in list order.
        offsets = {}
        for modality, spans in zip(output["mm_modalities"], output["mm_spans"]):
            offsets.setdefault(modality, []).extend(
                (start, end - 1) for start, end in spans
            )
        mm_items = self._collect_mm_items(output)
        for item in mm_items:
            item.offsets = offsets[item.modality]
        mm_items = self._serving_spec.split_mm_items(mm_items)
        # Hash the feature tensors themselves: transport may swap them for CUDA IPC handles.
        self._set_item_hashes(mm_items)
        mm_items = self._feature_transport.prepare_mm_items_for_transport(mm_items)
        input_ids_list = input_ids.tolist()
        params = self._core.strategy.mm_token_expansion_params
        # No padded_input_ids: caller mm_hashes may still replace the item hashes,
        # so the scheduler pads with the final pad values.
        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            mm_items=mm_items,
            **{
                field: params.get(param)
                for field, param in _OUTPUT_TOKEN_ID_PARAMS.items()
            },
            **{field: output[field] for field in _OUTPUT_FIELDS if field in output},
        )

    def _set_item_hashes(self, mm_items: list[MultimodalDataItem]) -> None:
        """Hash each item's feature scoped by its other strategy fields, e.g. grids.

        Features carry no layout: a solid 448x896 and 896x448 image give equal
        patches and differ only in image_grid_thw, so a feature-only hash would let
        one reuse the other's embedding and KV cache.
        """
        strategy = self._core.strategy
        layout_fields = sorted(
            set(strategy.mm_item_fields) - strategy.mm_feature_fields
        )
        for item in mm_items:
            feature = item.feature
            # tensor_hash copies a CUDA feature to the current device, cuda:0 on this thread.
            on_feature_device = (
                torch.cuda.device(feature.device)
                if isinstance(feature, torch.Tensor) and feature.is_cuda
                else contextlib.nullcontext()
            )
            with on_feature_device:
                item.set_hash(
                    resolve_multimodal_item_hash(
                        feature=feature,
                        namespace=_layout_digest(item, layout_fields),
                    )
                )

    # Adapted from sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor.collect_mm_items_from_processor_output.
    def _collect_mm_items(self, output: BatchFeature) -> list[MultimodalDataItem]:
        # One item per modality, from the fields the strategy declares.
        items: dict[Modality, MultimodalDataItem] = {}
        for name, value in output.items():
            modality = self._core.strategy.mm_item_fields.get(name)
            if modality is None:
                continue
            if modality not in items:
                items[modality] = MultimodalDataItem(modality=modality)
            field = "feature" if name in self._core.strategy.mm_feature_fields else name
            items[modality].set(field, value)
        return list(items.values())

    def compute_mrope_positions(self, input_ids, mm_items):
        """M-RoPE from bound items, for requests that arrive already processed.

        Only the scheduler's fallback calls this. It rebuilds the processor output's
        media fields from each item's strategy fields.
        """
        # TODO(rockdu): Corner case: requests that arrive without M-RoPE positions.
        # Check whether every frontend can send positions, so this fallback can be removed.
        strategy = self._core.strategy
        item_fields = {
            name: modality
            for name, modality in strategy.mm_item_fields.items()
            if name not in strategy.mm_feature_fields
        }
        media_items = [(item.modality,) for item in mm_items]
        item_features = [
            {
                name: torch.as_tensor(value)
                for name, value in item.model_specific_data.items()
                if item_fields.get(name) == item.modality
            }
            for item in mm_items
        ]
        output = dict(strategy.merge_media(media_items, item_features))
        output["input_ids"] = input_ids
        return strategy.compute_mrope_positions(output)

    def resolve_image_token_counts(self, images: list) -> list[int]:
        """Per-image expanded token counts, from the HF processor's own helper."""
        return resolve_hf_image_token_counts(self._core._processor, images)

    def clear_preprocess_cache(self) -> None:
        """No preprocess cache to clear; token-space output is never cached."""

    def shutdown(self) -> None:
        self.hash_executor.shutdown()
        self._core.shutdown()


def _layout_digest(item: MultimodalDataItem, field_names: list[str]) -> str:
    """SHA-256 identity of the named fields an item holds, in the given order."""
    hasher = hashlib.sha256()
    for name in field_names:
        if name not in item.model_specific_data:
            continue
        value = torch.as_tensor(item.model_specific_data[name]).detach().cpu()
        for part in (
            name.encode(),
            str(value.dtype).encode(),
            str(tuple(value.shape)).encode(),
            value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes(),
        ):
            hasher.update(len(part).to_bytes(8, "big"))
            hasher.update(part)
    return CONTENT_HASH_PREFIX + hasher.hexdigest()
