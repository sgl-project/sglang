import functools
from typing import Optional

import torch
from transformers import BaseImageProcessor, BatchFeature

from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.hash_executor import MultimodalHashExecutor
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.serving_preprocessing import (
    fast_image_processor_device,
    preprocessing_competes_with_the_scheduler,
    resolve_auto_mm_processor_worker_num,
    resolve_hf_image_token_counts,
)
from sglang.srt.multimodal.token_space_mm_process.processor import (
    TokenSpaceProcessor,
)
from sglang.srt.multimodal.transport.feature_transport import MultimodalFeatureTransport
from sglang.srt.runtime_context import get_mm

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

    @classmethod
    @functools.cache
    def for_strategy(cls, strategy_class):
        return type(
            f"{cls.__name__}[{strategy_class.__name__}]",
            (cls,),
            {"_strategy_class": strategy_class},
        )

    def __init__(self, hf_config, server_args, _processor, transport_mode, **kwargs):
        strategy_class = self._strategy_class
        self._server_args = server_args
        mm = get_mm()
        self._feature_transport = MultimodalFeatureTransport(
            mm.mm_feature_transport,
            base_gpu_id=server_args.base_gpu_id,
            skip_mm_pool=kwargs.get("skip_mm_pool", False),
        )
        # PositionalEmbedsTransport reuses this pool, looked up by this name.
        self.cudaipc_mmfeature_pool = self._feature_transport.cudaipc_mmfeature_pool
        self._disable_fast_image_processor = (
            mm.disable_fast_image_processor or mm.image_processor_backend == "pil"
        )
        processor_config = MultimodalProcessorConfig(
            image_processor_backend=mm.image_processor_backend,
            disable_fast_image_processor=mm.disable_fast_image_processor,
            mm_process_config=mm.mm_process_config,
            mm_processor_worker_num=mm.mm_processor_worker_num
            or resolve_auto_mm_processor_worker_num(
                competes_with_the_scheduler=preprocessing_competes_with_the_scheduler(
                    _processor,
                    disable_fast_image_processor=self._disable_fast_image_processor,
                ),
                declared=None,
            ),
            mm_io_worker_num=mm.mm_io_worker_num,
            allowed_media_domains=mm.allowed_media_domains,
            media_url_max_file_size_mb=mm.media_url_max_file_size_mb,
            keep_mm_features_on_device=self._feature_transport.keep_mm_features_on_device,
        )
        self._core = TokenSpaceProcessor.for_strategy(strategy_class)(
            hf_config,
            _processor,
            processor_config=processor_config,
            compute_mrope_positions=True,
            # Decode where the legacy processors do.
            pin_io_threads_to_current_device=False,
            **kwargs,
        )
        # The tokenizer manager sets pad values through it.
        self.hash_executor = MultimodalHashExecutor()

    # Adapted from the device choice in sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor.process_mm_data.
    def _serving_image_processor_device(self) -> Optional[str]:
        processor = self._core._processor
        if (
            hasattr(processor, "image_processor")
            and isinstance(processor.image_processor, BaseImageProcessor)
            and not self._disable_fast_image_processor
        ):
            return fast_image_processor_device(
                processor, base_gpu_id=self._server_args.base_gpu_id
            )
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
        from sglang.srt.managers.io_struct import GenerateReqInput

        mm_token_expansion_start_len = (
            request_obj.mm_token_expansion_start_len
            if isinstance(request_obj, GenerateReqInput)
            else 0
        )
        if video_data is None and request_obj is not None:
            video_data = request_obj.video_data
        if audio_data is None and request_obj is not None:
            audio_data = request_obj.audio_data
        kwargs.setdefault("image_device", self._serving_image_processor_device())
        return await self._core.process(
            image_data,
            video_data,
            audio_data,
            input_ids=input_ids,
            mm_token_expansion_start_len=mm_token_expansion_start_len,
            package_output=self._package_output,
            **kwargs,
        )

    def _package_output(self, output: BatchFeature) -> MultimodalProcessorOutput:
        """Package the token-space output for the scheduler; the same for every model.

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
        # Split bundled items into per-image/video items for better cache granularity.
        from sglang.srt.managers.mm_utils import get_new_expanded_mm_items

        mm_items = get_new_expanded_mm_items(mm_items)
        if self._core.precompute_hash_before_cpu_transfer:
            self._feature_transport.precompute_hashes_before_cpu_transfer(mm_items)
        mm_items = self._feature_transport.prepare_mm_items_for_transport(mm_items)
        input_ids_list = input_ids.tolist()
        params = self._core.strategy.mm_token_expansion_params
        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
                input_ids_list, mm_items
            ),
            mm_items=mm_items,
            **{
                field: params.get(param)
                for field, param in _OUTPUT_TOKEN_ID_PARAMS.items()
            },
            **{field: output[field] for field in _OUTPUT_FIELDS if field in output},
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
        """M-RoPE from bound items, for requests that arrive already processed."""
        media_features = {}
        for key, modality in (
            ("image_grid_thw", Modality.IMAGE),
            ("video_grid_thw", Modality.VIDEO),
        ):
            grids = [
                torch.as_tensor(item.model_specific_data[key]).reshape(-1, 3)
                for item in mm_items
                if item.is_modality(modality) and key in item.model_specific_data
            ]
            if grids:
                media_features[key] = torch.cat(grids)
        return self._core.strategy.compute_mrope_positions(input_ids, media_features)

    def resolve_image_token_counts(self, images: list) -> list[int]:
        """Per-image expanded token counts, from the HF processor's own helper."""
        return resolve_hf_image_token_counts(self._core._processor, images)

    def clear_preprocess_cache(self) -> None:
        """No preprocess cache to clear; token-space output is never cached."""

    def shutdown(self) -> None:
        self.hash_executor.shutdown()
        self._core.shutdown()
