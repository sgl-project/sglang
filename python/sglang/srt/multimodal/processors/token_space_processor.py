from typing import Optional

import torch
from transformers import BatchFeature

from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class MultimodalProcessorOutputBuilder:
    """Turn the processor BatchFeature and its bound items into the serving output."""

    def __init__(self, hf_config):
        self.hf_config = hf_config
        self.model_type = hf_config.model_type
        self.image_token_id = getattr(hf_config, "image_token_id", None)
        self.video_token_id = getattr(hf_config, "video_token_id", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

    def build(
        self,
        ret: BatchFeature,
        mm_items: list[MultimodalDataItem],
        input_ids: torch.Tensor,
        *,
        keep_mm_features_on_device: bool,
        base_input_ids: Optional[list[int]] = None,
    ) -> MultimodalProcessorOutput:
        input_ids_list = input_ids.flatten().tolist()
        return MultimodalProcessorOutput(
            input_ids=input_ids_list,
            padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
                input_ids_list, mm_items
            ),
            mm_items=mm_items,
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
        )

    def compute_mrope_positions(self, input_ids, mm_items):
        return None, None


class TokenSpaceMultimodalProcessor(BaseMultimodalProcessor):
    """Serve a model through its token-space strategy and output builder."""

    use_token_space_processor = True
    prefer_tokenized_input = True
    token_space_process_strategy_class = None
    output_builder_class = MultimodalProcessorOutputBuilder

    @classmethod
    def supports_token_space_processing(cls, hf_config):
        return cls.token_space_process_strategy_class.supports_token_space_processing(
            hf_config
        )

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.token_space_process_strategy = self.token_space_process_strategy_class(
            hf_config, _processor, mm_process_config=self.mm_process_config
        )
        self.output_builder = self.output_builder_class(hf_config)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=self.output_builder.image_token_id,
            video_token_id=self.output_builder.video_token_id,
            audio_token_id=self.output_builder.audio_token_id,
        ).build(_processor)

    def compute_mrope_positions(self, input_ids, mm_items):
        return self.output_builder.compute_mrope_positions(input_ids, mm_items)

    def sglang_post_process(
        self, ret: BatchFeature, processor=None
    ) -> MultimodalProcessorOutput:
        """Turn a processor-shaped BatchFeature into the serving output, as legacy does.

        `processor` is the worker's processor clone; post-processing does not use it.
        """
        input_ids = ret["input_ids"].flatten()
        mm_items = self.collect_mm_items_from_processor_output(ret)
        mm_items = self._bind_and_finalize_mm_items(
            mm_items,
            input_ids=input_ids,
            mm_tokens=self.mm_tokens,
            images=None,
        )
        return self.output_builder.build(
            ret,
            mm_items,
            input_ids,
            keep_mm_features_on_device=self.keep_mm_features_on_device,
            base_input_ids=input_ids.tolist(),
        )

    def _get_mm_token_expansion_start_len(self, request_obj, start_len):
        from sglang.srt.managers.io_struct import GenerateReqInput

        if start_len is None:
            start_len = (
                request_obj.mm_token_expansion_start_len or 0
                if isinstance(request_obj, GenerateReqInput)
                else 0
            )
        return start_len

    async def process_mm_data_async(
        self,
        image_data=None,
        input_text="",
        request_obj=None,
        *,
        input_ids=None,
        audio_data=None,
        video_data=None,
        mm_token_expansion_start_len=None,
        max_req_input_len=None,
        **kwargs,
    ) -> MultimodalProcessorOutput:
        """The processor entry; `input_text` holds the request's input_ids when it has them."""
        if input_ids is None and isinstance(input_text, list):
            input_ids = input_text
        mm_token_expansion_start_len = self._get_mm_token_expansion_start_len(
            request_obj, mm_token_expansion_start_len
        )
        if video_data is None and request_obj is not None:
            video_data = request_obj.video_data
        if audio_data is None and request_obj is not None:
            audio_data = request_obj.audio_data
        ret = await self.load_and_process_token_space_async(
            image_data,
            video_data,
            audio_data,
            input_ids=input_ids,
            mm_token_expansion_start_len=mm_token_expansion_start_len,
            **kwargs,
        )
        return await self._run_mm_processor(self.sglang_post_process, ret=ret)
