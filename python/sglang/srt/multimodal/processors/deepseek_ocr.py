from typing import List, Union

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM, _is_ocr2
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class DeepseekOCRProcessor(BaseMultimodalProcessor):
    models = [DeepseekOCRForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # OCR-2 runs 768px local crops (144 query_768 tokens each); OCR-1 uses 640.
        # The checkpoints only declare candidate_resolutions=[[1024, 1024]] (the
        # global base), so the local crop size is not derivable from the config.
        _processor.ocr2_mode = _is_ocr2(hf_config)
        _processor.image_size = 768 if _processor.ocr2_mode else 640
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>", image_token_id=self._processor.image_token_id
        ).build(_processor)

    async def process_mm_data_async(
        self, image_data: List[Union[str, bytes]], input_text, *args, **kwargs
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )

        mm_items, input_ids, _ = await self.process_and_combine_mm_data_async(
            base_output, self.mm_tokens
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.mm_tokens.image_token_id,
        )
