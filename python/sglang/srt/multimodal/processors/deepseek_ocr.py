from typing import List, Union

from sglang.srt.configs.deepseek_ocr import is_ocr2_config, local_crop_size
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


def apply_ocr_geometry(processor, hf_config) -> None:
    """Patch a checkpoint's local-crop geometry onto an already-built HF processor.

    `image_size` is overwritten unconditionally: neither checkpoint carries the
    crop size (see `local_crop_size`), so a value in `processor_config.json` would
    not survive.
    """
    processor.ocr2_mode = is_ocr2_config(hf_config)
    processor.image_size = local_crop_size(hf_config)


class DeepseekOCRProcessor(BaseMultimodalProcessor):
    models = [DeepseekOCRForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # The shared processor config's candidate_resolutions is the *global* base, so
        # the crop size has to come from the model identity instead.
        apply_ocr_geometry(_processor, hf_config)
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
