from typing import List, Union

from sglang.srt.configs.deepseek_ocr import is_ocr2_config, local_crop_size
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


def apply_ocr_geometry(_processor, hf_config) -> None:
    """Point an OCR processor at its checkpoint's local-crop geometry.

    The checkpoint config does not carry the crop size (see `local_crop_size`),
    and the HF processor is already built by the time SGLang sees it, so the
    geometry is patched onto the instance here. Kept as a function so that both
    the policy and this patching can be asserted without a server.

    Note this overwrites `image_size` unconditionally: the geometry is derived
    from the model identity, so a value declared in `processor_config.json` would
    not survive.
    """
    _processor.ocr2_mode = is_ocr2_config(hf_config)
    _processor.image_size = local_crop_size(hf_config)


class DeepseekOCRProcessor(BaseMultimodalProcessor):
    models = [DeepseekOCRForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # The local-crop geometry is not in the checkpoints: DeepSeek-OCR and
        # DeepSeek-OCR-2 ship identical processor configs with no `image_size` and
        # candidate_resolutions=[[1024, 1024]] (the *global* base). Taking the crop
        # size from there would serve OCR-2 with OCR-1's 640px crops, whose 100
        # visual tokens match neither tuned query table. Pick it per model.
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
