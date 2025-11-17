import logging
from typing import List, Union

from transformers.processing_utils import ProcessorMixin

from sglang.srt.models.phi4mm import Phi4MMForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


# It is an adapter of hf phi4 mm processor to make it work for sglang
# Ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py#L693
class Phi4MMProcessorAdapter(ProcessorMixin):
    def __init__(self, _processor) -> None:
        self._processor = _processor

    def __call__(self, **kwargs):
        result = self._processor(**kwargs)

        # Map HuggingFace output keys to sglang standard keys
        key_mapping = {
            "input_image_embeds": "pixel_values",
            "input_audio_embeds": "audio_features",
            "audio_embed_sizes": "audio_feature_lens",
        }
        for hf_key, sglang_key in key_mapping.items():
            if hf_key in result:
                result[sglang_key] = result[hf_key]
                del result[hf_key]

        # Filter out None or empty tensors from the result.
        # This prevents the sglang function base_processor.collect_mm_items_from_processor_output()
        # from misclassifying audio content as image content, and vice versa.
        filtered_result = {
            k: v
            for k, v in result.items()
            if v is not None and (not hasattr(v, "numel") or v.numel() > 0)
        }
        return filtered_result


class Phi4MMMultimodalProcessor(BaseMultimodalProcessor):
    models = [Phi4MMForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.processor = Phi4MMProcessorAdapter(_processor)
        super().__init__(hf_config, server_args, self.processor, *args, **kwargs)

        # the following CONSTANTS come from hugging-face microsoft/Phi-4-multimodal-instruct's processing_phi4mm.py file
        # ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py
        self.IMAGE_TOKEN = "<|endoftext10|>"
        self.AUDIO_TOKEN = "<|endoftext11|>"
        self.IM_TOKEN_ID = 200010
        self.AUDIO_TOKEN_ID = 200011
        self.AUDIO_SAMPLE_RATE = 16000

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.IMAGE_TOKEN,
            image_token_id=self.IM_TOKEN_ID,
            audio_token=self.AUDIO_TOKEN,
            audio_token_id=self.AUDIO_TOKEN_ID,
        ).build(self.processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=self.AUDIO_SAMPLE_RATE,
        )

        if base_output.audios is not None:
            # hugging-face microsoft/Phi-4-multimodal-instruct's processing_phi4mm.py file requires the audio input to be tuple of (audio, sample_rate)
            # ref: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/processing_phi4mm.py
            base_output.audios = [
                (audio, self.AUDIO_SAMPLE_RATE) for audio in base_output.audios
            ]

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
            "audio_token_id": self.mm_tokens.audio_token_id,
        }
