import re
from typing import Optional, Union

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalProcessorOutput,
)
from sglang.srt.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

# Default ASR prompt template for Qwen3-ASR transcription endpoint
_DEFAULT_ASR_PROMPT = (
    "<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|>"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):
    models = [Qwen3ASRForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # Access the thinker_config for token IDs
        if hasattr(hf_config, "thinker_config"):
            thinker_config = hf_config.thinker_config
        else:
            thinker_config = hf_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Audio special tokens
        self.AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_start\|>(?:<\|audio_pad\|>)+<\|audio_end\|>"
        )

        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update(
            {"feature_attention_mask": Modality.AUDIO}
        )

    def _build_transcription_prompt(self, input_text: Union[str, list]) -> str:
        """Build a prompt for the transcription endpoint.

        When the input text is empty (from /v1/audio/transcriptions),
        construct the default Qwen3-ASR chat prompt with an audio placeholder.
        """
        if isinstance(input_text, list):
            # Token IDs - decode to text first
            input_text = self._tokenizer.decode(input_text)

        if not input_text or not input_text.strip():
            return _DEFAULT_ASR_PROMPT
        return input_text

    def _compute_mrope_positions(
        self,
        input_ids: torch.Tensor,
        audio_feature_lengths: Optional[torch.Tensor] = None,
    ):
        """Compute MRoPE positions for Qwen3-ASR.

        For audio-only model, all 3 MRoPE dimensions get the same sequential
        positions. Audio tokens get sequential positions just like text tokens.
        """
        seq_len = input_ids.shape[0]
        if input_ids.dim() > 1:
            seq_len = input_ids.shape[-1]

        # For Qwen3-ASR, all 3 dimensions get identical sequential positions
        # since audio tokens don't have spatial structure
        positions = torch.arange(seq_len, dtype=torch.long)
        mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        mrope_position_delta = torch.tensor([0], dtype=torch.long)

        return mrope_positions, mrope_position_delta

    def compute_mrope_positions(self, input_ids, mm_items):
        """Compute M-RoPE positions for Qwen3-ASR.

        All 3 dimensions get identical sequential positions since audio
        tokens have no spatial structure.
        """
        if isinstance(input_ids, list):
            seq_len = len(input_ids)
        else:
            seq_len = input_ids.shape[-1] if input_ids.dim() > 1 else input_ids.shape[0]

        positions = torch.arange(seq_len, dtype=torch.long)
        mrope_positions = positions.unsqueeze(0).expand(3, -1).clone()
        return mrope_positions, torch.tensor([0], dtype=torch.long)

    async def process_mm_data_async(
        self,
        audio_data=None,
        input_text=None,
        request_obj=None,
        **kwargs,
    ):
        if not audio_data:
            return None

        # Build the prompt - handles empty text from transcription endpoint
        prompt = self._build_transcription_prompt(input_text)

        base_output = self.load_mm_data(
            prompt=prompt,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # The feature_attention_mask is automatically set on audio items
        # by the base processor's collect_mm_items_from_processor_output()
        # since we registered it in ATTR_NAME_TO_MODALITY

        # Compute MRoPE positions
        mrope_positions, mrope_position_delta = self._compute_mrope_positions(
            input_ids
        )

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            audio_start_id=self.audio_start_id,
            audio_token_id=self.audio_token_id,
            audio_end_id=self.audio_end_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
