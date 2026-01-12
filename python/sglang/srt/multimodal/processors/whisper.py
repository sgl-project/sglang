from typing import Any, Dict, Optional

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.whisper import WhisperForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import load_audio


class WhisperProcessor(BaseMultimodalProcessor):
    models = [WhisperForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not audio_data:
            return None

        audios = [load_audio(audio) for audio in audio_data]
        assert len(audios) == 1

        # For Whisper, ALWAYS use the proper transcription token sequence
        # and IGNORE any text prompt - Whisper is a pure speech-to-text model
        # The decoder_start_token_id and forced_decoder_ids from generation config
        # set up: <|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]

        # Get decoder start tokens from generation config or use defaults
        # Default: <|startoftranscript|>(50258) + <|en|>(50259) + <|transcribe|>(50360) + <|notimestamps|>(50364)
        decoder_start_token_id = getattr(
            self.hf_config, "decoder_start_token_id", 50258
        )

        # Try to get forced_decoder_ids from config
        forced_decoder_ids = getattr(self.hf_config, "forced_decoder_ids", None)
        if forced_decoder_ids:
            # forced_decoder_ids is list of [position, token_id]
            input_ids = [decoder_start_token_id]
            for _, token_id in forced_decoder_ids:
                if token_id is not None:
                    input_ids.append(token_id)
        else:
            # Default transcription tokens for English
            # <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
            input_ids = [50258, 50259, 50360, 50364]

        # Whisper expects input features padded to max_length (3000 frames = 30 seconds)
        # This is the standard context length for Whisper
        input_features = self._processor.feature_extractor(
            audios[0],
            sampling_rate=16000,
            padding="max_length",  # Pad to 3000 frames
            return_tensors="pt",
        )["input_features"][0]

        output = {}
        output["input_ids"] = input_ids
        output["mm_items"] = [
            MultimodalDataItem(
                feature=input_features,
                modality=Modality.AUDIO,
            )
        ]

        return output
