import logging
import re
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    _get_feat_extract_output_lengths,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_audio

logger = logging.getLogger(__name__)


class Qwen3ASRMultimodalProcessor(BaseMultimodalProcessor):
    models = [Qwen3ASRForConditionalGeneration]

    def __init__(
        self,
        hf_config: Any,
        server_args: Any,
        _processor: Any,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        thinker_config = getattr(hf_config, "thinker_config", hf_config)
        audio_config = getattr(thinker_config, "audio_config", None)
        self.n_window = getattr(audio_config, "n_window", 50) if audio_config else 50

        self.AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_start\|>(?:<\|audio_pad\|>)+<\|audio_end\|>"
        )

        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")

        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

    def _get_feature_extractor(self) -> Any:
        if hasattr(self._processor, "feature_extractor"):
            return self._processor.feature_extractor
        raise ValueError(
            "Qwen3-ASR processor does not have a feature_extractor attribute"
        )

    def _compute_audio_output_length(self, feature_len: int) -> int:
        """Compute the number of output tokens from the audio encoder."""
        t = torch.tensor([feature_len], dtype=torch.long)
        return _get_feat_extract_output_lengths(t, self.n_window).item()

    @staticmethod
    def _build_mrope_positions(
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build MRoPE positions for pure audio (no spatial structure).

        All 3 dimensions get identical sequential positions.
        Returns (mrope_positions [3, seq_len], mrope_position_delta [1]).
        """
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, -1)
        position_delta = torch.zeros(1, dtype=torch.long)
        return positions, position_delta

    def _build_transcription_input_ids(
        self, num_audio_tokens: int
    ) -> List[int]:
        """Build input_ids for transcription endpoint (empty text prompt).

        Produces: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n
                  <|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|>
                  <|im_end|>\n<|im_start|>assistant\n
        """
        tokenizer = self._processor.tokenizer

        system_tokens = tokenizer.encode(
            "<|im_start|>system\n<|im_end|>\n", add_special_tokens=False
        )
        user_start_tokens = tokenizer.encode(
            "<|im_start|>user\n", add_special_tokens=False
        )
        audio_tokens = (
            [self.audio_start_id]
            + [self.audio_token_id] * num_audio_tokens
            + [self.audio_end_id]
        )
        user_end_tokens = tokenizer.encode(
            "<|im_end|>\n", add_special_tokens=False
        )
        assistant_start_tokens = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

        input_ids = (
            system_tokens
            + user_start_tokens
            + audio_tokens
            + user_end_tokens
            + assistant_start_tokens
        )
        return input_ids

    @staticmethod
    def _pop_language(request_obj: Any) -> Optional[str]:
        """Pop language from sampling_params so it doesn't reach SamplingParams.__init__."""
        sampling_params = getattr(request_obj, "sampling_params", None) or {}
        return sampling_params.pop("language", None)

    async def process_mm_data_async(
        self,
        image_data: Any,
        audio_data: Any,
        input_text: str,
        request_obj: Any,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        if not audio_data:
            return None

        # Pop language before it reaches SamplingParams
        self._pop_language(request_obj)

        feature_extractor = self._get_feature_extractor()

        is_transcription = not input_text or not input_text.strip()

        if is_transcription:
            return await self._process_transcription(audio_data, feature_extractor)
        else:
            return await self._process_chat(
                audio_data, input_text, feature_extractor
            )

    async def _process_transcription(
        self,
        audio_data: Any,
        feature_extractor: Any,
    ) -> Optional[Dict[str, Any]]:
        """Handle transcription endpoint (empty text, raw audio)."""
        if len(audio_data) != 1:
            raise ValueError(
                f"Qwen3-ASR transcription expects exactly 1 audio input, got {len(audio_data)}"
            )

        audio = load_audio(audio_data[0])

        features = feature_extractor(
            audio,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = features["input_features"][0]
        if "attention_mask" in features:
            feature_len = features["attention_mask"].sum().item()
        else:
            feature_len = input_features.shape[-1]

        num_audio_tokens = self._compute_audio_output_length(feature_len)
        input_ids = self._build_transcription_input_ids(num_audio_tokens)

        feature_lens = torch.tensor([feature_len], dtype=torch.long)

        item = MultimodalDataItem(
            feature=input_features,
            modality=Modality.AUDIO,
        )
        item["audio_feature_lens"] = feature_lens
        mm_items = [item]

        mrope_positions, mrope_position_delta = self._build_mrope_positions(
            len(input_ids)
        )

        return {
            "input_ids": input_ids,
            "mm_items": mm_items,
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }

    async def _process_chat(
        self,
        audio_data: Any,
        input_text: str,
        feature_extractor: Any,
    ) -> Optional[Dict[str, Any]]:
        """Handle chat completions endpoint (text with audio tokens)."""
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        if "feature_attention_mask" in ret:
            feature_lens = ret["feature_attention_mask"].sum(dim=-1).long()
        elif "input_features" in ret:
            input_features_tensor = ret["input_features"]
            feature_lens = torch.tensor(
                [input_features_tensor.shape[-1]] * input_features_tensor.shape[0],
                dtype=torch.long,
            )
        else:
            feature_lens = torch.tensor(
                [feature_extractor.nb_max_frames] * len(audio_data),
                dtype=torch.long,
            )

        for i, item in enumerate(mm_items):
            if item.modality == Modality.AUDIO and i < len(feature_lens):
                item["audio_feature_lens"] = feature_lens[i : i + 1]

        input_ids_list = input_ids.tolist()
        mrope_positions, mrope_position_delta = self._build_mrope_positions(
            len(input_ids_list)
        )

        return {
            "mm_items": mm_items,
            "input_ids": input_ids_list,
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
