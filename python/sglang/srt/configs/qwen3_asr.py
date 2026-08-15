import torch
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    PretrainedConfig,
    ProcessorMixin,
)

from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.utils import logger


class Qwen3ASRProcessor(ProcessorMixin):
    """Minimal composite processor: WhisperFeatureExtractor + Qwen2Tokenizer.

    AutoProcessor.from_pretrained() for Qwen3-ASR returns just a tokenizer,
    but SGLang's multimodal pipeline needs a processor that handles audio.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def _get_feat_extract_output_lengths(self, input_lengths):
        if not isinstance(input_lengths, torch.Tensor):
            input_lengths = torch.tensor(input_lengths)
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    def __call__(self, text=None, audio=None, audio_kwargs=None, **kwargs):
        inputs = {}
        if audio is not None:
            audio_kwargs = audio_kwargs or {}
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors=kwargs.get("return_tensors"),
                **audio_kwargs,
            )
            inputs["input_features"] = audio_inputs["input_features"]
            if "attention_mask" in audio_inputs:
                inputs["feature_attention_mask"] = audio_inputs["attention_mask"]

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors"),
                padding=kwargs.get("padding", False),
            )
            input_ids = text_inputs["input_ids"]

            # Expand the single <|audio_pad|> placeholder in the prompt to N
            # copies, where N is the audio encoder's output length for this clip.
            # Without this, the model only sees 1 audio token for hundreds of
            # feature frames and can't align audio embeddings with token positions.
            if audio is not None and "feature_attention_mask" in inputs:
                audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
                feat_lengths = inputs["feature_attention_mask"].sum(dim=-1)
                audio_token_counts = self._get_feat_extract_output_lengths(feat_lengths)
                expanded = []
                for seq_idx in range(input_ids.shape[0]):
                    ids = input_ids[seq_idx].tolist()
                    audio_idx = 0
                    new_ids = []
                    for tid in ids:
                        if tid == audio_pad_id and audio_idx < len(audio_token_counts):
                            n = int(audio_token_counts[audio_idx].item())
                            new_ids.extend([audio_pad_id] * n)
                            audio_idx += 1
                        else:
                            new_ids.append(tid)
                    expanded.append(new_ids)
                max_len = max(len(s) for s in expanded)
                pad_id = self.tokenizer.pad_token_id or 0
                padded = [s + [pad_id] * (max_len - len(s)) for s in expanded]
                input_ids = torch.tensor(padded, dtype=torch.long)

            inputs["input_ids"] = input_ids
        return inputs


class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type = "qwen3_asr_thinker"
    sub_configs = {
        "audio_config": Qwen3OmniMoeAudioEncoderConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151676,
        audio_start_token_id=151669,
        audio_end_token_id=151670,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        self.audio_config = audio_config

        from transformers.models.qwen3.configuration_qwen3 import (
            Qwen3Config as HFQwen3Config,
        )

        if isinstance(text_config, dict):
            text_config = HFQwen3Config(**text_config)
        elif text_config is None:
            text_config = HFQwen3Config()

        self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id


@register_customized_processor(Qwen3ASRProcessor)
class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(self, thinker_config=None, **kwargs):
        if thinker_config is None:
            thinker_config = {}
            logger.info(
                "thinker_config is None. "
                "Initializing Qwen3-ASR thinker with default values"
            )
        if isinstance(thinker_config, dict):
            self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        else:
            self.thinker_config = thinker_config
        super().__init__(**kwargs)

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.thinker_config.text_config


AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoConfig.register("qwen3_asr_thinker", Qwen3ASRThinkerConfig)
