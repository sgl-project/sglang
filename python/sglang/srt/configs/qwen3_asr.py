# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Configuration and processor classes for Qwen3-ASR model."""

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
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initializer_range = initializer_range

        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            # Use the proper Qwen3Config so rope_parameters property works
            from transformers.models.qwen3.configuration_qwen3 import (
                Qwen3Config as HFQwen3Config,
            )

            text_config = HFQwen3Config(**text_config)
        elif text_config is None:
            text_config = PretrainedConfig()
        self.text_config = text_config

        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id


class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
            logger.info(
                "thinker_config is None. Initializing Qwen3-ASR thinker with default values"
            )

        if isinstance(thinker_config, dict):
            self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        else:
            self.thinker_config = thinker_config
        self.support_languages = support_languages or []

    def get_text_config(self, decoder=False) -> "PretrainedConfig":
        return self.thinker_config.text_config


class Qwen3ASRProcessor(ProcessorMixin):
    """Custom processor combining WhisperFeatureExtractor + Qwen2Tokenizer.

    AutoProcessor.from_pretrained() for Qwen3-ASR returns just a tokenizer
    because the model repo doesn't ship a proper ProcessorMixin class.
    This wrapper provides the composite processor that SGLang expects.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.audio_token = "<|audio_pad|>"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **{k: v for k, v in kwargs.items() if k in ("revision",)},
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **{k: v for k, v in kwargs.items() if k in ("revision", "use_fast")},
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def _get_feat_extract_output_lengths(self, input_lengths):
        """Compute the number of audio tokens from mel feature lengths."""
        import torch

        if not isinstance(input_lengths, torch.Tensor):
            input_lengths = torch.tensor(input_lengths)
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
            + (input_lengths // 100) * 13
        )
        return output_lengths

    def __call__(
        self,
        text=None,
        audio=None,
        audio_kwargs=None,
        **kwargs,
    ):
        import torch

        if audio_kwargs is None:
            audio_kwargs = {}

        if audio is not None:
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors=kwargs.get("return_tensors"),
                **audio_kwargs,
            )
            # Rename attention_mask -> feature_attention_mask
            inputs = {"input_features": audio_inputs["input_features"]}
            if "attention_mask" in audio_inputs:
                inputs["feature_attention_mask"] = audio_inputs["attention_mask"]
        else:
            inputs = {}

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors"),
                padding=kwargs.get("padding", False),
            )
            input_ids = text_inputs["input_ids"]

            # Expand <|audio_pad|> tokens based on audio feature lengths
            if audio is not None and "feature_attention_mask" in inputs:
                audio_pad_id = self.tokenizer.convert_tokens_to_ids(
                    self.audio_token
                )
                feat_mask = inputs["feature_attention_mask"]
                feat_lengths = feat_mask.sum(dim=-1)  # actual mel lengths
                audio_token_counts = self._get_feat_extract_output_lengths(
                    feat_lengths
                )

                # Expand each sequence's audio_pad tokens
                expanded_ids_list = []
                for seq_idx in range(input_ids.shape[0]):
                    seq_ids = input_ids[seq_idx].tolist()
                    audio_idx = 0
                    new_ids = []
                    for tid in seq_ids:
                        if tid == audio_pad_id and audio_idx < len(
                            audio_token_counts
                        ):
                            count = int(audio_token_counts[audio_idx].item())
                            new_ids.extend([audio_pad_id] * count)
                            audio_idx += 1
                        else:
                            new_ids.append(tid)
                    expanded_ids_list.append(new_ids)

                # Pad to same length and convert to tensor
                max_len = max(len(ids) for ids in expanded_ids_list)
                padded = [
                    ids + [self.tokenizer.pad_token_id or 0] * (max_len - len(ids))
                    for ids in expanded_ids_list
                ]
                input_ids = torch.tensor(padded, dtype=torch.long)

            inputs["input_ids"] = input_ids

        return inputs


AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoConfig.register("qwen3_asr_thinker", Qwen3ASRThinkerConfig)


@register_customized_processor(Qwen3ASRProcessor)
class _Qwen3ASRConfigForProcessorRegistration(Qwen3ASRConfig):
    """Shim so that ``_CUSTOMIZED_MM_PROCESSOR["qwen3_asr"]`` resolves to
    ``Qwen3ASRProcessor`` when ``get_processor()`` loads the model."""

    model_type = "qwen3_asr"
