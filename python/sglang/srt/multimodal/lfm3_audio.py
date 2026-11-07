# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tokenizer adapter for LFM3 audio models."""

from typing import ClassVar

from transformers import AutoFeatureExtractor, ProcessorMixin

# This is only a placeholder: every occurrence is replaced by encoder embeddings,
# with no start/end delimiter embeddings, matching the training input sequence.
AUDIO_TOKEN = "<|audio|>"


class Lfm3AudioProcessor(ProcessorMixin):
    attributes: ClassVar = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        super().__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        # get_processor may replace a TokenizersBackend after construction.
        if tokenizer.convert_tokens_to_ids(AUDIO_TOKEN) in (
            None,
            tokenizer.unk_token_id,
        ):
            tokenizer.add_tokens([AUDIO_TOKEN], special_tokens=True)
        template = tokenizer.chat_template
        if not isinstance(template, str):
            raise TypeError("LFM3 audio requires a checkpoint chat_template.jinja")
        self._tokenizer = tokenizer
        self.chat_template = tokenizer.chat_template

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        revision = kwargs.pop("revision", None)
        return cls(
            get_tokenizer(
                pretrained_model_name_or_path, tokenizer_revision=revision, **kwargs
            ),
            AutoFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path, revision=revision, **kwargs
            ),
        )
