# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team

from .base import TransformersBase
from .causal import CausalMixin
from .moe import MoEMixin
from .multimodal import MultiModalMixin
from .pooling import ClassificationMixin, EmbeddingMixin


class TransformersForCausalLM(CausalMixin, TransformersBase):
    pass


class TransformersMoEForCausalLM(MoEMixin, CausalMixin, TransformersBase):
    pass


class TransformersMultiModalForCausalLM(MultiModalMixin, CausalMixin, TransformersBase):
    pass


class TransformersMultiModalMoEForCausalLM(
    MultiModalMixin, MoEMixin, CausalMixin, TransformersBase
):
    pass


class TransformersEmbeddingModel(EmbeddingMixin, TransformersBase):
    pass


class TransformersMoEEmbeddingModel(MoEMixin, EmbeddingMixin, TransformersBase):
    pass


class TransformersMultiModalEmbeddingModel(
    MultiModalMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEEmbeddingModel(
    MultiModalMixin, MoEMixin, EmbeddingMixin, TransformersBase
):
    pass


class TransformersForSequenceClassification(ClassificationMixin, TransformersBase):
    pass


class TransformersMoEForSequenceClassification(
    MoEMixin, ClassificationMixin, TransformersBase
):
    pass


class TransformersMultiModalForSequenceClassification(
    MultiModalMixin, ClassificationMixin, TransformersBase
):
    pass


class TransformersMultiModalMoEForSequenceClassification(
    MultiModalMixin, MoEMixin, ClassificationMixin, TransformersBase
):
    pass


EntryClass = [
    TransformersForCausalLM,
    TransformersMoEForCausalLM,
    TransformersMultiModalForCausalLM,
    TransformersMultiModalMoEForCausalLM,
    TransformersEmbeddingModel,
    TransformersMoEEmbeddingModel,
    TransformersMultiModalEmbeddingModel,
    TransformersMultiModalMoEEmbeddingModel,
    TransformersForSequenceClassification,
    TransformersMoEForSequenceClassification,
    TransformersMultiModalForSequenceClassification,
    TransformersMultiModalMoEForSequenceClassification,
]
