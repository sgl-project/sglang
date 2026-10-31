# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification

from sglang.srt.configs.embedding_model_spec import (
    resolve_embedding_model_spec,
)
from sglang.srt.configs.transformers_task import (
    DenseSpec,
    resolve_embedding_pipeline,
    resolve_metadata_file,
)
from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    pool_hidden_states,
    score_and_pool,
)

_ACTIVATIONS = {
    "identity": nn.Identity,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


_CLASSIFICATION_PREFIXES = {
    "score.": "task_head.classifier.",
    "classifier.": "task_head.classifier.",
    "model.score.": "task_head.classifier.",
    "model.classifier.": "task_head.classifier.",
}


class Normalize(nn.Module):
    def forward(self, value):
        return nn.functional.normalize(value, p=2, dim=-1)


class TransformersPooler(nn.Module):
    def __init__(self, plan, *, supports_dimensions=False, device=None):
        super().__init__()
        self.plan = plan
        self.pooling_type = PoolingType[plan.pooling.name]
        self.supports_dimensions = supports_dimensions
        stages = []
        for stage in plan.stages:
            if stage == "normalize":
                stages.append(Normalize())
            else:
                stages.append(
                    nn.Sequential(
                        nn.Linear(
                            stage.in_features,
                            stage.out_features,
                            bias=stage.bias,
                            device=device,
                        ),
                        _ACTIVATIONS[stage.activation](),
                    )
                )
        self.stages = nn.ModuleList(stages)

    def forward(self, hidden_states, forward_batch):
        pooled = pool_hidden_states(self.pooling_type, hidden_states, forward_batch)
        value = pooled
        dimensions = getattr(forward_batch, "dimensions", None)
        if dimensions is not None:
            if not self.supports_dimensions:
                raise ValueError(
                    "This embedding checkpoint does not declare Matryoshka dimensions"
                )
            if len(dimensions) != value.shape[0] or any(
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or size > self.plan.output_dimension
                for size in dimensions
            ):
                raise ValueError(
                    "Embedding dimensions must fit the final projection size"
                )
        for index, stage in enumerate(self.stages):
            if (
                dimensions is not None
                and index == len(self.stages) - 1
                and isinstance(stage, Normalize)
            ):
                value = [stage(row[:size]) for row, size in zip(value, dimensions)]
            else:
                value = stage(value)
        if dimensions is not None and not isinstance(value, list):
            value = [row[:size] for row, size in zip(value, dimensions)]
        return EmbeddingPoolerOutput(
            embeddings=value,
            pooled_hidden_states=pooled
            if getattr(forward_batch, "return_pooled_hidden_states", False)
            else None,
        )

    def load_projection_weights(self, model_path, revision=None):
        loaded = set()
        for index, spec in enumerate(self.plan.stages):
            if not isinstance(spec, DenseSpec):
                continue
            path = resolve_metadata_file(
                model_path, f"{spec.path}/model.safetensors", revision, optional=True
            )
            if path is not None:
                from safetensors.torch import load_file

                state = load_file(path, device="cpu")
            else:
                path = resolve_metadata_file(
                    model_path, f"{spec.path}/pytorch_model.bin", revision
                )
                state = torch.load(path, map_location="cpu", weights_only=True)
            normalized = {}
            for name, value in state.items():
                if name.startswith("linear."):
                    name = name.removeprefix("linear.")
                elif name.startswith("dense."):
                    name = name.removeprefix("dense.")
                if name in normalized:
                    raise ValueError(f"Duplicate Dense tensor: {name}")
                normalized[name] = value
            self.stages[index][0].load_state_dict(normalized, strict=True)
            loaded.update(f"pooler.stages.{index}.0.{name}" for name in normalized)
        return loaded


class PackedClassificationHead(nn.Module):
    def __init__(self, task_model):
        super().__init__()
        model_type = task_model.config.model_type
        if model_type == "bert":
            self.kind = "bert"
            self.classifier = task_model.classifier
            self.dropout = task_model.dropout
        elif model_type in {"roberta", "xlm-roberta", "camembert"}:
            self.kind = "roberta"
            self.classifier = task_model.classifier
            self.dropout = nn.Identity()
        elif model_type in {
            "qwen2",
            "qwen3",
            "llama",
            "mistral",
            "gemma",
            "gemma2",
            "gemma3_text",
            "phi3",
            "qwen2_moe",
            "qwen3_moe",
        } and isinstance(getattr(task_model, "score", None), nn.Linear):
            self.kind = "decoder"
            self.classifier = task_model.score
            self.dropout = nn.Identity()
        else:
            raise ValueError(
                f"The Transformers backend has no packed sequence classification contract for {type(task_model).__name__}"
            )
        self.num_labels = task_model.config.num_labels
        activation = getattr(
            task_model.config, "sbert_ce_default_activation_function", None
        )
        activation_type = (
            activation.rsplit(".", 1)[-1].lower() if activation else "identity"
        )
        if activation_type not in _ACTIVATIONS:
            raise ValueError(f"Unsupported cross-encoder activation: {activation!r}")
        self.activation = _ACTIVATIONS[activation_type]()

    def forward(self, pooled, backbone_pooler=None):
        if self.kind == "bert":
            if backbone_pooler is None:
                raise ValueError(
                    "BERT classification requires its trained backbone pooler"
                )
            logits = self.classifier(self.dropout(backbone_pooler(pooled[:, None, :])))
        elif self.kind == "roberta":
            logits = self.classifier(pooled[:, None, :])
        else:
            logits = self.classifier(pooled)
        if logits.ndim != 2 or logits.shape[-1] != self.num_labels:
            raise ValueError("Classification head returned an invalid logits shape")
        return self.activation(logits)


def _task_spec(owner):
    model_config = getattr(owner, "model_config", None)
    spec = getattr(model_config, "embedding_model_spec", None)
    if spec is None:
        spec = resolve_embedding_model_spec(
            getattr(owner.config, "architectures", None),
            is_embedding_requested=True,
            is_embedding_gemma=owner.text_config.model_type == "gemma3_text"
            and getattr(owner.text_config, "use_bidirectional_attention", False),
            model_type=owner.text_config.model_type,
        )
    return spec


def _task_location(owner):
    model_config = getattr(owner, "model_config", None)
    return (
        getattr(model_config, "model_path", None)
        or getattr(owner.config, "_name_or_path", ""),
        getattr(model_config, "revision", None),
    )


def task_checkpoint_prefixes(backbone, *, classification=False):
    mappings = dict(_CLASSIFICATION_PREFIXES) if classification else {}
    prefix = getattr(backbone, "base_model_prefix", "")
    if prefix:
        mappings[f"{prefix}."] = "model."
    return mappings


def _install_backbone_mapper(owner, backbone, *, classification=False):
    from sglang.srt.models.utils import WeightsMapper

    owner.weight_mapper = owner.weight_mapper | WeightsMapper(
        orig_to_new_prefix=task_checkpoint_prefixes(
            backbone, classification=classification
        )
    )


class EmbeddingMixin:
    def _build_model(self, config):
        backbone = AutoModel.from_config(
            config,
            torch_dtype=torch.get_default_dtype(),
            trust_remote_code=self.trust_remote_code,
        )
        _install_backbone_mapper(self, backbone)
        if config.model_type in {"bert", "roberta", "xlm-roberta", "camembert"}:
            backbone.pooler = None
            self.skip_prefixes.append("model.pooler.")
        return backbone

    def _configure_task(self):
        self.ignore_unexpected_prefixes.append("lm_head.")
        if not self.pp_group.is_last_rank:
            return
        model_config = getattr(self, "model_config", None)
        plan = getattr(model_config, "transformers_embedding_plan", None)
        if plan is None:
            model_path, revision = _task_location(self)
            plan = resolve_embedding_pipeline(
                self.text_config, _task_spec(self), model_path, revision
            )
        self.pooler = TransformersPooler(
            plan,
            supports_dimensions=bool(getattr(model_config, "is_matryoshka", False)),
            device="meta",
        )

    def _pool_output(self, hidden_states, forward_batch):
        return self.pooler(hidden_states, forward_batch)

    def _validate_task_weights(self, loaded):
        if self.pooler is not None and not self._weights_loaded:
            loaded.update(self.pooler.load_projection_weights(*_task_location(self)))


class ClassificationMixin:
    def _build_model(self, config):
        task_model = AutoModelForSequenceClassification.from_config(
            config,
            torch_dtype=torch.get_default_dtype(),
            trust_remote_code=self.trust_remote_code,
        )
        backbone = task_model.base_model
        if backbone is task_model:
            raise ValueError(
                "Sequence classification models must expose a separate backbone"
            )
        self.task_head = PackedClassificationHead(task_model)
        _install_backbone_mapper(self, backbone, classification=True)
        self.ignore_unexpected_prefixes = [
            prefix
            for prefix in self.ignore_unexpected_prefixes
            if prefix not in {"classifier.", "score."}
        ]
        return backbone

    def _configure_task(self):
        if not self.pp_group.is_last_rank:
            self.task_head = None
            self.skip_prefixes.append("task_head.")
            return
        pooling_type = (
            PoolingType.LAST if self.task_head.kind == "decoder" else PoolingType.CLS
        )
        self.pooler = Pooler(pooling_type, normalize=False)

    def _pool_output(self, hidden_states, forward_batch):
        if getattr(forward_batch, "dimensions", None) is not None:
            raise ValueError(
                "Classification logits do not support embedding dimension truncation"
            )
        if self.task_head.kind == "decoder":
            return score_and_pool(
                self.task_head,
                self.pooler,
                hidden_states,
                forward_batch,
                forward_batch.input_ids,
            )
        if (
            getattr(forward_batch, "token_indices_to_pool", None) is not None
            or getattr(forward_batch, "multi_item_delimiter_indices", None) is not None
        ):
            raise ValueError(
                "Encoder classifiers do not support decoder token-position scoring"
            )
        pooled = pool_hidden_states(PoolingType.CLS, hidden_states, forward_batch)
        scores = self.task_head(pooled, getattr(self.model, "pooler", None))
        return EmbeddingPoolerOutput(
            embeddings=scores,
            pooled_hidden_states=pooled
            if getattr(forward_batch, "return_pooled_hidden_states", False)
            else None,
        )

    def _validate_task_weights(self, loaded):
        if self.task_head is None or getattr(self, "_weights_loaded", False):
            return
        required = {
            f"task_head.{name}" for name, _ in self.task_head.named_parameters()
        }
        if self.task_head.kind == "bert":
            required.update(
                f"model.pooler.{name}"
                for name, _ in self.model.pooler.named_parameters()
            )
        missing = required - loaded
        if missing:
            raise ValueError(
                f"Classification checkpoint is missing required tensors: {sorted(missing)}"
            )
