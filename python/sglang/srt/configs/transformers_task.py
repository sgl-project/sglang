# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from sglang.srt.configs.embedding_model_spec import (
    BCGPrefillPolicy,
    EmbeddingExecution,
    EmbeddingTask,
    PoolingStrategy,
)


@dataclass(frozen=True)
class DenseSpec:
    path: str
    in_features: int
    out_features: int
    bias: bool
    activation: str


@dataclass(frozen=True)
class EmbeddingPipeline:
    pooling: PoolingStrategy
    stages: tuple[DenseSpec | str, ...]
    output_dimension: int
    sentence_transformers: bool = False
    max_seq_length: int | None = None

    @property
    def normalize(self):
        return bool(self.stages and self.stages[-1] == "normalize")


_ACTIVATIONS = {
    "torch.nn.modules.linear.Identity": "identity",
    "torch.nn.Identity": "identity",
    "torch.nn.modules.activation.Tanh": "tanh",
    "torch.nn.Tanh": "tanh",
    "torch.nn.modules.activation.ReLU": "relu",
    "torch.nn.ReLU": "relu",
    "torch.nn.modules.activation.GELU": "gelu",
    "torch.nn.GELU": "gelu",
    "torch.nn.modules.activation.Sigmoid": "sigmoid",
    "torch.nn.Sigmoid": "sigmoid",
}


def resolve_metadata_file(model_path, filename, revision=None, *, optional=False):
    if not model_path:
        if optional:
            return None
        raise ValueError(f"A model path is required to load {filename}")
    root = Path(model_path)
    if root.is_dir():
        result = root / filename
        if result.is_file():
            return str(result)
        if optional:
            return None
        raise FileNotFoundError(result)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(model_path, filename, revision=revision)
    except EntryNotFoundError:
        if optional:
            return None
        raise


def _read_json(model_path, filename, revision):
    with open(resolve_metadata_file(model_path, filename, revision)) as stream:
        return json.load(stream)


def _module_path(value):
    if not isinstance(value, str):
        raise ValueError("Sentence Transformers module paths must be strings")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Invalid Sentence Transformers module path: {value!r}")
    return value


def resolve_embedding_pipeline(config, spec, model_path="", revision=None):
    hidden_size = getattr(config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("Embedding models must declare a positive hidden_size")
    configured_pooling = getattr(config, "pooling_type", None)
    if configured_pooling is not None:
        try:
            pooling = PoolingStrategy(str(configured_pooling).lower())
        except ValueError as exc:
            raise ValueError(
                f"Unsupported pooling_type: {configured_pooling!r}"
            ) from exc
        if pooling == PoolingStrategy.MODEL_DEFINED:
            raise ValueError("pooling_type must be CLS, LAST, or MEAN")
    else:
        pooling = spec.pooling
        if pooling == PoolingStrategy.MODEL_DEFINED:
            pooling = PoolingStrategy.LAST
    normalize = getattr(config, "normalize", spec.normalize)
    if not isinstance(normalize, bool):
        raise ValueError("Embedding normalize must be a boolean")
    modules_path = resolve_metadata_file(
        model_path, "modules.json", revision, optional=True
    )
    if modules_path is None:
        return EmbeddingPipeline(
            pooling, ("normalize",) if normalize else (), hidden_size
        )
    with open(modules_path) as stream:
        modules = json.load(stream)
    if not isinstance(modules, list) or not modules:
        raise ValueError("Sentence Transformers modules.json must be a nonempty list")
    stages = []
    pooled = False
    dimension = hidden_size
    indices = set()
    for position, module in enumerate(modules):
        if not isinstance(module, dict):
            raise ValueError("Each Sentence Transformers module must be an object")
        index = module.get("idx")
        if type(index) is not int or index in indices or index != position:
            raise ValueError("Sentence Transformers module indices must be consecutive")
        indices.add(index)
        path = _module_path(module.get("path"))
        kind = module.get("type")
        if kind == "sentence_transformers.models.Transformer":
            if position != 0 or path not in ("", "."):
                raise ValueError(
                    "The Sentence Transformers backbone must be at the model root"
                )
            continue
        if position == 0:
            raise ValueError(
                "The first Sentence Transformers module must be Transformer"
            )
        if kind == "sentence_transformers.models.Pooling":
            if pooled or stages:
                raise ValueError(
                    "Exactly one Pooling module must precede postprocessing"
                )
            data = _read_json(model_path, f"{path}/config.json", revision)
            modes = {
                "pooling_mode_cls_token": PoolingStrategy.CLS,
                "pooling_mode_mean_tokens": PoolingStrategy.MEAN,
                "pooling_mode_lasttoken": PoolingStrategy.LAST,
            }
            if any(
                not isinstance(value, bool)
                for key, value in data.items()
                if key.startswith("pooling_mode_")
            ):
                raise ValueError(
                    "Sentence Transformers pooling mode flags must be booleans"
                )
            active = [
                key
                for key, value in data.items()
                if key.startswith("pooling_mode_") and value
            ]
            if len(active) != 1 or active[0] not in modes:
                raise ValueError(
                    f"Unsupported Sentence Transformers pooling modes: {active}"
                )
            if data.get("word_embedding_dimension", hidden_size) != hidden_size:
                raise ValueError(
                    "Pooling dimension does not match the backbone hidden_size"
                )
            if data.get("include_prompt", True) is not True:
                raise ValueError(
                    "Sentence Transformers pooling with include_prompt=false is unsupported"
                )
            metadata_pooling = modes[active[0]]
            if configured_pooling is not None and pooling != metadata_pooling:
                raise ValueError(
                    "pooling_type conflicts with Sentence Transformers metadata"
                )
            pooling = metadata_pooling
            pooled = True
        elif kind == "sentence_transformers.models.Dense":
            if not pooled:
                raise ValueError("Dense modules must follow Pooling")
            data = _read_json(model_path, f"{path}/config.json", revision)
            in_features, out_features = (
                data.get("in_features"),
                data.get("out_features"),
            )
            if (
                in_features != dimension
                or not isinstance(out_features, int)
                or out_features <= 0
            ):
                raise ValueError(
                    "Sentence Transformers Dense dimensions do not compose"
                )
            activation_name = data.get(
                "activation_function", "torch.nn.modules.activation.Tanh"
            )
            if activation_name not in _ACTIVATIONS:
                raise ValueError(f"Unsupported Dense activation: {activation_name!r}")
            bias = data.get("bias", True)
            if not isinstance(bias, bool):
                raise ValueError("Sentence Transformers Dense bias must be a boolean")
            stages.append(
                DenseSpec(
                    path, in_features, out_features, bias, _ACTIVATIONS[activation_name]
                )
            )
            dimension = out_features
        elif kind == "sentence_transformers.models.Normalize":
            if not pooled:
                raise ValueError("Normalize modules must follow Pooling")
            stages.append("normalize")
        else:
            raise ValueError(f"Unsupported Sentence Transformers module: {kind!r}")
    if not pooled:
        raise ValueError("Sentence Transformers metadata is missing Pooling")
    metadata_normalize = bool(stages and stages[-1] == "normalize")
    if hasattr(config, "normalize") and normalize != metadata_normalize:
        raise ValueError("normalize conflicts with Sentence Transformers metadata")
    max_seq_length = None
    backbone_config = resolve_metadata_file(
        model_path, "sentence_bert_config.json", revision, optional=True
    )
    if backbone_config is not None:
        with open(backbone_config) as stream:
            max_seq_length = json.load(stream).get("max_seq_length")
        if max_seq_length is not None and (
            type(max_seq_length) is not int or max_seq_length <= 0
        ):
            raise ValueError("Sentence Transformers max_seq_length must be positive")
    return EmbeddingPipeline(pooling, tuple(stages), dimension, True, max_seq_length)


def apply_transformers_embedding_pipeline(spec, plan):
    full_sequence = spec.bidirectional_attention or plan.pooling != PoolingStrategy.LAST
    return replace(
        spec,
        task=EmbeddingTask.EMBED,
        pooling=plan.pooling,
        normalize=plan.normalize,
        postprocessor="sentence_transformers"
        if plan.sentence_transformers
        else "model_defined",
        safe_disable_radix_cache=full_sequence,
        safe_disable_chunked_prefill=full_sequence,
        safe_disable_kv_cache=False,
        bcg_prefill_policy=BCGPrefillPolicy.DEFAULT,
        execution=(
            EmbeddingExecution.ENCODER_ONLY
            if spec.bidirectional_attention
            else spec.execution
        ),
    )


def resolve_transformers_task(model_config):
    spec = model_config.embedding_model_spec
    if spec.task != EmbeddingTask.EMBED:
        return
    plan = getattr(model_config, "transformers_embedding_plan", None)
    if plan is None:
        plan = resolve_embedding_pipeline(
            model_config.hf_text_config,
            spec,
            model_config.model_path,
            model_config.revision,
        )
        model_config.transformers_embedding_plan = plan
    model_config.embedding_model_spec = apply_transformers_embedding_pipeline(
        spec, plan
    )
