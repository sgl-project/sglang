"""Gemma 4 assistant loading, lifecycle, and read-only target-KV sharing.

The MVP uses ``mlx-vlm==0.5.0`` as the narrow assistant provider.  All
provider details stay behind :class:`Gemma4MTPAssistantRuntime`; scheduler and
worker code see only generation-bound seeds, request-bound KV views, and a
single ``propose_one`` operation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten

from sglang.srt.hardware_backend.mlx.model_adapter import MlxTargetSeed
from sglang.srt.hardware_backend.mlx.spec_config import load_assistant_config_dict

logger = logging.getLogger(__name__)

_ASSISTANT_MODEL_TYPES = frozenset({"gemma4_assistant"})
_ASSISTANT_ARCHITECTURES = frozenset(
    {"Gemma4AssistantForCausalLM", "Gemma4UnifiedAssistantForCausalLM"}
)
_TARGET_MODEL_TYPES = frozenset({"gemma4", "gemma4_text"})
_LAYER_TYPES = frozenset({"sliding_attention", "full_attention"})


def _text_target(model: Any) -> tuple[Any, Any, Any]:
    causal = getattr(model, "language_model", model)
    backbone = getattr(causal, "model", None)
    args = getattr(causal, "args", None) or getattr(backbone, "config", None)
    if backbone is None or args is None:
        raise TypeError("expected an mlx-lm Gemma 4 text target")
    model_type = getattr(causal, "model_type", None) or getattr(
        args, "model_type", None
    )
    if model_type not in _TARGET_MODEL_TYPES:
        raise ValueError(
            f"Gemma 4 MTP target family must be one of {sorted(_TARGET_MODEL_TYPES)}; "
            f"got {model_type!r}"
        )
    return causal, backbone, args


def _reject_remote_hooks(config: Mapping[str, Any], prefix: str = "config") -> None:
    for key, value in config.items():
        path = f"{prefix}.{key}"
        if key in {"auto_map", "model_file"} and value not in (None, {}, [], ""):
            raise ValueError(
                f"Gemma 4 assistant custom/remote code hook {path!r} is not allowed"
            )
        if isinstance(value, Mapping):
            _reject_remote_hooks(value, path)


@dataclass(frozen=True)
class Gemma4MTPAssistantMetadata:
    model_type: str
    architecture: str
    target_family: str
    vocab_size: int
    backbone_hidden_size: int
    assistant_hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    tie_word_embeddings: bool
    use_ordered_embeddings: bool
    num_centroids: int
    centroid_intermediate_top_k: int
    sliding_window: int
    head_dim: int
    global_head_dim: int
    num_key_value_heads: int
    num_global_key_value_heads: int | None
    final_logit_softcapping: float | None

    def head_dim_for(self, layer_type: str) -> int:
        return self.global_head_dim if layer_type == "full_attention" else self.head_dim


def validate_gemma4_assistant_config(
    config: Mapping[str, Any], target_model: Any
) -> Gemma4MTPAssistantMetadata:
    """Validate provider-independent assistant/target compatibility."""

    _reject_remote_hooks(config)
    model_type = str(config.get("model_type", ""))
    if model_type not in _ASSISTANT_MODEL_TYPES:
        raise ValueError("assistant model_type must be 'gemma4_assistant'")

    architectures = tuple(config.get("architectures") or ())
    if len(architectures) != 1 or architectures[0] not in _ASSISTANT_ARCHITECTURES:
        raise ValueError(
            "assistant architecture must be Gemma4AssistantForCausalLM "
            "(or the explicit Gemma4UnifiedAssistantForCausalLM compatibility alias)"
        )
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("assistant config requires a text_config object")
    if text.get("model_type") != "gemma4_text":
        raise ValueError("assistant text_config.model_type must be 'gemma4_text'")

    causal, _backbone, target = _text_target(target_model)
    target_layers = tuple(getattr(target, "layer_types", ()) or ())
    layer_types = tuple(text.get("layer_types") or ())
    num_layers = int(text.get("num_hidden_layers", -1))
    if num_layers <= 0 or len(layer_types) != num_layers:
        raise ValueError("assistant layer_types must match num_hidden_layers")
    if any(layer_type not in _LAYER_TYPES for layer_type in layer_types):
        raise ValueError("assistant contains an unsupported attention layer type")
    if (
        len(target_layers) < num_layers
        or tuple(target_layers[-num_layers:]) != layer_types
    ):
        raise ValueError(
            "assistant sliding/full layer layout must match the target's final tail"
        )

    vocab_size = int(text.get("vocab_size", -1))
    target_vocab = int(getattr(target, "vocab_size", -1))
    if vocab_size <= 0 or vocab_size != target_vocab:
        raise ValueError(
            f"assistant vocab_size {vocab_size} does not match target {target_vocab}"
        )
    backbone_hidden = int(config.get("backbone_hidden_size", -1))
    target_hidden = int(getattr(target, "hidden_size", -1))
    if backbone_hidden <= 0 or backbone_hidden != target_hidden:
        raise ValueError(
            "assistant backbone_hidden_size must match the target hidden_size"
        )
    assistant_hidden = int(text.get("hidden_size", -1))
    if assistant_hidden <= 0:
        raise ValueError("assistant hidden_size must be positive")

    tie = bool(config.get("tie_word_embeddings", False))
    if tie != bool(text.get("tie_word_embeddings", tie)):
        raise ValueError(
            "top-level and text assistant tied-embedding metadata disagree"
        )
    if tie != bool(getattr(causal, "tie_word_embeddings", True)):
        raise ValueError("target and assistant tied-embedding settings must match")
    if int(text.get("num_kv_shared_layers", 0)) != num_layers:
        raise ValueError("every assistant layer must be query-only/KV-shared")

    head_dim = int(text.get("head_dim", -1))
    global_head_dim = int(text.get("global_head_dim", head_dim))
    if head_dim != int(getattr(target, "head_dim", -1)):
        raise ValueError("assistant sliding-attention head_dim does not match target")
    if global_head_dim != int(getattr(target, "global_head_dim", head_dim)):
        raise ValueError("assistant full-attention head_dim does not match target")
    sliding_window = int(text.get("sliding_window", -1))
    if sliding_window != int(getattr(target, "sliding_window", -1)):
        raise ValueError("assistant sliding_window does not match target")

    ordered = bool(config.get("use_ordered_embeddings", False))
    centroids = int(config.get("num_centroids", 0))
    centroid_topk = int(config.get("centroid_intermediate_top_k", 0))
    if ordered:
        if centroids <= 0 or vocab_size % centroids:
            raise ValueError(
                "ordered assistant vocabulary must be divisible by num_centroids"
            )
        if centroid_topk <= 0 or centroid_topk > centroids:
            raise ValueError(
                "centroid_intermediate_top_k must be in [1, num_centroids]"
            )
    elif centroids < 0 or centroid_topk < 0:
        raise ValueError("centroid metadata cannot be negative")

    return Gemma4MTPAssistantMetadata(
        model_type=model_type,
        architecture=architectures[0],
        target_family=str(getattr(causal, "model_type", "gemma4_text")),
        vocab_size=vocab_size,
        backbone_hidden_size=backbone_hidden,
        assistant_hidden_size=assistant_hidden,
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        tie_word_embeddings=tie,
        use_ordered_embeddings=ordered,
        num_centroids=centroids,
        centroid_intermediate_top_k=centroid_topk,
        sliding_window=sliding_window,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        num_key_value_heads=int(text.get("num_key_value_heads", 1)),
        num_global_key_value_heads=(
            None
            if text.get("num_global_key_value_heads") is None
            else int(text["num_global_key_value_heads"])
        ),
        final_logit_softcapping=(
            None
            if text.get("final_logit_softcapping") is None
            else float(text["final_logit_softcapping"])
        ),
    )


@dataclass(frozen=True)
class Gemma4MTPKVSharingPlan:
    """Assistant layer -> target logical owner -> compact cache index."""

    assistant_layer_types: tuple[str, ...]
    target_logical_layers: tuple[int, ...]
    target_owner_layers: tuple[int, ...]
    compact_cache_indices: tuple[int, ...]
    compact_owner_layers: tuple[int, ...]

    @classmethod
    def from_target(
        cls, target_model: Any, metadata: Gemma4MTPAssistantMetadata
    ) -> Gemma4MTPKVSharingPlan:
        _causal, backbone, target = _text_target(target_model)
        layers = tuple(backbone.layers)
        target_types = tuple(getattr(target, "layer_types", ()) or ())
        if len(layers) != len(target_types):
            raise ValueError("target layer metadata cardinality is inconsistent")
        if tuple(target_types[-metadata.num_hidden_layers :]) != metadata.layer_types:
            raise ValueError("assistant layout is not the target's final layer tail")

        previous = tuple(getattr(backbone, "previous_kvs", ()))
        if len(previous) != len(layers):
            raise ValueError("target does not expose complete YOCO owner metadata")
        compact_owners = tuple(
            index
            for index, layer in enumerate(layers)
            if bool(getattr(layer.self_attn, "has_kv", True))
        )
        owner_to_compact = {owner: index for index, owner in enumerate(compact_owners)}

        logical_layers: list[int] = []
        owner_layers: list[int] = []
        compact_indices: list[int] = []
        for layer_type in metadata.layer_types:
            candidates = [
                index for index, value in enumerate(target_types) if value == layer_type
            ]
            if not candidates:
                raise ValueError(f"target has no {layer_type!r} layer for assistant")
            logical = candidates[-1]
            owner = int(previous[logical])
            if owner < 0 or owner >= len(layers):
                raise ValueError("target YOCO owner is out of range")
            if int(previous[owner]) != owner:
                raise ValueError("nested/shared YOCO owners are not supported")
            if target_types[owner] != layer_type:
                raise ValueError("target YOCO owner changes the attention layer type")
            if owner not in owner_to_compact:
                raise ValueError("target YOCO owner has no compact native cache entry")
            logical_layers.append(logical)
            owner_layers.append(owner)
            compact_indices.append(owner_to_compact[owner])

        return cls(
            assistant_layer_types=metadata.layer_types,
            target_logical_layers=tuple(logical_layers),
            target_owner_layers=tuple(owner_layers),
            compact_cache_indices=tuple(compact_indices),
            compact_owner_layers=compact_owners,
        )

    @property
    def expected_cache_entries(self) -> int:
        return len(self.compact_owner_layers)


class _RuntimeToken:
    def __init__(self, runtime: Gemma4MTPAssistantRuntime, request_id: str):
        self.runtime = runtime
        self.generation = runtime.generation
        self.request_id = request_id
        self.active = True

    def validate(self) -> None:
        self.runtime._validate_generation(self.generation)
        if not self.active:
            raise RuntimeError(
                f"stale Gemma 4 MTP request binding for {self.request_id!r}"
            )

    def invalidate(self) -> None:
        self.active = False


class Gemma4MTPKVView:
    """Generation- and request-bound read-only view over compact native caches."""

    def __init__(
        self,
        token: _RuntimeToken,
        cache: Sequence[Any],
        plan: Gemma4MTPKVSharingPlan,
        metadata: Gemma4MTPAssistantMetadata,
    ):
        if len(cache) != plan.expected_cache_entries:
            raise ValueError(
                "native target cache cardinality does not match the KV sharing plan: "
                f"{len(cache)} != {plan.expected_cache_entries}"
            )
        self._token = token
        self._cache = tuple(cache)
        self._plan = plan
        self._metadata = metadata

    @property
    def request_id(self) -> str:
        return self._token.request_id

    @property
    def position(self) -> int:
        self._token.validate()
        offsets = {int(entry.offset) for entry in self._cache}
        if len(offsets) != 1:
            raise RuntimeError(
                f"native target cache offsets disagree: {sorted(offsets)}"
            )
        return offsets.pop()

    @staticmethod
    def _logical_kv(entry: Any) -> tuple[mx.array, mx.array]:
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is None or values is None:
            raise RuntimeError("assistant cannot bind an empty target KV cache")
        offset = int(entry.offset)
        if hasattr(entry, "_temporal_order"):
            keys = entry._temporal_order(keys)
            values = entry._temporal_order(values)
            valid = min(offset, int(entry.max_size))
            keys = keys[..., -valid:, :] if valid else keys[..., :0, :]
            values = values[..., -valid:, :] if valid else values[..., :0, :]
            # Provider masks index K/V positions from zero while RoPE uses the
            # absolute query offset.  Restore absolute positions with a zero
            # prefix; its mask hides the prefix and exposes only the window.
            padding = offset - valid
            if padding > 0:
                k_pad = mx.zeros(
                    (*keys.shape[:-2], padding, keys.shape[-1]), keys.dtype
                )
                v_pad = mx.zeros(
                    (*values.shape[:-2], padding, values.shape[-1]), values.dtype
                )
                keys = mx.concatenate((k_pad, keys), axis=-2)
                values = mx.concatenate((v_pad, values), axis=-2)
            return keys, values
        return keys[..., :offset, :], values[..., :offset, :]

    def shared_kv_states(self) -> dict[str, tuple[mx.array, mx.array]]:
        self._token.validate()
        shared: dict[str, tuple[mx.array, mx.array]] = {}
        for layer_type, cache_index in zip(
            self._plan.assistant_layer_types, self._plan.compact_cache_indices
        ):
            if cache_index < 0 or cache_index >= len(self._cache):
                raise RuntimeError(
                    "KV sharing plan contains an out-of-range cache index"
                )
            if layer_type in shared:
                continue
            keys, values = self._logical_kv(self._cache[cache_index])
            expected_dim = self._metadata.head_dim_for(layer_type)
            if keys.shape[-1] != expected_dim or values.shape[-1] != expected_dim:
                raise ValueError(
                    f"{layer_type} target KV head width does not match assistant: "
                    f"K={keys.shape[-1]}, V={values.shape[-1]}, expected={expected_dim}"
                )
            shared[layer_type] = (keys, values)
        return shared


class Gemma4MTPProposerSeed:
    def __init__(self, token: _RuntimeToken, target_seed: MlxTargetSeed):
        self._token = token
        self.target_seed = target_seed

    @property
    def request_id(self) -> str:
        return self._token.request_id

    def validate(self) -> None:
        self._token.validate()


class Gemma4MTPModelHandle:
    """Guarded model handle; retained handles cannot use unloaded weights."""

    def __init__(self, runtime: Gemma4MTPAssistantRuntime):
        self._runtime = runtime

    @property
    def fingerprint(self) -> str:
        self._runtime._validate_generation(self._runtime.generation)
        return self._runtime.fingerprint


class Gemma4MTPAssistantRuntime:
    """Provider-independent, generation-checked one-token assistant runtime."""

    def __init__(
        self,
        *,
        owner: Gemma4MTPAssistantLoader,
        generation: int,
        model: Any,
        metadata: Gemma4MTPAssistantMetadata,
        sharing_plan: Gemma4MTPKVSharingPlan,
        fingerprint: str,
        checkpoint: str,
        revision: str | None,
    ):
        self._owner = owner
        self.generation = generation
        self._model = model
        self.metadata = metadata
        self.sharing_plan = sharing_plan
        self.fingerprint = fingerprint
        self.checkpoint = checkpoint
        self.revision = revision
        self._active = True
        self._request_tokens: dict[str, list[_RuntimeToken]] = {}
        self.model_handle = Gemma4MTPModelHandle(self)

    def _validate_generation(self, generation: int) -> None:
        if (
            not self._active
            or generation != self.generation
            or self._owner.generation != self.generation
            or self._owner.runtime is not self
        ):
            raise RuntimeError(
                f"stale Gemma 4 MTP runtime generation {generation}; "
                f"current generation is {self._owner.generation}"
            )

    def _new_token(self, request_id: str) -> _RuntimeToken:
        self._validate_generation(self.generation)
        token = _RuntimeToken(self, request_id)
        self._request_tokens.setdefault(request_id, []).append(token)
        return token

    def bind_request(self, request_id: str, cache: Sequence[Any]) -> Gemma4MTPKVView:
        return Gemma4MTPKVView(
            self._new_token(request_id), cache, self.sharing_plan, self.metadata
        )

    def bind_seed(
        self, request_id: str, target_seed: MlxTargetSeed
    ) -> Gemma4MTPProposerSeed:
        return Gemma4MTPProposerSeed(self._new_token(request_id), target_seed)

    def release_request(self, request_id: str) -> None:
        for token in self._request_tokens.pop(request_id, ()):
            token.invalidate()

    def clear_request_bindings(self) -> None:
        for request_id in tuple(self._request_tokens):
            self.release_request(request_id)

    @property
    def request_binding_count(self) -> int:
        return sum(
            token.active for values in self._request_tokens.values() for token in values
        )

    def invalidate(self) -> None:
        self.clear_request_bindings()
        self._active = False

    def propose_one(self, seed: Gemma4MTPProposerSeed, kv_view: Gemma4MTPKVView) -> int:
        self._validate_generation(self.generation)
        seed.validate()
        # Accessing shared K/V validates the view and its request lifecycle.
        if seed.request_id != kv_view.request_id:
            raise ValueError("assistant seed and KV view belong to different requests")
        target_seed = seed.target_seed
        hidden = target_seed.hidden_state
        embedding = target_seed.token_embedding
        expected = (1, 1, self.metadata.backbone_hidden_size)
        if tuple(hidden.shape) != expected or tuple(embedding.shape) != expected:
            raise ValueError(
                "assistant seed hidden/embedding shapes must both be "
                f"{expected}; got {hidden.shape} and {embedding.shape}"
            )
        shared = kv_view.shared_kv_states()
        inputs = mx.concatenate((embedding, hidden), axis=-1)
        positions = mx.array([[kv_view.position]], dtype=mx.int32)
        _projected, logits = self._model(inputs, shared, positions)
        if self.metadata.final_logit_softcapping is not None:
            cap = self.metadata.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        return int(token.item())


def _resolve_checkpoint(path_or_repo: str, revision: str | None) -> Path:
    local = Path(path_or_repo).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.is_file():
        return local.parent.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=path_or_repo,
            revision=revision,
            allow_patterns=("config.json", "*.safetensors", "*.safetensors.index.json"),
        )
    )


def _load_strict_provider_model(
    checkpoint: Path, config: Mapping[str, Any], target_model: Any
) -> tuple[Any, str]:
    try:
        from mlx_vlm.speculative.drafters.gemma4_assistant.config import (
            Gemma4AssistantConfig,
        )
        from mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant import (
            Gemma4AssistantDraftModel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Gemma 4 MTP requires the MLX optional dependency mlx-vlm==0.5.0"
        ) from exc

    provider_config = Gemma4AssistantConfig.from_dict(dict(config))
    model = Gemma4AssistantDraftModel(provider_config)
    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(
            f"no assistant .safetensors files found in {checkpoint}"
        )
    weights: dict[str, mx.array] = {}
    duplicates: list[str] = []
    for weight_file in weight_files:
        for name, value in mx.load(str(weight_file)).items():
            if name in weights:
                duplicates.append(name)
            weights[name] = value
    if duplicates:
        raise ValueError(
            "duplicate assistant weights: " + ", ".join(sorted(set(duplicates)))
        )
    weights = model.sanitize(weights)
    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    wrong_shape = sorted(
        name
        for name in set(expected) & set(weights)
        if tuple(expected[name].shape) != tuple(weights[name].shape)
    )
    if missing or unexpected or wrong_shape:
        parts = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if unexpected:
            parts.append("unexpected=" + ",".join(unexpected))
        if wrong_shape:
            details = ",".join(
                f"{name}:{tuple(weights[name].shape)}!={tuple(expected[name].shape)}"
                for name in wrong_shape
            )
            parts.append("wrong_shape=" + details)
        raise ValueError("assistant weight validation failed: " + "; ".join(parts))

    model.load_weights(list(weights.items()), strict=True)
    model.bind(target_model)
    model.eval()
    mx.eval(model.parameters())

    digest = hashlib.sha256()
    for name in sorted(weights):
        value = weights[name]
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        if value.size:
            sample = value.reshape(-1)[mx.array([0, value.size - 1])]
            mx.eval(sample)
            digest.update(repr(sample.tolist()).encode())
    return model, digest.hexdigest()


class Gemma4MTPAssistantLoader:
    """Strict assistant loader with monotonic lifecycle invalidation."""

    def __init__(self, target_model: Any):
        _text_target(target_model)
        self.target_model = target_model
        self.generation = 0
        self.runtime: Gemma4MTPAssistantRuntime | None = None
        self.load_count = 0
        self._provider_cache: dict[Any, Any] = {}
        self._model_cache: dict[Any, Any] = {}

    def _invalidate_current(self) -> None:
        if self.runtime is not None:
            self.runtime.invalidate()
        self.runtime = None
        self._provider_cache.clear()
        self._model_cache.clear()
        self.generation += 1

    def load(
        self, path_or_repo: str, *, revision: str | None = None
    ) -> Gemma4MTPAssistantRuntime:
        # Compatibility is checked from JSON before the old generation is
        # invalidated or any new weights are materialized.
        config = load_assistant_config_dict(path_or_repo, revision=revision)
        metadata = validate_gemma4_assistant_config(config, self.target_model)
        plan = Gemma4MTPKVSharingPlan.from_target(self.target_model, metadata)

        self._invalidate_current()
        checkpoint = _resolve_checkpoint(path_or_repo, revision)
        try:
            model, fingerprint = _load_strict_provider_model(
                checkpoint, config, self.target_model
            )
        except BaseException:
            # The previous generation is already invalid by design; leave no
            # partially constructed provider/model cache behind.
            self._provider_cache.clear()
            self._model_cache.clear()
            raise

        runtime = Gemma4MTPAssistantRuntime(
            owner=self,
            generation=self.generation,
            model=model,
            metadata=metadata,
            sharing_plan=plan,
            fingerprint=fingerprint,
            checkpoint=path_or_repo,
            revision=revision,
        )
        self.runtime = runtime
        self._model_cache[(path_or_repo, revision, self.generation)] = model
        self.load_count += 1
        return runtime

    def reload_assistant(self) -> Gemma4MTPAssistantRuntime:
        if self.runtime is None:
            raise RuntimeError("cannot reload an assistant before it is loaded")
        return self.load(self.runtime.checkpoint, revision=self.runtime.revision)

    def replace_assistant(
        self, path_or_repo: str, *, revision: str | None = None
    ) -> Gemma4MTPAssistantRuntime:
        return self.load(path_or_repo, revision=revision)

    def unload_assistant(self) -> None:
        self._invalidate_current()

    def clear_request_bindings(self) -> None:
        if self.runtime is not None:
            self.runtime.clear_request_bindings()
