# Copyright 2023-2026 SGLang Team
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
"""Steering vector registry, batch planner, and decoder-layer forward hooks."""

from __future__ import annotations

import contextlib
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.steering.spec import SteeringVectorSpec

_LAYER_KEY_PATTERNS = (
    re.compile(r"^(\d+)$"),
    re.compile(r"^(?:model\.)?(?:layers|layer|h|blocks)\.(\d+)$"),
    re.compile(r"^(?:model\.)?decoder\.layers\.(\d+)$"),
)

_DEFAULT_LAYER_ATTR_PATHS = (
    "model.layers",
    "layers",
    "transformer.h",
    "model.decoder.layers",
    "decoder.layers",
)


def _parse_layer_key(key: Union[int, str]) -> int:
    """Parse a layer index from an int or standard checkpoint tensor key."""
    if isinstance(key, bool):
        raise ValueError(f"Invalid steering vector layer key: {key!r}")
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        for pattern in _LAYER_KEY_PATTERNS:
            match = pattern.match(key.strip())
            if match is not None:
                return int(match.group(1))
    raise ValueError(
        f"Unrecognized steering vector layer key {key!r}. Expected an int or a "
        "string like '12', 'layer.12', 'layers.12', or 'model.layers.12'."
    )


def resolve_decoder_layers(
    model: nn.Module,
    layers_attr: Optional[str] = None,
) -> Sequence[nn.Module]:
    """Locate the sequence of decoder layers on a model."""
    candidate_paths = (
        (layers_attr,) if layers_attr is not None else _DEFAULT_LAYER_ATTR_PATHS
    )
    for attr_path in candidate_paths:
        obj: Any = model
        found = True
        for part in attr_path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and isinstance(obj, (nn.ModuleList, list, tuple)):
            return obj
    raise ValueError(
        f"Could not locate decoder layers on {type(model).__name__} "
        f"(checked {list(candidate_paths)})."
    )


def _is_pp_missing_layer(module: nn.Module) -> bool:
    return type(module).__name__ == "PPMissingLayer"


def _reshape_delta_for_target(
    delta: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Align a 2D ``[num_tokens, hidden]`` delta with a 2D or 3D activation tensor."""
    if target.shape[-1] != delta.shape[-1]:
        raise ValueError(
            f"Steering vector hidden size ({delta.shape[-1]}) does not match "
            f"layer output hidden size ({target.shape[-1]})."
        )
    if target.ndim == 2:
        if delta.shape[0] not in (1, target.shape[0]):
            raise ValueError(
                f"Steering delta token count ({delta.shape[0]}) does not match "
                f"layer output token count ({target.shape[0]})."
            )
        return delta
    if target.ndim == 3:
        bs, seq_len, hidden = target.shape
        if delta.shape[0] == 1:
            return delta.unsqueeze(1)
        if delta.shape[0] == bs:
            return delta.unsqueeze(1)
        if delta.shape[0] == bs * seq_len:
            return delta.view(bs, seq_len, hidden)
        raise ValueError(
            f"Steering delta token count ({delta.shape[0]}) cannot broadcast "
            f"to layer output shape {tuple(target.shape)}."
        )
    raise ValueError(
        f"Unsupported layer output rank {target.ndim} for steering application "
        f"(expected 2D or 3D tensor)."
    )


def _prepare_delta(delta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    aligned = _reshape_delta_for_target(delta, target)
    if aligned.device != target.device or aligned.dtype != target.dtype:
        aligned = aligned.to(device=target.device, dtype=target.dtype)
    return aligned


@dataclass(frozen=True)
class SteeringVector:
    """A registered multi-layer additive steering vector."""

    name: str
    vectors: Dict[int, torch.Tensor]
    hidden_size: int
    default_scale: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def layer_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.vectors.keys()))


@dataclass
class SteeringBatchPlan:
    """Precomputed per-layer additive deltas for one forward pass."""

    layer_deltas: Dict[int, torch.Tensor]
    num_tokens: int

    @property
    def is_empty(self) -> bool:
        return not self.layer_deltas

    def apply_to_layer_output(
        self,
        layer_idx: int,
        output: Any,
        forward_batch: Any = None,
    ) -> Any:
        """Apply this plan's additive steering delta for ``layer_idx`` to a layer output.

        Supports:
        - SGLang ``ResidualStream`` models (e.g. Llama, Qwen2, Mixtral) while
          preserving object identity required by ``ResidualStream.check()``.
        - Classic SGLang models returning ``(hidden_states, residual)`` tuples.
        - Plain ``torch.Tensor`` decoder layer outputs.
        """
        delta = self.layer_deltas.get(layer_idx)
        if delta is None:
            return output

        stream = (
            getattr(forward_batch, "residual_stream", None)
            if forward_batch is not None
            else None
        )
        if stream is not None:
            return self._apply_with_residual_stream(delta, output, stream)

        if isinstance(output, torch.Tensor):
            return output + _prepare_delta(delta, output)

        if (
            isinstance(output, tuple)
            and len(output) >= 1
            and isinstance(output[0], torch.Tensor)
        ):
            steered = output[0] + _prepare_delta(delta, output[0])
            return (steered, *output[1:])

        raise TypeError(
            f"Unsupported decoder layer output type {type(output).__name__} "
            f"for steering at layer {layer_idx}."
        )

    @staticmethod
    def _apply_with_residual_stream(
        delta: torch.Tensor,
        output: Any,
        stream: Any,
    ) -> Any:
        """Apply delta in-place to preserve ResidualStream identity invariants."""
        pending = stream.pending
        if pending is None:
            if isinstance(output, torch.Tensor):
                output.add_(_prepare_delta(delta, output))
                return output
            raise TypeError(
                f"Expected tensor output for written residual stream, "
                f"got {type(output).__name__}."
            )

        if pending.owed is None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Expected tensor output for completed residual contribution, "
                    f"got {type(output).__name__}."
                )
            output.add_(_prepare_delta(delta, output))
            return output

        # OwedOutput (pending TP/MoE reduction or deferred finalize): add once
        # to the replicated residual buffer so TP all-reduce does not multiply
        # the steering delta by tp_size.
        if stream.residual is not None and isinstance(stream.residual, torch.Tensor):
            stream.residual.add_(_prepare_delta(delta, stream.residual))
            return output

        completed = pending.complete()
        if not isinstance(completed, torch.Tensor):
            raise TypeError(
                f"Completed owed output is not a tensor: {type(completed).__name__}."
            )
        completed.add_(_prepare_delta(delta, completed))
        return output


class SteeringVectorManager:
    """Manages registered steering vectors and per-forward decoder-layer hooks.

    When a forward batch carries no active steering specs (``steering_vectors``
    is ``None`` or all specs are inactive for the batch's forward phase),
    :meth:`forward_context` is a no-op and installs zero hooks on the model.
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        layers_attr: Optional[str] = None,
    ) -> None:
        if isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size!r}.")
        if isinstance(num_hidden_layers, bool) or num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive, got {num_hidden_layers!r}."
            )
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(
                f"dtype must be a floating-point torch.dtype, got {dtype}."
            )

        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.device = torch.device(device)
        self.dtype = dtype
        self.layers_attr = layers_attr
        self._vectors: Dict[str, SteeringVector] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._vectors

    def list_vectors(self) -> List[str]:
        return sorted(self._vectors.keys())

    def get(self, name: str) -> SteeringVector:
        if name not in self._vectors:
            raise KeyError(f"Steering vector {name!r} is not registered.")
        return self._vectors[name]

    def register(
        self,
        name: str,
        vectors: Mapping[Union[int, str], torch.Tensor],
        *,
        default_scale: float = 1.0,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SteeringVector:
        """Validate and register a per-layer steering vector set."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Steering vector name must be a non-empty string.")
        normalized_name = name.strip()
        if not overwrite and normalized_name in self._vectors:
            raise ValueError(
                f"Steering vector {normalized_name!r} is already registered. "
                "Pass overwrite=True to replace it."
            )
        if isinstance(default_scale, bool) or not isinstance(
            default_scale, (int, float)
        ):
            raise ValueError(
                f"default_scale must be a finite number, got {default_scale!r}."
            )
        default_scale_f = float(default_scale)
        if not math.isfinite(default_scale_f):
            raise ValueError(f"default_scale must be finite, got {default_scale_f!r}.")
        if not isinstance(vectors, Mapping) or len(vectors) == 0:
            raise ValueError(
                "vectors must be a non-empty mapping of layer index to 1D tensor."
            )

        validated: Dict[int, torch.Tensor] = {}
        for raw_key, tensor in vectors.items():
            layer_idx = _parse_layer_key(raw_key)
            if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
                raise ValueError(
                    f"Layer index {layer_idx} out of range [0, {self.num_hidden_layers})."
                )
            if layer_idx in validated:
                raise ValueError(f"Duplicate layer index {layer_idx} in vectors.")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Layer {layer_idx} steering vector must be a torch.Tensor, "
                    f"got {type(tensor).__name__}."
                )
            if tensor.ndim != 1 or tensor.shape[0] != self.hidden_size:
                raise ValueError(
                    f"Layer {layer_idx} steering vector must have shape "
                    f"({self.hidden_size},), got {tuple(tensor.shape)}."
                )
            if not tensor.is_floating_point():
                raise ValueError(
                    f"Layer {layer_idx} steering vector must have a floating-point "
                    f"dtype, got {tensor.dtype}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Layer {layer_idx} steering vector contains NaN or Inf values."
                )

            target_dtype = self.dtype if self.dtype is not None else tensor.dtype
            validated[layer_idx] = (
                tensor.detach()
                .to(device=self.device, dtype=target_dtype)
                .contiguous()
                .clone()
            )

        entry = SteeringVector(
            name=normalized_name,
            vectors=validated,
            hidden_size=self.hidden_size,
            default_scale=default_scale_f,
            metadata=dict(metadata) if metadata is not None else {},
        )
        self._vectors[normalized_name] = entry
        return entry

    def load_from_file(
        self,
        name: str,
        path: Union[str, os.PathLike],
        *,
        default_scale: float = 1.0,
        overwrite: bool = False,
    ) -> SteeringVector:
        """Load a steering vector from a ``.safetensors`` or ``.pt``/``.pth`` file.

        Supported layouts:
        - Mapping of layer keys (``0``, ``"12"``, ``"layer.12"``, ``"layers.12"``,
          ``"model.layers.12"``) to 1D tensors of shape ``(hidden_size,)``.
        - Single 2D tensor of shape ``(num_hidden_layers, hidden_size)`` (stored
          directly in ``.pt`` or under key ``"vectors"`` in ``.safetensors``/``.pt``),
          registering all non-zero rows (or all rows if none are non-zero).
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Steering vector file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".safetensors":
            from safetensors.torch import load_file

            raw: Any = load_file(str(file_path), device="cpu")
        elif suffix in (".pt", ".pth"):
            raw = torch.load(file_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(
                f"Unsupported steering vector file format {suffix!r}. "
                "Supported formats: '.safetensors', '.pt', '.pth'."
            )

        vectors = self._extract_vectors_from_loaded_object(raw)
        return self.register(
            name,
            vectors,
            default_scale=default_scale,
            overwrite=overwrite,
            metadata={"source_path": str(file_path)},
        )

    def _extract_vectors_from_loaded_object(
        self, raw: Any
    ) -> Dict[Union[int, str], torch.Tensor]:
        if isinstance(raw, torch.Tensor):
            return self._unpack_2d_tensor(raw)
        if isinstance(raw, Mapping):
            if (
                len(raw) == 1
                and "vectors" in raw
                and isinstance(raw["vectors"], torch.Tensor)
                and raw["vectors"].ndim == 2
            ):
                return self._unpack_2d_tensor(raw["vectors"])
            return dict(raw)
        raise ValueError(
            f"Loaded steering vector file must contain a mapping or 2D tensor, "
            f"got {type(raw).__name__}."
        )

    def _unpack_2d_tensor(self, tensor: torch.Tensor) -> Dict[int, torch.Tensor]:
        if tensor.ndim != 2 or tensor.shape != (
            self.num_hidden_layers,
            self.hidden_size,
        ):
            raise ValueError(
                f"2D steering tensor must have shape "
                f"({self.num_hidden_layers}, {self.hidden_size}), "
                f"got {tuple(tensor.shape)}."
            )
        vectors: Dict[int, torch.Tensor] = {}
        for layer_idx in range(self.num_hidden_layers):
            row = tensor[layer_idx]
            if torch.any(row != 0):
                vectors[layer_idx] = row
        if not vectors:
            vectors = {i: tensor[i] for i in range(self.num_hidden_layers)}
        return vectors

    def unregister(self, name: str) -> None:
        if name not in self._vectors:
            raise KeyError(f"Steering vector {name!r} is not registered.")
        del self._vectors[name]

    def clear(self) -> None:
        self._vectors.clear()

    def validate_spec(self, spec: SteeringVectorSpec) -> SteeringVector:
        """Verify that ``spec`` references a registered vector and valid layers."""
        vec = self.get(spec.name)
        if spec.layers is not None:
            for layer_idx in spec.layers:
                if layer_idx < 0 or layer_idx >= self.num_hidden_layers:
                    raise ValueError(
                        f"Steering spec layer {layer_idx} out of range "
                        f"[0, {self.num_hidden_layers})."
                    )
        return vec

    def prepare_batch(self, forward_batch: Any) -> Optional[SteeringBatchPlan]:
        """Build per-layer additive deltas for ``forward_batch``, or ``None`` if inactive."""
        if forward_batch is None:
            return None
        specs: Optional[Sequence[Optional[SteeringVectorSpec]]] = getattr(
            forward_batch, "steering_vectors", None
        )
        if not specs:
            return None

        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and forward_mode.is_idle():
            return None

        num_reqs = len(specs)
        req_token_lens = self._resolve_req_token_lens(forward_batch, num_reqs)
        active_num_tokens = sum(req_token_lens)
        if active_num_tokens == 0:
            return None

        total_num_tokens = self._resolve_total_num_tokens(
            forward_batch, active_num_tokens
        )
        req_is_decode = self._resolve_req_is_decode(forward_batch, num_reqs)

        # Collect active spans per layer: layer_idx -> list of (start, end, vec, scale)
        layer_entries: Dict[int, List[Tuple[int, int, SteeringVector, float]]] = {}
        token_offset = 0
        for req_idx, spec in enumerate(specs):
            tok_len = req_token_lens[req_idx]
            start = token_offset
            end = start + tok_len
            token_offset = end

            if spec is None or tok_len <= 0:
                continue

            vec = self.validate_spec(spec)
            if not spec.applies_to_phase(is_decode=req_is_decode[req_idx]):
                continue

            effective_scale = spec.scale * vec.default_scale
            if effective_scale == 0.0:
                continue

            target_layers = (
                [idx for idx in spec.layers if idx in vec.vectors]
                if spec.layers is not None
                else list(vec.vectors.keys())
            )
            for layer_idx in target_layers:
                layer_entries.setdefault(layer_idx, []).append(
                    (start, end, vec, effective_scale)
                )

        if not layer_entries:
            return None

        layer_deltas: Dict[int, torch.Tensor] = {}
        for layer_idx, entries in layer_entries.items():
            # Fast path: every token in the batch gets the exact same vector & scale.
            first_vec = entries[0][2]
            first_scale = entries[0][3]
            covered_tokens = sum(end - start for start, end, _, _ in entries)
            is_uniform = covered_tokens == total_num_tokens and all(
                vec is first_vec and scale == first_scale
                for _, _, vec, scale in entries
            )
            if is_uniform:
                layer_deltas[layer_idx] = (
                    first_vec.vectors[layer_idx] * first_scale
                ).unsqueeze(0)
            else:
                ref_vec = first_vec.vectors[layer_idx]
                delta = torch.zeros(
                    (total_num_tokens, self.hidden_size),
                    device=ref_vec.device,
                    dtype=ref_vec.dtype,
                )
                for start, end, vec, scale in entries:
                    src = vec.vectors[layer_idx]
                    if src.dtype != delta.dtype or src.device != delta.device:
                        src = src.to(device=delta.device, dtype=delta.dtype)
                    delta[start:end].add_(src, alpha=scale)
                layer_deltas[layer_idx] = delta

        return SteeringBatchPlan(layer_deltas=layer_deltas, num_tokens=total_num_tokens)

    @staticmethod
    def _resolve_req_token_lens(forward_batch: Any, num_reqs: int) -> List[int]:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and forward_mode.is_decode():
            return [1] * num_reqs

        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            return [int(x) for x in extend_seq_lens_cpu[:num_reqs]]

        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is not None:
            return [int(x) for x in extend_seq_lens[:num_reqs].tolist()]

        return [1] * num_reqs

    @staticmethod
    def _resolve_total_num_tokens(forward_batch: Any, active_num_tokens: int) -> int:
        input_ids = getattr(forward_batch, "input_ids", None)
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 1:
            return max(int(input_ids.shape[0]), active_num_tokens)
        input_embeds = getattr(forward_batch, "input_embeds", None)
        if isinstance(input_embeds, torch.Tensor) and input_embeds.ndim >= 1:
            return max(int(input_embeds.shape[0]), active_num_tokens)
        return active_num_tokens

    @staticmethod
    def _resolve_req_is_decode(forward_batch: Any, num_reqs: int) -> List[bool]:
        steering_is_decode = getattr(forward_batch, "steering_is_decode", None)
        if steering_is_decode is not None:
            return [bool(x) for x in steering_is_decode[:num_reqs]]
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and forward_mode.is_decode():
            return [True] * num_reqs
        return [False] * num_reqs

    @contextlib.contextmanager
    def forward_context(
        self,
        model: nn.Module,
        forward_batch: Any,
    ) -> Iterator[Optional[SteeringBatchPlan]]:
        """Context manager that installs decoder-layer steering hooks for one forward pass.

        Zero hooks are registered when ``forward_batch`` has no active steering
        specs. Installed hooks are always removed on exit, including exceptions.
        """
        plan = self.prepare_batch(forward_batch)
        if plan is None or plan.is_empty:
            yield None
            return

        with self.apply_plan(model, plan, forward_batch=forward_batch):
            yield plan

    @contextlib.contextmanager
    def apply_plan(
        self,
        model: nn.Module,
        plan: SteeringBatchPlan,
        *,
        forward_batch: Any = None,
    ) -> Iterator[SteeringBatchPlan]:
        """Install forward hooks on only the layers targeted by ``plan`` and always clean up."""
        if plan.is_empty:
            yield plan
            return

        layers = resolve_decoder_layers(model, layers_attr=self.layers_attr)
        handles: List[torch.utils.hooks.RemovableHandle] = []
        try:
            for layer_idx in sorted(plan.layer_deltas.keys()):
                if layer_idx >= len(layers):
                    raise ValueError(
                        f"Steering layer index {layer_idx} exceeds model layer count "
                        f"({len(layers)})."
                    )
                layer_module = layers[layer_idx]
                if _is_pp_missing_layer(layer_module):
                    continue

                def _make_hook(idx: int):
                    def _hook(
                        _module: nn.Module,
                        inputs: Tuple[Any, ...],
                        kwargs: Dict[str, Any],
                        output: Any,
                    ):
                        live_fb = kwargs.get("forward_batch")
                        if live_fb is None:
                            for arg in inputs:
                                if hasattr(arg, "residual_stream"):
                                    live_fb = arg
                                    break
                        if live_fb is None:
                            live_fb = forward_batch
                        return plan.apply_to_layer_output(
                            idx, output, forward_batch=live_fb
                        )

                    return _hook

                handles.append(
                    layer_module.register_forward_hook(
                        _make_hook(layer_idx), with_kwargs=True
                    )
                )

            yield plan
        finally:
            for handle in handles:
                handle.remove()
