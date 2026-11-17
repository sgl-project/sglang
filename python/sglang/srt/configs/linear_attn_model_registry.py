"""Registry for linear attention hybrid models (softmax + linear attention).

External models can register themselves without modifying SGLang core files:

    from sglang.srt.configs.linear_attn_model_registry import (
        register_linear_attn_model, LinearAttnModelSpec,
    )

    register_linear_attn_model(LinearAttnModelSpec(
        config_class=MyLinearAttnConfig,
        backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
        arch_names=["MyLinearAttnForCausalLM"],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
    ))

Prefix caching keeps a request's linear attention (conv and recurrent) state in the
radix tree as a copy of a state pool slot, and the strategy that takes those copies
decides whether overlap scheduling can run:

- ``no_buffer`` copies (or donates) the request's own slot when the scheduler
  processes a batch's results. With overlap scheduling the next batch, which may
  hold the same request, is already queued and advances that slot in place, so the
  copy would hold a later state than its key. SGLang turns overlap scheduling off
  for this strategy. It needs nothing from the backend.
- ``extra_buffer`` (``support_mamba_cache_extra_buffer=True``) keeps overlap
  scheduling: each request gets two track slots (which the state pool sizing
  reserves per running request), and the scheduler picks the
  tracked positions (in an extend, the last ``mamba_cache_chunk_size`` boundary of
  its new tokens, relative to the extend start; in decode, every
  ``mamba_track_interval`` tokens) and alternates between the slots. The backend
  must fill a request's track slot with its state at the tracked position, in the
  same forward, for every row of ``forward_batch.mamba_track_mask``:

  - Extend: the conv state there, from the inputs before it
    (``forward_metadata.track_conv_indices``; set ``conv_states_shape`` so its last
    dim is the conv window length), and the recurrent state there, which
    ``forward_batch.mamba_track_aligned_lens()`` new tokens into the extend need
    not end (a kernel output at a chunk boundary).
    ``MambaAttnBackendBase._track_mamba_state_extend`` copies it from per-chunk
    states or the final state.
  - Decode: the slot's states after the step;
    ``MambaAttnBackendBase._track_mamba_state_decode`` copies them.

  The destinations are ``MambaAttnBackendBase.state_tracking_indices``. A backend that
  declares the strategy without filling the track slots leaves stale states in the
  prefix cache, which later requests resume from.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LinearAttnModelSpec:
    """Specification for a hybrid (softmax + linear attention) model."""

    config_class: type
    backend_class_name: str  # fully-qualified class name, lazily imported
    arch_names: list[str] = field(default_factory=list)
    uses_mamba_radix_cache: bool = True
    support_mamba_cache: bool = True
    # Whether the backend fills the extra_buffer strategy's track slots (see the
    # module docstring), which keeps overlap scheduling on with the radix cache.
    support_mamba_cache_extra_buffer: bool = False
    unwrap_text_config: bool = False  # call get_text_config() before isinstance check
    hybrid_backend_class_name: str | None = None
    config_predicate: Callable[[Any], bool] | None = None


_LINEAR_ATTN_MODEL_REGISTRY: list[LinearAttnModelSpec] = []


def register_linear_attn_model(spec: LinearAttnModelSpec) -> None:
    _LINEAR_ATTN_MODEL_REGISTRY.append(spec)
    logger.debug(
        "Registered linear attn model: config=%s, backend=%s, archs=%s",
        spec.config_class.__name__,
        spec.backend_class_name.rsplit(".", 1)[-1],
        spec.arch_names,
    )


def get_linear_attn_config(hf_config: Any) -> Optional[tuple[LinearAttnModelSpec, Any]]:
    for spec in _LINEAR_ATTN_MODEL_REGISTRY:
        config = hf_config.get_text_config() if spec.unwrap_text_config else hf_config
        if isinstance(config, spec.config_class) and (
            spec.config_predicate is None or spec.config_predicate(config)
        ):
            return spec, config
    return None


def get_linear_attn_spec_by_arch(arch_name: str) -> Optional[LinearAttnModelSpec]:
    for spec in _LINEAR_ATTN_MODEL_REGISTRY:
        if arch_name in spec.arch_names:
            return spec
    return None


def get_linear_attn_spec(hf_config: Any) -> Optional[LinearAttnModelSpec]:
    """Spec for ``hf_config``: by architecture name, else by config class and predicate."""
    spec = get_linear_attn_spec_by_arch(hf_config.architectures[0])
    if spec is not None:
        return spec
    result = get_linear_attn_config(hf_config)
    return result[0] if result is not None else None


def import_backend_class(dotted_name: str) -> type:
    module_path, class_name = dotted_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
