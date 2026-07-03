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
"""Declarative model-override registry.

Model-identity adjustments to the server configuration are DECLARED here and
resolved into the flags tier through the ``apply_model_overrides`` gate —
model code never mutates ``ServerArgs``, which stays the pristine user input.

Two declaration forms, keyed on ``hf_config.architectures[0]``:

- ``MODEL_OVERRIDES``: pure-constant cases — ``arch -> {field: value}``.
- ``@register_model_override(arch)``: derived cases — a callable
  ``fn(server_args, hf_config) -> dict`` that faithfully carries today's
  conditional logic. ``server_args`` is pristine and must be treated
  read-only: the callable returns declarations, it never writes.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.arg_groups.arg_utils import model_overridable_fields
from sglang.srt.runtime_context import resolve_flag_leaf
from sglang.srt.utils.common import is_xpu

logger = logging.getLogger(__name__)

# Constant per-architecture overrides (populated by the migration sweeps).
MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # These models run in bfloat16 regardless of the requested dtype
    # (faithful port of the legacy unconditional arch branch).
    "MistralLarge3ForCausalLM": {"dtype": "bfloat16"},
    "PixtralForConditionalGeneration": {"dtype": "bfloat16"},
}

# Derived per-architecture override providers, in registration order.
_MODEL_OVERRIDE_FNS: Dict[str, List[Callable[..., dict]]] = {}

# Predicate-keyed providers, in registration order — for legacy branches
# matched by substring/predicate on the architecture string rather than an
# exact name (e.g. '"Step3p5ForCausalLM" in model_arch').
_PREDICATE_OVERRIDE_FNS: List[Tuple[Callable[[str], bool], Callable[..., dict]]] = []


def register_model_override(architecture: str):
    """Register a derived-override provider for ``architecture``.

    The decorated callable receives ``(server_args, hf_config)``, must not
    mutate either, and returns a ``{field: resolved_value}`` dict (possibly
    empty when nothing applies). Providers needing derived model data beyond
    the HF config go through ``server_args.get_model_config()`` (cached,
    read-only) — never anything mutating.
    """

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        _MODEL_OVERRIDE_FNS.setdefault(architecture, []).append(fn)
        return fn

    return decorator


def register_model_override_predicate(predicate: Callable[[str], bool]):
    """Register a derived-override provider keyed by an architecture
    predicate. Same callable contract as ``register_model_override``."""

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        _PREDICATE_OVERRIDE_FNS.append((predicate, fn))
        return fn

    return decorator


def _invoke_provider(
    fn: Callable[..., dict], server_args: Any, hf_config: Any
) -> Dict[str, Any]:
    declared = fn(server_args, hf_config)
    if not isinstance(declared, dict):
        raise TypeError(
            f"model override provider {fn.__qualname__} must return a dict, "
            f"got {type(declared).__name__}"
        )
    return declared


def collect_model_override_declarations(
    architecture: str, server_args: Any, hf_config: Any
) -> List[Tuple[str, Dict[str, Any]]]:
    """Collect ``(source, declaration)`` pairs for one architecture.

    Application order (last writer wins downstream in the gate): the constant
    ``MODEL_OVERRIDES`` entry first, then exact-keyed callables in
    registration order, then matching predicate-keyed callables in
    registration order. Empty declarations are dropped.
    """
    declarations: List[Tuple[str, Dict[str, Any]]] = []
    const = MODEL_OVERRIDES.get(architecture)
    if const:
        declarations.append((f"MODEL_OVERRIDES[{architecture!r}]", dict(const)))
    for fn in _MODEL_OVERRIDE_FNS.get(architecture, ()):
        declared = _invoke_provider(fn, server_args, hf_config)
        if declared:
            declarations.append((fn.__qualname__, dict(declared)))
    for predicate, fn in _PREDICATE_OVERRIDE_FNS:
        if predicate(architecture):
            declared = _invoke_provider(fn, server_args, hf_config)
            if declared:
                declarations.append((fn.__qualname__, dict(declared)))
    return declarations


# ---------------------------------------------------------------------------
# Derived per-family declarations (faithful ports of legacy arch branches).
# Callables read the PRISTINE server_args, never write; logging is kept
# verbatim from the legacy branch for operator-visible fidelity.
# ---------------------------------------------------------------------------


def _register_for(*architectures: str):
    """Register one provider for several architectures (family lists)."""

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        for architecture in architectures:
            register_model_override(architecture)(fn)
        return fn

    return decorator


# Keep in sync with MIMO_V2_MODEL_ARCHS (server_args.py / configs/hf_config.py).
@_register_for("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")
def _mimo_v2_overrides(server_args: Any, hf_config: Any) -> dict:
    if server_args.speculative_algorithm == "EAGLE":
        logger.info("Enable multi-layer EAGLE speculative decoding for MiMoV2 model.")
        return {"enable_multi_layer_eagle": True}
    return {}


@_register_for("MiniMaxM2ForCausalLM")
def _minimax_m2_overrides(server_args: Any, hf_config: Any) -> dict:
    logger.info(
        "Enable TF32 matmul for MiniMaxM2ForCausalLM model to improve gate gemm performance."
    )
    return {"enable_tf32_matmul": True}


@_register_for(
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
)
def _gemma2_gemma3_overrides(server_args: Any, hf_config: Any) -> dict:
    # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with gemma2 model.
    # It failed at this test: https://github.com/sgl-project/sglang/actions/runs/16255155597/job/45890331952#step:4:736
    logger.warning(
        f"Disable hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    return {"disable_hybrid_swa_memory": True}


@_register_for("Exaone4ForCausalLM", "ExaoneMoEForCausalLM")
def _exaone_overrides(server_args: Any, hf_config: Any) -> dict:
    if hf_config.sliding_window_pattern is not None:
        logger.warning(
            f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
        )
        return {"disable_hybrid_swa_memory": True}
    return {}


@_register_for("GptOssForCausalLM")
def _gpt_oss_overrides(server_args: Any, hf_config: Any) -> dict:
    if is_xpu():
        # Check for bf16 dtype on Intel XPU. Reads the pristine dtype request,
        # which equals the legacy mid-branch read: dtype had no earlier writer
        # for this arch.
        if server_args.dtype == "auto":
            logger.warning(
                "GptOssForCausalLM on Intel XPU currently supports bfloat16 dtype only"
            )
        elif server_args.dtype not in ["bfloat16"]:
            raise NotImplementedError(
                f"GptOssForCausalLM on Intel XPU only supports bfloat16 dtype, "
                f"but got '{server_args.dtype}'. Please use --dtype bfloat16 or remove --dtype to use auto."
            )
    quantization_config = getattr(hf_config, "quantization_config", None)
    if (
        quantization_config is not None
        and quantization_config.get("quant_method") == "mxfp4"
    ):
        # use bf16 for mxfp4 triton kernels
        return {"dtype": "bfloat16"}
    return {}


@_register_for("Olmo2ForCausalLM")
def _olmo2_overrides(server_args: Any, hf_config: Any) -> dict:
    # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with Olmo3 model.
    logger.warning(
        f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    return {"disable_hybrid_swa_memory": True}


@register_model_override_predicate(
    lambda arch: "Step3p5ForCausalLM" in arch
    or "Step3p7ForConditionalGeneration" in arch
)
def _step3p_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if server_args.speculative_algorithm == "EAGLE":
        logger.info(
            "Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model."
        )
        overrides["enable_multi_layer_eagle"] = True
    if server_args.enable_hierarchical_cache:
        logger.warning(
            "Reset swa_full_tokens_ratio to 1.0 for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["swa_full_tokens_ratio"] = 1.0
        logger.warning(
            "Disable hybrid SWA memory for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["disable_hybrid_swa_memory"] = True
    return overrides


@dataclasses.dataclass(frozen=True)
class OverrideRecord:
    """Provenance of one resolved write: ``base`` is the value before this
    declaration applied (the pristine value for the first writer)."""

    source: str
    field: str
    base: Any
    resolved: Any


def apply_model_overrides(
    flags: Any,
    server_args: Any,
    declarations: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    terminal: Sequence[Tuple[str, Dict[str, Any]]] = (),
    whitelist: Optional[Iterable[str]] = None,
    leaf_map: Optional[Dict[str, str]] = None,
) -> List[OverrideRecord]:
    """Resolve model-override declarations into the flags tier.

    - **Transactional**: every declaration (``terminal`` included) is
      validated against the whitelist and the flag-leaf layout BEFORE any
      write; on error nothing is applied.
    - **Ordering**: ``declarations`` apply in order (last writer wins), then
      ``terminal`` (the enforce-disable pass) applies after everything.
    - **Materialization**: every whitelisted field becomes a flag leaf —
      declared fields carry the resolved value, undeclared ones the pristine
      ``server_args`` value — so readers only ever read flags, never a
      "flag or fallback to config" combination.
    - ``server_args`` is read-only here: resolution output lives on flags.

    Returns the provenance log, one record per declared write.
    """
    if whitelist is None:
        whitelist = model_overridable_fields(type(server_args))
    whitelist = frozenset(whitelist)

    ordered = list(declarations) + list(terminal)

    problems = [
        f"{source}: {sorted(set(decl) - whitelist)} not model-overridable"
        for source, decl in ordered
        if set(decl) - whitelist
    ]
    if problems:
        raise ValueError(
            "model override validation failed (nothing was applied): "
            + "; ".join(problems)
        )
    for field in sorted(whitelist):
        owner, leaf = resolve_flag_leaf(flags, field, leaf_map=leaf_map)
        if leaf not in type(owner).__dataclass_fields__:
            raise ValueError(
                f"flag leaf for '{field}' is not declared on "
                f"{type(owner).__name__} (declare the dataclass field and map "
                "it in FLAG_LEAF_MAP); nothing was applied"
            )
        if getattr(owner, "_frozen", False):
            raise RuntimeError(
                f"cannot resolve '{field}': {type(owner).__name__} is frozen; "
                "nothing was applied"
            )

    resolved = {field: getattr(server_args, field) for field in whitelist}
    records: List[OverrideRecord] = []
    for source, decl in ordered:
        for field, value in decl.items():
            records.append(OverrideRecord(source, field, resolved[field], value))
            resolved[field] = value

    for field, value in resolved.items():
        owner, leaf = resolve_flag_leaf(flags, field, leaf_map=leaf_map)
        setattr(owner, leaf, value)
    return records


def apply_declarations_to_server_args(
    server_args: Any,
    declarations: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    terminal: Sequence[Tuple[str, Dict[str, Any]]] = (),
) -> None:
    """Transition-period dual-apply: replay declarations onto ``server_args``
    in gate order, byte-identical to the legacy imperative writes.

    Retired per field once that field's readers have all flipped to the flags
    tier (at which point the server_args field returns to pristine).

    Validates against the same whitelist as the publish gate BEFORE any write:
    a registry typo or a not-yet-resolvable field must fail fast here, not
    mutate ``server_args`` and only be rejected at publish time.
    """
    # Non-dataclass fixtures carry no Arg metadata (mirrors the
    # model_overridable_fields escape); only real ServerArgs is validated.
    if dataclasses.is_dataclass(type(server_args)):
        whitelist = model_overridable_fields(type(server_args))
        for source, decl in list(declarations) + list(terminal):
            unknown = set(decl) - whitelist
            if unknown:
                raise ValueError(
                    f"{source}: {sorted(unknown)} not model-overridable; the "
                    "transition dual-apply refuses fields the publish gate "
                    "would reject."
                )
    for _source, decl in list(declarations) + list(terminal):
        for field, value in decl.items():
            setattr(server_args, field, value)


def assert_flag_parity(
    flags: Any,
    server_args: Any,
    fields: Iterable[str],
    *,
    leaf_map: Optional[Dict[str, str]] = None,
) -> None:
    """Dual-apply drift guard: each migrated field's flag leaf must equal the
    (dual-applied) ``server_args`` value."""
    mismatches = []
    for field in fields:
        owner, leaf = resolve_flag_leaf(flags, field, leaf_map=leaf_map)
        flag_value = getattr(owner, leaf)
        args_value = getattr(server_args, field)
        if flag_value != args_value:
            mismatches.append(
                f"{field}: flags={flag_value!r} server_args={args_value!r}"
            )
    if mismatches:
        raise AssertionError("flag/server_args parity broken: " + "; ".join(mismatches))
