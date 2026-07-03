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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.arg_groups.arg_utils import model_overridable_fields
from sglang.srt.runtime_context import resolve_flag_leaf

# Constant per-architecture overrides (populated by the migration sweeps).
MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = {}

# Derived per-architecture override providers, in registration order.
_MODEL_OVERRIDE_FNS: Dict[str, List[Callable[..., dict]]] = {}


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


def collect_model_override_declarations(
    architecture: str, server_args: Any, hf_config: Any
) -> List[Tuple[str, Dict[str, Any]]]:
    """Collect ``(source, declaration)`` pairs for one architecture.

    Application order (last writer wins downstream in the gate): the constant
    ``MODEL_OVERRIDES`` entry first, then registered callables in registration
    order. Empty declarations are dropped.
    """
    declarations: List[Tuple[str, Dict[str, Any]]] = []
    const = MODEL_OVERRIDES.get(architecture)
    if const:
        declarations.append((f"MODEL_OVERRIDES[{architecture!r}]", dict(const)))
    for fn in _MODEL_OVERRIDE_FNS.get(architecture, ()):
        declared = fn(server_args, hf_config)
        if not isinstance(declared, dict):
            raise TypeError(
                f"model override provider {fn.__qualname__} must return a dict, "
                f"got {type(declared).__name__}"
            )
        if declared:
            declarations.append((fn.__qualname__, dict(declared)))
    return declarations


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
    """
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
