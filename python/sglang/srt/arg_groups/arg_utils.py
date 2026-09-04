# Copyright 2023-2024 SGLang Team
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
"""Utilities for auto-deriving argparse CLI arguments from dataclass fields.

Usage::

    from sglang.srt.arg_groups.arg_utils import A, Arg, add_cli_args_from_dataclass

    @dataclasses.dataclass
    class ServerArgs:
        # Simple fields — bare string is the help text:
        host: A[str, "The host of the HTTP server."] = "127.0.0.1"
        port: A[int, "The port of the HTTP server."] = 30000
        trust_remote_code: A[bool, "Whether to allow custom models."] = False
        tokenizer_path: A[Optional[str], "The path of the tokenizer."] = None

        # Fields with extra metadata — use Arg(...):
        model_path: A[str, Arg(help="Path to model weights.", aliases=["--model"])]
        load_format: A[str, Arg(help="Format.", choices=CHOICES)] = "auto"

        @staticmethod
        def add_cli_args(parser):
            add_cli_args_from_dataclass(parser, ServerArgs)

``A`` is a short alias for ``typing.Annotated``. A bare ``str`` inside the
annotation is equivalent to ``Arg(help=that_string)``.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import types
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

A = Annotated


class _NoFallback:
    """Sentinel for ``Arg.fallback``: this field declares none.

    ``None`` cannot serve, because ``None`` is what a field *holds* when the
    operator did not type it -- the state a fallback answers for.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<no fallback>"


NO_FALLBACK = _NoFallback()


@dataclasses.dataclass(frozen=True)
class Arg:
    """CLI argument metadata attached to a dataclass field via ``Annotated``."""

    help: str = ""
    choices: list | None = None
    aliases: list[str] | None = None
    cli_name: str | None = None
    type_parser: Callable | None = None
    nargs: str | None = None
    required: bool | None = None
    action: Any | None = None
    action_kwargs: dict | None = None
    const: Any | None = None
    # When True, this field is skipped by add_cli_args_from_dataclass.
    # Use for fields that have no CLI surface (e.g. injected via Python only).
    no_cli: bool = False
    # When True, config resolution (model overrides and post-process passes)
    # may decide this field: the declaration stash accepts the name, and
    # `resolution_result` and the config bags answer with the decision. The
    # field keeps what the operator passed.
    resolvable: bool = False
    # What the field means when nobody said anything: the operator did not type
    # it and resolution did not decide it. Not the dataclass default -- that
    # stays `None`, because `None` is how the record spells "not typed" and the
    # record is the wire format. This is the bottom of the read chain instead:
    # override, then decision, then input, then this. A model family that
    # declares the field still wins, because a decision sits above it.
    #
    # Only a value fixed for the life of the configuration belongs here. A
    # default that depends on the machine (`get_device()`), on another field
    # (`tokenizer_path` following `model_path`), or on anything impure
    # (`random.randint`) is a decision, and decisions stay in a hook where
    # their order is visible.
    fallback: Any = NO_FALLBACK


@dataclasses.dataclass(frozen=True)
class NS:
    """Namespace-path marker for a ServerArgs field, attached alongside the
    field's metadata in ``Annotated``:

        field: A[int, "help", NS("parallel")] = 1
        field: A[str, Arg(help="…"), NS("exec.moe")] = "auto"

    ``ServerArgs`` no longer uses it: its fields are declared in the
    ``arg_groups/fields/`` classes, each of which carries the ``_NS_PATH`` it
    stands for, so the module a declaration lives in *is* its namespace. What
    is left for this marker is the case a class cannot express -- one ad-hoc
    dataclass whose fields span several namespaces, which is what the
    config-bag tests build."""

    path: str


@functools.cache
def namespace_of(cls) -> dict:
    """``{field_name: dotted namespace path}``, read from the declaring class.

    A field's namespace is where it is declared: each class in
    ``arg_groups/fields/`` carries the ``_NS_PATH`` it stands for, and
    ``ServerArgs`` composes them. Walking the MRO therefore answers "which
    namespace owns this field" without a per-field marker -- the file the
    declaration sits in is the marker.

    A class that is not built that way -- an ad-hoc dataclass spanning several
    namespaces, which is what the config-bag tests construct -- falls back to
    the per-field ``NS`` marker. A field with neither is absent from the map
    (the coverage lint flags them). Non-dataclass types yield an empty map.
    """
    if not dataclasses.is_dataclass(cls):
        return {}
    # An assembled record: the collector recorded who declared each field,
    # because there are no base classes left to ask.
    out = dict(getattr(cls, "_NS_BY_FIELD", None) or {})
    # A class that still inherits its namespaces: nearest declaration wins, so
    # walk the MRO front to back and keep the first answer.
    for base in cls.__mro__:
        path = base.__dict__.get("_NS_PATH")
        if path is None:
            continue
        for name in getattr(base, "__annotations__", {}):
            out.setdefault(name, path)
    if len(out) == len(dataclasses.fields(cls)):
        return out
    hints = get_type_hints(cls, include_extras=True)
    for field in dataclasses.fields(cls):
        if field.name in out:
            continue
        tp = hints.get(field.name, field.type)
        if get_origin(tp) is Annotated:
            for a in get_args(tp)[1:]:
                if isinstance(a, NS):
                    out[field.name] = a.path
                    break
    return out


@functools.cache
def field_names(cls) -> frozenset:
    """Names of ``cls`` dataclass fields — what a declaration may name."""
    if not dataclasses.is_dataclass(cls):
        return frozenset()
    return frozenset(field.name for field in dataclasses.fields(cls))


@functools.cache
def resolvable_fields(cls) -> frozenset:
    """Names of ``cls`` dataclass fields whose ``Arg`` metadata declares
    ``resolvable=True`` — the whitelist for config resolution.

    Non-dataclass types (e.g. mock config objects in tests) have no Arg
    metadata and yield an empty whitelist."""
    if not dataclasses.is_dataclass(cls):
        return frozenset()
    hints = get_type_hints(cls, include_extras=True)
    names = set()
    for field in dataclasses.fields(cls):
        _, arg = _unwrap_annotated(hints.get(field.name, field.type))
        if arg is not None and arg.resolvable:
            names.add(field.name)
    return frozenset(names)


@functools.cache
def fallbacks_of(cls) -> dict:
    """``{field_name: value}`` for every field of ``cls`` that declares one.

    Read the same way `resolvable_fields` reads its flag, so a fallback lives
    beside the help text of the field it belongs to rather than in whatever
    hook used to fill it in.
    """
    if not dataclasses.is_dataclass(cls):
        return {}
    hints = get_type_hints(cls, include_extras=True)
    out = {}
    for field in dataclasses.fields(cls):
        _, arg = _unwrap_annotated(hints.get(field.name, field.type))
        if arg is not None and arg.fallback is not NO_FALLBACK:
            out[field.name] = arg.fallback
    return out


def with_fallback(cls, name: str, value: Any) -> Any:
    """``value``, or the declared fallback when nothing has answered.

    `resolution_result` calls this as its last step -- the effective surface,
    which the config bags and `/server_info` read through.

    Deliberately not the views a resolution pass is handed. Those answer with
    what has been decided over what the operator typed, and a pass reads them
    *while deciding*: `model_overrides/inkling.py` branches on
    `if cfg.swa_full_tokens_ratio is None`, and a fallback answering there
    would make that branch dead and replace the family's value with the
    generic one. `test_declared_fallbacks.py` pins both halves.

    A container fallback is copied, so a reader that mutates what it got does
    not edit the declaration for every other reader -- the reason a dataclass
    spells this `default_factory`.
    """
    if value is not None:
        return value
    fallback = fallbacks_of(cls).get(name, NO_FALLBACK)
    if fallback is NO_FALLBACK:
        return value
    if isinstance(fallback, (list, dict, set)):
        return copy.deepcopy(fallback)
    return fallback


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MISSING = dataclasses.MISSING


def _unwrap_annotated(tp):
    """Return (inner_type, Arg | None) from ``Annotated[T, Arg(...)]``.

    Also accepts a bare string as shorthand: ``Annotated[T, "help text"]``
    is equivalent to ``Annotated[T, Arg(help="help text")]``.
    """
    origin = get_origin(tp)
    if origin is Annotated:
        args = get_args(tp)
        inner = args[0]
        for a in args[1:]:
            if isinstance(a, Arg):
                return inner, a
            if isinstance(a, str):
                return inner, Arg(help=a)
        return inner, None
    return tp, None


def _unwrap_optional(tp):
    """If tp is Optional[X] (i.e. Union[X, None]), return (X, True). Else (tp, False)."""
    origin = get_origin(tp)
    is_union = origin is Union or (
        hasattr(types, "UnionType") and origin is types.UnionType
    )
    if is_union:
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return tp, False


def _unwrap_literal(tp):
    """If tp is Literal[...], return list of values. Else None."""
    origin = get_origin(tp)
    if origin is Literal:
        return list(get_args(tp))
    return None


def _infer_type_func(tp):
    """Map a Python type annotation to an argparse ``type=`` callable."""
    if tp is str:
        return str
    if tp is int:
        return int
    if tp is float:
        return float
    return str


def _field_default(field):
    """Return the default value for a dataclass field, or _MISSING."""
    if field.default is not _MISSING:
        return field.default
    if field.default_factory is not _MISSING:
        return field.default_factory()
    return _MISSING


def _field_to_cli_name(name: str) -> str:
    """Convert a field name like ``model_path`` to ``--model-path``."""
    return "--" + name.replace("_", "-")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_cli_args_from_dataclass(parser, cls, *, fields: list[str] | None = None):
    """Add argparse arguments for every ``A[T, "help"]`` or ``A[T, Arg(...)]`` field.

    Fields without an ``Arg`` or bare-string annotation are silently skipped —
    they must still be registered manually (this allows incremental migration).

    Parameters
    ----------
    parser : argparse.ArgumentParser
    cls : dataclass type
    fields : optional list of field names to include. If None, all fields with
        ``Arg`` annotations are included.
    """
    hints = get_type_hints(cls, include_extras=True)

    for field in dataclasses.fields(cls):
        if fields is not None and field.name not in fields:
            continue

        hint = hints.get(field.name)
        if hint is None:
            continue

        raw_type, arg_meta = _unwrap_annotated(hint)
        if arg_meta is None:
            continue
        if arg_meta.no_cli:
            continue

        cli_name = arg_meta.cli_name or _field_to_cli_name(field.name)
        names = [cli_name] + (arg_meta.aliases or [])
        default = _field_default(field)
        # Anchor dest to the field name so argparse stores the value
        # under the dataclass attribute directly, even when cli_name
        # differs (e.g. --tensor-parallel-size → tp_size).
        auto_dest = cli_name.lstrip("-").replace("-", "_")
        dest_kwarg = {"dest": field.name} if field.name != auto_dest else {}

        # Handle custom action
        if arg_meta.action is not None:
            kwargs = {
                "action": arg_meta.action,
                "help": arg_meta.help,
                **dest_kwarg,
            }
            if default is not _MISSING:
                kwargs["default"] = default
            if arg_meta.action_kwargs:
                kwargs.update(arg_meta.action_kwargs)
            parser.add_argument(*names, **kwargs)
            continue

        # Unwrap Optional
        inner_type, is_optional = _unwrap_optional(raw_type)

        # Check for Literal — auto-derive choices
        literal_vals = _unwrap_literal(inner_type)
        if literal_vals is not None:
            choices = arg_meta.choices or literal_vals
            # Infer type from first literal value
            val_type = type(literal_vals[0]) if literal_vals else str
            type_func = arg_meta.type_parser or _infer_type_func(val_type)
            kwargs = dict(
                type=type_func, choices=choices, help=arg_meta.help, **dest_kwarg
            )
            if default is not _MISSING:
                kwargs["default"] = default
            if arg_meta.const is not None:
                kwargs["const"] = arg_meta.const
            parser.add_argument(*names, **kwargs)
            continue

        # Check for List[X] — but skip if type_parser is set (the parser
        # handles the whole value as a single string, e.g. json_list_type).
        origin = get_origin(inner_type)
        if origin is list and arg_meta.type_parser is None:
            elem_args = get_args(inner_type)
            elem_type = elem_args[0] if elem_args else str
            type_func = _infer_type_func(elem_type)
            nargs = arg_meta.nargs or "+"
            kwargs = dict(
                type=type_func,
                nargs=nargs,
                help=arg_meta.help,
                **dest_kwarg,
            )
            if arg_meta.choices:
                kwargs["choices"] = arg_meta.choices
            if default is not _MISSING:
                kwargs["default"] = default
            if arg_meta.const is not None:
                kwargs["const"] = arg_meta.const
            parser.add_argument(*names, **kwargs)
            continue

        # Bool → store_true
        if inner_type is bool:
            kwargs = dict(action="store_true", help=arg_meta.help, **dest_kwarg)
            if default is not _MISSING:
                kwargs["default"] = default
            parser.add_argument(*names, **kwargs)
            continue

        # Scalar types (str, int, float, etc.)
        type_func = arg_meta.type_parser or _infer_type_func(inner_type)
        kwargs = dict(type=type_func, help=arg_meta.help, **dest_kwarg)
        if arg_meta.choices:
            kwargs["choices"] = arg_meta.choices
        if arg_meta.nargs:
            kwargs["nargs"] = arg_meta.nargs
        if default is not _MISSING:
            kwargs["default"] = default
        if arg_meta.const is not None:
            kwargs["const"] = arg_meta.const
        if (
            arg_meta.required is True
            or (arg_meta.required is None and default is _MISSING)
        ) and any(name.startswith("-") for name in names):
            kwargs["required"] = True
        parser.add_argument(*names, **kwargs)
