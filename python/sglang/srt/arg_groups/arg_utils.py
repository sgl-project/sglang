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

import dataclasses
import functools
import types
from typing import (
    Annotated,
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

A = Annotated


@dataclasses.dataclass(frozen=True)
class Arg:
    """CLI argument metadata attached to a dataclass field via ``Annotated``."""

    help: str = ""
    choices: Optional[list] = None
    aliases: Optional[List[str]] = None
    cli_name: Optional[str] = None
    type_parser: Optional[Callable] = None
    nargs: Optional[str] = None
    required: Optional[bool] = None
    action: Optional[Any] = None
    action_kwargs: Optional[dict] = None
    const: Optional[Any] = None
    # When True, this field is skipped by add_cli_args_from_dataclass.
    # Use for fields that have no CLI surface (e.g. injected via Python only).
    no_cli: bool = False
    # When True, this field may be written by config resolution (model
    # overrides and post-process passes): it is part of the whitelist accepted
    # by the apply_model_overrides gate, and its resolved value lives on the
    # flags tier (the server_args field itself stays the pristine user input).
    resolvable: bool = False


@functools.lru_cache(maxsize=None)
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


def add_cli_args_from_dataclass(parser, cls, *, fields: Optional[List[str]] = None):
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
        if (origin is list or origin is List) and arg_meta.type_parser is None:
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
