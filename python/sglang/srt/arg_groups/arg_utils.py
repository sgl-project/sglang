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
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Literal,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import msgspec
import msgspec.structs

A = Annotated

# What the readbacks publish in a credential's place. A fixed string, so a
# reader diffing two dumps of the same configuration does not see a change.
# `launch_command` uses it as typed; the record-side readbacks use
# `redacted_value`, which keeps the count of values the record holds.
REDACTED = "<redacted>"


def redacted_value(value: Any) -> str:
    """The marker a record-side readback publishes for a configured credential:
    ``<redacted:1 value>`` for one, ``<redacted:N values>`` for a field that
    holds several (a list, tuple or set of credentials).

    The count is the one diagnostic a redacted field can keep: whether auth is
    configured at all, and, for a field that carries a set of keys, how many the
    parse produced. An operator who passed six keys and sees ``1 value`` learns
    that the flag did not fan out, without a key reaching the log. Callers pass
    only set values; ``None`` stays ``None`` so an unset credential still reads
    as unset. Same configuration, same string, so diffs of two dumps stay quiet."""
    n = len(value) if isinstance(value, (list, tuple, set, frozenset)) else 1
    return f"<redacted:{n} value{'' if n == 1 else 's'}>"


class Arg(msgspec.Struct, frozen=True):
    """CLI argument metadata attached to a field via ``Annotated``."""

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
    # What the field means when nobody said anything -- the bottom of the read
    # chain: override, decision, input, then this. Not the dataclass default,
    # which stays `None` because that is how the record spells "not typed".
    # `None` here means the field declares no fallback, which is the same
    # answer the read chain gives without one.
    #
    # Only a value fixed for the life of the configuration belongs here. One
    # that depends on the machine, on another field, or on anything impure is a
    # decision, and decisions stay in a hook where their order is visible.
    fallback: Any = None
    # When True, the field holds a credential. `resolved_dict` and the launch
    # command publish `REDACTED` in its place; the record keeps the real value,
    # so the auth middleware and the SSL loader read the field as before.
    secret: bool = False


class Derived(msgspec.Struct, frozen=True):
    """Metadata for a field the configuration implies, not one anyone types.

    The other half of a namespace. An ``Arg`` field is the operator's input and
    is collected into ``ServerArgs``; a ``Derived`` field carries no annotation,
    so it is not a dataclass field and never reaches the record -- which is
    right, because it has no input to preserve and the record is what crosses a
    process boundary.

    ``fn`` names what computes it, as a dotted path resolved lazily so that a
    declaration module stays free of runtime imports. Such a field is a pure
    function of the published configuration, so it is computed once at
    ``publish`` and stored as an ordinary bag leaf -- a plain attribute load,
    which is what a read inside compiled model code needs.

    Every declaration carries ``fn`` today, the parallel quotients included:
    they are a function of the configured leaves, so they are computed at
    publish like the rest. What is special about them is not how they are
    computed but that a stamp can move one afterwards -- an elastic scale-up
    restamps ``attn_dp_size`` -- which ``ParallelContext`` answers above the
    published leaf.
    """

    doc: str = ""
    fn: str = ""


class NS(msgspec.Struct, frozen=True):
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
    if not is_record(cls):
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
    if len(out) == len(record_fields(cls)):
        return out
    hints = get_type_hints(cls, include_extras=True)
    for field in record_fields(cls):
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
    if not is_record(cls):
        return frozenset()
    return frozenset(field.name for field in record_fields(cls))


@functools.cache
def resolvable_fields(cls) -> frozenset:
    """Names of ``cls`` dataclass fields whose ``Arg`` metadata declares
    ``resolvable=True`` — the whitelist for config resolution.

    Non-dataclass types (e.g. mock config objects in tests) have no Arg
    metadata and yield an empty whitelist."""
    if not is_record(cls):
        return frozenset()
    hints = get_type_hints(cls, include_extras=True)
    names = set()
    for field in record_fields(cls):
        _, arg = _unwrap_annotated(hints.get(field.name, field.type))
        if arg is not None and arg.resolvable:
            names.add(field.name)
    return frozenset(names)


@functools.cache
def secret_fields(cls) -> frozenset:
    """Names of ``cls`` fields whose ``Arg`` metadata declares ``secret=True``:
    the credentials the readbacks redact.

    Non-record types (e.g. mock config objects in tests) have no Arg metadata
    and yield an empty set."""
    if not is_record(cls):
        return frozenset()
    hints = get_type_hints(cls, include_extras=True)
    names = set()
    for field in record_fields(cls):
        _, arg = _unwrap_annotated(hints.get(field.name, field.type))
        if arg is not None and arg.secret:
            names.add(field.name)
    return frozenset(names)


def _cli_spellings(cls, names) -> frozenset:
    """Every option string ``add_cli_args_from_dataclass`` registers for the
    fields in ``names``: the ``cli_name`` (or the derived ``--field-name``)
    and each alias. ``no_cli`` fields register nothing."""
    hints = get_type_hints(cls, include_extras=True)
    flags = set()
    for field in record_fields(cls):
        if field.name not in names:
            continue
        _, arg = _unwrap_annotated(hints.get(field.name, field.type))
        if arg is None or arg.no_cli:
            continue
        flags.add(arg.cli_name or _field_to_cli_name(field.name))
        flags.update(arg.aliases or ())
    return frozenset(flags)


@functools.cache
def cli_flags(cls) -> frozenset:
    """Every CLI spelling the record registers. What an abbreviated flag is
    resolved against."""
    if not is_record(cls):
        return frozenset()
    return _cli_spellings(cls, {field.name for field in record_fields(cls)})


@functools.cache
def secret_cli_flags(cls) -> frozenset:
    """The CLI spellings of ``secret_fields(cls)``: ``--api-key`` and any
    alias the field declares. What ``redacted_argv`` looks for."""
    if not is_record(cls):
        return frozenset()
    return _cli_spellings(cls, secret_fields(cls))


def _resolve_flag(cls, flag: str) -> str | None:
    """The registered spelling argparse read ``flag`` as, or ``None``.

    An exact spelling or alias resolves to itself. So does a proper prefix of
    exactly one registered spelling: argparse accepts abbreviations
    (``allow_abbrev`` defaults to True) and resolves a prefix only when it is
    unique among every option string, so once the parse has succeeded a prefix
    unique among the record's spellings resolved to that spelling. A prefix
    that also matched a flag registered outside the record was ambiguous to
    argparse, and there was no parse to redact. A value, or a flag the
    metadata does not know, resolves to nothing."""
    known = cli_flags(cls)
    if flag in known:
        return flag
    if not flag.startswith("--"):
        return None
    matches = [spelling for spelling in known if spelling.startswith(flag)]
    return matches[0] if len(matches) == 1 else None


def _spells_a_secret_flag(cls, flag: str) -> bool:
    """Whether argparse read ``flag`` as a credential flag of ``cls``."""
    return _resolve_flag(cls, flag) in secret_cli_flags(cls)


def _spells_a_public_flag(cls, flag: str) -> bool:
    """Whether argparse read ``flag`` as a non-credential flag of ``cls``: the
    one position where the value pass leaves a matching token alone."""
    resolved = _resolve_flag(cls, flag)
    return resolved is not None and resolved not in secret_cli_flags(cls)


def redacted_argv(cls, argv: Sequence[str], record: Any = None) -> list[str]:
    """``argv`` with every credential of ``cls`` replaced by ``REDACTED``.
    Everything else is returned as typed.

    Two passes. The first reads the flags the way argparse did: a token that
    spells a credential flag (exact, alias, or unique abbreviation) hides the
    value after it, or after its ``=``. The second runs when ``record`` (the
    parsed record, or the argparse namespace) is given, and hides the parsed
    credential values themselves wherever a whole token, or the value half of
    a ``--flag=VALUE`` token, equals one. It closes any spelling the first pass
    cannot know, such as a flag registered outside the record with a
    credential ``dest``. It leaves one position alone: the value of a flag
    that resolves (exact, alias, or unique abbreviation) to a non-credential
    field of ``cls``, as the token after it or as its ``=`` half. argparse
    bound that token to a field the metadata knows is public, so
    ``--api-key 1 --tp 1`` keeps ``--tp 1`` readable. A matching token
    anywhere else (after a flag the metadata does not know, standalone) is
    hidden, because nothing says what argparse bound it to, and there the
    alternative under-hides a key."""
    out: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            out.append(REDACTED)
            hide_next = False
            continue
        flag, sep, _value = token.partition("=")
        if _spells_a_secret_flag(cls, flag):
            out.append(f"{flag}={REDACTED}" if sep else flag)
            hide_next = not sep
            continue
        out.append(token)
    if record is None:
        return out

    values = {
        value
        for name in secret_fields(cls)
        if isinstance(value := getattr(record, name, None), str) and value
    }
    if not values:
        return out
    for i, token in enumerate(out):
        if token in values and not token.startswith("-"):
            prev_flag, prev_sep, _ = out[i - 1].partition("=") if i else ("", "", "")
            if not prev_sep and _spells_a_public_flag(cls, prev_flag):
                continue
            out[i] = REDACTED
            continue
        flag, sep, value = token.partition("=")
        if sep and value in values and flag.startswith("-"):
            if _spells_a_public_flag(cls, flag):
                continue
            out[i] = f"{flag}={REDACTED}"
    return out


def redacted_call(cls, name: str, kwargs: Mapping[str, Any]) -> str:
    """``name(k=v, ...)`` spelled with ``repr`` values, every credential kwarg
    of ``cls`` replaced by its ``redacted_value`` marker. How the in-process
    ``Engine`` records the call that built its record."""
    secrets = secret_fields(cls)
    parts = (
        f"{k}={redacted_value(v)!r}" if k in secrets and v is not None else f"{k}={v!r}"
        for k, v in kwargs.items()
    )
    return f"{name}(" + ", ".join(parts) + ")"


@functools.cache
def fallbacks_of(cls) -> dict:
    """``{field_name: value}`` for every field of ``cls`` that declares one.

    Read the same way `resolvable_fields` reads its flag, so a fallback lives
    beside the help text of the field it belongs to rather than in whatever
    hook used to fill it in.
    """
    if not is_record(cls):
        return {}
    hints = get_type_hints(cls, include_extras=True)
    out = {}
    for field in record_fields(cls):
        _, arg = _unwrap_annotated(hints.get(field.name, field.type))
        if arg is not None and arg.fallback is not None:
            # Two things `with_fallback` relies on and cannot check itself,
            # asserted where a new declaration passes through. This function is
            # cached, so a mutable fallback would hand one shared object to
            # every reader; and a field whose dataclass default is not `None`
            # can never reach the fallback, which makes the declaration dead.
            assert not isinstance(arg.fallback, (list, dict, set)), (
                f"{cls.__name__}.{field.name}: a mutable fallback would be "
                "shared by every reader -- use a scalar"
            )
            assert field.default is None, (
                f"{cls.__name__}.{field.name}: declares a fallback but defaults "
                f"to {field.default!r}, so the fallback is unreachable"
            )
            out[field.name] = arg.fallback
    return out


def with_fallback(cls, name: str, value: Any) -> Any:
    """``value``, or the declared fallback when nothing has answered.

    `resolution_result` calls this as its last step -- the effective surface,
    which the config bags and `/server_info` read through. Deliberately not the
    views a pass reads *while deciding*: `model_overrides/inkling.py` branches
    on `if cfg.swa_full_tokens_ratio is None`, and a fallback answering there
    would make that branch dead. `test_declared_fallbacks.py` pins both halves.

    A mutable fallback would need copying per read, for the reason a dataclass
    spells this `default_factory`. Every declared one is a scalar.
    """
    if value is not None:
        return value
    return fallbacks_of(cls).get(name, value)


def record_fields(cls):
    """The declared fields of a record, Struct or dataclass.

    `ServerArgs` and the namespace classes are `msgspec.Struct`; the config-bag
    tests build ad-hoc dataclasses spanning namespaces, and the helpers here are
    driven with both. Anything else yields nothing.
    """
    if isinstance(cls, type) and issubclass(cls, msgspec.Struct):
        return msgspec.structs.fields(cls)
    if dataclasses.is_dataclass(cls):
        return dataclasses.fields(cls)
    return ()


def is_record(cls) -> bool:
    """Whether ``cls`` declares fields the way a record does."""
    target = cls if isinstance(cls, type) else type(cls)
    return issubclass(target, msgspec.Struct) or dataclasses.is_dataclass(cls)


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
    """Return the default value for a field, or `_MISSING`.

    The two record shapes spell "no default" differently -- a Struct field says
    `msgspec.NODEFAULT`, a dataclass field `dataclasses.MISSING` -- so both are
    normalized here and every caller below tests against `_MISSING` alone.
    """
    absent = (_MISSING, msgspec.NODEFAULT)
    if field.default not in absent:
        return field.default
    if field.default_factory not in absent:
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

    for field in record_fields(cls):
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
