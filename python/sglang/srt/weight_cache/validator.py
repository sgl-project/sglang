# SPDX-License-Identifier: Apache-2.0
"""Validator: is a snapshot-loaded rank equal to the owner's raw load?

It shares no logic with load-diff and never imports it: a check made with
load-diff's own walk would only show that install repeated export.

The owner describes its raw-loaded model, the reader describes its installed
model the same way, and any difference refuses the load. One walk follows every
object the model reaches and describes it under the path of its first visit:

- a module: its class, its mode, and the names of its parameters, buffers and
  children in registration order;
- a tensor: class, dtype, shape, stride, device kind and requires_grad, and, if
  it has data, the allocation and byte offset where the data starts. Every
  allocation is digested once, in full, with BLAKE3: a published storage under
  its name, an allocation of the model's own by the order of first use;
- an object: its class and its fields. A function: its code and what it closes
  over;
- a list, tuple or dict that holds such objects, member by member;
- another way to an object already seen, as a link to the first.

All of these must be equal. Plain values (numbers, strings, containers of them)
are compared only where either side's load changed them since construction: the
others may differ between correct processes, such as a checkpoint path as typed.

The walk guards against a load that leaves another state. It does not guard
against the model's own classes, which are trusted code. Still, it runs none of
their code where it can read without it: fields come from instance dicts and
slots, members from the built-in containers' own iteration, and what a tensor
is from torch.Tensor itself, so that reading a model does not change it.

What the walk cannot describe stops the attestation, so no rank loads the
snapshot. One kind is let through: an object that keeps its state where Python
cannot see it, such as a lock, a process group or a random generator, must be
the very object that construction left there. So must a logger, which leads to
every logger of the process, and a path object.

The expert map is state of the process, not of the model: the validator
compares the map of its process whichever model it is asked about.

Not seen:

- a plain value that construction derived from the device and that no load
  changes (a reader constructs on the meta device);
- what happens inside an object that must only be the same object;
- whether two names hold one plain container or two equal ones;
- an order that move_to_end gave an ordered dict, such as that of a hook
  registered with prepend: dicts are read in the order of insertion;
- where an empty tensor sits, and the bytes of an allocation beyond its
  published storage;
- classes, Python modules and other state outside the model, gradients.
"""

from __future__ import annotations

import array
import collections
import contextlib
import enum
import functools
import logging
import pathlib
import re
import struct
import types
import zlib
from typing import Any, Iterator, Optional

import blake3
import msgspec
import numpy as np
import torch
from torch import nn

_VERSION = 1
_MAX_DEPTH = 64  # what is nested deeper is not described
_MAX_OBJECTS = 20_000_000  # about four for each field of each module
_MAX_REPORTED = 20
_MAX_ATTESTATION = 1 << 30
_MAX_LEAF = 256  # a longer plain value is kept as its digest
_MAX_SPLIT = 64  # a longer plain list or dict is one value
_DIGESTED = b"\xc1"  # the one byte that starts no msgpack value
_NO = object()
_WORD = re.compile(r"[A-Za-z0-9_]+")
_REGISTRIES = ("_parameters", "_buffers", "_modules")
_HOOK_REGISTRIES = frozenset(
    {
        "_backward_pre_hooks",
        "_backward_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    }
)
# Options of a hook, keyed like its registry by a handle id of this process.
_HOOK_FLAGS = {
    "_forward_hooks_with_kwargs": "_forward_hooks",
    "_forward_hooks_always_called": "_forward_hooks",
    "_forward_pre_hooks_with_kwargs": "_forward_pre_hooks",
}
_TENSOR_FIELDS = (
    "class",
    "dtype",
    "shape",
    "stride",
    "device",
    "requires_grad",
    "allocation",
    "offset",
)

# What a class or an object is gets read through these, never by asking it.
_TYPE = type.__dict__
_MODULE_DICT = types.ModuleType.__dict__["__dict__"]
_PARTIAL = functools.partial.__dict__
_T = torch.Tensor
_S = torch.UntypedStorage
_HEADER = object.__basicsize__
_POINTER = struct.calcsize("P")
_HEAP = 1 << 9  # Py_TPFLAGS_HEAPTYPE: the class was made at run time
# Classes that were not made at run time and still keep nothing of their own:
# object, SimpleNamespace, and what every msgspec.Struct inherits.
_BARE = (object, types.SimpleNamespace, *msgspec.Struct.__mro__[1:])
_CHUNK = 64 << 20  # host bytes are hashed from copies of this size
_ENUM_FIELDS = frozenset({"__objclass__", "_name_", "_sort_order_", "_value_"})
# How values of a class are described. A class is looked up by its id: hashing
# or comparing a class would ask its metaclass.
_EXACT = {
    id(cls): category
    for category, classes in (
        ("none", (type(None),)),
        ("bool", (bool,)),
        ("int", (int,)),
        ("float", (float,)),
        ("complex", (complex,)),
        ("str", (str,)),
        ("bytes", (bytes,)),
        ("bytearray", (bytearray,)),
        ("ellipsis", (type(...),)),
        ("range", (range,)),
        ("slice", (slice,)),
        ("list", (list,)),
        ("tuple", (tuple,)),
        ("dict", (dict,)),
        ("ordered dict", (collections.OrderedDict,)),
        ("set", (set,)),
        ("frozenset", (frozenset,)),
        ("device", (torch.device,)),
        ("torch", (torch.dtype, torch.layout, torch.memory_format)),
        ("size", (torch.Size,)),
        ("ndarray", (np.ndarray,)),
        ("array", (array.array,)),
        (
            "numpy scalar",
            tuple(set(np.sctypeDict.values()) - {np.void, np.object_}),
        ),
        ("tensor", (torch.Tensor, nn.Parameter)),
        ("function", (types.FunctionType,)),
        ("builtin", (types.BuiltinFunctionType,)),
        ("method", (types.MethodType,)),
        ("code", (types.CodeType,)),
        ("partial", (functools.partial,)),
    )
    for cls in classes
}
# A class that is none of those is described by the first of these it inherits.
_BASES = (
    (nn.Module, "module"),
    (torch.Tensor, "other tensor"),
    (enum.Enum, "enum"),
    (pathlib.PurePath, "path"),
    (type, "class"),
    (types.ModuleType, "python module"),
    (logging.Logger, "logger"),
    (functools.partial, "partial"),
    (list, "other list"),
    (tuple, "other tuple"),
    (dict, "other dict"),
    (set, "other set"),
    (frozenset, "other set"),
    (int, "other scalar"),
    (float, "other scalar"),
    (complex, "other scalar"),
    (str, "other scalar"),
    (bytes, "other scalar"),
    (bytearray, "other scalar"),
)
# The members of a function and of a partial that are described on their own.
_FUNCTION_MEMBERS = frozenset(
    name
    for name, entry in types.FunctionType.__dict__.items()
    if type(entry) is types.MemberDescriptorType
)
_PARTIAL_MEMBERS = frozenset(
    name
    for name, entry in _PARTIAL.items()
    if type(entry) is types.MemberDescriptorType
)


class ValidationError(RuntimeError):
    """A model is not the owner's raw load, or cannot be described."""


class _Attestation(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    facts: dict[str, Any]  # by path: what must be equal
    allocations: dict[str, tuple[int, bytes]]  # size and digest
    leaves: dict[str, bytes]  # by path: a digest of each plain value held now
    changed: dict[str, str]  # the plain values the load changed, for the report
    expert_map: Optional[dict[str, Any]]


class _Constructed:
    def __init__(self, leaves: dict[str, bytes], opaque: dict[str, Any]):
        self.leaves, self.opaque = leaves, opaque


class _Unrecorded:
    def __init__(self, why: str):
        self.why = why


class _Unsupported(Exception):
    """What the walk met and cannot describe. The text follows the path."""


def record_construction(model: nn.Module) -> Any:
    """What ``model`` holds right after construction, by the validator's own count.

    Never raises: a model it cannot record is refused by attest and validate.
    """
    try:
        walk = _Walk(published=None)
        walk.visit(model, "")
        if walk.problems:
            return _Unrecorded(_cut("; ".join(walk.problems[:3]), 400))
        return _Constructed(walk.leaves, walk.opaque)
    except BaseException as e:
        if not _isa(e, Exception):
            raise
        return _Unrecorded(_why(e))


def attest(
    *, model: nn.Module, storages: dict[str, torch.Tensor], constructed: Any
) -> bytes:
    """Owner: describe the raw-loaded model and the storages it publishes."""
    with _refusing("This model cannot be described"):
        described, _ = _describe(
            model=model, storages=storages, constructed=constructed
        )
        packed = msgspec.msgpack.encode(described)
        return bytes([_VERSION]) + zlib.compress(packed, 1)


def validate(
    *,
    model: nn.Module,
    storages: dict[str, torch.Tensor],
    attestation: bytes,
    constructed: Any,
) -> None:
    """Reader: raise unless the installed model matches the owner's attestation."""
    theirs = _read(attestation)
    with _refusing("This model cannot be compared with the owner's"):
        mine, walk = _describe(model=model, storages=storages, constructed=constructed)
        lines = _differences(theirs, mine, walk)
    if lines.count:
        raise ValidationError(
            lines.report("The installed model is not the owner's raw load")
        )


def _differences(theirs: _Attestation, mine: _Attestation, walk: _Walk) -> _Lines:
    lines = _Lines()
    _compare_facts(lines, "", theirs.facts, mine.facts)
    for label in sorted(theirs.allocations.keys() | mine.allocations.keys()):
        owner, here = theirs.allocations.get(label), mine.allocations.get(label)
        kind, _, name = label.partition(":")
        what = f"storage {name}" if kind == "published" else f"own allocation {name}"
        if label in walk.users:
            what += f", first used by {walk.users[label]}"
        if here is None:
            lines.add(f"{what} is only on the owner")
        elif owner is None:
            lines.add(f"{what} is only here")
        elif owner[0] != here[0]:
            lines.add(f"{what}: {owner[0]} bytes on the owner, {here[0]} here")
        elif owner[1] != here[1]:
            lines.add(f"{what}: its bytes differ from the owner's")
    for path in sorted(theirs.changed.keys() | mine.changed.keys()):
        if theirs.leaves.get(path) == mine.leaves.get(path):
            continue
        here = _shown(walk.leaves.get(path))
        if path in theirs.changed:
            was = theirs.changed[path]
            lines.add(f"{path}: the owner's load left {was}, here it is {here}")
        else:
            lines.add(f"{path} changed only here, to {here}")
    if (theirs.expert_map is None) != (mine.expert_map is None):
        lines.add("only one of the owner and this rank has an expert map")
    elif theirs.expert_map is not None:
        _compare_facts(lines, "expert map ", theirs.expert_map, mine.expert_map)
    return lines


@contextlib.contextmanager
def _refusing(what: str) -> Iterator[None]:
    """Whatever goes wrong inside is a refusal."""
    try:
        yield
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"{what}: {_why(e)}") from e


def _mro(cls: type) -> tuple:
    return _TYPE["__mro__"].__get__(cls)


def _isa(value: Any, classes: Any) -> bool:
    """isinstance by the real type, without asking the object or its class."""
    wanted = classes if type(classes) is tuple else (classes,)
    return any(base is cls for base in _mro(type(value)) for cls in wanted)


def _text(value: Any) -> str:
    """A name as text that can be sent."""
    if not _isa(value, str):
        return "?"
    return str.__str__(value).encode("utf-8", "backslashreplace").decode()


def _type_name(cls: type) -> str:
    try:
        module = _TYPE["__module__"].__get__(cls)
    except AttributeError:
        module = None
    return f"{_text(module)}.{_text(_TYPE['__qualname__'].__get__(cls))}"


def _cut(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def _why(error: BaseException) -> str:
    name = _text(_TYPE["__name__"].__get__(type(error)))
    try:
        return _cut(f"{name}: {error}", 400)
    except Exception:
        return name


class _Lines:
    """The differences: all counted, the first few kept."""

    def __init__(self):
        self.count, self.kept = 0, []

    def add(self, line: str) -> None:
        self.count += 1
        if len(self.kept) < _MAX_REPORTED:
            self.kept.append(_cut(line, 400))

    def report(self, head: str, what: str = "difference") -> str:
        more = self.count - len(self.kept)
        return (
            f"{head}, {self.count} {what}(s): "
            + "; ".join(self.kept)
            + (f"; and {more} more" if more else "")
        )


class _Budget:
    """What one walk may still look at, and what it learned about classes."""

    def __init__(self, final: bool = False):
        self.final = final  # a loaded model is described: nothing in it is on meta
        self.limit = self.left = _MAX_OBJECTS
        self.kinds: dict[int, str] = {}  # id of a class -> how it is described
        self.slots: dict[int, list] = {}  # id of a class -> the slots it declares

    def spend(self, count: int = 1) -> None:
        self.left -= count
        if self.left < 0:
            raise ValidationError(f"The model reaches more than {self.limit} values")


def _category(kind: type, budget: _Budget) -> str:
    """How values of a class are described, decided from its real bases."""
    category = _EXACT.get(id(kind)) or budget.kinds.get(id(kind))
    if category is None:
        mro = _mro(kind)
        category = "object"
        for base, name in _BASES:
            if any(cls is base for cls in mro):
                category = name
                break
        budget.kinds[id(kind)] = category
    return category


def _native(cls: type, slots: int) -> bool:
    """Whether instances keep state that Python cannot see: a class that was not
    made at run time, or more memory than their slots, their dict and their weak
    references take."""
    for base in _mro(cls):
        static = not _TYPE["__flags__"].__get__(base) & _HEAP
        if static and not any(base is bare for bare in _BARE):
            return True
    size = _HEADER + _POINTER * slots
    size += _POINTER * (_TYPE["__dictoffset__"].__get__(cls) > 0)
    size += _POINTER * (_TYPE["__weakrefoffset__"].__get__(cls) > 0)
    return (
        _TYPE["__itemsize__"].__get__(cls) != 0
        or _TYPE["__basicsize__"].__get__(cls) > size
    )


def _instance_dict(value: Any) -> Optional[dict]:
    """The instance dict of an object, from the slot that holds it."""
    kind = type(value)
    for cls in _mro(kind):
        slot = _TYPE["__dict__"].__get__(cls).get("__dict__", _NO)
        if slot is _NO:
            continue
        # A property of that name would run its code, and another slot under
        # that name would hide the fields.
        made_in_c = (
            type(slot) is types.GetSetDescriptorType
            or type(slot) is types.MemberDescriptorType
        )
        if made_in_c and slot.__name__ == "__dict__" and slot.__objclass__ is cls:
            held = slot.__get__(value, cls)
            if held is None or _isa(held, dict):
                return held
        raise _Unsupported("cannot be read without running its code")
    if _TYPE["__dictoffset__"].__get__(kind) != 0:
        raise _Unsupported("cannot be read without running its code")
    return None


def _slots(
    value: Any, budget: _Budget, skip: frozenset
) -> tuple[int, list[tuple[str, Any]]]:
    """How many slots the classes of an object declare, and the ones that hold
    something. A name that several classes declare is told apart by the place of
    the class among the bases."""
    kind = type(value)
    declared = budget.slots.get(id(kind))
    if declared is None:
        found = []
        for place, cls in enumerate(_mro(kind)):
            entries = _TYPE["__dict__"].__get__(cls)
            budget.spend(len(entries))
            found.extend(
                (name, place, cls, entry)
                for name, entry in entries.items()
                if type(entry) is types.MemberDescriptorType
                and type(name) is str
                and name not in ("__dict__", "__weakref__")
            )
        names = [name for name, _, _, _ in found]
        declared = budget.slots[id(kind)] = [
            (name if names.count(name) == 1 else f"{place}:{name}", name, cls, entry)
            for name, place, cls, entry in found
        ]
    held = []
    for label, name, cls, entry in declared:
        if name in skip:
            continue
        try:
            held.append((label, entry.__get__(value, cls)))
        except AttributeError:  # a slot that holds nothing
            continue
    return len(declared), held


def _fields(
    value: Any,
    budget: _Budget,
    skip: frozenset = frozenset(),
    described: bool = False,
) -> tuple[list[tuple[str, Any]], Optional[dict], bool]:
    """The slots of an object and its instance dict, read without running its
    code, and whether they are all the state it keeps. They are not for an
    object with state where Python cannot see it, unless that state is
    ``described`` in another way."""
    declared, slots = _slots(value, budget, skip)
    if not described and _native(type(value), declared):
        return [], None, False
    held = _instance_dict(value)
    if held is not None:
        budget.spend(dict.__len__(held))
        for name in dict.__iter__(held):
            if type(name) is not str:
                raise _Unsupported("has a field whose name is not a str")
    return slots, held, held is not None or declared > 0


def _members(value: Any, budget: _Budget) -> list:
    """The members of a list, tuple or set, by the built-in class's own iteration."""
    for base in (list, tuple, set, frozenset):
        if _isa(value, base):
            budget.spend(base.__len__(value))
            return list(base.__iter__(value))
    raise TypeError("not a list, tuple or set")


def _entries(value: dict, budget: _Budget) -> list[tuple[Any, Any]]:
    """The entries of a dict in the order of insertion, by the built-in class's
    own iteration: an ordered dict's own would hash its keys."""
    budget.spend(dict.__len__(value))
    return list(dict.items(value))


def _kind(value: Any, category: str) -> str:
    exact = category in ("list", "tuple", "dict", "set", "frozenset")
    return category if exact else _type_name(type(value))


def _base_value(value: Any, depth: int, budget: _Budget) -> Any:
    """What a subclass of a number or a string holds, read by the built-in class."""
    if _isa(value, int):
        return _atom(int.__int__(value), depth, budget)
    if _isa(value, float):
        return _atom(float.__float__(value), depth, budget)
    if _isa(value, complex):
        parts = complex.real.__get__(value), complex.imag.__get__(value)
        return _atom(complex(*parts), depth, budget)
    if _isa(value, str):
        return _atom(str.__str__(value), depth, budget)
    if _isa(value, bytearray):
        return _atom(bytearray.__getitem__(value, slice(None)), depth, budget)
    return _atom(bytes.__getitem__(value, slice(None)), depth, budget)


def _atom(value: Any, depth: int, budget: _Budget) -> Any:
    """A value without parts, as data that is equal in two processes that hold
    the same value; _NO for anything else."""
    category = _category(type(value), budget)
    if category in ("none", "bool", "bytes"):
        return value
    if category == "int":
        if -(2**63) <= value < 2**64:
            return value
        size = value.bit_length() // 8 + 1
        return ["int", value.to_bytes(size, "little", signed=True)]
    if category == "str":
        try:
            value.encode()
        except UnicodeEncodeError:
            return ["str", value.encode("utf-8", "surrogatepass")]
        return value
    if category == "float":  # by its bits: a NaN equals itself, and has a sign
        return ["float", struct.pack("<d", value)]
    if category == "complex":
        return ["complex", struct.pack("<dd", value.real, value.imag)]
    if category == "bytearray":
        return ["bytearray", bytes(value)]
    if category == "ellipsis":
        return ["ellipsis"]
    if category == "range":
        parts = (value.start, value.stop, value.step)
        return ["range", [_atom(part, depth, budget) for part in parts]]
    if category == "device":
        if value.type == "meta" and budget.final:
            raise _Unsupported("holds the meta device")
        return ["device", value.type]  # the index is this process's own
    if category == "torch":
        return ["torch", str(value)]
    if category == "size":
        budget.spend(len(value))
        return ["size", [_atom(n, depth, budget) for n in tuple.__iter__(value)]]
    if category == "ndarray":
        dtype = value.dtype
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            return _NO  # naming its fields could run their code
        digest = blake3.blake3(np.ascontiguousarray(value).tobytes()).digest()
        return ["ndarray", dtype.str, list(value.shape), digest]
    if category == "array":
        return ["array", value.typecode, blake3.blake3(value.tobytes()).digest()]
    if category == "numpy scalar":
        return ["numpy", value.dtype.str, value.tobytes()]
    if category == "enum":
        # A member with fields of its own, or with a value that is not plain,
        # is described as an object.
        held = _instance_dict(value)
        if held is None or dict.__len__(held) != len(_ENUM_FIELDS):
            return _NO
        if any(
            type(k) is not str or k not in _ENUM_FIELDS for k in dict.__iter__(held)
        ):
            return _NO
        name = dict.get(held, "_name_")
        member = _plain(dict.get(held, "_value_"), depth + 1, budget)
        if type(name) is not str or member is _NO:
            return _NO
        name = _atom(name, depth, budget)
        return ["enum", _type_name(type(value)), name, member]
    if category == "class":
        return ["class", _type_name(value)]
    if category == "python module":
        name = dict.get(_MODULE_DICT.__get__(value), "__name__")
        return ["python module", _text(name)]
    return _NO


def _plain(value: Any, depth: int = 0, budget: Optional[_Budget] = None) -> Any:
    """A value made of atoms and of lists, tuples, dicts, sets and slices of
    them, as data; _NO for anything else."""
    budget = budget or _Budget()
    budget.spend()
    encoded = _atom(value, depth, budget)
    if encoded is not _NO:
        return encoded
    category = _category(type(value), budget)
    if category not in ("list", "tuple", "dict", "set", "frozenset", "slice"):
        return _NO
    if depth >= _MAX_DEPTH:
        raise _Unsupported(f"is nested deeper than {_MAX_DEPTH}")
    if category == "slice":
        parts = (value.start, value.stop, value.step)
        parts = [_plain(part, depth + 1, budget) for part in parts]
        return _NO if any(part is _NO for part in parts) else ["slice", parts]
    budget.spend(len(value))
    items = []
    if category == "dict":
        for key, item in value.items():
            key = _plain(key, depth + 1, budget)
            item = _plain(item, depth + 1, budget)
            if key is _NO or item is _NO:
                return _NO
            items.append([key, item])
        return [category, items]  # in its order: code may depend on it
    for item in value:
        items.append(_plain(item, depth + 1, budget))
        if items[-1] is _NO:
            return _NO
    if category in ("set", "frozenset"):
        items.sort(key=msgspec.msgpack.encode)
    return [category, items]


def _segment(name: str) -> str:
    """A name as a part of a path, unlike every other name and every nesting."""
    return name if _WORD.fullmatch(name) else f"<{name!r}>"


def _code(code: types.CodeType, depth: int, budget: _Budget) -> list:
    """What a function runs and how it takes its arguments, without where its
    file is."""
    tables = (
        code.co_consts,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
    )
    budget.spend(1 + sum(len(table) for table in tables))
    if depth >= _MAX_DEPTH:
        raise _Unsupported(f"is nested deeper than {_MAX_DEPTH}")
    consts = []
    for const in code.co_consts:
        if type(const) is types.CodeType:
            consts.append(_code(const, depth + 1, budget))
            continue
        consts.append(_plain(const, depth + 1, budget))
        if consts[-1] is _NO:
            raise _Unsupported("has code with a constant that is not a plain value")
    return [
        code.co_code,
        *(list(table) for table in tables[1:]),
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        getattr(code, "co_exceptiontable", b""),
        consts,
    ]


class _Walk:
    """One pass over what a model reaches.

    ``facts`` must be equal on owner and reader; ``leaves`` are plain values,
    compared where a load changed them; ``opaque`` is what must only be the
    same object. ``published`` maps an allocation to the storages published
    from it; None makes a construction record, which locates no tensor.
    ``strict`` turns every plain value into a fact.
    """

    def __init__(self, *, published: Optional[dict], strict: bool = False):
        self.published, self.strict = published, strict
        self.facts: dict[str, Any] = {}
        self.leaves: dict[str, bytes] = {}
        self.opaque: dict[str, Any] = {}
        self.allocations: dict[str, torch.Tensor] = {}
        self.users: dict[str, str] = {}  # an own allocation -> its first tensor
        self.spans: list[tuple[str, int, int, str]] = []  # device, bytes, what
        self.problems: list[str] = []
        self.budget = _Budget(final=published is not None)
        self._first: dict[int, str] = {}  # id of an object -> its first path
        self._alive: list[Any] = []  # so that no id is reused during the walk
        self._own: dict[tuple[str, int], tuple[int, int]] = {}

    def _put(self, table: dict, path: str, value: Any) -> None:
        if path in self.facts or path in self.leaves:
            self.problems.append(f"two places in the model have the path {path!r}")
        else:
            table[path] = value

    def _leaf(self, path: str, encoded: Any) -> None:
        packed = msgspec.msgpack.encode(encoded)
        if len(packed) > _MAX_LEAF:
            digest = blake3.blake3(packed).digest(16)
            encoded = ["digest", len(packed), digest]
            packed = _DIGESTED + msgspec.msgpack.encode(encoded[1:])
        if self.strict:
            self._put(self.facts, path, ["value", encoded])
        else:
            self._put(self.leaves, path, packed)

    def _seen(self, value: Any, path: str, tag: str) -> bool:
        """Whether an object was met before. Then this path is a link to the
        first one."""
        first = self._first.get(id(value))
        if first is not None:
            self._put(self.facts, path, [tag, first])
            return True
        self._first[id(value)] = path
        self._alive.append(value)
        return False

    def visit(self, value: Any, path: str, depth: int = 0) -> None:
        try:
            encoded = _plain(value, depth, self.budget)
            category = _category(type(value), self.budget)
            plain = encoded is not _NO
            split = category in ("list", "dict") and 0 < len(value) <= _MAX_SPLIT
            if plain and not split:
                self._leaf(path, encoded)
                return
            # A plain container is a value: it has no identity here.
            if not plain and self._seen(value, path, "same as"):
                return
            if depth >= _MAX_DEPTH:
                raise _Unsupported(f"is nested deeper than {_MAX_DEPTH}")
            self._describe(value, category, path, depth + 1, plain)
        except ValidationError:
            raise
        except _Unsupported as e:
            self.problems.append(f"{path or '<root>'} {e}")
        except Exception as e:
            name = _text(_TYPE["__name__"].__get__(type(e)))
            self.problems.append(f"{path or '<root>'} cannot be described: {name}")

    def _shape(self, path: str, shape: list, plain: bool) -> None:
        if plain:
            self._leaf(path, shape)
        else:
            self._put(self.facts, path, shape)

    def _visit_fields(
        self, slots: list, held: Optional[dict], path: str, depth: int
    ) -> None:
        """The slots of an object, and its instance dict, which is one object
        too: two objects may share it."""
        fields = list(slots)
        if held is not None:
            if not self._seen(held, f"{path}::<fields>", "same fields as"):
                fields.extend(dict.items(held))
        for name, item in sorted(fields, key=lambda field: _segment(field[0])):
            self.visit(item, f"{path}::{_segment(name)}", depth)

    def _held(
        self, value: Any, path: str, depth: int, skip: frozenset = frozenset()
    ) -> None:
        """The fields of a value that is described by more than its fields."""
        slots, held, _ = _fields(value, self.budget, skip, described=True)
        self._visit_fields(slots, held, path, depth)

    def _describe(
        self, value: Any, category: str, path: str, depth: int, plain: bool
    ) -> None:
        budget = self.budget
        if category == "module":
            self._module(value, path, depth)
        elif category == "tensor":
            self._tensor(value, path, depth)
        elif category == "other tensor":
            if budget.final:
                name = _type_name(type(value))
                raise _Unsupported(f"is a {name}, a tensor class that is not described")
            self._held(value, path, depth)
        elif category in ("list", "tuple", "other list", "other tuple"):
            members = _members(value, budget)
            self._shape(path, [_kind(value, category), len(members)], plain)
            for i, item in enumerate(members):
                self.visit(item, f"{path}[{i}]", depth)
            if category.startswith("other "):
                self._held(value, path, depth)
        elif category in ("dict", "ordered dict", "other dict"):
            entries = _entries(value, budget)
            keys = [_plain(key, depth, budget) for key, _ in entries]
            if any(key is _NO for key in keys):
                raise _Unsupported("holds a dict with a key that is not a plain value")
            self._shape(path, [_kind(value, category), keys], plain)
            for key, (_, item) in zip(keys, entries):
                self.visit(item, f"{path}[{key!r}]", depth)
            if category != "dict":
                self._held(value, path, depth)
        elif category in ("set", "frozenset", "other set"):
            items = [_plain(item, depth, budget) for item in _members(value, budget)]
            if any(item is _NO for item in items):
                raise _Unsupported(
                    "holds a set with a member that is not a plain value"
                )
            items.sort(key=msgspec.msgpack.encode)
            self._put(self.facts, path, [_kind(value, category), items])
            self._held(value, path, depth)
        elif category == "slice":
            self._put(self.facts, path, ["slice"])
            self.visit(value.start, f"{path}::<start>", depth)
            self.visit(value.stop, f"{path}::<stop>", depth)
            self.visit(value.step, f"{path}::<step>", depth)
        elif category == "function":
            self._function(value, path, depth)
        elif category == "builtin":
            # Its qualified name would be asked of the object it is bound to.
            bound = value.__self__
            free = bound is None or _isa(bound, types.ModuleType)
            receiver = None if free else _type_name(type(bound))
            module, name = _text(value.__module__), _text(value.__name__)
            self._put(self.facts, path, ["builtin", module, receiver, name])
            if not free:
                self.visit(bound, f"{path}::<self>", depth)
        elif category == "method":
            self._put(self.facts, path, ["method"])
            self.visit(value.__self__, f"{path}::<self>", depth)
            self.visit(value.__func__, f"{path}::<function>", depth)
        elif category == "partial":
            self._put(self.facts, path, ["partial", _type_name(type(value))])
            self.visit(_PARTIAL["func"].__get__(value), f"{path}::<function>", depth)
            self.visit(_PARTIAL["args"].__get__(value), f"{path}::<args>", depth)
            keywords = _PARTIAL["keywords"].__get__(value)
            self.visit(keywords, f"{path}::<keywords>", depth)
            self._held(value, path, depth, _PARTIAL_MEMBERS)
        elif category == "code":
            code = msgspec.msgpack.encode(_code(value, depth, budget))
            self._put(self.facts, path, ["code", blake3.blake3(code).digest(8)])
        elif category == "other scalar":
            self._put(self.facts, path, ["object", _type_name(type(value))])
            self._leaf(f"{path}::<value>", _base_value(value, depth, budget))
            self._held(value, path, depth)
        else:
            whole = False
            if category not in ("logger", "path"):
                slots, held, whole = _fields(value, budget)
            if not whole:
                self._put(self.facts, path, ["opaque", _type_name(type(value))])
                self.opaque[path] = value
                return
            self._put(self.facts, path, ["object", _type_name(type(value))])
            self._visit_fields(slots, held, path, depth)

    def _function(self, function: types.FunctionType, path: str, depth: int) -> None:
        code = msgspec.msgpack.encode(_code(function.__code__, depth, self.budget))
        module, name = _text(function.__module__), _text(function.__qualname__)
        digest = blake3.blake3(code).digest(8)
        self._put(self.facts, path, ["function", module, name, digest])
        for i, cell in enumerate(function.__closure__ or ()):
            where = f"{path}::<closure {i}>"
            if self._seen(cell, where, "same cell as"):
                continue  # two functions that share a cell see each other's writes
            try:
                held = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            self.visit(held, where, depth)
        self.visit(function.__defaults__, f"{path}::<defaults>", depth)
        self.visit(function.__kwdefaults__, f"{path}::<kwdefaults>", depth)
        self._held(function, path, depth, _FUNCTION_MEMBERS)

    def _table(self, table: Any, where: str) -> Optional[list]:
        """The entries of a registry of a module; None if another module's
        registry is this very dict. A dict of another class could answer for
        its entries, so it is no registry."""
        if _category(type(table), self.budget) not in ("dict", "ordered dict"):
            raise _Unsupported(f"has no plain dict at {where!r}")
        if self._seen(table, where, "same table as"):
            return None
        return _entries(table, self.budget)

    def _module(self, module: nn.Module, path: str, depth: int) -> None:
        slots, held, whole = _fields(module, self.budget)
        if not whole or held is None:
            raise _Unsupported("is a module with state that its fields do not show")
        if self._seen(held, f"{path}::<fields>", "same fields as"):
            raise _Unsupported("is a module that shares its fields with another")
        fields = dict(slots)
        fields.update(dict.items(held))
        if len(fields) != len(slots) + dict.__len__(held):
            raise _Unsupported("is a module with two fields of one name")
        mode = fields.get("training")
        mode = mode if type(mode) is bool else None
        self._put(self.facts, path, ["module", _type_name(type(module)), mode])
        registered = set()
        for registry in _REGISTRIES:
            where = f"{path}<{registry[1:]}>"
            entries = self._table(fields.get(registry), where)
            if not entries:
                continue
            if any(type(name) is not str for name, _ in entries):
                raise _Unsupported(f"has {registry[1:]} with a name that is not a str")
            order = [[_segment(name), entry is not None] for name, entry in entries]
            self._put(self.facts, where, order)
            for name, entry in entries:
                registered.add(name)
                if entry is not None:
                    name = _segment(name)
                    self.visit(entry, f"{path}.{name}" if path else name, depth)
        for name in sorted(fields, key=_segment):
            if name in _REGISTRIES or (name == "training" and mode is not None):
                continue
            value, where = fields[name], f"{path}::{_segment(name)}"
            if name in registered:
                self.problems.append(
                    f"{where} shadows a registered parameter, buffer or module"
                )
            elif name in _HOOK_REGISTRIES:
                # In call order: the keys are handle ids of this process.
                hooks = self._table(value, where) or []
                if hooks:
                    self._put(self.facts, where, ["hooks", len(hooks)])
                for i, (_, hook) in enumerate(hooks):
                    self.visit(hook, f"{where}[{i}]", depth)
            elif name in _HOOK_FLAGS:
                flags = self._table(value, where) or []
                registry = fields.get(_HOOK_FLAGS[name])
                order = _entries(registry, self.budget) if _isa(registry, dict) else []
                order = [key for key, _ in order]
                if any(type(key) is not int for key in order) or any(
                    type(key) is not int or type(flag) is not bool
                    for key, flag in flags
                ):
                    self.problems.append(f"{where} is not a table of hook options")
                elif flags:
                    flags = [
                        [order.index(key) if key in order else -1, flag]
                        for key, flag in flags
                    ]
                    self._put(self.facts, where, ["hook flags", flags])
            else:
                self.visit(value, where, depth)

    def _tensor(self, tensor: torch.Tensor, path: str, depth: int) -> None:
        if not self.budget.final:
            self._put(self.facts, path, ["tensor"])
        else:
            if tensor.layout is not torch.strided or tensor.is_quantized:
                raise _Unsupported("is not a plain dense tensor")
            if _T.is_conj(tensor) or _T.is_neg(tensor):
                raise _Unsupported("is a lazy conjugate or negative view")
            if tensor.device.type == "meta":
                raise _Unsupported("has no data: it is on the meta device")
            allocation, offset = self._locate(tensor, path)
            self._put(
                self.facts,
                path,
                [
                    "tensor",
                    _type_name(type(tensor)),
                    str(tensor.dtype),
                    list(tensor.shape),
                    list(_T.stride(tensor)),
                    tensor.device.type,
                    tensor.requires_grad,
                    allocation,
                    offset,
                ],
            )
        self._held(tensor, path, depth)

    def _locate(self, tensor: torch.Tensor, path: str) -> tuple[list, int]:
        """The allocation a tensor is a view of, and its first byte there."""
        if _T.numel(tensor) == 0:
            return ["none"], 0  # it reads no byte, wherever it sits
        storage = _T.untyped_storage(tensor)
        size = _T.element_size(tensor)
        first = _T.storage_offset(tensor) * size
        reach = sum((n - 1) * s for n, s in zip(tensor.shape, _T.stride(tensor)))
        end = first + (reach + 1) * size
        nbytes = _S.nbytes(storage)
        if end > nbytes:
            raise _Unsupported("reaches beyond its allocation")
        device, base = str(tensor.device), _S.data_ptr(storage)
        if (device, base) in self.published:
            for name, start, stop in self.published[device, base]:
                if start <= first and end <= stop:
                    return ["published", name], first - start
            raise _Unsupported("is not inside a published storage of its allocation")
        index, known = self._own.get((device, base), (len(self._own), nbytes))
        label = f"own:{index}"
        if (device, base) not in self._own:
            self._own[device, base] = (index, nbytes)
            whole = torch.empty(0, dtype=torch.uint8, device=tensor.device)
            self.allocations[label] = _T.set_(whole, storage)
            self.users[label] = path
            self.spans.append(
                (device, base, base + nbytes, f"the allocation of {path}")
            )
        elif known != nbytes:
            raise _Unsupported(
                f"and {self.users[label]} start at one address in allocations of "
                "different sizes"
            )
        return ["own", index], first


def _published(
    storages: dict[str, torch.Tensor], walk: _Walk
) -> dict[tuple[str, int], list[tuple[str, int, int]]]:
    """Each allocation's published byte ranges: (name, first byte, end)."""
    ranges: dict[tuple[str, int], list[tuple[str, int, int]]] = {}
    for name, storage in dict.items(storages):
        if type(name) is not str:
            walk.problems.append("a storage has a name that is not a str")
            continue
        try:
            name.encode()
        except UnicodeEncodeError:
            walk.problems.append(f"storage {name!r} has a name that cannot be sent")
            continue
        if not (
            _category(type(storage), walk.budget) == "tensor"
            and storage.layout is torch.strided
            and storage.dtype is torch.uint8
            and _T.dim(storage) == 1
            and (_T.numel(storage) < 2 or _T.stride(storage)[0] == 1)
            and storage.device.type != "meta"
            and not _T.is_conj(storage)
            and not _T.is_neg(storage)
        ):
            walk.problems.append(f"storage {name!r} is not a 1-D uint8 tensor of bytes")
            continue
        if _T.numel(storage) == 0:
            continue  # a name and no byte: it sits nowhere
        backing = _T.untyped_storage(storage)
        first = _T.storage_offset(storage)
        end = first + _T.numel(storage)
        if end > _S.nbytes(backing):
            walk.problems.append(f"storage {name!r} reaches beyond its allocation")
            continue
        device, base = str(storage.device), _S.data_ptr(backing)
        ranges.setdefault((device, base), []).append((name, first, end))
        walk.spans.append((device, base + first, base + end, f"storage {name!r}"))
    return ranges


def _overlaps(spans: list[tuple[str, int, int, str]]) -> list[str]:
    """The pairs of allocations that share bytes of memory."""
    problems, last = [], None
    for span in sorted(spans):
        if last is not None and span[0] == last[0] and span[1] < last[2]:
            problems.append(f"{last[3]} and {span[3]} overlap in memory")
        if last is None or span[0] != last[0] or span[2] > last[2]:
            last = span
    return problems


def _digests(allocations: dict[str, torch.Tensor]) -> dict[str, bytes]:
    """BLAKE3 of each 1-D uint8 tensor. CUDA tensors are hashed on their device.
    Host bytes are hashed from copies: a storage that NumPy has been shown can
    no longer be resized."""
    digests, on_cuda = {}, {}
    for label, data in allocations.items():
        data = _T.detach(data)
        size = _T.numel(data)
        if size == 0:
            digests[label] = blake3.blake3(b"").digest()
        elif data.is_cuda:
            on_cuda.setdefault(str(data.device), []).append((label, data))
        else:
            threads = blake3.blake3.AUTO if size >= 1 << 20 else 1
            hasher = blake3.blake3(max_threads=threads)
            for start in range(0, size, _CHUNK):
                chunk = _T.narrow(data, 0, start, min(_CHUNK, size - start))
                hasher.update(_T.numpy(_T.clone(_T.cpu(chunk))))
            digests[label] = hasher.digest()
    if on_cuda:
        from sglang.kernels.ops.memory.gpu_tensor_hash_blake3 import gpu_tensors_blake3

        for batch in on_cuda.values():
            hashed = gpu_tensors_blake3([data for _, data in batch])
            digests.update((label, d) for (label, _), d in zip(batch, hashed))
    return digests


def _expert_map() -> Optional[dict[str, Any]]:
    """Every field of this process's expert map, tensors with their bytes."""
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return None
    walk = _Walk(published={}, strict=True)
    walk.visit(metadata, "")
    problems = walk.problems + _overlaps(walk.spans)
    problems += [f"{path} is not described" for path in walk.opaque]
    if problems:
        raise ValidationError(
            f"The expert map cannot be described: {_cut('; '.join(problems), 400)}"
        )
    digests = _digests(walk.allocations)
    facts = dict(walk.facts)
    for label, data in walk.allocations.items():
        facts[f"<{label}>"] = [_T.numel(data), digests[label]]
    return facts


def _describe(
    *, model: nn.Module, storages: dict[str, torch.Tensor], constructed: Any
) -> tuple[_Attestation, _Walk]:
    """The model as an attestation, and the walk that made it."""
    if _isa(constructed, _Unrecorded):
        raise ValidationError(
            f"The construction of this model was not recorded: {constructed.why}"
        )
    if not _isa(constructed, _Constructed):
        raise ValidationError("The model has no construction record of the validator")
    walk = _Walk(published={})
    walk.published = _published(storages, walk)
    problems = walk.problems
    if not problems:
        for name, storage in dict.items(storages):
            walk.allocations[f"published:{name}"] = storage
        walk.visit(model, "")
        problems.extend(_overlaps(walk.spans))
        for path, held in walk.opaque.items():
            if constructed.opaque.get(path, _NO) is not held:
                problems.append(
                    f"{path} holds a {_type_name(type(held))} that is not described "
                    "and is not the object that construction left there"
                )
    if problems:
        lines = _Lines()
        for problem in problems:
            lines.add(problem)
        raise ValidationError(lines.report("This model cannot be described", "problem"))
    digests = _digests(walk.allocations)
    held = walk.leaves
    changed = sorted(
        path
        for path in constructed.leaves.keys() | held.keys()
        if constructed.leaves.get(path) != held.get(path)
    )
    described = _Attestation(
        facts=dict(sorted(walk.facts.items())),
        allocations={
            label: (_T.numel(data), digests[label])
            for label, data in sorted(walk.allocations.items())
        },
        leaves={
            path: blake3.blake3(value).digest(8) for path, value in sorted(held.items())
        },
        changed={path: _shown(held.get(path)) for path in changed},
        expert_map=_expert_map(),
    )
    return described, walk


def _read(attestation: bytes) -> _Attestation:
    try:
        data = bytes(attestation)
        if not data or data[0] != _VERSION:
            raise ValueError(
                f"it has version {data[:1].hex() or 'none'}, "
                f"this validator reads version {_VERSION:02x}"
            )
        inflate = zlib.decompressobj()
        packed = inflate.decompress(data[1:], _MAX_ATTESTATION)
        if inflate.unconsumed_tail or inflate.unused_data or not inflate.eof:
            raise ValueError("it is cut short, too large, or followed by more")
        return msgspec.msgpack.decode(packed, type=_Attestation)
    except Exception as e:
        raise ValidationError(f"Cannot read the owner's attestation: {_why(e)}") from e


def _pretty(encoded: Any) -> str:
    """A plain value as it is shown in a report."""
    if type(encoded) is not list:
        return repr(encoded)
    if encoded[:1] == ["float"] and len(encoded) == 2:
        bits = encoded[1]
        if type(bits) is bytes and len(bits) == 8:
            number = struct.unpack("<d", bits)[0]
            # Two NaNs print alike and may differ.
            return repr(number) if number == number else f"nan({bits.hex()})"
    return f"[{', '.join(_pretty(item) for item in encoded)}]"


def _shown(leaf: Optional[bytes]) -> str:
    if leaf is None:
        return "absent"
    if leaf.startswith(_DIGESTED):
        size, digest = msgspec.msgpack.decode(leaf[1:])
        return f"a value of {size} bytes with digest {digest.hex()[:16]}"
    return _cut(_pretty(msgspec.msgpack.decode(leaf)), 120)


def _compare_facts(lines: _Lines, what: str, owner: dict, reader: dict) -> None:
    for path in sorted(owner.keys() | reader.keys()):
        ours, here = owner.get(path, _NO), reader.get(path, _NO)
        both = ours is not _NO and here is not _NO
        # By their encoding: 1 is not True.
        if both and msgspec.msgpack.encode(ours) == msgspec.msgpack.encode(here):
            continue
        where = f"{what}{path or '<root>'}"
        if here is _NO:
            lines.add(f"{where} is only on the owner: {_cut(repr(ours))}")
            continue
        if ours is _NO:
            lines.add(f"{where} is only here: {_cut(repr(here))}")
            continue
        parts = []
        if (
            type(ours) is list
            and type(here) is list
            and len(ours) == len(here) == len(_TENSOR_FIELDS) + 1
            and ours[0] == here[0] == "tensor"
        ):
            parts = [
                f"{field} is {a!r} on the owner, {b!r} here"
                for field, a, b in zip(_TENSOR_FIELDS, ours[1:], here[1:])
                if a != b
            ]
        if parts:
            lines.add(f"{where}: {', '.join(parts)}")
        else:
            lines.add(
                f"{where}: {_cut(repr(ours))} on the owner, {_cut(repr(here))} here"
            )
