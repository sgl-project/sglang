# SPDX-License-Identifier: Apache-2.0
"""The load-diff format: what a loaded model holds, as data that a reader installs.

D1 of the snapshot-loading roadmap (#50943). An owner loads a model; a reader
builds the same model on the meta device and installs the owner's snapshot
instead of loading it. This module defines that snapshot (the ``load_diff``
bytes of a SnapshotManifest) and checks it; load_diff.py makes and installs it.

- Every tensor the model holds is a view of a storage: one allocation,
  numbered by first use, independent of how a transport moves it. Views keep
  their byte offsets, dtypes, shapes and strides, so aliases survive. A
  transport moves whole storages, on the model's device or the CPU, and
  nothing else.
- Every plain value that loading changed since construction is encoded as
  data. Decoding never imports, never runs code of the model or of a library,
  and never creates an instance of a class named in the snapshot. Besides
  data, an encoding may name an object the reader already has: one of its
  modules, an object at a path in its model, or a member of an enum its
  process has already imported.

``walk_model`` decides where to look, the same way for owner and reader: the
registered parameters and buffers of every module, then its plain attributes
and what they hold: tensors (with their own Python attributes), tuples, lists
and dicts, and the ``__dict__`` of other objects. A container is visited
along every path that reaches it, so that every place that holds a tensor is
found; another object is visited once, at its first path. An object bound in
a Python module's namespace is looked at, not into: it is not the model's.
Anything else is a leaf, compared by identity.

A storage's bytes are checked with a BLAKE3-256 digest, computed on the GPU
with the kernel the multimodal cache uses. A reader that rebuilds, from the
tensors it actually installed, the same storage of each tensor at the same
byte offset with the same dtype, shape and stride, on storages whose digests
match the owner's, holds the same bytes in every tensor.
"""

from __future__ import annotations

import enum
import functools
import hashlib
import logging
import struct
import sys
import types
from collections import OrderedDict, defaultdict
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import msgspec
import torch
from torch import nn

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# nn.Module's own bookkeeping (_parameters, _buffers, _modules, hooks, ...):
# never a module's plain attribute.
MODULE_FIELDS = frozenset(vars(nn.Module()))

# How deep the walk goes into containers and objects, and how many values it
# looks at in all; a model past either is refused.
MAX_DEPTH = 64
MAX_NODES = 2_000_000
# A bytes value in a plain attribute, at most.
MAX_BYTES_VALUE = 1 << 20
# Items of a container that count, at most, as a module-level object's state
# or as module-level themselves.
_MAX_ITEMS = 10_000
_INT64_MAX = (1 << 63) - 1

_ATOMS = (type(None), bool, int, float, str, bytes)
_OPAQUE = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.BuiltinMethodType,
    functools.partial,
)
# dtype names a snapshot may use; looked up here, never on the torch module.
_DTYPES = {
    str(value).removeprefix("torch."): value
    for value in vars(torch).values()
    if type(value) is torch.dtype
}
# The factories a defaultdict in a snapshot may have.
FACTORIES = {f.__name__: f for f in (list, dict, set, int, float, str, tuple)}


class SnapshotError(RuntimeError):
    """A snapshot that cannot be made or installed; ``problems`` lists why,
    one line per path."""

    def __init__(self, what: str, problems: Sequence[str]):
        self.problems = list(problems)
        shown = "\n  ".join(self.problems[:20])
        more = len(self.problems) - 20
        tail = f"\n  ... and {more} more" if more > 0 else ""
        super().__init__(f"{what} ({len(self.problems)} problem(s)):\n  {shown}{tail}")


class Unencodable(ValueError):
    """A value that is not data and names nothing a reader has."""


def parse_dtype(name: Any) -> torch.dtype:
    if type(name) is not str or name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}")
    return _DTYPES[name]


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def view_nbytes(shape, stride, storage_offset: int, itemsize: int) -> int:
    """Bytes a view needs from the start of its storage (0 for an empty
    view, as torch's own bounds check)."""
    if any(size == 0 for size in shape):
        return 0
    extent = 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    return (storage_offset + extent) * itemsize


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#
# A path is a tuple: the name of a module of the model ("" for the model
# itself), then steps. A step is ("p", name) for a parameter or buffer of
# that module, ("a", name) for an entry of an object's __dict__ (for a module,
# a plain attribute), ("k", key) for a dict item and ("i", index) for a tuple
# or list item.


def path_str(path: tuple) -> str:
    out = path[0] or "<model>"
    for kind, key in path[1:]:
        out += f".{key}" if kind in ("p", "a") else f"[{key!r}]"
    return out


def resolve_path(modules: Dict[str, nn.Module], path: tuple) -> Any:
    """What the model whose modules (by every name) are ``modules`` holds at
    ``path``. Reads module registries, ``__dict__`` entries and the items of
    builtin containers only: no attribute lookup, property or item access of
    the model's own classes runs. Raises KeyError, IndexError or TypeError."""
    obj = modules[path[0]]
    for kind, key in path[1:]:
        obj = step_into(obj, kind, key)
    return obj


def step_into(obj: Any, kind: str, key: Any) -> Any:
    if kind == "p":
        if key in obj._parameters:
            return obj._parameters[key]
        return obj._buffers[key]
    if kind == "a":
        return vars_of(obj)[key]
    if kind in ("k", "i"):
        return _item(obj, key)
    raise KeyError(f"unknown step {kind!r}")


# A class's MRO and namespace, read through ``type`` itself: a metaclass of
# the model's cannot run code on these reads.
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_DICT = type.__dict__["__dict__"]


def vars_of(obj: Any) -> dict:
    """``obj.__dict__``, without running code of its class: only through the
    slot Python itself defines, never a ``__dict__`` the class overrides."""
    for klass in _TYPE_MRO.__get__(type(obj)):
        slot = _TYPE_DICT.__get__(klass).get("__dict__")
        if slot is not None:
            if type(slot) in (types.GetSetDescriptorType, types.MemberDescriptorType):
                d = slot.__get__(obj)
                if type(d) is dict:
                    return d
            break
    raise TypeError(f"{type(obj).__name__} has no plain __dict__")


def process_local_fields(kind: type) -> frozenset:
    """The attribute names ``kind`` or a base class declares in a class-level
    ``_process_local_fields``: tensors valid in one process only, such as
    scratch a kernel writes on every call. A snapshot carries their dtype and
    shape, never their bytes, and a reader makes its own, zeroed."""
    names: Set[str] = set()
    for klass in _TYPE_MRO.__get__(kind):
        declared = _TYPE_DICT.__get__(klass).get("_process_local_fields")
        if type(declared) in (frozenset, set, tuple):
            names.update(name for name in declared if type(name) is str)
    return frozenset(names)


def _has_dict(obj: Any) -> bool:
    try:
        vars_of(obj)
        return True
    except (TypeError, AttributeError):
        return False


def items_of(container: Any) -> List[Tuple[Any, Any]]:
    """(key or index, item) of a tuple, list or dict, as the builtin type (or
    OrderedDict) stores them."""
    kind = type(container)
    if issubclass(kind, OrderedDict):
        return list(OrderedDict.items(container))
    if issubclass(kind, dict):
        return list(dict.items(container))
    if issubclass(kind, list):
        return list(enumerate(list.__iter__(container)))
    if issubclass(kind, tuple):
        return list(enumerate(tuple.__iter__(container)))
    raise TypeError(f"{kind.__name__} is not a tuple, list or dict")


def _members(container: Any) -> List[Any]:
    """The items of a tuple, list, set or frozenset, or the values of a dict,
    as the builtin type stores them."""
    kind = type(container)
    if issubclass(kind, dict):
        return list(dict.values(container))
    for base in (list, tuple, set, frozenset):
        if issubclass(kind, base):
            return list(base.__iter__(container))
    raise TypeError(f"{kind.__name__} is not a builtin container")


def _item(container: Any, key: Any) -> Any:
    kind = type(container)
    if issubclass(kind, dict):
        if not dict.__contains__(container, key):
            raise KeyError(key)
        return dict.__getitem__(container, key)
    if issubclass(kind, list):
        return list.__getitem__(container, key)
    if issubclass(kind, tuple):
        return tuple.__getitem__(container, key)
    raise TypeError(f"{kind.__name__} is not a tuple, list or dict")


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


class ModuleLevel:
    """The objects bound in Python modules' namespaces, and the items of the
    builtin containers bound there. It holds them, so that no other object
    takes one of their ids while it lives."""

    def __init__(self):
        self._held: List[Any] = []
        self.ids: Set[int] = set()
        for module in list(sys.modules.values()):
            if not issubclass(type(module), types.ModuleType):
                continue
            for value in list(vars(module).values()):
                self._add(value)
                if _is_container(value) and _len(value) <= _MAX_ITEMS:
                    for item in _members(value):
                        self._add(item)

    def _add(self, value: Any) -> None:
        if id(value) not in self.ids:
            self.ids.add(id(value))
            self._held.append(value)

    def __contains__(self, value: Any) -> bool:
        return id(value) in self.ids


def is_data_leaf(value: Any) -> bool:
    kind = type(value)
    if kind in _ATOMS or kind in (torch.dtype, torch.device, torch.Size):
        return True
    if issubclass(kind, enum.Enum):
        return True
    if kind in (set, frozenset):
        return all(type(item) in _ATOMS for item in value)
    return False


def leaf_key(value: Any) -> Any:
    """What a data leaf is compared by: its type and value, floats by their
    IEEE bytes (NaN equals NaN, -0.0 is not 0.0)."""
    if type(value) is float:
        return (float, struct.pack("<d", value))
    if type(value) in (set, frozenset):
        return (type(value), frozenset(value))
    return (type(value), value)


def _identity(value: Any) -> tuple:
    if is_data_leaf(value):
        return ("v", leaf_key(value))
    return ("x", type(value), id(value))


_CONTAINERS = (tuple, list, dict, set, frozenset)


def _is_container(value: Any) -> bool:
    return issubclass(type(value), _CONTAINERS)


def _len(container: Any) -> int:
    for base in _CONTAINERS:
        if issubclass(type(container), base):
            return base.__len__(container)
    raise TypeError(f"{type(container).__name__} is not a builtin container")


def _global_state(value: Any) -> tuple:
    """What a module-level object is compared by: its items or ``__dict__``
    entries, then theirs, and so on, data by value and anything else (tensors,
    functions, modules of the model) by identity, as far as _MAX_ITEMS values
    and MAX_DEPTH levels; past either, by identity and size."""
    budget = [_MAX_ITEMS]
    on_path: Set[int] = set()

    def state(v: Any, depth: int) -> tuple:
        budget[0] -= 1
        kind = type(v)
        if issubclass(kind, (torch.Tensor, nn.Module) + _OPAQUE) or id(v) in on_path:
            return ("x", kind, id(v))
        big = _len(v) if _is_container(v) else len(vars_of(v)) if _has_dict(v) else 0
        if budget[0] < big or depth >= MAX_DEPTH:
            return ("big", kind, id(v), big)
        budget[0] -= big
        if is_data_leaf(v):
            return ("v", leaf_key(v))
        if issubclass(kind, (set, frozenset)):
            return ("set", kind, id(v), frozenset(_identity(m) for m in _members(v)))
        if _is_container(v):
            items = items_of(v)
        elif _has_dict(v):
            items = list(vars_of(v).items())
        else:
            return ("x", kind, id(v))
        on_path.add(id(v))
        try:
            pairs = [(k, state(x, depth + 1)) for k, x in items]
        finally:
            on_path.discard(id(v))
        # The order of a plain dict's keys or an object's attributes is not
        # part of its state; a sequence's or an OrderedDict's is.
        ordered = issubclass(kind, (tuple, list, OrderedDict))
        return (kind, id(v), tuple(pairs) if ordered else frozenset(pairs))

    return state(value, 0)


class Walk(msgspec.Struct):
    """What ``walk_model`` found, in walk order.

    ``values`` describes every place it looked at, by path: ("v", leaf key)
    for data, ("t",) for a tensor, ("m", name) for a module of the model,
    ("c", type, id, keys or length) for a container, ("o", type, id) for an
    object whose ``__dict__`` it walked and ("r", first path) where it saw
    that object again, ("g", type, id, state) for a module-level object,
    ("y", type, id) for a container already on the path (a cycle), ("l",)
    for a process-local tensor and ("x", type, id) for any other leaf. ``objects`` holds the values themselves
    (only with ``keep_objects``). ``tensors`` holds every place that holds a
    tensor, registered or not, and ``homes`` the first such place of each
    tensor, where its own attributes were walked; ``local`` every place of a
    process-local tensor (process_local_fields), which is in neither. ``registries`` holds every
    module's parameters and buffers in registration order, by the module's
    first name: (name, "p" or "b", tensor or None, persistent).
    """

    modules: List[Tuple[str, nn.Module]] = []
    registries: List[Tuple[str, List[tuple]]] = []
    values: Dict[tuple, tuple] = {}
    objects: Dict[tuple, Any] = {}
    tensors: Dict[tuple, torch.Tensor] = {}
    homes: Dict[int, tuple] = {}
    local: Dict[tuple, torch.Tensor] = {}
    boundaries: Dict[tuple, Any] = {}
    problems: List[str] = []


def walk_model(
    model: nn.Module,
    *,
    module_level: Optional[ModuleLevel] = None,
    boundaries: Optional[Set[tuple]] = None,
    keep_objects: bool = False,
) -> Walk:
    """Walk ``model``. An owner passes ``module_level``; a reader passes the
    ``boundaries`` (paths) at which the owner's walk stopped instead, so that
    both stop at the same places."""
    walk = Walk()
    walk.modules = list(model.named_modules(remove_duplicate=False))
    module_names: Dict[int, str] = {}
    for name, module in walk.modules:
        module_names.setdefault(id(module), name)
    first_path: Dict[int, tuple] = {}
    on_path: Set[int] = set()
    states: Dict[int, tuple] = {}
    declared: Dict[type, frozenset] = {}
    nodes = 0

    def local_of(kind: type) -> frozenset:
        if kind not in declared:
            declared[kind] = process_local_fields(kind)
        return declared[kind]

    def global_value(path: tuple, value: Any) -> None:
        # Only an owner compares the state; a reader just stops here.
        if id(value) not in states:
            states[id(value)] = () if module_level is None else _global_state(value)
        walk.values[path] = ("g", type(value), id(value), states[id(value)])
        walk.boundaries[path] = value

    def visit(path: tuple, value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_NODES:
            if nodes == MAX_NODES + 1:
                walk.problems.append(f"{path_str(path)}: more than {MAX_NODES} values")
            return
        if keep_objects:
            walk.objects[path] = value
        kind = type(value)
        # Immutable values (atoms, tuples such as (), frozensets) are shared
        # freely and cannot be changed in place: never module-level objects.
        is_global = not issubclass(kind, _ATOMS + (tuple, frozenset)) and (
            (boundaries is not None and path in boundaries)
            or (module_level is not None and value in module_level)
        )
        if is_global and kind is set:  # data, but shared
            global_value(path, value)
        elif is_data_leaf(value):
            walk.values[path] = ("v", leaf_key(value))
        elif issubclass(kind, torch.Tensor):
            walk.values[path] = ("t",)
            walk.tensors[path] = value
            if id(value) not in walk.homes:
                walk.homes[id(value)] = path
                children(path, list(vars_of(value).items()), "a", depth)
        elif issubclass(kind, nn.Module) and id(value) in module_names:
            walk.values[path] = ("m", module_names[id(value)])
        elif depth >= MAX_DEPTH:
            walk.problems.append(f"{path_str(path)}: deeper than {MAX_DEPTH}")
        elif issubclass(kind, _OPAQUE):
            walk.values[path] = ("x", kind, id(value))
        elif is_global:
            global_value(path, value)
        elif issubclass(kind, (tuple, list, dict)):
            if id(value) in on_path:
                walk.values[path] = ("y", kind, id(value))
                return
            items = items_of(value)
            shape = tuple(k for k, _ in items) if issubclass(kind, dict) else len(items)
            if issubclass(kind, defaultdict):
                shape = (shape, value.default_factory)
            walk.values[path] = ("c", kind, id(value), shape)
            on_path.add(id(value))
            try:
                children(path, items, "k" if issubclass(kind, dict) else "i", depth)
            finally:
                on_path.discard(id(value))
        elif id(value) in first_path:
            walk.values[path] = ("r", first_path[id(value)])
        elif _has_dict(value):
            first_path[id(value)] = path
            walk.values[path] = ("o", kind, id(value))
            children(path, list(vars_of(value).items()), "a", depth, local_of(kind))
        else:
            walk.values[path] = ("x", kind, id(value))

    def children(path, items, step, depth, local=frozenset()):
        for key, item in items:
            if key in local and issubclass(type(item), torch.Tensor):
                walk.values[path + ((step, key),)] = ("l",)
                walk.local[path + ((step, key),)] = item
            else:
                visit(path + ((step, key),), item, depth + 1)

    # Every registration first, so that a registered tensor's home (where its
    # own attributes are walked) is its registered name; then their
    # attributes; then the modules' plain attributes.
    for path, tensor in _walk_registrations(model, walk, keep_objects):
        if walk.homes[id(tensor)] == path:
            children(path, list(vars_of(tensor).items()), "a", 0)
    for name, module in model.named_modules():
        local = local_of(type(module))
        for kind, store in (("p", module._parameters), ("b", module._buffers)):
            for key in sorted(local.intersection(store)):
                if store[key] is not None:  # a reader would get the owner's bytes
                    path = path_str((name, ("p", key)))
                    walk.problems.append(
                        f"{path}: declared process-local, but registered"
                    )
        attrs = [(k, v) for k, v in vars_of(module).items() if k not in MODULE_FIELDS]
        children((name,), attrs, "a", -1, local)
    return walk


def _walk_registrations(
    model: nn.Module, walk: Walk, keep_objects: bool
) -> List[Tuple[tuple, torch.Tensor]]:
    """Record every module's registry, and every registered tensor at its
    registered name, in ``walk``. Returns (path, tensor) of each one."""
    registered: List[Tuple[tuple, torch.Tensor]] = []
    for name, module in model.named_modules():
        registry = []
        for kind, store in (("p", module._parameters), ("b", module._buffers)):
            for key, tensor in list(store.items()):
                persistent = key not in module._non_persistent_buffers_set
                registry.append((key, kind, tensor, persistent))
                if tensor is not None:
                    path = (name, ("p", key))
                    walk.values[path] = ("t",)
                    walk.tensors[path] = tensor
                    walk.homes.setdefault(id(tensor), path)
                    registered.append((path, tensor))
                    if keep_objects:
                        walk.objects[path] = tensor
        walk.registries.append((name, registry))
    return registered


# ---------------------------------------------------------------------------
# Storages and views
# ---------------------------------------------------------------------------


class StorageTable(msgspec.Struct):
    """Every storage behind a set of tensors, and each tensor as a view of one.

    ``storages[sid]`` is (where, nbytes, a uint8 tensor that covers the
    storage), where is "model" for the model's device and "cpu" for host
    memory. Storage ids follow first use. ``views[id(tensor)]`` is (sid, byte
    offset, dtype name, shape, stride). A storage whose bytes lie inside
    another's becomes part of it; any other overlap is a problem.
    """

    storages: List[Tuple[str, int, torch.Tensor]] = []
    views: Dict[int, Tuple[int, int, str, List[int], List[int]]] = {}
    problems: List[str] = []


def _unsupported(tensor: torch.Tensor, device: torch.device) -> Optional[str]:
    if tensor.layout != torch.strided:
        return f"a {tensor.layout} tensor"
    if tensor.is_quantized or tensor.is_nested:
        return "a quantized or nested tensor"
    if tensor.is_conj() or tensor.is_neg():
        return "a conjugate or negative view"
    if tensor.device.type == "meta":
        return "a tensor still on the meta device"
    if tensor.device != device and tensor.device.type != "cpu":
        return f"a tensor on {tensor.device}, not on the model's {device}"
    return None


def build_storage_table(
    places: Dict[tuple, torch.Tensor], device: torch.device
) -> StorageTable:
    """The storages and views of the tensors at ``places`` (path -> tensor, in
    walk order) on ``device`` or the CPU."""
    table = StorageTable()
    first: Dict[int, Tuple[tuple, torch.Tensor]] = {}
    for path, tensor in places.items():
        first.setdefault(id(tensor), (path, tensor))
    # One span per distinct byte range, (device, start, nbytes); a storage of
    # no bytes is a span of its own.
    spans: Dict[tuple, torch.UntypedStorage] = {}
    span_path: Dict[tuple, tuple] = {}
    span_of: Dict[int, tuple] = {}
    for key, (path, tensor) in first.items():
        why = _unsupported(tensor, device)
        if why is not None:
            table.problems.append(f"{path_str(path)}: {why}")
            continue
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            span = (str(storage.device), -1, storage._cdata)
        else:
            span = (str(storage.device), storage.data_ptr(), storage.nbytes())
        spans.setdefault(span, storage)
        span_path.setdefault(span, path)
        span_of[key] = span
    # A span inside another belongs to it, at an offset.
    owner: Dict[tuple, Tuple[tuple, int]] = {}
    by_device: Dict[str, List[tuple]] = {}
    for span in spans:
        if span[1] < 0:
            owner[span] = (span, 0)
        else:
            by_device.setdefault(span[0], []).append(span)
    for here in by_device.values():
        here.sort(key=lambda s: (s[1], -s[2]))
        outer = None
        for span in here:
            if outer is not None and span[1] < outer[1] + outer[2]:
                if span[1] + span[2] > outer[1] + outer[2]:
                    table.problems.append(
                        f"{path_str(span_path[span])}: its storage overlaps that of "
                        f"{path_str(span_path[outer])} from {span[1] - outer[1]} "
                        f"bytes in, and ends past it"
                    )
                owner[span] = (outer, span[1] - outer[1])
                continue
            outer = span
            owner[span] = (span, 0)
    sid_of: Dict[tuple, int] = {}
    for key, (path, tensor) in first.items():
        if key not in span_of:
            continue
        outer, shift = owner[span_of[key]]
        if outer not in sid_of:
            storage = spans[outer]
            where = "model" if storage.device == device else "cpu"
            whole = torch.empty(0, dtype=torch.uint8, device=storage.device)
            whole.set_(storage)
            sid_of[outer] = len(table.storages)
            table.storages.append((where, storage.nbytes(), whole))
        offset = shift + tensor.storage_offset() * tensor.element_size()
        if offset % tensor.element_size():
            table.problems.append(
                f"{path_str(path)}: starts {offset} bytes into its storage, not a "
                f"multiple of its {tensor.element_size()}-byte elements"
            )
        table.views[key] = (
            sid_of[outer],
            offset,
            dtype_name(tensor.dtype),
            list(tensor.shape),
            list(tensor.stride()),
        )
    return table


def make_view(
    storage: torch.UntypedStorage,
    offset: int,
    dtype: str,
    shape: Sequence[int],
    stride: Sequence[int],
) -> torch.Tensor:
    """The tensor at byte ``offset`` of ``storage`` with this dtype, shape and
    stride, on the storage's device."""
    torch_dtype = parse_dtype(dtype)
    tensor = torch.empty(0, dtype=torch_dtype, device=storage.device)
    return tensor.set_(storage, offset // torch_dtype.itemsize, shape, stride)


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def storage_digests(storages: Sequence[torch.Tensor]) -> List[bytes]:
    """BLAKE3-256 of every byte of each storage, given as uint8 tensors that
    cover their storages. CUDA storages are hashed on their GPU, with one
    host synchronization per device; CPU ones on the host; others are copied
    to the host first, a piece at a time. The caller makes sure the bytes
    are written."""
    import blake3

    digests: List[Optional[bytes]] = [None] * len(storages)
    on_gpu: Dict[torch.device, List[int]] = {}
    for i, storage in enumerate(storages):
        if storage.numel() == 0:
            digests[i] = blake3.blake3(b"").digest()
        elif storage.device.type == "cuda":
            on_gpu.setdefault(storage.device, []).append(i)
        else:
            hasher = blake3.blake3()
            for piece in storage.split(64 << 20):
                hasher.update(memoryview(piece.cpu().numpy()))
            digests[i] = hasher.digest()
    if on_gpu:
        from sglang.kernels.selector import get_kernel

        hash_batch = get_kernel("memory.gpu_tensors_blake3")
        for indices in on_gpu.values():
            batch = hash_batch([storages[i] for i in indices])
            for i, digest in zip(indices, batch):
                digests[i] = digest
    return digests


def structure_digest(model: nn.Module) -> Tuple[bytes, int, int]:
    """(SHA-256, module names, registrations) of what construction decides:
    every module's name, class and which name it was first reached by, and
    its parameters' and buffers' names, kinds, shapes and dtypes. Plain
    values are left out: two correct processes may differ in them."""
    digest = hashlib.sha256()
    first_name: Dict[int, str] = {}
    modules = registrations = 0
    for name, module in model.named_modules(remove_duplicate=False):
        cls = type(module)
        shared = first_name.setdefault(id(module), name)
        digest.update(
            f"M {name} {cls.__module__}.{cls.__qualname__} {shared}\n".encode()
        )
        modules += 1
        if shared != name:
            continue
        for kind, store in (("p", module._parameters), ("b", module._buffers)):
            for key, tensor in store.items():
                if tensor is None:
                    what = "None"
                else:
                    what = f"{list(tensor.shape)} {dtype_name(tensor.dtype)}"
                digest.update(f"{kind} {key} {what}\n".encode())
                registrations += 1
    return digest.digest(), modules, registrations


# ---------------------------------------------------------------------------
# Values as data
# ---------------------------------------------------------------------------


def _lookup_enum(module_name: Any, qualname: Any) -> type:
    """The enum class that a module this process has already imported defines
    at ``qualname``. Reads module and class ``__dict__``s only; imports
    nothing. Raises KeyError."""
    if type(module_name) is not str or type(qualname) is not str:
        raise KeyError("an enum is named by two strings")
    obj = sys.modules.get(module_name)
    if not issubclass(type(obj), types.ModuleType):
        raise KeyError(f"module {module_name} is not imported in this process")
    for part in qualname.split("."):
        if not issubclass(type(obj), (types.ModuleType, type)):
            raise KeyError(f"{module_name}:{qualname} is not in a module or class")
        try:
            obj = vars(obj)[part]
        except KeyError:
            raise KeyError(f"{module_name}:{qualname} does not exist here") from None
    if not issubclass(type(obj), enum.EnumMeta):
        raise KeyError(f"{module_name}:{qualname} is not an enum")
    return obj


class EncodeContext(msgspec.Struct, kw_only=True):
    """What the owner's encoder refers to.

    ``tensor_ids``: id(tensor) -> its number in the snapshot.
    ``first_paths``: id(object) -> the first path of each object the walk
    visited once, and of each list or dict (first_paths()); anywhere else
    such an object is a reference to that path, so that aliases survive.
    ``module_names``: id(module) -> its first name in the model.
    ``device``: the model's device.
    ``keep(path, value)``: whether the reader keeps the value it already has
    at ``path`` (the same object the model held there when it was built).
    ``addresses``: sorted (start, end) byte ranges of the model's storages
    (exported or process-local); an int inside one is refused, as an address
    that means nothing elsewhere.
    """

    tensor_ids: Dict[int, int] = {}
    first_paths: Dict[int, tuple] = {}
    module_names: Dict[int, str] = {}
    device: torch.device = torch.device("cpu")
    keep: Optional[Callable[[tuple, Any], bool]] = None
    addresses: Sequence[Tuple[int, int]] = ()


def _in_ranges(value: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    lo, hi = 0, len(ranges)
    while lo < hi:
        mid = (lo + hi) // 2
        if ranges[mid][1] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(ranges) and ranges[lo][0] <= value


def encode_value(value: Any, path: tuple, ctx: EncodeContext) -> Any:
    """``value``, found at ``path``, as data. Raises Unencodable naming what
    is not."""
    building: List[int] = []

    def enc(v, here):
        kind = type(v)
        if here != path and ctx.keep is not None and ctx.keep(here, v):
            return ["keep"]
        if kind is int:
            _not_an_address(v, ctx)
            return v
        if kind in (type(None), bool, float, str):
            return v
        if kind is bytes:
            if len(v) > MAX_BYTES_VALUE:
                raise Unencodable(f"{len(v)} bytes, more than {MAX_BYTES_VALUE}")
            return v
        if kind is torch.dtype:
            return ["dtype", dtype_name(v)]
        if kind is torch.device:
            if v == ctx.device:
                return ["device", "model"]
            if v.index is None or v.type == "cpu":
                return ["device", v.type]
            raise Unencodable(f"device {v}, not the model's {ctx.device}")
        if kind is torch.Size:
            return ["size", list(v)]
        if issubclass(kind, enum.Enum):
            cls = kind.__module__, kind.__qualname__
            try:
                if vars(_lookup_enum(*cls))["_member_map_"].get(v.name) is v:
                    return ["enum", cls[0], cls[1], v.name]
            except KeyError:
                pass
            raise Unencodable(f"a member of {kind.__qualname__}, not found by its name")
        if kind in (set, frozenset):
            if not all(type(item) in _ATOMS for item in v):
                raise Unencodable(f"a {kind.__name__} of objects")
            for item in v:
                if type(item) is int:
                    _not_an_address(item, ctx)
            items = sorted(v, key=lambda item: (type(item).__name__, repr(item)))
            return ["set" if kind is set else "fset", items]
        if issubclass(kind, torch.Tensor):
            tid = ctx.tensor_ids.get(id(v))
            if tid is None:
                raise Unencodable("a tensor outside the snapshot")
            return ["tensor", tid]
        if issubclass(kind, nn.Module) and id(v) in ctx.module_names:
            return ["module", ctx.module_names[id(v)]]
        first = ctx.first_paths.get(id(v))
        if first is not None and first != here:
            if here[: len(first)] == first:
                raise Unencodable("a container that holds itself")
            return ["ref", encode_path(first, ctx)]
        if kind in (tuple, list, dict, OrderedDict, defaultdict):
            if id(v) in building:
                raise Unencodable("a container that holds itself")
            building.append(id(v))
            try:
                if kind in (tuple, list):
                    items = [enc(item, here + (("i", i),)) for i, item in enumerate(v)]
                    return ["tuple" if kind is tuple else "list", items]
                pairs = [
                    [encode_key(key, ctx), enc(item, here + (("k", key),))]
                    for key, item in items_of(v)
                ]
                if kind is defaultdict:
                    factory = v.default_factory
                    if (
                        factory is not None
                        and FACTORIES.get(getattr(factory, "__name__", None))
                        is not factory
                    ):
                        raise Unencodable(f"a defaultdict whose factory is {factory!r}")
                    return [
                        "ddict",
                        None if factory is None else factory.__name__,
                        pairs,
                    ]
                return ["odict" if kind is OrderedDict else "dict", pairs]
            finally:
                building.pop()
        raise Unencodable(f"a {kind.__module__}.{kind.__qualname__}")

    return enc(value, path)


def first_paths(walk: Walk) -> Dict[int, tuple]:
    """id -> first path, of every object ``walk`` walked into and of every
    list or dict it saw (EncodeContext.first_paths)."""
    out: Dict[int, tuple] = {}
    for path, desc in walk.values.items():
        if desc[0] == "o" or (desc[0] == "c" and desc[1] in MUTABLE_CONTAINERS):
            out.setdefault(desc[2], path)
    return out


MUTABLE_CONTAINERS = (list, dict, OrderedDict, defaultdict)


def _not_an_address(value: int, ctx: EncodeContext) -> None:
    if ctx.addresses and _in_ranges(value, ctx.addresses):
        raise Unencodable(f"{value:#x}, an address inside a storage of the model")


def encode_key(key: Any, ctx: Optional[EncodeContext] = None) -> Any:
    """A dict key as data: atoms, dtypes, sizes, enum members and tuples of
    these."""
    kind = type(key)
    if kind is int and ctx is not None:
        _not_an_address(key, ctx)
    if kind in (type(None), bool, int, float, str, bytes):
        return key
    if kind is tuple:
        return ["tuple", [encode_key(item, ctx) for item in key]]
    if kind in (torch.dtype, torch.Size) or issubclass(kind, enum.Enum):
        return encode_value(key, ("",), EncodeContext())
    raise Unencodable(f"a dict key of type {kind.__qualname__}")


def encode_path(path: tuple, ctx: Optional[EncodeContext] = None) -> list:
    steps = [
        [kind, encode_key(key, ctx) if kind == "k" else key] for kind, key in path[1:]
    ]
    return [path[0], steps]


class DecodeContext(msgspec.Struct):
    """What a reader's decoder refers to: its modules by every name, the
    tensors it installed by number, and its model's device."""

    modules: Dict[str, nn.Module] = {}
    tensors: Sequence[torch.Tensor] = ()
    device: torch.device = torch.device("cpu")


def decode_path(encoded: list) -> tuple:
    root, steps = encoded
    return (root,) + tuple(
        (kind, decode_key(key) if kind == "k" else key) for kind, key in steps
    )


def decode_key(encoded: Any) -> Any:
    if type(encoded) is list and encoded[:1] == ["tuple"]:
        return tuple(decode_key(item) for item in encoded[1])
    return decode_value(encoded, ("",), DecodeContext())


def decode_value(encoded: Any, path: tuple, ctx: DecodeContext) -> Any:
    """The object ``encoded`` describes, built for the reader's model at
    ``path``. ``["keep"]`` gives what the reader's model holds at that place
    now, ``["ref", path]`` what it holds at another path."""

    made: Dict[tuple, Any] = {}  # containers decoded so far, by path

    def dec(e, here):
        out = dec_one(e, here)
        if type(e) is list and e[0] in ("list", "dict", "odict", "ddict"):
            made[here] = out
        return out

    def dec_one(e, here):
        if type(e) is not list:
            return e
        tag, args = e[0], e[1:]
        if tag == "keep":
            return resolve_path(ctx.modules, here)
        if tag == "dtype":
            return parse_dtype(args[0])
        if tag == "device":
            return ctx.device if args[0] == "model" else torch.device(args[0])
        if tag == "size":
            return torch.Size(args[0])
        if tag == "enum":
            return vars(_lookup_enum(args[0], args[1]))["_member_map_"][args[2]]
        if tag in ("set", "fset"):
            return (set if tag == "set" else frozenset)(args[0])
        if tag == "tensor":
            return ctx.tensors[args[0]]
        if tag == "module":
            return ctx.modules[args[0]]
        if tag == "ref":
            target = decode_path(args[0])
            if target in made:
                return made[target]
            return resolve_path(ctx.modules, target)
        if tag in ("tuple", "list"):
            items = [dec(item, here + (("i", i),)) for i, item in enumerate(args[0])]
            return tuple(items) if tag == "tuple" else items
        if tag in ("dict", "odict", "ddict"):
            if tag == "ddict":
                out = defaultdict(None if args[0] is None else FACTORIES[args[0]])
            else:
                out = OrderedDict() if tag == "odict" else {}
            for key, item in args[-1]:
                key = decode_key(key)
                out[key] = dec(item, here + (("k", key),))
            return out
        raise KeyError(f"unknown encoding {tag!r}")

    return dec(encoded, path)


# ---------------------------------------------------------------------------
# The manifest: the ``load_diff`` bytes of a SnapshotManifest
# ---------------------------------------------------------------------------

# Sizes, offsets and ids: non-negative and within what torch holds.
Count = Annotated[int, msgspec.Meta(ge=0, le=_INT64_MAX)]
Digest = Annotated[bytes, msgspec.Meta(min_length=32, max_length=32)]


class StorageEntry(
    msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True
):
    """A storage, by storage id. A transport moves each one with bytes under
    the name str(storage id); the reader makes empty ones itself."""

    nbytes: Count
    where: Literal["model", "cpu"]  # the model's device, or host memory
    digest: Digest  # BLAKE3-256 of its bytes


class TensorEntry(
    msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True
):
    """A tensor, by tensor id: a view of a storage. ``cls`` is "tensor",
    "parameter", or the module.qualname of the reader's own tensor at
    ``home``, the path where the tensor's own attributes are kept."""

    sid: Count
    offset: Count  # bytes
    dtype: str
    shape: List[Count]
    stride: List[Count]
    cls: str
    requires_grad: bool
    home: list


class Registration(
    msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True
):
    name: str
    kind: Literal["p", "b"]
    tid: Optional[Count]
    persistent: bool


class Registry(
    msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True
):
    """A module's parameters and buffers after loading, in order."""

    module: str
    entries: List[Registration]


class Place(msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True):
    """Another place of a tensor: a plain attribute, a container item."""

    path: list
    tid: Count


class Local(msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True):
    """A process-local tensor: every attribute that holds it, and its dtype
    and shape; a reader makes it, zeroed, on the model's device or the CPU."""

    dtype: str
    shape: List[Count]
    where: Literal["model", "cpu"]
    places: List[list]


class Structure(
    msgspec.Struct, array_like=True, frozen=True, forbid_unknown_fields=True
):
    """What construction decided (structure_digest)."""

    digest: Digest
    modules: Count
    registrations: Count


class Manifest(msgspec.Struct, kw_only=True):
    """What a reader installs. ``ops`` are what loading changed, in the
    owner's walk order: ["set", path, value], ["fill", path, value] or
    ["del", path], with paths from encode_path and values from encode_value.
    ``boundaries`` are the paths of the module-level objects the walk did not
    enter; ``removed_modules`` the modules loading removed; ``local`` the
    process-local tensors."""

    version: int
    structure: Structure
    storages: List[StorageEntry]
    tensors: List[TensorEntry]
    registered: List[Registry]
    places: List[Place]
    ops: List[list]
    boundaries: List[list]
    removed_modules: List[str]
    local: List[Local]


def encode_manifest(manifest: Manifest) -> bytes:
    """``manifest`` as msgpack bytes. Raises SnapshotError for a value msgpack
    cannot hold (an int past 64 bits)."""
    try:
        return msgspec.msgpack.encode(manifest)
    except (msgspec.EncodeError, OverflowError, TypeError) as e:
        raise SnapshotError("cannot encode the load-diff manifest", [str(e)]) from e


def decode_manifest(data: Any) -> Manifest:
    """The manifest in msgpack ``data``: typed by msgspec (a wrong type is
    refused with its place), then checked by validate_manifest."""
    if type(data) is not bytes:
        raise SnapshotError("invalid load-diff manifest", ["not bytes"])
    try:
        manifest = msgspec.msgpack.decode(data, type=Manifest)
    except (msgspec.DecodeError, RecursionError) as e:
        raise SnapshotError("invalid load-diff manifest", [str(e)]) from e
    validate_manifest(manifest)
    return manifest


def validate_manifest(m: Manifest) -> None:
    """What the types do not say: the version, ids in range, every view inside
    its storage, every storage used, no name registered twice, every tensor's
    home one of its places, and the paths and values, which are untyped.
    Raises SnapshotError."""
    if m.version != FORMAT_VERSION:
        raise SnapshotError("invalid load-diff manifest", [f"version {m.version!r}"])
    problems: List[str] = []
    n = len(m.tensors)
    for tid, view in enumerate(m.tensors):
        if not _valid_view(view, m.storages):
            problems.append(f"tensors[{tid}]")
    unused = set(range(len(m.storages))) - {view.sid for view in m.tensors}
    if unused:
        problems.append(f"storages no tensor uses: {sorted(unused)[:5]}")
    owned = set()  # (path, tensor id) of every place of a tensor
    modules: Set[str] = set()
    names: Set[Tuple[str, str]] = set()
    for i, registry in enumerate(m.registered):
        if registry.module in modules:
            problems.append(f"registered[{i}]: {registry.module!r} twice")
        modules.add(registry.module)
        for entry in registry.entries:
            name = (registry.module, entry.name)
            if name in names or (entry.tid is not None and entry.tid >= n):
                problems.append(f"registered[{i}]: {entry.name!r}")
            names.add(name)
            owned.add((repr([registry.module, [["p", entry.name]]]), entry.tid))
    paths: Set[str] = set()
    for i, place in enumerate(m.places):
        path = repr(place.path)
        if path in paths or not (
            _valid_path(place.path) and place.path[1] and place.tid < n
        ):
            problems.append(f"places[{i}]")
        paths.add(path)
        owned.add((path, place.tid))
    for i, entry in enumerate(m.local):
        if not (
            entry.dtype in _DTYPES and len(entry.shape) <= MAX_DEPTH and entry.places
        ):
            problems.append(f"local[{i}]")
        for place in entry.places:
            path = repr(place)
            if path in paths or not (_valid_path(place) and place[1]):
                problems.append(f"local[{i}]: a place")
            paths.add(path)
    for tid, view in enumerate(m.tensors):
        if (repr(view.home), tid) not in owned:
            problems.append(f"tensors[{tid}]: its home is not its place")
    op_paths: Set[str] = set()
    for i, op in enumerate(m.ops):
        if not _valid_op(op, n, op_paths):
            problems.append(f"ops[{i}]")
    for i, path in enumerate(m.boundaries):
        if not _valid_path(path):
            problems.append(f"boundaries[{i}]")
    if problems:
        raise SnapshotError("invalid load-diff manifest", problems[:20])


def _is_int(value: Any) -> bool:
    return type(value) is int and 0 <= value <= _INT64_MAX


def _valid_view(view: TensorEntry, storages: List[StorageEntry]) -> bool:
    dtype = _DTYPES.get(view.dtype)
    if dtype is None or view.sid >= len(storages):
        return False
    if not (
        len(view.shape) == len(view.stride) <= MAX_DEPTH
        and view.offset % dtype.itemsize == 0
        and (not view.requires_grad or dtype.is_floating_point or dtype.is_complex)
        and _valid_path(view.home)
    ):
        return False
    first = view.offset // dtype.itemsize
    needed = view_nbytes(view.shape, view.stride, first, dtype.itemsize)
    return needed <= storages[view.sid].nbytes


def _valid_op(op: list, n_tensors: int, seen: Set[str]) -> bool:
    """["set" | "fill", path, value] or ["del", path], each path once."""
    if not (len(op) >= 2 and op[0] in ("set", "fill", "del") and _valid_path(op[1])):
        return False
    path = repr(op[1])
    if not op[1][1] or path in seen:
        return False
    seen.add(path)
    if op[0] == "del":
        return len(op) == 2
    return len(op) == 3 and _valid_encoding(op[2], n_tensors, 0)


def _valid_path(path: Any, depth: int = 0) -> bool:
    if not (type(path) is list and len(path) == 2 and type(path[0]) is str):
        return False
    steps = path[1]
    if type(steps) is not list or len(steps) > MAX_DEPTH + 2 or depth > MAX_DEPTH:
        return False
    for step in steps:
        if not (type(step) is list and len(step) == 2):
            return False
        kind, key = step
        if kind in ("p", "a"):
            ok = type(key) is str
        elif kind == "i":
            ok = _is_int(key)
        elif kind == "k":
            ok = _valid_key(key, depth + 1)
        else:
            ok = False
        if not ok:
            return False
    return True


def _valid_key(e: Any, depth: int) -> bool:
    """A dict key: an atom, or a tuple, dtype, size or enum member of atoms."""
    if type(e) is list and len(e) == 2 and e[0] == "tuple" and type(e[1]) is list:
        return depth <= MAX_DEPTH and all(_valid_key(item, depth + 1) for item in e[1])
    if type(e) is list and e[:1] not in (["dtype"], ["size"], ["enum"]):
        return False
    return _valid_encoding(e, 0, depth)


# tag -> number of arguments
_TAGS = {
    "keep": 0, "dtype": 1, "device": 1, "size": 1, "enum": 3, "set": 1,
    "fset": 1, "tensor": 1, "module": 1, "ref": 1, "tuple": 1, "list": 1,
    "dict": 1, "odict": 1, "ddict": 2,
}  # fmt: skip


def _valid_encoding(e: Any, n_tensors: int, depth: int) -> bool:
    if depth > MAX_DEPTH:
        return False
    kind = type(e)
    if kind in (type(None), bool, int, float, str):
        return True
    if kind is bytes:
        return len(e) <= MAX_BYTES_VALUE
    if not (kind is list and e and type(e[0]) is str and e[0] in _TAGS):
        return False
    tag, args = e[0], e[1:]
    if len(args) != _TAGS[tag]:
        return False
    if tag == "keep":
        return True
    if tag == "dtype":
        return type(args[0]) is str and args[0] in _DTYPES
    if tag == "device":
        return args[0] in ("model", "cpu", "cuda", "npu", "xpu", "hpu", "mps")
    if tag == "size":
        return type(args[0]) is list and all(_is_int(v) for v in args[0])
    if tag == "enum":
        return all(type(a) is str for a in args)
    if tag in ("set", "fset"):
        return type(args[0]) is list and all(type(v) in _ATOMS for v in args[0])
    if tag == "tensor":
        return _is_int(args[0]) and args[0] < n_tensors
    if tag == "module":
        return type(args[0]) is str
    if tag == "ref":
        return _valid_path(args[0], depth + 1)
    if tag in ("tuple", "list"):
        return type(args[0]) is list and all(
            _valid_encoding(v, n_tensors, depth + 1) for v in args[0]
        )
    if tag == "ddict" and not (
        args[0] is None or (type(args[0]) is str and args[0] in FACTORIES)
    ):
        return False
    pairs = args[-1]
    return type(pairs) is list and all(
        type(p) is list
        and len(p) == 2
        and _valid_key(p[0], depth + 1)
        and _valid_encoding(p[1], n_tensors, depth + 1)
        for p in pairs
    )
