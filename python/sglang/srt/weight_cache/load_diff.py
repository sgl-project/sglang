# SPDX-License-Identifier: Apache-2.0
"""Load-diff: what loading produced on an owner, and installing exactly that.

D1 of #50943. ``record_construction`` takes the model as built, before any
weight is loaded; ``export`` compares the loaded model with it and returns
every storage the model holds and the ``load_diff`` bytes of the manifest
(load_diff_format.Manifest); ``install`` builds the owner's model again on a
skeleton the reader built on the meta device, from those storages, sets the
values that loading changed, and checks the result. Nothing is ever
post-processed again, nothing is imported by a name in a snapshot, and no
code in a snapshot runs.

It shares nothing with the validator but the schema.
"""

from __future__ import annotations

import itertools
import logging
import sys
import time
import types
import weakref
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import msgspec
import torch
from torch import nn

from sglang.srt.weight_cache.load_diff_format import (
    FACTORIES,
    FORMAT_VERSION,
    MODULE_FIELDS,
    MUTABLE_CONTAINERS,
    DecodeContext,
    EncodeContext,
    Local,
    Manifest,
    ModuleLevel,
    Place,
    Registration,
    Registry,
    SnapshotError,
    StorageEntry,
    StorageTable,
    Structure,
    TensorEntry,
    Unencodable,
    Walk,
    build_storage_table,
    decode_key,
    decode_manifest,
    decode_path,
    decode_value,
    dtype_name,
    encode_manifest,
    encode_path,
    encode_value,
    first_paths,
    is_data_leaf,
    items_of,
    leaf_key,
    make_view,
    parse_dtype,
    path_str,
    resolve_path,
    step_into,
    storage_digests,
    structure_digest,
    vars_of,
    walk_model,
)
from sglang.srt.weight_cache.schema import SnapshotManifest, StorageMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The interface: record_construction, export, install
# ---------------------------------------------------------------------------


def record_construction(model: nn.Module) -> Any:
    """Owner: what ``model`` holds right after construction, before weights load.

    Called from ``on_model_constructed``. The record refers to the model and
    its objects weakly; ``export`` takes it."""
    module_level = ModuleLevel()
    walk = walk_model(model, module_level=module_level, keep_objects=True)
    refs = {}
    for path, desc in walk.values.items():
        if desc[0] in ("c", "o", "x", "g") and (ref := _weak(walk.objects[path])):
            refs[path] = ref
    hooks = {
        name: tuple((f, k, _weak(h) or h) for f, k, h in _hooks(m))
        for name, m in model.named_modules()
    }
    return Baseline(
        model=weakref.ref(model),
        structure=structure_digest(model),
        modules={name: (type(m), _weak(m)) for name, m in walk.modules},
        hooks=hooks,
        values=walk.values,
        refs=refs,
        module_level=module_level,
        problems=walk.problems,
    )


def export(
    *, model: nn.Module, constructed: Any
) -> tuple[dict[str, torch.Tensor], bytes]:
    """Owner: every storage the loaded model holds, and the manifest's ``load_diff``.

    ``model`` was loaded since ``constructed`` was recorded, post-processing
    included, and has not run. The storages are uint8 tensors, each covering
    one allocation (contiguous, offset 0) on the model's device or the CPU,
    named by first use; storages of no bytes are left out (the reader makes
    them). Raises SnapshotError listing everything a reader could not get: a
    value that is not data, a module added or replaced, hooks added or
    removed, a tensor on another device, a module-level object that loading
    changed."""
    if not isinstance(constructed, Baseline) or constructed.model() is not model:
        raise SnapshotError(
            "cannot hand the loaded model to a reader",
            ["its construction was not recorded (record_construction)"],
        )
    tic = time.perf_counter()
    device = _model_device(model)
    after = walk_model(model, module_level=constructed.module_level, keep_objects=True)
    modules_now, removed, problems = _module_changes(constructed, after, model)
    problems += _localize(after, device)
    table = build_storage_table(after.tensors, device)
    problems += table.problems
    tensor_ids, tensors = _tensor_entries(after, table)
    ctx = _encode_context(constructed, after, table, tensor_ids, device)
    ops = _diff(constructed, after, ctx, modules_now, problems)
    problems += _address_keys(after, ctx)
    if problems:
        raise SnapshotError("cannot hand the loaded model to a reader", problems)
    t_digest = time.perf_counter()
    digests = storage_digests([whole for _, _, whole in table.storages])
    digest_s = time.perf_counter() - t_digest
    manifest = Manifest(
        version=FORMAT_VERSION,
        structure=Structure(*constructed.structure),
        storages=[
            StorageEntry(nbytes, where, digest)
            for (where, nbytes, _), digest in zip(table.storages, digests)
        ],
        tensors=tensors,
        registered=_registries(after, tensor_ids),
        places=_places(after, tensor_ids),
        ops=ops,
        boundaries=[encode_path(path) for path in after.boundaries],
        removed_modules=removed,
        local=_locals(after),
    )
    payload = encode_manifest(manifest)
    decode_manifest(payload)  # what a reader gets passes a reader's checks
    stats = {
        "storages": len(table.storages),
        "bytes": sum(nbytes for _, nbytes, _ in table.storages),
        "tensors": len(tensors),
        "places": len(manifest.places),
        "ops": len(ops),
        "local": len(manifest.local),
        "manifest_bytes": len(payload),
        "digest_s": digest_s,
        "export_s": time.perf_counter() - tic,
    }
    logger.info("load-diff: exported %s", stats)
    return {
        str(sid): whole
        for sid, (_, nbytes, whole) in enumerate(table.storages)
        if nbytes
    }, payload


def install(
    *,
    skeleton: nn.Module,
    manifest: SnapshotManifest,
    storages: dict[str, torch.Tensor],
) -> nn.Module:
    """Reader: attach ``storages`` to the meta-built ``skeleton`` and set the values
    loading changed on the owner. Never post-processes.

    ``storages`` must be exactly ``manifest.storages``: the same names, sizes
    and devices. A registered tensor keeps the identity of the one the reader
    built where it can (``torch.utils.swap_tensors``). The result is checked
    against the owner's export (check_install: views, registrations, values,
    BLAKE3 of every storage). Nothing is put back on failure: drop the
    skeleton and keep the storages mapped until the process exits, since it,
    or a cache it shares, may still refer to them."""
    m = decode_manifest(manifest.load_diff)
    device = _check_storages(m, manifest.storages, storages)
    tensors, swapped = _install(skeleton, m, storages, device)
    stats = check_install(skeleton, m, tensors, device)
    stats.update(
        tensors=len(tensors), swapped=swapped, ops=len(m.ops), local=len(m.local)
    )
    logger.info("load-diff: installed %s", stats)
    return skeleton


def _model_device(model: nn.Module) -> torch.device:
    """The device of the model's tensors that are not on the CPU (the first
    one found), or the CPU."""
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        if tensor.device.type not in ("cpu", "meta"):
            return tensor.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Owner: the model as built, and what loading changed
# ---------------------------------------------------------------------------


# PyTorch's own hook tables on a module.
_HOOK_FIELDS = tuple(sorted(f for f in vars(nn.Module()) if "hook" in f))


def _hooks(module: nn.Module) -> tuple:
    """The hooks registered on ``module``, in the order they run: (table,
    key, hook) for each."""
    fields = vars_of(module)
    return tuple(
        (field, key, hook)
        for field in _HOOK_FIELDS
        if issubclass(type(fields.get(field)), dict)
        for key, hook in items_of(fields[field])
    )


def _same_hooks(built: tuple, now: tuple) -> bool:
    return len(built) == len(now) and all(
        b[:2] == n[:2] and (b[2]() if isinstance(b[2], weakref.ref) else b[2]) is n[2]
        for b, n in zip(built, now)
    )


def _weak(value: Any) -> Optional[weakref.ref]:
    try:
        return weakref.ref(value)
    except TypeError:
        return None


class Baseline(msgspec.Struct):
    """A model as it was built, before loading: what export compares with.
    Besides weak references, it holds only what Python modules' namespaces
    hold anyway, so it keeps nothing of the model alive. ``refs`` holds a weak
    reference to each object it saw that allows one, so that a new object at
    a reused address is not taken for the old one."""

    model: Any  # a weak reference to the model
    structure: Tuple[bytes, int, int]
    modules: Dict[str, Tuple[type, Optional[weakref.ref]]]
    hooks: Dict[str, tuple]
    values: Dict[tuple, tuple]
    refs: Dict[tuple, Any]
    module_level: ModuleLevel
    problems: List[str]

    def same(self, path: tuple, value: Any) -> bool:
        """Whether ``value`` is the object the model held at ``path`` when it
        was built (as far as ids and weak references tell)."""
        built = self.values.get(path)
        if built is None or len(built) < 3 or built[1] is not type(value):
            return False
        ref = self.refs.get(path)
        return built[2] == id(value) and (ref is None or ref() is value)


def _module_changes(
    baseline: Baseline, after: Walk, model: nn.Module
) -> Tuple[Dict[str, int], List[str], List[str]]:
    """The modules now (name -> id), the names loading removed, and the
    problems: modules added, replaced or re-classed, hooks changed, and what
    the walks could not read."""
    problems = list(after.problems) + list(baseline.problems)
    modules_now: Dict[str, int] = {}
    for name, module in after.modules:
        modules_now[name] = id(module)
        built = baseline.modules.get(name)
        if built is None or built[1] is None or built[1]() is not module:
            what = "added" if built is None else "replaced"
        elif built[0] is not type(module):
            what = "re-classed"
        else:
            continue
        problems.append(
            f"{name or '<model>'}: a {type(module).__qualname__} module that "
            f"loading {what}"
        )
    removed = [name for name in baseline.modules if name not in modules_now]
    for name, module in model.named_modules():
        if not _same_hooks(baseline.hooks.get(name, ()), _hooks(module)):
            problems.append(
                f"{name or '<model>'}: loading added, removed or reordered hooks"
            )
    return modules_now, removed, problems


def _class_name(tensor: torch.Tensor) -> str:
    if type(tensor) is torch.Tensor:
        return "tensor"
    if type(tensor) is nn.Parameter:
        return "parameter"
    return f"{type(tensor).__module__}.{type(tensor).__qualname__}"


def _tensor_entries(
    after: Walk, table: StorageTable
) -> Tuple[Dict[int, int], List[TensorEntry]]:
    """Every tensor once, at its first place: id(tensor) -> tensor id, and
    the entries by tensor id."""
    tensor_ids: Dict[int, int] = {}
    entries: List[TensorEntry] = []
    for path, tensor in after.tensors.items():
        if id(tensor) in tensor_ids or id(tensor) not in table.views:
            continue
        tensor_ids[id(tensor)] = len(entries)
        sid, offset, dtype, shape, stride = table.views[id(tensor)]
        entries.append(
            TensorEntry(
                sid,
                offset,
                dtype,
                shape,
                stride,
                _class_name(tensor),
                bool(tensor.requires_grad),
                encode_path(path),
            )
        )
    return tensor_ids, entries


def _address_keys(after: Walk, ctx: EncodeContext) -> List[str]:
    """Every place of a tensor whose path has a dict key that is an address
    in an exported storage (``cache[w.data_ptr()] = w``): a reader's own
    addresses differ, so it would not find the tensor there."""
    problems = []
    for path in after.tensors:
        try:
            encode_path(path, ctx)
        except Unencodable as e:
            problems.append(
                f"{path_str(path)}: loading left {e}; a reader cannot get it"
            )
    return problems


def _registries(after: Walk, tensor_ids: Dict[int, int]) -> List[Registry]:
    return [
        Registry(
            module_name,
            [
                Registration(
                    key, kind, None if t is None else tensor_ids[id(t)], persistent
                )
                for key, kind, t, persistent in registry
            ],
        )
        for module_name, registry in after.registries
    ]


def _places(after: Walk, tensor_ids: Dict[int, int]) -> List[Place]:
    """Every place of a tensor but its registrations."""
    return [
        Place(encode_path(path), tensor_ids[id(tensor)])
        for path, tensor in after.tensors.items()
        if not (len(path) == 2 and path[1][0] == "p")
    ]


def _localize(walk: Walk, device: torch.device) -> List[str]:
    """Every place of a process-local tensor becomes a process-local place,
    also one the walk took for an ordinary tensor (another reference to it).
    Returns the problems: such a tensor registered, on another device, or not
    the whole of its storage (a reader makes it as a new contiguous tensor),
    and a tensor that shares a storage with one (a view of it would still be
    sent)."""
    local_ids = {id(tensor) for tensor in walk.local.values()}
    problems = []
    for path, tensor in list(walk.tensors.items()):
        if id(tensor) not in local_ids:
            continue
        if len(path) == 2 and path[1][0] == "p":
            problems.append(f"{path_str(path)}: a process-local tensor registered here")
        else:
            walk.local[path] = walk.tensors.pop(path)
            walk.values[path] = ("l",)
            walk.homes.pop(id(tensor), None)
    storages: Dict[tuple, tuple] = {}
    for path, tensor in walk.local.items():
        if tensor.device.type != "cpu" and tensor.device != device:
            problems.append(
                f"{path_str(path)}: a process-local tensor on {tensor.device}"
            )
        elif tensor.numel():
            whole = tensor.untyped_storage()
            if not (
                tensor.is_contiguous()
                and tensor.storage_offset() == 0
                and whole.nbytes() == tensor.numel() * tensor.element_size()
            ):
                problems.append(
                    f"{path_str(path)}: a process-local tensor that is not the "
                    "whole of its storage"
                )
            storages.setdefault((tensor.device, whole.data_ptr()), path)
    for path, tensor in walk.tensors.items():
        if tensor.numel() and tensor.device.type != "meta":
            key = (tensor.device, tensor.untyped_storage().data_ptr())
            if key in storages:
                problems.append(
                    f"{path_str(path)}: shares its storage with the process-local "
                    f"tensor at {path_str(storages[key])}"
                )
    return problems


def _locals(after: Walk) -> List[Local]:
    """Each process-local tensor once: its dtype, shape, device and places."""
    places: Dict[int, List[tuple]] = {}
    held: Dict[int, torch.Tensor] = {}
    for path, tensor in after.local.items():
        places.setdefault(id(tensor), []).append(path)
        held[id(tensor)] = tensor
    return [
        Local(
            dtype_name(held[i].dtype),
            list(held[i].shape),
            "cpu" if held[i].device.type == "cpu" else "model",
            [encode_path(path) for path in paths],
        )
        for i, paths in places.items()
    ]


def _encode_context(
    baseline: Baseline,
    after: Walk,
    table: StorageTable,
    tensor_ids: Dict[int, int],
    device: torch.device,
) -> EncodeContext:
    def keep(here, value):
        built = baseline.values.get(here)
        if built is None:
            return False
        if is_data_leaf(value):  # unchanged data: the reader's own may differ
            return built == ("v", leaf_key(value))
        return built[0] in ("c", "o", "g", "x", "y") and baseline.same(here, value)

    # A list or dict the reader already holds somewhere (the one the model
    # was built with) is referred to there; any other one, at its first place.
    paths = first_paths(after)
    for path, desc in reversed(list(after.values.items())):
        if (
            desc[0] == "c"
            and desc[1] in MUTABLE_CONTAINERS
            and baseline.same(path, after.objects[path])
        ):
            paths[desc[2]] = path
    return EncodeContext(
        tensor_ids=tensor_ids,
        first_paths=paths,
        module_names={id(m): name for name, m in reversed(after.modules)},
        device=device,
        keep=keep,
        addresses=sorted(
            {
                (whole.data_ptr(), whole.data_ptr() + nbytes)
                for _, nbytes, whole in table.storages
                if nbytes
            }
            | {
                (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                for t in after.local.values()
                if t.numel()
            }
        ),
    )


def _global_name(value: Any) -> str:
    for module_name, module in list(sys.modules.items()):
        if issubclass(type(module), types.ModuleType):
            for key, item in list(vars(module).items()):
                if item is value:
                    return f"{module_name}:{key}"
    return f"a module-level {type(value).__qualname__}"


def _global_change(path: tuple, old: tuple, new: tuple, value: Any) -> str:
    fields = "its items"
    a, b = old[3], new[3]
    if type(a[0]) is not str and type(b[0]) is not str:  # (type, id, items)
        before, after = dict(a[2]), dict(b[2])
        changed = [
            k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
        ]
        fields = ", ".join(sorted(map(str, changed))[:10])
    return (
        f"{path_str(path)}: loading changed {_global_name(value)}, a module-level "
        f"object the model refers to ({fields}); a reader keeps its own"
    )


def _refill(old: tuple, new: tuple) -> bool:
    """Whether the same container changed in a way per-item operations do
    not repeat: a list's length, an OrderedDict's order, a defaultdict's
    factory."""
    return old[3] != new[3] and issubclass(new[1], (list, OrderedDict, defaultdict))


def _diff(
    baseline: Baseline,
    after: Walk,
    ctx: EncodeContext,
    modules_now: Dict[str, int],
    problems: List[str],
) -> List[list]:
    """The operations that turn a model as built into ``after``: set a value
    where one changed or appeared, refill a container whose length, order or
    factory changed, delete what is gone. A tensor is never an operation:
    every place of a tensor is installed. A change of a module-level object
    is a problem.

    An operation stands for everything under its path except the items it
    keeps (the reader's own object, see EncodeContext.keep) and the
    attributes of tensors: those are compared on their own."""
    before = baseline.values
    ops: List[list] = []
    op_paths: Set[tuple] = set()
    homes: Set[tuple] = set(after.homes.values())
    apart: Set[tuple] = set(homes)  # compared on their own
    registered = {
        (name, key) for name, registry in after.registries for key, *_ in registry
    }

    def shadowed(path):
        if path in apart:
            return False
        for n in range(len(path) - 1, 1, -1):
            if path[:n] in apart:
                return False
            if path[:n] in op_paths:
                return True
        return False

    def emit(kind, path, value=None):
        try:
            op = [kind, encode_path(path, ctx)]
            if kind != "del":
                op.append(encode_value(value, path, ctx))
        except Unencodable as e:
            problems.append(
                f"{path_str(path)}: loading left {e}; a reader cannot get it"
            )
            return
        ops.append(op)
        op_paths.add(path)
        if kind != "del":
            apart.update(_keep_paths(op[2], path, set()))

    for path, new in after.values.items():
        if shadowed(path):
            continue
        old = before.get(path)
        if (
            path[-1][0] == "a"
            and path[:-1] in homes
            and before.get(path[:-1]) != ("t",)
        ):
            old = None  # an attribute of a tensor where the model held none
        tag = new[0]
        same = (
            old is not None
            and old[0] == tag
            and tag in ("c", "o", "g", "x", "y")
            and baseline.same(path, after.objects[path])
        )
        if tag in ("t", "l") or (
            old == new and (same or tag not in ("c", "o", "g", "x", "y"))
        ):
            continue  # unchanged, or a tensor: every place of a tensor is installed
        if same and tag == "g":
            problems.append(_global_change(path, old, new, after.objects[path]))
        elif same and tag == "c" and _refill(old, new):
            emit("fill", path, after.objects[path])
        elif not same:
            emit("set", path, after.objects[path])
        # else: the same container or object, compared item by item

    for path in before:
        if path in after.values or path[-1][0] == "p" or shadowed(path):
            continue
        parent = path[:-1]
        if len(parent) == 1:
            same_parent = parent[0] in modules_now and (
                (parent[0], path[-1][1]) not in registered
            )
        else:
            now = after.values.get(parent)
            same_parent = now is not None and (
                (now[0] in ("c", "o") and baseline.same(parent, after.objects[parent]))
                or (now[0] == "t" and parent in after.homes.values())
            )
        if same_parent:
            emit("del", path)
    return ops


def _keep_paths(encoded: Any, path: tuple, out: Set[tuple]) -> Set[tuple]:
    if type(encoded) is list and encoded:
        if encoded[0] == "keep":
            out.add(path)
        elif encoded[0] in ("tuple", "list"):
            for i, item in enumerate(encoded[1]):
                _keep_paths(item, path + (("i", i),), out)
        elif encoded[0] in ("dict", "odict", "ddict"):
            for key, item in encoded[-1]:
                _keep_paths(item, path + (("k", decode_key(key)),), out)
    return out


# ---------------------------------------------------------------------------
# Reader: install a manifest in a model built on the meta device
# ---------------------------------------------------------------------------

_MISSING = object()


def _check_storages(
    m: Manifest, metas: Mapping[str, StorageMeta], given: Mapping[str, torch.Tensor]
) -> torch.device:
    """``given`` and the SnapshotManifest's ``metas`` must both be exactly the
    storages with bytes: each a whole uint8 storage of its size, on the CPU
    or, for the model's ones, all on one device. Returns that device."""
    expected = {str(sid): e for sid, e in enumerate(m.storages) if e.nbytes}
    problems = []
    if set(given) != set(expected) or set(metas) != set(expected):
        problems.append(
            f"{len(given)} storages given and {len(metas)} listed for {len(expected)}"
        )
    device = None
    for name, entry in expected.items():
        storage, meta = given.get(name), metas.get(name)
        if storage is None or meta is None:
            problems.append(f"storage {name}: missing")
            continue
        if entry.where == "model" and device is None:
            device = storage.device
        on = device if entry.where == "model" else torch.device("cpu")
        if not (
            type(storage) is torch.Tensor
            and storage.dtype == torch.uint8
            and storage.device == on
            and storage.dim() == 1
            and storage.is_contiguous()
            and storage.numel() == entry.nbytes
            and storage.storage_offset() == 0
            and storage.untyped_storage().nbytes() == entry.nbytes
            and isinstance(meta, StorageMeta)
            and meta.nbytes == entry.nbytes
            and meta.device == storage.device.type
        ):
            problems.append(f"storage {name}: not {entry.nbytes} bytes on {on}")
    if problems:
        raise SnapshotError("the storages are not the manifest's", problems)
    if device is None:  # the model has no bytes off the CPU
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    return device


def _install(
    model: nn.Module,
    m: Manifest,
    given: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Tuple[List[torch.Tensor], int]:
    """Install ``m`` in ``model``: the tensors (views of the storages), the
    registries, the modules loading removed, then the places and operations.
    Returns the tensors by tensor id, and how many kept the identity of the
    tensor the reader built."""
    _check_structure(model, m.structure)
    modules = dict(model.named_modules(remove_duplicate=False))
    storages = [
        given[str(sid)]
        if e.nbytes
        else torch.empty(
            0, dtype=torch.uint8, device=device if e.where == "model" else "cpu"
        )
        for sid, e in enumerate(m.storages)
    ]
    tensors, swapped = _build_tensors(modules, m.tensors, storages)
    _set_registries(modules, m.registered, tensors)
    for name in m.removed_modules:
        parent, _, leaf = name.rpartition(".")
        if parent in modules:
            _dict_pop(modules[parent]._modules, leaf)
    _apply_changes(modules, m, tensors, device)
    return tensors, swapped


def _check_structure(model: nn.Module, want: Structure) -> None:
    digest, modules, registrations = structure_digest(model)
    if digest != want.digest:
        raise SnapshotError(
            "this reader built another model than the owner",
            [
                f"{modules} module names and {registrations} registrations here, "
                f"{want.modules} and {want.registrations} there"
            ],
        )


def _build_tensors(
    modules: Dict[str, nn.Module],
    entries: Sequence[TensorEntry],
    storages: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], int]:
    tensors: List[torch.Tensor] = []
    used: Set[int] = set()
    swapped, not_swapped = 0, []
    for e in entries:
        view = make_view(
            storages[e.sid].untyped_storage(), e.offset, e.dtype, e.shape, e.stride
        )
        home = decode_path(e.home)
        try:
            built = resolve_path(modules, home)
        except (KeyError, IndexError, TypeError):
            built = None
        if not issubclass(type(built), torch.Tensor) or id(built) in used:
            built = None
        tensor = _new_tensor(view, e.cls, e.requires_grad, built)
        if built is not None:
            used.add(id(built))
            if type(built) is type(tensor) and _swap(built, tensor):
                tensor = built
                swapped += 1
            else:
                vars_of(tensor).update(vars_of(built))
                not_swapped.append(path_str(home))
        tensors.append(tensor)
    if not_swapped:
        # Whatever still refers to the tensor built there keeps a meta
        # tensor; it fails when used, not silently.
        logger.warning(
            "load-diff: %d tensor(s) replaced instead of swapped, e.g. %s",
            len(not_swapped),
            ", ".join(not_swapped[:5]),
        )
    return tensors, swapped


def _new_tensor(view: torch.Tensor, cls: str, requires_grad: bool, built: Any):
    if cls == "tensor":
        return view.requires_grad_(requires_grad) if requires_grad else view
    if cls == "parameter":
        return nn.Parameter(view, requires_grad=requires_grad)
    if (
        built is not None
        and f"{type(built).__module__}.{type(built).__qualname__}" == cls
    ):
        return torch.Tensor._make_subclass(type(built), view, requires_grad)
    raise SnapshotError(
        "cannot install",
        [f"a {cls} tensor, and this reader built no such tensor there"],
    )


def _swap(built: torch.Tensor, new: torch.Tensor) -> bool:
    """Give ``built`` the data of ``new`` and keep its own attributes."""
    saved = dict(vars_of(built))
    try:
        torch.utils.swap_tensors(built, new)
    except Exception:  # e.g. another tensor still refers to it
        return False
    vars_of(built).clear()
    vars_of(built).update(saved)
    return True


def _set_registries(
    modules: Dict[str, nn.Module],
    registered: Sequence[Registry],
    tensors: Sequence[torch.Tensor],
) -> None:
    """Every module's parameters and buffers as the owner's after loading."""
    for registry in registered:
        module = modules[registry.module]
        names = set()
        for e in registry.entries:
            names.add(e.name)
            tensor = None if e.tid is None else tensors[e.tid]
            if e.kind == "p" and tensor is not None:
                if not isinstance(tensor, nn.Parameter):
                    raise SnapshotError(
                        "cannot install",
                        [f"{registry.module}.{e.name}: not a parameter"],
                    )
            if e.kind == "p":
                store, other = module._parameters, module._buffers
            else:
                store, other = module._buffers, module._parameters
            _dict_pop(other, e.name)
            _dict_pop(vars_of(module), e.name)
            _dict_set(store, e.name, tensor)
            non_persistent = module._non_persistent_buffers_set
            if e.kind == "b" and (e.name in non_persistent) == e.persistent:
                _toggle(non_persistent, e.name)
        for store in (module._parameters, module._buffers):
            for key in [k for k in store if k not in names]:
                _dict_pop(store, key)
        for key in list(module._non_persistent_buffers_set):
            if key not in module._buffers:
                _toggle(module._non_persistent_buffers_set, key)


def _apply_changes(
    modules: Dict[str, nn.Module],
    m: Manifest,
    tensors: Sequence[torch.Tensor],
    device: torch.device,
) -> None:
    """The places whose holder exists already (so that the operations on a
    tensor's attributes find the tensor), the operations in the owner's walk
    order (each one's holder comes before it), then the places in what the
    operations made. A process-local tensor is the reader's own, zeroed."""
    boundaries = {decode_path(path) for path in m.boundaries}
    ctx = DecodeContext(modules, tensors, device)
    local = [
        torch.zeros(
            e.shape,
            dtype=parse_dtype(e.dtype),
            device=device if e.where == "model" else "cpu",
        )
        for e in m.local
    ]
    placed = [(decode_path(p.path), tensors[p.tid]) for p in m.places] + [
        (decode_path(path), local[i])
        for i, e in enumerate(m.local)
        for path in e.places
    ]
    later = []
    for path, tensor in placed:
        try:  # nothing is written when the holder or the index is missing
            _place(modules, path, tensor, boundaries)
        except (KeyError, IndexError, TypeError):
            later.append((path, tensor))
    for op in m.ops:
        path = decode_path(op[1])
        with _at(path):
            _apply(op, path, ctx, boundaries)
    for path, tensor in later:
        with _at(path):
            _place(modules, path, tensor, boundaries)


class _at:
    """Name the path when installing there fails on a missing holder."""

    def __init__(self, path: tuple):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        if kind is not None and issubclass(kind, (KeyError, IndexError, TypeError)):
            raise SnapshotError(
                "cannot install",
                [f"{path_str(self.path)}: {kind.__name__}: {error}"],
            ) from error
        return False


def _exact_dict(d: Any) -> None:
    if type(d) not in (dict, OrderedDict, defaultdict):
        raise SnapshotError(
            "cannot install", [f"a {type(d).__qualname__} is not a dict"]
        )


def _dict_set(d: dict, key: Any, value: Any) -> None:
    _exact_dict(d)
    type(d).__setitem__(d, key, value)


def _dict_pop(d: dict, key: Any) -> None:
    _exact_dict(d)
    if dict.__contains__(d, key):
        type(d).__delitem__(d, key)


def _toggle(members: set, key: str) -> None:
    if key in members:
        members.discard(key)
    else:
        members.add(key)


def _writable(
    modules: Dict[str, nn.Module],
    path: tuple,
    boundaries: Set[tuple],
    in_place: bool = False,
) -> None:
    """Refuse an operation that would write a module's own bookkeeping, a
    registration, or into a module-level object (the owner's boundaries).
    Setting a field rebinds it and writes into its holder; ``in_place``
    (fill) also writes into the object the field holds."""
    steps = path[1:]
    last = len(path) + 1 if in_place else len(path)
    bad = (
        not steps
        or steps[-1][0] == "p"  # a registration: installed as one
        or any(kind == "p" for kind, _ in steps[1:])
        or any(path[:n] in boundaries for n in range(2, last))
    )
    holder = modules[path[0]]
    for n, (kind, key) in enumerate(steps):
        if bad:
            break
        if kind == "a" and issubclass(type(holder), nn.Module):
            bad = key in MODULE_FIELDS or any(
                key in d for d in (holder._parameters, holder._buffers, holder._modules)
            )
        if not bad and n + 1 < len(steps):
            holder = step_into(holder, kind, key)
    if bad:
        raise SnapshotError(
            "cannot install", [f"{path_str(path)}: not a place to write"]
        )


def _assign(modules: Dict[str, nn.Module], path: tuple, value: Any) -> None:
    holder = resolve_path(modules, path[:-1])
    kind, key = path[-1]
    if kind == "a":
        _dict_set(vars_of(holder), key, value)
    elif kind == "k":
        _dict_set(holder, key, value)
    elif kind == "i" and type(holder) is list:
        list.__setitem__(holder, key, value)
    elif kind == "i" and issubclass(type(holder), tuple):
        items = list(tuple.__iter__(holder))
        items[key] = value
        _assign(modules, path[:-1], tuple.__new__(type(holder), items))
    else:
        raise SnapshotError("cannot install", [f"{path_str(path)}: cannot write there"])


def _apply(op: list, path: tuple, ctx: DecodeContext, boundaries) -> None:
    _writable(ctx.modules, path, boundaries, in_place=op[0] == "fill")
    if op[0] == "del":
        holder = resolve_path(ctx.modules, path[:-1])
        kind, key = path[-1]
        if kind not in ("a", "k"):
            raise SnapshotError("cannot install", [f"{path_str(path)}: cannot delete"])
        _dict_pop(vars_of(holder) if kind == "a" else holder, key)
        return
    value = decode_value(op[2], path, ctx)
    if op[0] == "set":
        _assign(ctx.modules, path, value)
        return
    # fill: the same list or dict, with the owner's items (and factory)
    target = resolve_path(ctx.modules, path)
    kind = type(target)
    if kind is not type(value) or kind not in MUTABLE_CONTAINERS:
        raise SnapshotError("cannot install", [f"{path_str(path)}: not a {kind}"])
    # ``value`` may be ``target`` itself (a reference to it): read it first.
    items = items_of(value)
    factory = getattr(value, "default_factory", None)
    kind.clear(target)
    if kind is list:
        list.extend(target, [v for _, v in items])
    else:
        for k, v in items:
            kind.__setitem__(target, k, v)
    if kind is defaultdict:
        target.default_factory = factory


def _place(modules, path, tensor, boundaries) -> None:
    """Point ``path`` at ``tensor`` (a place of it), unless it is there."""
    try:
        here = resolve_path(modules, path)
    except (KeyError, IndexError):  # a place that loading added
        here = _MISSING
    if here is not tensor:
        _writable(modules, path, boundaries)
        _assign(modules, path, tensor)


# ---------------------------------------------------------------------------
# Reader: the self-test
# ---------------------------------------------------------------------------


def _holds(want: Any, value: Any, ctx: DecodeContext) -> bool:
    """Whether ``value`` is what the encoding ``want`` describes: data equal
    (floats by their bits), tensors, modules and references the very same
    objects, containers of the same type item by item; ``["keep"]`` stands
    for whatever the reader holds there."""
    if type(want) is not list:
        return type(value) is type(want) and leaf_key(value) == leaf_key(want)
    tag, args = want[0], want[1:]
    if tag == "keep":
        return True
    if tag == "tensor":
        return value is ctx.tensors[args[0]]
    if tag == "module":
        return value is ctx.modules.get(args[0])
    if tag == "ref":
        try:
            return value is resolve_path(ctx.modules, decode_path(args[0]))
        except (KeyError, IndexError, TypeError):
            return False
    if tag in ("tuple", "list"):
        kind = tuple if tag == "tuple" else list
        return (
            type(value) is kind
            and len(value) == len(args[0])
            and all(_holds(w, v, ctx) for w, v in zip(args[0], value))
        )
    if tag in ("dict", "odict", "ddict"):
        kind = {"dict": dict, "odict": OrderedDict, "ddict": defaultdict}[tag]
        if type(value) is not kind:
            return False
        items = items_of(value)
        if [k for k, _ in items] != [decode_key(k) for k, _ in args[-1]]:
            return False
        if tag == "ddict" and value.default_factory is not (
            None if args[0] is None else FACTORIES[args[0]]
        ):
            return False
        return all(_holds(w, v, ctx) for (_, w), (_, v) in zip(args[-1], items))
    decoded = decode_value(want, ("",), ctx)  # dtype, device, size, enum, set
    return type(value) is type(decoded) and value == decoded


def check_install(
    model: nn.Module, m: Manifest, tensors: Sequence[torch.Tensor], device
) -> Dict[str, Any]:
    """The self-test: walk the installed model as the owner walked it, and
    require, from the tensors actually there, every place to hold the
    tensor the manifest names, as the same view (storage, byte offset,
    dtype, shape, stride) of a storage with the owner's BLAKE3 digest; every
    registration as the owner's; every value loading set to read back as
    set, and every deleted one gone. It shows that install repeated export;
    it does not validate the owner. Raises SnapshotError."""
    tic = time.perf_counter()
    boundaries = {decode_path(path) for path in m.boundaries}
    walk = walk_model(model, boundaries=boundaries, keep_objects=True)
    modules = dict(walk.modules)
    problems = list(walk.problems) + _localize(walk, device)
    table = build_storage_table(walk.tensors, device)
    problems += table.problems + _local_problems(walk, m, table, device)
    expected: Dict[tuple, int] = {}
    for registry in m.registered:
        for e in registry.entries:
            if e.tid is not None:
                expected[(registry.module, ("p", e.name))] = e.tid
    for place in m.places:
        expected[decode_path(place.path)] = place.tid
    sids: Dict[int, int] = {}  # reader's storage id -> owner's, and back
    back: Dict[int, int] = {}
    for path in walk.tensors.keys() | expected.keys():
        tid, tensor = expected.get(path), walk.tensors.get(path)
        if tid is None:
            problems.append(f"{path_str(path)}: a tensor the owner does not have here")
        elif tensor is None:
            problems.append(f"{path_str(path)}: no tensor here")
        elif tensor is not tensors[tid] or id(tensor) not in table.views:
            problems.append(f"{path_str(path)}: not the tensor installed for it")
        else:
            sid, *geometry = table.views[id(tensor)]
            want = m.tensors[tid]
            if (
                sids.setdefault(sid, want.sid) != want.sid
                or back.setdefault(want.sid, sid) != sid
                or geometry != [want.offset, want.dtype, want.shape, want.stride]
            ):
                problems.append(f"{path_str(path)}: another view than the owner's")
    for sid, (where, nbytes, _) in enumerate(table.storages):
        owner = m.storages[sids[sid]] if sid in sids else None
        if owner is not None and (owner.nbytes, owner.where) != (nbytes, where):
            problems.append(f"storage {sids[sid]}: {nbytes} bytes on {where} here")
    problems += _registration_problems(walk, m, tensors)
    ctx = DecodeContext(modules, tensors, device)
    for op in m.ops:
        path = decode_path(op[1])
        try:
            value = resolve_path(modules, path)
        except (KeyError, IndexError, TypeError):
            if op[0] != "del":
                problems.append(f"{path_str(path)}: missing")
            continue
        if op[0] == "del":
            problems.append(f"{path_str(path)}: still here")
        elif not _holds(op[2], value, ctx):
            problems.append(f"{path_str(path)}: not the value the owner set")
    t_walk = time.perf_counter()
    if not problems:
        digests = storage_digests([whole for _, _, whole in table.storages])
        for sid, digest in enumerate(digests):
            owner_sid = sids.get(sid)
            if owner_sid is None or m.storages[owner_sid].digest != digest:
                problems.append(f"storage {owner_sid}: its bytes are not the owner's")
    if problems:
        raise SnapshotError(
            "the installed model is not the one the owner exported", problems
        )
    return {"check_s": t_walk - tic, "digest_s": time.perf_counter() - t_walk}


def _local_problems(
    walk: Walk, m: Manifest, table: StorageTable, device: torch.device
) -> List[str]:
    """Every place of a process-local tensor holds the reader's own tensor of
    the owner's dtype and shape, one per entry, in no storage it was sent."""
    expected = {
        decode_path(path): i for i, e in enumerate(m.local) for path in e.places
    }
    sent = {whole.untyped_storage().data_ptr() for _, _, whole in table.storages}
    held: Dict[int, int] = {}
    problems = []
    for path in walk.local.keys() | expected.keys():
        i, tensor = expected.get(path), walk.local.get(path)
        if i is None:
            problems.append(
                f"{path_str(path)}: a process-local tensor the owner does not have here"
            )
            continue
        if tensor is None:
            problems.append(f"{path_str(path)}: no process-local tensor here")
            continue
        e = m.local[i]
        on = device if e.where == "model" else torch.device("cpu")
        if not (
            tensor.dtype == parse_dtype(e.dtype)
            and list(tensor.shape) == e.shape
            and tensor.device == on
            and held.setdefault(i, id(tensor)) == id(tensor)
            and (tensor.numel() == 0 or tensor.untyped_storage().data_ptr() not in sent)
        ):
            problems.append(
                f"{path_str(path)}: not the reader's own process-local tensor"
            )
    return problems


def _registration_problems(
    walk: Walk, m: Manifest, tensors: Sequence[torch.Tensor]
) -> List[str]:
    """Registrations, by name: install keeps the reader's order of the names."""
    got = {
        name: {key: (kind, id(t), persistent) for key, kind, t, persistent in registry}
        for name, registry in walk.registries
    }
    problems = []
    for registry in m.registered:
        want = {
            e.name: (
                e.kind,
                id(None if e.tid is None else tensors[e.tid]),
                e.persistent,
            )
            for e in registry.entries
        }
        if got.pop(registry.module, None) != want:
            problems.append(
                f"{registry.module or '<model>'}: its registrations are not the owner's"
            )
    for name in got:
        problems.append(f"{name or '<model>'}: a module the owner does not have")
    return problems
