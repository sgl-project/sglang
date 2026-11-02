# SPDX-License-Identifier: Apache-2.0
"""Where a model holds its tensors and other values, and how to reach them.

A loader that replaces a model's parameters and buffers, instead of copying
into them, must keep valid whatever else in the model refers to them: views
and second references kept in plain attributes, second registrations of the
same tensor, values that post-load processing sets. This module decides where
to look and how to go through what it finds, the same way for every caller.
What to do with the values (compare, encode, rebuild), when to look, and how
deep, is up to the caller.

Where it looks:

- the parameters and buffers of every module, under every name they are
  registered under (``registered_tensors``, ``tensor_names``);
- the plain attributes of every module: its ``__dict__`` without nn.Module's
  own bookkeeping (``plain_attributes``), and the tuples, lists and dicts
  they hold (``walk_plain_attributes``). A module registered under several
  names is looked at once, under the first.

How it goes through containers:

- Containers are tuples, lists and dicts, subclasses included (namedtuple,
  OrderedDict, defaultdict, torch.Size, an enum with a tuple mixin). Items are
  read and replaced as the builtin type stores them, so a subclass that
  overrides iteration or item access can neither hide an item nor change
  what is stored; its other attributes are not looked at.
- Dict keys of any type are followed; a path holds the key object itself.
- Everything else is a leaf, not looked into: tensors, modules, enum members
  (but container ones), strings, sets, and other objects with their
  attributes.
- A container reached along several paths is visited along each, since each
  path is a place a caller may have to update. A container already on the
  current path (a cycle) is not entered again, nor is one below
  ``max_depth`` containers; ``on_cut`` hears of both.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn

__all__ = [
    "MODULE_FIELDS",
    "Held",
    "Place",
    "container_items",
    "get_at",
    "get_item",
    "is_container",
    "plain_attributes",
    "registered_tensors",
    "replace_item",
    "tensor_names",
    "walk_plain_attributes",
    "walk_value",
]

# nn.Module's own bookkeeping (_parameters, _buffers, _modules, hooks, ...).
MODULE_FIELDS = frozenset(vars(nn.Module()))

_CONTAINERS = (tuple, list, dict)
_MISSING = object()


@dataclasses.dataclass(frozen=True)
class Place:
    """Where a value is held: in the module named ``owner`` ("" for the model
    itself), at ``path``, an attribute name followed by the indices and keys
    of the containers below it. When ``registered``, ``path`` is the name of
    one of the module's parameters or buffers."""

    owner: str
    path: Tuple[Any, ...]
    registered: bool = False

    @classmethod
    def of_registered(cls, name: str) -> Place:
        """The place of the parameter or buffer with dotted name ``name``."""
        owner, _, attr = name.rpartition(".")
        return cls(owner, (attr,), registered=True)

    @property
    def name(self) -> str:
        """The dotted name of the attribute that ``path`` starts at."""
        return f"{self.owner}.{self.path[0]}" if self.owner else str(self.path[0])

    @property
    def where(self) -> str:
        """``name`` followed by the container indices and keys."""
        return self.name + "".join(f"[{key!r}]" for key in self.path[1:])


@dataclasses.dataclass(frozen=True)
class Held:
    """A value that a walk found, where, and the containers on the way to it
    from the attribute, outermost first (none if the attribute holds it)."""

    place: Place
    value: Any
    containers: Tuple[Any, ...] = ()


def is_container(value: Any) -> bool:
    """Whether a walk goes into ``value``: a tuple, list or dict, or a
    subclass of one."""
    return isinstance(value, _CONTAINERS)


def container_items(container: Any) -> List[Tuple[Any, Any]]:
    """(index or key, item) of a tuple, list or dict, as the builtin type
    stores them."""
    if isinstance(container, dict):
        return list(dict.items(container))
    if isinstance(container, list):
        return list(enumerate(list.__iter__(container)))
    if isinstance(container, tuple):
        return list(enumerate(tuple.__iter__(container)))
    raise TypeError(f"not a tuple, list or dict: {type(container).__name__}")


def get_item(container: Any, key: Any) -> Any:
    """The item at ``key`` of a tuple, list or dict, as the builtin type
    stores it. Raises KeyError or IndexError if there is none (without
    calling a dict subclass's ``__missing__``), TypeError for other values."""
    if isinstance(container, dict):
        item = dict.get(container, key, _MISSING)
        if item is _MISSING:
            raise KeyError(key)
        return item
    if isinstance(container, list):
        return list.__getitem__(container, key)
    if isinstance(container, tuple):
        return tuple.__getitem__(container, key)
    raise TypeError(f"not a tuple, list or dict: {type(container).__name__}")


def replace_item(container: Any, key: Any, value: Any) -> None:
    """Replace the item at ``key`` of a list or dict as the builtin type
    stores it. Raises KeyError or IndexError if there is none; a tuple cannot
    be changed (TypeError): its holder needs a new one."""
    if isinstance(container, dict):
        if not dict.__contains__(container, key):
            raise KeyError(key)
        dict.__setitem__(container, key, value)
    elif isinstance(container, list):
        list.__setitem__(container, key, value)
    else:
        raise TypeError(f"cannot replace an item of a {type(container).__name__}")


def walk_value(
    value: Any,
    path: Tuple[Any, ...] = (),
    *,
    max_depth: int,
    on_cut: Optional[Callable[[Tuple[Any, ...], str], None]] = None,
) -> Iterator[Tuple[Tuple[Any, ...], Any, Tuple[Any, ...]]]:
    """Yield (path, leaf, containers) for ``value`` if it is a leaf, else for
    every leaf in it: ``path`` extended by the indices and keys that lead to
    the leaf, and the containers on the way, outermost first. At most
    ``max_depth`` levels of containers are looked into. ``on_cut(path,
    reason)`` hears of each container that is not: ``"cycle"`` if it is on
    its own path, ``"depth"`` if it is too deep."""
    on_path: set = set()
    containers: List[Any] = []

    def walk(value, path):
        if not isinstance(value, _CONTAINERS):
            yield path, value, tuple(containers)
            return
        if id(value) in on_path or len(containers) >= max_depth:
            if on_cut is not None:
                on_cut(path, "cycle" if id(value) in on_path else "depth")
            return
        on_path.add(id(value))
        containers.append(value)
        try:
            for key, item in container_items(value):
                yield from walk(item, path + (key,))
        finally:
            containers.pop()
            on_path.discard(id(value))

    return walk(value, tuple(path))


def plain_attributes(module: nn.Module) -> List[Tuple[str, Any]]:
    """(name, value) of the plain attributes of ``module``: its ``__dict__``
    without nn.Module's own bookkeeping. A list, so that a caller may change
    the module while going through it."""
    return [
        (attr, value)
        for attr, value in vars(module).items()
        if attr not in MODULE_FIELDS
    ]


def walk_plain_attributes(
    model: nn.Module,
    *,
    max_depth: int,
    on_cut: Optional[Callable[[Place, str], None]] = None,
) -> Iterator[Held]:
    """Every leaf in the plain attributes of the modules of ``model`` (see
    walk_value), where it is held, and the containers on the way."""
    for owner, module in model.named_modules():
        cut = None
        if on_cut is not None:
            cut = lambda path, reason, owner=owner: on_cut(  # noqa: E731
                Place(owner, path), reason
            )
        for attr, value in plain_attributes(module):
            for path, leaf, containers in walk_value(
                value, (attr,), max_depth=max_depth, on_cut=cut
            ):
                yield Held(Place(owner, path), leaf, containers)


def registered_tensors(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Every parameter and buffer of ``model``, by every name it is
    registered under: one registered under several names (tied weights, a
    buffer that several modules share) appears under each. Parameters come
    first."""
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    return tensors


def tensor_names(tensors: Dict[str, torch.Tensor]) -> Dict[int, List[str]]:
    """The names each tensor in ``tensors`` (a name -> tensor map, such as
    registered_tensors gives) has, by its id, in the map's order."""
    names: Dict[int, List[str]] = {}
    for name, tensor in tensors.items():
        names.setdefault(id(tensor), []).append(name)
    return names


def get_at(model: nn.Module, place: Place) -> Any:
    """What ``model`` holds at ``place`` now. Raises AttributeError,
    KeyError, IndexError or TypeError if it holds nothing there."""
    module = model.get_submodule(place.owner)
    if place.registered:
        return getattr(module, place.path[0])
    value = vars(module)[place.path[0]]
    for key in place.path[1:]:
        value = get_item(value, key)
    return value
