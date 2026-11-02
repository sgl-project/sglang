# SPDX-License-Identifier: Apache-2.0
"""Keep tensors that alias a parameter or buffer valid when a loader replaces
that parameter or buffer.

A module may keep a view of one of its parameters as a plain attribute
(``RadixLinearAttention.conv_weights`` is a 2-D view of its layer's
convolution weight), a second reference to it in a list or tuple, a view of
it registered as a buffer, or the same parameter registered under a second
name (``RadixLinearAttention.A_log``). A loader that replaces the parameter
with a new tensor instead of copying into it leaves all of these on the old
tensor: the weight cache loader builds the model on the meta device and then
points every parameter at the daemon's memory, and quantization
post-processing may replace parameters before the presharded loader copies
its checkpoint in.

``find_tensor_aliases`` records these aliases before the replacement, by the
name of the parameter or buffer each one aliases and its geometry relative
to it. ``rebase_tensor_aliases`` rebuilds, on whatever tensor has that name
afterwards, the aliases that still hold what they held when recorded, and
returns those it cannot rebuild. Nothing here depends on attribute names.
Rebuilt views carry no autograd history; they are for inference.

What is looked at is what ``model_walk`` goes through: every module's
registered parameters and buffers, and its plain attributes together with the
tuples, lists and dicts they hold, up to ``_MAX_DEPTH`` levels deep. Tensors
held by other objects (a quant method, a kernel's cached arguments) are not
seen.
"""

from __future__ import annotations

import dataclasses
import math
import weakref
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from sglang.srt.model_loader.model_walk import (
    Place,
    container_items,
    get_at,
    get_item,
    registered_tensors,
    replace_item,
    tensor_names,
    walk_plain_attributes,
)

__all__ = ["TensorAlias", "find_tensor_aliases", "rebase_tensor_aliases"]

# How deep to look into the tuples, lists and dicts a module holds.
_MAX_DEPTH = 5


@dataclasses.dataclass(frozen=True)
class TensorAlias:
    """One tensor that aliases the parameter or buffer named ``target``.

    It lives at ``path`` (an attribute, then container indices or keys) in
    the module ``owner``; when ``registered``, ``path`` names a parameter or
    buffer of that module.
    """

    owner: str
    path: Tuple[Any, ...]
    target: str
    registered: bool
    # The alias is the target tensor itself rather than a view of it.
    same_object: bool
    # Bytes from the target's first element to the alias's first element.
    byte_offset: int
    size: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    target_size: Tuple[int, ...]
    target_stride: Tuple[int, ...]
    target_dtype: torch.dtype
    # The alias is the whole target, contiguous, possibly reshaped: it can
    # follow a target whose shape changed but whose elements did not.
    whole: bool
    # Other tensors over exactly the same bytes, when nothing shows which one
    # the alias was made from.
    alternates: Tuple[str, ...]
    # Why the alias cannot be rebuilt, if it cannot.
    problem: Optional[str]
    # What the alias held when recorded. Weak, so recording keeps no memory
    # allocated; it tells a stale alias from one replaced or rebuilt since.
    tensor: weakref.ref = dataclasses.field(compare=False, repr=False)
    storage: Optional[StorageWeakRef] = dataclasses.field(compare=False, repr=False)

    @property
    def place(self) -> Place:
        return Place(self.owner, self.path, self.registered)

    @property
    def where(self) -> str:
        return self.place.where


def _storage_key(tensor: torch.Tensor) -> Optional[int]:
    # The StorageImpl address identifies a storage on every device, including
    # meta, where every data pointer is 0.
    try:
        return tensor.untyped_storage()._cdata
    except (NotImplementedError, RuntimeError):
        return None


def _byte_span(tensor: torch.Tensor) -> Tuple[int, int]:
    """[start, end) bytes of the storage that ``tensor`` can address."""
    itemsize = tensor.element_size()
    start = tensor.storage_offset() * itemsize
    if not tensor.numel():
        return start, start
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return start, start + extent * itemsize


def _is_dense(size: Sequence[int], stride: Sequence[int]) -> bool:
    """Whether a tensor's elements cover its byte span with no gaps or overlap."""
    expected = 1
    for step, length in sorted((st, n) for n, st in zip(size, stride) if n != 1):
        if step != expected:
            return False
        expected *= length
    return True


def _same_layout(
    tensor: torch.Tensor, size: Sequence[int], stride: Sequence[int]
) -> bool:
    # The stride of a dimension of size 1 never addresses anything.
    return tuple(tensor.shape) == tuple(size) and all(
        n == 1 or a == b for n, a, b in zip(size, tensor.stride(), stride)
    )


def _make_alias(
    owner,
    path,
    tensor,
    target_name,
    target,
    *,
    registered,
    alternates=(),
    problem=None,
):
    same_object = tensor is target
    alias_lo, alias_hi = _byte_span(tensor)
    target_lo, target_hi = _byte_span(target)
    whole = False
    if not same_object and problem is None:
        if alias_lo < target_lo or alias_hi > target_hi:
            problem = f"it spans more than {target_name}"
        elif not _is_dense(target.shape, target.stride()):
            problem = f"{target_name} does not cover its storage densely"
        elif any(t.is_conj() or t.is_neg() for t in (tensor, target)):
            # The bits change what the bytes mean; a strided view cannot
            # carry them over.
            problem = f"it or {target_name} is a conjugate or negative view"
        else:
            whole = (
                tensor.dtype == target.dtype
                and alias_lo == target_lo
                and tensor.numel() == target.numel()
                and tensor.is_contiguous()
                and target.is_contiguous()
            )
    key = _storage_key(tensor)
    return TensorAlias(
        owner=owner,
        path=path,
        target=target_name,
        registered=registered,
        same_object=same_object,
        byte_offset=alias_lo - target_lo,
        size=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        target_size=tuple(target.shape),
        target_stride=tuple(target.stride()),
        target_dtype=target.dtype,
        whole=whole,
        alternates=tuple(alternates),
        problem=problem,
        tensor=weakref.ref(tensor),
        storage=StorageWeakRef(tensor.untyped_storage()) if key is not None else None,
    )


def _choose_target(tensor, first_name, by_storage):
    """The registered tensor that ``tensor`` aliases: itself if it is one,
    else the one it is a view of, else the smallest one whose bytes contain
    its bytes. Returns (name, target, alternates, problem), or None."""
    if id(tensor) in first_name:
        return first_name[id(tensor)], tensor, (), None
    if _storage_key(tensor) is None:
        return None
    base = tensor._base
    if base is not None and id(base) in first_name:
        return first_name[id(base)], base, (), None
    candidates = by_storage.get(_storage_key(tensor))
    if not candidates:
        return None
    lo, hi = _byte_span(tensor)
    span = lambda t: _byte_span(t)[1] - _byte_span(t)[0]  # noqa: E731
    containing = [
        (n, t)
        for n, t in candidates
        if _byte_span(t)[0] <= lo and hi <= _byte_span(t)[1]
    ]
    if not containing:
        names = ", ".join(n for n, _ in candidates)
        name, target = candidates[0]
        return name, target, (), f"it spans more than one tensor ({names})"
    smallest = min(span(t) for _, t in containing)
    same = [(n, t) for n, t in containing if span(t) == smallest]
    (name, target), others = same[0], same[1:]
    return name, target, tuple(n for n, _ in others), None


def find_tensor_aliases(model: nn.Module) -> List[TensorAlias]:
    """Record every tensor in ``model`` that aliases one of its parameters or
    buffers: a tensor registered under more than one name, a registered
    tensor that is a view of another one, and plain attributes (and the
    tuples, lists and dicts they hold) that share a parameter's or buffer's
    storage. Call before a loader replaces any of them."""
    registered = registered_tensors(model)
    first_name: Dict[int, str] = {}
    by_storage: Dict[int, List[Tuple[str, torch.Tensor]]] = {}
    for tensor_id, names in tensor_names(registered).items():
        name = names[0]
        first_name[tensor_id] = name
        key = _storage_key(registered[name])
        if key is not None:
            by_storage.setdefault(key, []).append((name, registered[name]))

    aliases: List[TensorAlias] = []
    for name, tensor in registered.items():
        owner, _, attr = name.rpartition(".")
        if first_name[id(tensor)] != name:
            # The same tensor registered under a later name.
            aliases.append(
                _make_alias(
                    owner,
                    (attr,),
                    tensor,
                    first_name[id(tensor)],
                    tensor,
                    registered=True,
                )
            )
        elif (
            tensor._base is not None
            and id(tensor._base) in first_name
            and _storage_key(tensor) is not None
        ):
            # A registered view of another registered tensor.
            aliases.append(
                _make_alias(
                    owner,
                    (attr,),
                    tensor,
                    first_name[id(tensor._base)],
                    tensor._base,
                    registered=True,
                )
            )

    for held in walk_plain_attributes(model, max_depth=_MAX_DEPTH):
        tensor = held.value
        if not isinstance(tensor, torch.Tensor):
            continue
        choice = _choose_target(tensor, first_name, by_storage)
        if choice is None:
            continue
        name, target, alternates, problem = choice
        aliases.append(
            _make_alias(
                held.place.owner,
                held.place.path,
                tensor,
                name,
                target,
                registered=False,
                alternates=alternates,
                problem=problem,
            )
        )
    return aliases


def _get(model: nn.Module, alias: TensorAlias) -> Any:
    return get_at(model, alias.place)


def _set(model: nn.Module, alias: TensorAlias, new: torch.Tensor) -> None:
    module = model.get_submodule(alias.owner)
    if alias.registered:
        current = getattr(module, alias.path[0])
        if isinstance(current, nn.Parameter) and not isinstance(new, nn.Parameter):
            new = nn.Parameter(new, requires_grad=current.requires_grad)
        # Registers a parameter as a parameter and a buffer as a buffer.
        setattr(module, alias.path[0], new)
        return
    chain = [vars(module)[alias.path[0]]]
    for item in alias.path[1:-1]:
        chain.append(get_item(chain[-1], item))
    for container, item in reversed(list(zip(chain, alias.path[1:]))):
        if isinstance(container, tuple):
            items = [value for _, value in container_items(container)]
            items[item] = new
            new = (
                type(container)._make(items)
                if hasattr(container, "_make")
                else type(container)(items)
            )
        else:
            replace_item(container, item, new)
            return
    # A plain attribute: keep it out of the parameter registry even if
    # ``new`` is a parameter.
    vars(module)[alias.path[0]] = new


def _rebuild(alias: TensorAlias, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if alias.dtype == target.dtype:
            return target.as_strided(
                alias.size,
                alias.stride,
                target.storage_offset() + alias.byte_offset // target.element_size(),
            )
        start = target.storage_offset() * target.element_size() + alias.byte_offset
        if start % alias.dtype.itemsize:
            raise RuntimeError("it would start inside one of its elements")
        return torch.empty(0, dtype=alias.dtype, device=target.device).set_(
            target.untyped_storage(),
            start // alias.dtype.itemsize,
            alias.size,
            alias.stride,
        )


def _moved_apart(other: Optional[torch.Tensor], target: torch.Tensor) -> bool:
    """Whether ``other``, which covered the same bytes as ``target`` when
    recorded, no longer does."""
    return (
        other is None
        or _storage_key(other) != _storage_key(target)
        or _byte_span(other) != _byte_span(target)
    )


def _rebase_ties(
    model: nn.Module,
    registered: Dict[str, torch.Tensor],
    target: str,
    members: Sequence[TensorAlias],
) -> Tuple[int, List[str]]:
    """A tensor registered under several names: if exactly one new tensor
    has replaced it under some of them, register that one under the others
    too. Several different replacements were loaded separately and stay, as
    long as no name still holds the old tensor."""
    old = members[0].tensor()
    if old is None:
        return 0, []
    names = [target] + [m.place.name for m in members]
    current = {name: registered.get(name) for name in names}
    replacements = {
        id(t): t for t in current.values() if t is not None and t is not old
    }
    stale = [name for name, tensor in current.items() if tensor is old]
    if not replacements or not stale:
        return 0, []
    if len(replacements) > 1:
        return 0, [
            f"{', '.join(stale)}: still the tensor that {len(replacements)} "
            f"different tensors replaced under its other names"
        ]
    (new,) = replacements.values()
    if isinstance(new, nn.Parameter) != isinstance(old, nn.Parameter):
        return 0, [f"{target}: changed between parameter and buffer"]
    for name in stale:
        owner, _, attr = name.rpartition(".")
        setattr(model.get_submodule(owner), attr, new)
    return len(stale), []


def _rebase_one(
    model: nn.Module, registered: Dict[str, torch.Tensor], alias: TensorAlias
) -> Tuple[int, Optional[str]]:
    try:
        current = _get(model, alias)
    except (AttributeError, KeyError, IndexError, TypeError):
        return 0, None
    # Only an alias that still holds what it held when recorded is stale; one
    # the model or the loader has replaced or rebuilt since is left alone.
    if not isinstance(current, torch.Tensor) or current is not alias.tensor():
        return 0, None
    if not alias.same_object and _storage_key(current) != (
        alias.storage.cdata if alias.storage is not None else None
    ):
        return 0, None
    target = registered.get(alias.target)
    if target is None:
        return 0, f"{alias.where}: {alias.target} no longer exists"
    if alias.same_object:
        if target is current:
            return 0, None
        new = target
    elif _storage_key(target) == alias.storage.cdata:
        # The loader copied into the target: the alias is still valid.
        return 0, None
    elif alias.problem is not None:
        return 0, f"{alias.where}: {alias.problem}"
    elif any(_moved_apart(registered.get(n), target) for n in alias.alternates):
        return 0, (
            f"{alias.where}: it could have been made from {alias.target} or "
            f"{', '.join(alias.alternates)}, which no longer share memory"
        )
    elif target.is_meta:
        return 0, f"{alias.where}: {alias.target} is still on the meta device"
    elif target.is_conj() or target.is_neg():
        return 0, f"{alias.where}: {alias.target} is now a conjugate or negative view"
    elif target.dtype == alias.target_dtype and _same_layout(
        target, alias.target_size, alias.target_stride
    ):
        try:
            new = _rebuild(alias, target)
        except RuntimeError as e:
            return 0, f"{alias.where}: cannot rebuild it on {alias.target}: {e}"
    elif (
        alias.whole
        and target.dtype == alias.dtype
        and target.numel() == math.prod(alias.size)
        and target.is_contiguous()
    ):
        # The target's elements are the same, in the same order, in another
        # shape: the alias is still all of it.
        with torch.no_grad():
            new = target.view(alias.size)
    else:
        return 0, (
            f"{alias.where}: {alias.target} changed from "
            f"{list(alias.target_size)}/{alias.target_dtype} to "
            f"{list(target.shape)}/{target.dtype}"
        )
    try:
        _set(model, alias, new)
    except (TypeError, ValueError) as e:
        return 0, f"{alias.where}: cannot store the rebuilt tensor: {e}"
    return 1, None


def rebase_tensor_aliases(
    model: nn.Module, aliases: Sequence[TensorAlias]
) -> Tuple[int, List[str]]:
    """Rebuild every alias in ``aliases`` that still holds what it held when
    recorded while its target has been replaced since.

    Names that shared one tensor are rejoined first, since that can replace
    the target of a view. An alias the loader left valid (it copied into the
    target), or that the model or the loader replaced or rebuilt itself, is
    left alone. Returns the number of aliases rebuilt and a description of
    each stale alias that could not be.
    """
    rebuilt = 0
    problems: List[str] = []
    registered = registered_tensors(model)
    ties: Dict[str, List[TensorAlias]] = {}
    for alias in aliases:
        if alias.registered and alias.same_object:
            ties.setdefault(alias.target, []).append(alias)
    for target, members in ties.items():
        count, tie_problems = _rebase_ties(model, registered, target, members)
        rebuilt += count
        problems += tie_problems
    if rebuilt:
        registered = registered_tensors(model)
    for alias in aliases:
        if alias.registered and alias.same_object:
            continue
        count, problem = _rebase_one(model, registered, alias)
        rebuilt += count
        if problem is not None:
            problems.append(problem)
    return rebuilt, problems
