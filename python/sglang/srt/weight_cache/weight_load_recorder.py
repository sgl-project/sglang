# SPDX-License-Identifier: Apache-2.0
"""Record checkpoint-to-runtime weight placement from native weight loaders.

The recorder treats ``model.load_weights`` as the source of truth.  Source
daemons record the real load while allowing writes to execute.  Target daemons
replay the same metadata with meta tensors and exact scalar values, suppressing
registered tensor writes.  Both modes produce model-independent logical boxes
backed by final registered parameter or buffer storage.

Supported loader operations must leave logical regions representable as
contiguous destination ranges. Unsupported mappings fail explicitly.

Mappings share ``shape`` and ``resolve(flat_index)``: a logical box, an index
within that box, and a contiguous run length. Views, concatenations and static
index selections compose this mapping and share final-copy lowering; they never
allocate an element-sized provenance tensor.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import inspect
import weakref
from bisect import bisect_right
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class WeightLoadRecordingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LogicalWeightMetadata:
    tensor_id: str
    shape: tuple[int, ...]
    dtype: str
    itemsize: int
    scalar_value: bool | int | float | None = None


@dataclass(frozen=True)
class RecordedWeightView:
    tensor_id: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    parameter: Any
    parameter_names: tuple[str, ...]
    byte_offset: int
    shard_dims: tuple[int, ...]
    expert_id: int | None
    layout_fingerprint: str

    @property
    def nbytes(self) -> int:
        return prod(self.local_shape) * int(self.parameter.element_size())


@dataclass(frozen=True)
class RecordedWeightInitialization:
    parameter: Any
    parameter_names: tuple[str, ...]
    byte_offset: int
    nbytes: int
    value: bool | int | float


@dataclass(frozen=True)
class WeightLoadPlan:
    logical_weights: tuple[LogicalWeightMetadata, ...]
    views: tuple[RecordedWeightView, ...]
    initializations: tuple[RecordedWeightInitialization, ...] = ()

    def initialize_local_regions(self) -> None:
        """Replay native constant writes after layout validation, never checkpoint copies."""
        with torch.no_grad():
            for region in self.initializations:
                itemsize = region.parameter.element_size()
                region.parameter.view(-1).narrow(
                    0, region.byte_offset // itemsize, region.nbytes // itemsize
                ).fill_(region.value)


@dataclass(frozen=True)
class _LogicalProvenance:
    metadata: LogicalWeightMetadata
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    layout_ops: tuple[str, ...] = ()
    unsupported_operation: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.local_shape

    def resolve(self, index: int) -> tuple[_LogicalProvenance, int, int]:
        return self, index, prod(self.local_shape) - index

    def unsupported(self, operation: str) -> _LogicalProvenance:
        return _LogicalProvenance(
            metadata=self.metadata,
            global_shape=self.global_shape,
            global_offset=self.global_offset,
            local_shape=self.local_shape,
            layout_ops=self.layout_ops,
            unsupported_operation=operation,
        )


@dataclass(frozen=True)
class _CompositePiece:
    provenance: _TensorProvenance
    tensor_offset: tuple[int, ...]


@dataclass(frozen=True)
class _CompositeProvenance:
    tensor_shape: tuple[int, ...]
    pieces: tuple[_CompositePiece, ...]
    dim: int
    starts: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor_shape

    def resolve(self, index: int) -> tuple[_LogicalProvenance, int, int]:
        coordinates = _unravel_index(index, self.shape)
        piece = self.pieces[bisect_right(self.starts, coordinates[self.dim]) - 1]
        relative = tuple(a - b for a, b in zip(coordinates, piece.tensor_offset))
        local = 0
        for i, n in zip(relative, piece.provenance.shape):
            local = local * n + i
        root, root_index, run = piece.provenance.resolve(local)
        return (
            root,
            root_index,
            min(
                run,
                _contiguous_tail(
                    local, piece.provenance.shape, _contiguous_strides(self.shape)
                ),
            ),
        )

    def unsupported(self, operation: str) -> _CompositeProvenance:
        return replace(
            self,
            pieces=tuple(
                _CompositePiece(
                    provenance=piece.provenance.unsupported(operation),
                    tensor_offset=piece.tensor_offset,
                )
                for piece in self.pieces
            ),
        )


@dataclass(frozen=True)
class _ViewProvenance:
    """Map a view's flat iteration order back to its input, without payloads.

    Keep each aliasing step until the final copy. Materializing a view preserves
    this iteration order, so clone/contiguous need no element-sized index map.
    """

    parent: _TensorProvenance
    input_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_stride: tuple[int, ...]
    offset: int

    @property
    def shape(self) -> tuple[int, ...]:
        return self.output_shape

    def unsupported(self, operation: str) -> _ViewProvenance:
        return replace(self, parent=self.parent.unsupported(operation))

    def resolve(self, index: int) -> tuple[_LogicalProvenance, int, int]:
        coordinates = _unravel_index(index, self.output_shape)
        address = self.offset + sum(
            i * stride for i, stride in zip(coordinates, self.output_stride)
        )
        input_coordinates = [0] * len(self.input_shape)
        for dim in sorted(
            range(len(self.input_shape)),
            key=self.input_stride.__getitem__,
            reverse=True,
        ):
            if self.input_shape[dim] == 1:
                continue
            input_coordinates[dim], address = divmod(address, self.input_stride[dim])
            if not 0 <= input_coordinates[dim] < self.input_shape[dim]:
                raise WeightLoadRecordingError("view escapes its tracked input region")
        if address != 0:
            raise WeightLoadRecordingError("view accesses a hole in its tracked input")
        input_index = 0
        for coordinate, size in zip(input_coordinates, self.input_shape):
            input_index = input_index * size + coordinate
        run = min(
            _contiguous_tail(index, self.output_shape, self.output_stride),
            _contiguous_tail(input_index, self.input_shape, self.input_stride),
        )
        root, root_index, parent_run = self.parent.resolve(input_index)
        return root, root_index, min(run, parent_run)


@dataclass(frozen=True)
class _IndexProvenance:
    """A bounded, static index selection over any existing source mapping."""

    parent: _TensorProvenance
    dim: int
    indices: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            self.parent.shape[: self.dim]
            + (len(self.indices),)
            + self.parent.shape[self.dim + 1 :]
        )

    def unsupported(self, operation: str) -> _IndexProvenance:
        return replace(self, parent=self.parent.unsupported(operation))

    def resolve(self, index: int) -> tuple[_LogicalProvenance, int, int]:
        coordinates = list(_unravel_index(index, self.shape))
        coordinates[self.dim] = self.indices[coordinates[self.dim]]
        parent_index = 0
        for i, size in zip(coordinates, self.parent.shape):
            parent_index = parent_index * size + i
        root, root_index, run = self.parent.resolve(parent_index)
        tail = prod(self.shape[self.dim + 1 :])
        return root, root_index, min(run, tail - index % tail)


_TensorProvenance = (
    _LogicalProvenance | _CompositeProvenance | _ViewProvenance | _IndexProvenance
)

# Bound metadata expansion for irregular views; never allocate per-weight indices.
_MAX_VIEW_FRAGMENTS = 4096


def _has_view_mapping(provenance: _TensorProvenance) -> bool:
    if isinstance(provenance, _CompositeProvenance):
        return any(_has_view_mapping(piece.provenance) for piece in provenance.pieces)
    return isinstance(provenance, (_ViewProvenance, _IndexProvenance))


def _unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates = []
    for size in reversed(shape):
        index, coordinate = divmod(index, size)
        coordinates.append(coordinate)
    return tuple(reversed(coordinates))


def _contiguous_tail(
    index: int, shape: tuple[int, ...], strides: tuple[int, ...]
) -> int:
    size = 1
    for extent, stride in zip(reversed(shape), reversed(strides)):
        if extent == 1:
            continue
        if stride != size:
            break
        size *= extent
    return size - index % size


def _view_provenance(
    tensor: torch.Tensor,
    output: torch.Tensor,
    parent: _TensorProvenance,
) -> _ViewProvenance | None:
    if (
        tensor.dtype != output.dtype
        or _storage_id(tensor) != _storage_id(output)
        or tensor.is_conj() != output.is_conj()
        or tensor.is_neg() != output.is_neg()
    ):
        return None
    # Inverting the input's physical coordinates requires a non-overlapping,
    # positive-stride layout. Sliced inputs may contain holes; resolve checks them.
    span = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size <= 1:
            continue
        if stride < span:
            return None
        span += (size - 1) * stride
    if tensor.numel() != prod(parent.shape):
        return None
    return _ViewProvenance(
        parent=parent,
        input_shape=tuple(tensor.shape),
        input_stride=tuple(tensor.stride()),
        output_shape=tuple(output.shape),
        output_stride=tuple(output.stride()),
        offset=output.storage_offset() - tensor.storage_offset(),
    )


def _indexed_region(
    tensor: torch.Tensor, output: torch.Tensor, dim: int, *, reduced: bool = False
) -> tuple[int, int] | None:
    """Read an indexing op's normalized bounds from the actual output view.

    Slice/narrow/select/split/unbind share this rule. Torch already resolved
    negative indices, clipping and split sizes; do not reimplement them here.
    """
    stride = tensor.stride(dim)
    if stride <= 0 or (not reduced and output.stride(dim) != stride):
        return None
    start, remainder = divmod(output.storage_offset() - tensor.storage_offset(), stride)
    extent = 1 if reduced else output.size(dim)
    if remainder or start < 0 or start + extent > tensor.size(dim):
        return None
    return start, extent


def _cast_provenance(
    provenance: _TensorProvenance, source_dtype: torch.dtype, target_dtype: torch.dtype
) -> _TensorProvenance:
    if source_dtype == target_dtype:
        return provenance
    if isinstance(provenance, _LogicalProvenance):
        return _replace_provenance_shape(
            provenance,
            layout_op=f"cast({_dtype_name(source_dtype)}->{_dtype_name(target_dtype)})",
        )
    if isinstance(provenance, (_ViewProvenance, _IndexProvenance)):
        return replace(
            provenance,
            parent=_cast_provenance(provenance.parent, source_dtype, target_dtype),
        )
    return replace(
        provenance,
        pieces=tuple(
            replace(
                piece,
                provenance=_cast_provenance(
                    piece.provenance, source_dtype, target_dtype
                ),
            )
            for piece in provenance.pieces
        ),
    )


@dataclass(frozen=True)
class _Destination:
    parameter: Any
    names: tuple[str, ...]
    storage_id: int
    begin: int
    end: int
    dtype: torch.dtype


_CURRENT_LOGICAL: contextvars.ContextVar[_LogicalProvenance | None] = (
    contextvars.ContextVar("sglang_current_logical_weight", default=None)
)
_CURRENT_EXPERT_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "sglang_current_weight_expert_id", default=None
)
_DISPATCH_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "sglang_weight_load_dispatch_active", default=False
)


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _storage_id(tensor: Any) -> int:
    storage = tensor.untyped_storage()
    value = getattr(storage, "_cdata", None)
    return int(value if value is not None else storage.data_ptr())


def _tensor_address(tensor: Any) -> int:
    return int(tensor.data_ptr())


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    result, stride = [], 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


def _reshape_provenance(
    provenance: _TensorProvenance, shape: tuple[int, ...]
) -> _ViewProvenance:
    return _ViewProvenance(
        parent=provenance,
        input_shape=provenance.shape,
        input_stride=_contiguous_strides(provenance.shape),
        output_shape=shape,
        output_stride=_contiguous_strides(shape),
        offset=0,
    )


def _names_an_expert_slot(tensor_id: str, expert_id: int) -> bool:
    """Report whether a checkpoint name indexes exactly this expert slot.

    ``expert_id`` comes from the loader signature and is authoritative; this only
    filters the fused-shared-expert case, where the generic MoE loader passes an
    out-of-range slot for a tensor that is really shared. Matching the number
    anywhere is wrong: ``layers.3.mlp.experts.7.w1`` would then credit expert 3
    to the layer index. Compare against the innermost index instead, which needs
    no knowledge of container names.

    Still a naming heuristic: it assumes the expert index is the last numeric
    path segment. Only the loader reporting the slot's valid range could remove
    the assumption entirely.
    """
    innermost = next(
        (segment for segment in reversed(tensor_id.split(".")) if segment.isdigit()),
        None,
    )
    return innermost == str(expert_id)


def _tensor_geometry(tensor: Any) -> tuple[Any, ...]:
    """Describe only the position and layout of a tensor, never its values.

    Used to decide observationally whether a mutating operation moved a tracked
    tensor. A reshard plan records where bytes live, so a mutation that leaves
    this tuple unchanged cannot invalidate anything already recorded.
    """
    geometry: tuple[Any, ...] = (
        tuple(int(value) for value in tensor.shape),
        tensor.dtype,
        str(tensor.device),
    )
    try:
        return geometry + (
            tuple(int(value) for value in tensor.stride()),
            int(tensor.storage_offset()),
            _storage_id(tensor),
            _tensor_address(tensor),
        )
    except (NotImplementedError, RuntimeError):
        return geometry + (None,)


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _iter_registered_tensors(
    model: Any,
) -> Iterator[tuple[str, torch.Tensor, bool]]:
    for name, parameter in model.named_parameters(remove_duplicate=False):
        yield name, parameter, True
    for name, buffer in model.named_buffers(remove_duplicate=False):
        yield name, buffer, False


def _runtime_tensor_state(
    tensor: torch.Tensor, *, require_storage: bool
) -> tuple[Any, ...]:
    common = (
        id(tensor),
        str(tensor.layout),
        str(tensor.device),
        tuple(int(value) for value in tensor.shape),
        tensor.dtype,
    )
    try:
        storage = (
            _storage_id(tensor),
            _tensor_address(tensor),
            tuple(int(value) for value in tensor.stride()),
        )
    except (NotImplementedError, RuntimeError):
        if require_storage:
            raise
        storage = None
    return common + (storage,)


def _replace_provenance_shape(
    provenance: _LogicalProvenance,
    *,
    global_shape: tuple[int, ...] | None = None,
    global_offset: tuple[int, ...] | None = None,
    local_shape: tuple[int, ...] | None = None,
    layout_op: str | None = None,
) -> _LogicalProvenance:
    return _LogicalProvenance(
        metadata=provenance.metadata,
        global_shape=global_shape or provenance.global_shape,
        global_offset=(
            provenance.global_offset if global_offset is None else global_offset
        ),
        local_shape=local_shape or provenance.local_shape,
        layout_ops=(
            provenance.layout_ops
            if layout_op is None
            else provenance.layout_ops + (layout_op,)
        ),
        unsupported_operation=provenance.unsupported_operation,
    )


class _WeightLoadDispatchMode(TorchDispatchMode):
    _TRANSPARENT_OPS = {
        "aten::alias",
        "aten::clone",
        "aten::contiguous",
        "aten::detach",
        "aten::_to_copy",
    }
    _RESHAPE_OPS = {
        "aten::view",
        "aten::_unsafe_view",
        "aten::reshape",
        "aten::squeeze",
        "aten::unsqueeze",
    }

    def __init__(self, recorder: WeightLoadRecorder, *, execute_writes: bool) -> None:
        super().__init__()
        self._recorder = recorder
        self._execute_writes = execute_writes

    @staticmethod
    def _name(func: Any) -> str:
        schema = getattr(func, "_schema", None)
        return str(getattr(schema, "name", func))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        del types
        kwargs = kwargs or {}
        name = self._name(func)

        if name == "aten::copy_":
            destination, source = args[:2]
            self._recorder.record_copy(destination=destination, source=source)
            if self._execute_writes:
                return func(*args, **kwargs)
            return destination

        if name in ("aten::fill_", "aten::zero_"):
            # default_weight_loader broadcasts single-element checkpoint tensors
            # with fill_(loaded_weight.item()), which passes a Python scalar and
            # so carries no provenance of its own.
            destination = args[0]
            if self._recorder.record_fill(
                destination=destination, value=0 if name == "aten::zero_" else args[1]
            ):
                if self._execute_writes:
                    return func(*args, **kwargs)
                return destination

        provenance_inputs = [
            (tensor, provenance)
            for tensor in _iter_tensors((args, kwargs))
            if (provenance := self._recorder.provenance_for(tensor)) is not None
        ]

        schema = getattr(func, "_schema", None)
        is_mutable = bool(getattr(schema, "is_mutable", False))
        geometry_before = (
            tuple(_tensor_geometry(tensor) for tensor, _ in provenance_inputs)
            if is_mutable
            else ()
        )

        result = func(*args, **kwargs)
        if not provenance_inputs:
            return result

        if is_mutable:
            return self._propagate_mutation(
                name=name,
                args=args,
                result=result,
                provenance_inputs=provenance_inputs,
                geometry_before=geometry_before,
            )

        if name in self._TRANSPARENT_OPS and len(provenance_inputs) == 1:
            input_tensor, provenance = provenance_inputs[0]
            for output in _iter_tensors(result):
                self._recorder.register_provenance(
                    output,
                    _cast_provenance(provenance, input_tensor.dtype, output.dtype),
                )
            return result

        if name == "aten::index_select":
            tensor, dim, indices = args[:3]
            provenance = self._recorder.provenance_for(tensor)
            if provenance is None:
                return result
            if (
                indices.device.type != "cpu"
                or indices.numel() > _MAX_VIEW_FRAGMENTS
                or self._recorder.provenance_for(indices) is not None
            ):
                mapped = provenance.unsupported(
                    "index_select requires bounded, static CPU indices"
                )
            else:
                values = tuple(indices.tolist())
                dim = int(dim) % tensor.dim()
                if len(set(values)) != len(values) or any(
                    i < 0 or i >= tensor.size(dim) for i in values
                ):
                    mapped = provenance.unsupported(
                        "index_select requires unique in-range indices"
                    )
                else:
                    mapped = _IndexProvenance(provenance, dim, values)
            self._recorder.register_outputs(result, mapped)
            return result

        if name in ("aten::cat", "aten::stack"):
            self._recorder.register_outputs(
                result,
                self._record_cat(args, kwargs, result, stack=name == "aten::stack"),
            )
            return result

        mapped_inputs = [
            (t, p)
            for t, p in provenance_inputs
            if not isinstance(p, _LogicalProvenance)
        ]
        if mapped_inputs:
            input_tensor, provenance = mapped_inputs[0]
            for output in _iter_tensors(result):
                mapped = None
                if len(provenance_inputs) == 1:
                    if (
                        name in self._RESHAPE_OPS
                        and output.dtype == input_tensor.dtype
                        and output.numel() == input_tensor.numel()
                    ):
                        mapped = _reshape_provenance(provenance, tuple(output.shape))
                    else:
                        mapped = _view_provenance(input_tensor, output, provenance)
                self._recorder.register_provenance(
                    output, mapped or provenance.unsupported(name)
                )
            return result

        roots = {id(item.metadata): item for _, item in provenance_inputs}
        if len(roots) != 1:
            reason = f"{name} combines multiple logical checkpoint tensors"
            output_provenance = provenance_inputs[0][1].unsupported(reason)
            self._recorder.register_outputs(result, output_provenance)
            return result

        input_tensor, provenance = provenance_inputs[0]
        try:
            output_provenances = self._propagate(
                name=name,
                args=args,
                kwargs=kwargs,
                result=result,
                input_tensor=input_tensor,
                provenance=provenance,
            )
        except (IndexError, TypeError, ValueError) as error:
            output_provenances = (provenance.unsupported(f"{name}: {error}"),)

        outputs = tuple(_iter_tensors(result))
        # Preserve existing compact box/permute paths. If one cannot describe a
        # view composition, derive its map from observed alias geometry instead
        # of adding another operator-specific indexing rule.
        if provenance.unsupported_operation is None and any(
            item.unsupported_operation is not None for item in output_provenances
        ):
            output_provenances = tuple(
                _view_provenance(input_tensor, output, provenance)
                or output_provenances[0]
                for output in outputs
            )
        if len(output_provenances) == 1:
            for output in outputs:
                self._recorder.register_provenance(output, output_provenances[0])
        elif len(outputs) == len(output_provenances):
            for output, output_provenance in zip(outputs, output_provenances):
                self._recorder.register_provenance(output, output_provenance)
        else:
            unsupported = provenance.unsupported(
                f"{name} returned an unsupported tensor structure"
            )
            for output in outputs:
                self._recorder.register_provenance(output, unsupported)
        return result

    def _propagate_mutation(
        self,
        *,
        name: str,
        args: tuple[Any, ...],
        result: Any,
        provenance_inputs: list[tuple[torch.Tensor, Any]],
        geometry_before: tuple[Any, ...],
    ) -> Any:
        """Decide observationally whether an in-place op invalidated provenance.

        Reshard planning only needs where each byte lives, and a transfer moves
        the source's final bytes, so a mutation that changes values alone is
        harmless. Rather than judging operations by name, compare the observed
        geometry: ``resize_``/``set_``/``transpose_`` move a tracked tensor while
        ``mul_``/``add_``/``clamp_`` do not.
        """
        geometry_after = tuple(
            _tensor_geometry(tensor) for tensor, _ in provenance_inputs
        )
        reason = None
        if any(_has_view_mapping(p) for _, p in provenance_inputs):
            # The new view path describes reads, not arbitrary in-place writes.
            # Unchanged shape/stride does not prove that scatter/index_put (or
            # a custom kernel) preserved element identity.
            reason = f"mutable composed view: {name}"
        elif geometry_after != geometry_before:
            reason = f"geometry-changing in-place operation: {name}"
        if reason is not None:
            for tensor, provenance in provenance_inputs:
                self._recorder.register_provenance(
                    tensor, provenance.unsupported(reason)
                )
            self._recorder.register_outputs(
                result, provenance_inputs[0][1].unsupported(reason)
            )
            return result

        # In-place operations return the mutated operand, so the output inherits
        # that operand's provenance rather than an arbitrary input's.
        mutated = args[0] if args else None
        inherited = next(
            (
                provenance
                for tensor, provenance in provenance_inputs
                if tensor is mutated
            ),
            None,
        )
        if inherited is not None:
            self._recorder.register_outputs(result, inherited)
        return result

    def _record_cat(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        *,
        stack: bool = False,
    ) -> _CompositeProvenance:
        """Join mappings; stack first inserts a singleton axis in each input."""
        output = next(_iter_tensors(result))
        dim = int(args[1] if len(args) > 1 else kwargs.get("dim", 0)) % output.dim()
        pieces, consumed = [], 0
        for tensor in args[0]:
            provenance = self._recorder.provenance_for(tensor)
            if provenance is None:
                raise WeightLoadRecordingError("aten::cat contains an untracked tensor")
            if stack:
                shape = provenance.shape[:dim] + (1,) + provenance.shape[dim:]
                provenance = _reshape_provenance(provenance, shape)
            offset = [0] * output.dim()
            offset[dim] = consumed
            if isinstance(provenance, _CompositeProvenance) and provenance.dim == dim:
                for piece in provenance.pieces:
                    translated = list(piece.tensor_offset)
                    translated[dim] += consumed
                    pieces.append(_CompositePiece(piece.provenance, tuple(translated)))
            else:
                pieces.append(_CompositePiece(provenance, tuple(offset)))
            consumed += provenance.shape[dim]
        return _CompositeProvenance(
            tuple(output.shape),
            tuple(pieces),
            dim,
            tuple(piece.tensor_offset[dim] for piece in pieces),
        )

    def _propagate(
        self,
        *,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        input_tensor: torch.Tensor,
        provenance: _LogicalProvenance,
    ) -> tuple[_LogicalProvenance, ...]:
        if provenance.unsupported_operation is not None:
            return (provenance,)

        if name in self._RESHAPE_OPS:
            if int(input_tensor.numel()) != prod(provenance.local_shape):
                return (provenance.unsupported(f"{name} after indexed view"),)
            return (provenance,)

        if name in (
            "aten::slice",
            "aten::narrow",
            "aten::select",
            "aten::split",
            "aten::split_with_sizes",
            "aten::unbind",
        ):
            if tuple(input_tensor.shape) != provenance.local_shape:
                return (provenance.unsupported(f"{name} after shape-changing view"),)
            dim_arg = 2 if name in ("aten::split", "aten::split_with_sizes") else 1
            dim = int(args[dim_arg] if len(args) > dim_arg else kwargs.get("dim", 0))
            dim %= input_tensor.dim()
            reduced = name in ("aten::select", "aten::unbind")
            mapped = []
            for output in _iter_tensors(result):
                region = _indexed_region(input_tensor, output, dim, reduced=reduced)
                if region is None:
                    return (
                        provenance.unsupported(f"{name} is not a unit-step region"),
                    )
                start, extent = region
                offset, shape = (
                    list(provenance.global_offset),
                    list(provenance.local_shape),
                )
                offset[dim] += start
                shape[dim] = extent
                mapped.append(
                    _replace_provenance_shape(
                        provenance,
                        global_offset=tuple(offset),
                        local_shape=tuple(shape),
                    )
                )
            return tuple(mapped)

        if name in ("aten::transpose", "aten::permute", "aten::t"):
            ndim = len(provenance.local_shape)
            if input_tensor.dim() != ndim:
                return (provenance.unsupported("permutation after rank-changing view"),)
            if tuple(provenance.local_shape) != tuple(
                int(v) for v in input_tensor.shape
            ):
                return (provenance.unsupported("permutation after indexed view"),)
            if name == "aten::t":
                # torch does not decompose Tensor.t() into aten::transpose, so it
                # has to be handled as its own 0/1 swap.
                if ndim > 2:
                    return (provenance.unsupported("aten::t on a rank>2 tensor"),)
                order = list(range(ndim))
                if ndim == 2:
                    order = [1, 0]
            elif name == "aten::transpose":
                first, second = int(args[1]), int(args[2])
                if first < 0:
                    first += ndim
                if second < 0:
                    second += ndim
                order = list(range(ndim))
                order[first], order[second] = order[second], order[first]
            else:
                raw_order = args[1]
                order = [int(value) for value in raw_order]
                order = [value + ndim if value < 0 else value for value in order]
            if sorted(order) != list(range(ndim)):
                return (provenance.unsupported("invalid dimension permutation"),)
            return (
                _replace_provenance_shape(
                    provenance,
                    global_shape=tuple(provenance.global_shape[i] for i in order),
                    global_offset=tuple(provenance.global_offset[i] for i in order),
                    local_shape=tuple(provenance.local_shape[i] for i in order),
                    layout_op="permute(" + ",".join(str(i) for i in order) + ")",
                ),
            )

        return (provenance.unsupported(name),)


class WeightLoadRecorder:
    def __init__(self) -> None:
        self._metadata: dict[str, LogicalWeightMetadata] = {}
        self._events: list[RecordedWeightView] = []
        self._initializations: list[RecordedWeightInitialization] = []
        self._provenance: dict[
            int, tuple[weakref.ReferenceType[torch.Tensor], _TensorProvenance]
        ] = {}
        self._destinations: dict[int, tuple[_Destination, ...]] = {}
        self._destination_aliases: dict[int, torch.Tensor] = {}
        self._unsupported_destinations: dict[int, tuple[str, ...]] = {}
        self._unsupported_destination_ids: dict[int, str] = {}
        self._model: Any | None = None

    def record_model_load(
        self,
        model: Any,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        execute_writes: bool,
        load_weights_fn: Callable[[Any], Any] | None = None,
    ) -> Any:
        if self._model is None:
            self._model = model
        elif self._model is not model:
            raise WeightLoadRecordingError(
                "one recorder cannot span multiple model instances"
            )
        self._index_destinations(model)
        tracked_weights = self._tracked_weights(weights)
        with (
            self._wrap_parameter_loaders(model, execute_writes=execute_writes),
            self._propagate_dispatch_to_thread_pool(execute_writes=execute_writes),
        ):
            token = _DISPATCH_ACTIVE.set(True)
            try:
                with _WeightLoadDispatchMode(self, execute_writes=execute_writes):
                    result = (load_weights_fn or model.load_weights)(tracked_weights)
            finally:
                _DISPATCH_ACTIVE.reset(token)
        return result

    @contextlib.contextmanager
    def _propagate_dispatch_to_thread_pool(self, *, execute_writes: bool):
        """Install recorder dispatch around tasks submitted during model load."""

        native_submit = concurrent.futures.ThreadPoolExecutor.submit

        def recording_submit(executor, function, /, *args, **kwargs):
            if not _DISPATCH_ACTIVE.get():
                return native_submit(executor, function, *args, **kwargs)

            submit_context = contextvars.copy_context()

            def run_with_dispatch():
                with _WeightLoadDispatchMode(self, execute_writes=execute_writes):
                    return submit_context.run(function, *args, **kwargs)

            return native_submit(executor, run_with_dispatch)

        concurrent.futures.ThreadPoolExecutor.submit = recording_submit
        try:
            yield
        finally:
            concurrent.futures.ThreadPoolExecutor.submit = native_submit

    def _tracked_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise WeightLoadRecordingError(
                    "weight loaders must consume (str, Tensor) entries"
                )
            scalar_value = None
            if (
                tensor.numel() == 1
                and tensor.device.type != "meta"
                and not tensor.is_complex()
            ):
                scalar_value = tensor.item()
            metadata = LogicalWeightMetadata(
                tensor_id=name,
                shape=tuple(int(value) for value in tensor.shape),
                dtype=_dtype_name(tensor.dtype),
                itemsize=int(tensor.element_size()),
                scalar_value=scalar_value,
            )
            previous = self._metadata.get(name)
            if previous is not None and (
                replace(previous, scalar_value=scalar_value) != metadata
            ):
                raise WeightLoadRecordingError(
                    f"checkpoint tensor metadata changed during loading: {name}"
                )
            self._metadata[name] = metadata
            provenance = _LogicalProvenance(
                metadata=metadata,
                global_shape=metadata.shape,
                global_offset=(0,) * len(metadata.shape),
                local_shape=metadata.shape,
            )
            self.register_provenance(tensor, provenance)
            token = _CURRENT_LOGICAL.set(provenance)
            try:
                yield name, tensor
            finally:
                _CURRENT_LOGICAL.reset(token)

    def _index_destinations(self, model: Any) -> None:
        grouped: dict[tuple[Any, ...], tuple[Any, list[str]]] = {}
        aliases: dict[int, torch.Tensor] = {}
        unsupported_by_storage: dict[int, list[str]] = {}
        unsupported_by_id: dict[int, str] = {}

        def mark_unsupported(name: str, tensor: torch.Tensor) -> None:
            unsupported_by_id[id(tensor)] = name
            try:
                storage_id = _storage_id(tensor)
            except (NotImplementedError, RuntimeError):
                return
            unsupported_by_storage.setdefault(storage_id, []).append(name)

        for name, tensor, is_parameter in _iter_registered_tensors(model):
            if tensor.device.type == "meta":
                if not is_parameter:
                    mark_unsupported(name, tensor)
                    continue
                raise WeightLoadRecordingError(
                    f"runtime parameter remains on meta device: {name}"
                )
            try:
                is_contiguous = tensor.is_contiguous()
            except (NotImplementedError, RuntimeError) as error:
                if not is_parameter:
                    mark_unsupported(name, tensor)
                    continue
                raise WeightLoadRecordingError(
                    f"cannot inspect runtime parameter layout: {name}: {error}"
                ) from error
            if not is_contiguous:
                if not is_parameter:
                    mark_unsupported(name, tensor)
                    continue
                raise WeightLoadRecordingError(
                    f"non-contiguous runtime parameter is unsupported: {name}"
                )
            try:
                key = (
                    _storage_id(tensor),
                    int(tensor.storage_offset()),
                    tuple(int(value) for value in tensor.shape),
                    tuple(int(value) for value in tensor.stride()),
                    tensor.dtype,
                )
                _tensor_address(tensor)
            except (NotImplementedError, RuntimeError) as error:
                if not is_parameter:
                    mark_unsupported(name, tensor)
                    continue
                raise WeightLoadRecordingError(
                    f"cannot inspect runtime parameter storage: {name}: {error}"
                ) from error
            if key not in grouped:
                grouped[key] = (tensor, [])
            grouped[key][1].append(name)
            aliases[id(tensor)] = grouped[key][0]

        by_storage: dict[int, list[_Destination]] = {}
        for parameter, names in grouped.values():
            begin = _tensor_address(parameter)
            end = begin + int(parameter.numel()) * int(parameter.element_size())
            destination = _Destination(
                parameter=parameter,
                names=tuple(sorted(set(names))),
                storage_id=_storage_id(parameter),
                begin=begin,
                end=end,
                dtype=parameter.dtype,
            )
            by_storage.setdefault(destination.storage_id, []).append(destination)
        self._destinations = {
            key: tuple(
                sorted(value, key=lambda item: (item.end - item.begin, item.names))
            )
            for key, value in by_storage.items()
        }
        self._destination_aliases = aliases
        self._unsupported_destinations = {
            key: tuple(sorted(set(value)))
            for key, value in unsupported_by_storage.items()
        }
        self._unsupported_destination_ids = unsupported_by_id

    def _reject_unsupported_destination(
        self, destination: torch.Tensor, storage_id: int | None = None
    ) -> None:
        names = (
            self._unsupported_destinations.get(storage_id)
            if storage_id is not None
            else None
        )
        name = self._unsupported_destination_ids.get(id(destination))
        if names or name is not None:
            runtime_name = names[0] if names else name
            raise WeightLoadRecordingError(
                "checkpoint tensor writes an unsupported registered buffer: "
                f"{runtime_name}"
            )

    @contextlib.contextmanager
    def _wrap_parameter_loaders(self, model: Any, *, execute_writes: bool):
        restorations: list[tuple[Any, str, Any]] = []
        visited = set()
        for parameter in model.parameters():
            if id(parameter) in visited:
                continue
            visited.add(id(parameter))
            attr_name = (
                "_weight_loader"
                if hasattr(parameter, "_weight_loader")
                else "weight_loader"
            )
            original = getattr(parameter, attr_name, None)
            if not callable(original):
                continue
            try:
                signature = inspect.signature(original)
            except (TypeError, ValueError):
                signature = None

            def wrapped(
                *args,
                __original=original,
                __signature=signature,
                __execute_writes=execute_writes,
                **kwargs,
            ):
                expert_id = None
                if __signature is not None:
                    try:
                        bound = __signature.bind_partial(*args, **kwargs)
                        expert_id = bound.arguments.get("expert_id")
                    except TypeError:
                        pass
                if expert_id is None:
                    expert_id = kwargs.get("expert_id")
                token = _CURRENT_EXPERT_ID.set(
                    int(expert_id) if expert_id is not None else None
                )
                try:
                    if _DISPATCH_ACTIVE.get():
                        return __original(*args, **kwargs)
                    dispatch_token = _DISPATCH_ACTIVE.set(True)
                    try:
                        with _WeightLoadDispatchMode(
                            self, execute_writes=__execute_writes
                        ):
                            return __original(*args, **kwargs)
                    finally:
                        _DISPATCH_ACTIVE.reset(dispatch_token)
                finally:
                    _CURRENT_EXPERT_ID.reset(token)

            try:
                setattr(parameter, attr_name, wrapped)
            except (AttributeError, RuntimeError) as error:
                raise WeightLoadRecordingError(
                    f"cannot instrument parameter weight loader: {error}"
                ) from error
            restorations.append((parameter, attr_name, original))
        try:
            yield
        finally:
            for parameter, attr_name, original in reversed(restorations):
                setattr(parameter, attr_name, original)

    def register_provenance(
        self, tensor: torch.Tensor, provenance: _TensorProvenance
    ) -> None:
        if tuple(tensor.shape) != provenance.shape and tensor.numel() == prod(
            provenance.shape
        ):
            provenance = _reshape_provenance(provenance, tuple(tensor.shape))
        self._provenance[id(tensor)] = (weakref.ref(tensor), provenance)

    def register_outputs(self, value: Any, provenance: _TensorProvenance) -> None:
        for tensor in _iter_tensors(value):
            self.register_provenance(tensor, provenance)

    def provenance_for(self, tensor: torch.Tensor) -> _TensorProvenance | None:
        entry = self._provenance.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            return entry[1]
        current = _CURRENT_LOGICAL.get()
        if current is None or tensor.device.type != "meta":
            return None
        # Target replay feeds meta checkpoint tensors, and a loader may derive
        # further meta tensors the registry never saw. Only adopt the logical
        # weight in scope when the shape still matches it, so an unrelated
        # scratch tensor cannot inherit another tensor's position.
        if isinstance(current, _CompositeProvenance):
            expected = current.tensor_shape
        else:
            expected = current.local_shape
        if tuple(int(value) for value in tensor.shape) != tuple(expected):
            return None
        return current

    def record_fill(self, *, destination: torch.Tensor, value: Any) -> bool:
        if destination.numel() == 0:
            return True
        owner = self._owner_for(destination)
        if owner is None:
            return False
        if not destination.is_contiguous():
            raise WeightLoadRecordingError(
                "constant fill writes a non-contiguous target"
            )
        if not isinstance(value, (bool, int, float)):
            raise WeightLoadRecordingError("constant fill requires a concrete scalar")
        byte_offset = _tensor_address(destination) - owner.begin
        provenance = _CURRENT_LOGICAL.get()
        # Preserve the checkpoint identity of native scalar loaders. Other
        # fills (e.g. vocabulary padding) are target-local initialization.
        if (
            provenance is not None
            and prod(provenance.local_shape) == 1
            and destination.numel() == 1
            and value == provenance.metadata.scalar_value
        ):
            if provenance.unsupported_operation is not None:
                raise WeightLoadRecordingError(
                    f"unsupported scalar fill: {provenance.unsupported_operation}"
                )
            self._append_event(
                provenance=provenance, owner=owner, byte_offset=byte_offset
            )
        else:
            self._initializations.append(
                RecordedWeightInitialization(
                    parameter=owner.parameter,
                    parameter_names=owner.names,
                    byte_offset=byte_offset,
                    nbytes=destination.numel() * destination.element_size(),
                    value=value,
                )
            )
        return True

    def _owner_for(self, destination: torch.Tensor) -> _Destination | None:
        """Find the registered tensor whose byte range contains this write."""
        try:
            storage_id = _storage_id(destination)
            begin = _tensor_address(destination)
        except (NotImplementedError, RuntimeError):
            return None
        end = begin + int(destination.numel()) * int(destination.element_size())
        owner = next(
            (
                candidate
                for candidate in self._destinations.get(storage_id, ())
                if candidate.dtype == destination.dtype
                and candidate.begin <= begin
                and end <= candidate.end
            ),
            None,
        )
        if owner is None:
            self._reject_unsupported_destination(destination, storage_id)
        return owner

    def record_copy(self, *, destination: torch.Tensor, source: torch.Tensor) -> None:
        provenance = self.provenance_for(source)
        if provenance is None:
            return
        provenance = _cast_provenance(provenance, source.dtype, destination.dtype)
        if not isinstance(provenance, _LogicalProvenance):
            self._record_mapped_copy(destination, source, provenance)
            return
        if provenance.unsupported_operation is not None:
            raise WeightLoadRecordingError(
                f"{provenance.metadata.tensor_id} uses an unsupported loader "
                f"operation: {provenance.unsupported_operation}"
            )
        if not destination.is_contiguous():
            raise WeightLoadRecordingError(
                f"{provenance.metadata.tensor_id} writes a non-contiguous target view"
            )
        logical_elements = prod(provenance.local_shape)
        if int(destination.numel()) != logical_elements:
            raise WeightLoadRecordingError(
                f"{provenance.metadata.tensor_id} changes tensor element count: "
                f"logical={logical_elements}, target={destination.numel()}"
            )
        self._reject_unsupported_destination(destination)
        owner = self._owner_for(destination)
        if owner is None:
            # Temporary tensors are allowed; only final writes into registered
            # model tensors contribute placement records.
            return

        self._append_event(
            provenance=provenance,
            owner=owner,
            byte_offset=_tensor_address(destination) - owner.begin,
        )

    def _record_mapped_copy(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        provenance: _TensorProvenance,
    ) -> None:
        if not destination.is_contiguous():
            raise WeightLoadRecordingError("view copy writes a non-contiguous target")
        if tuple(destination.shape) != tuple(source.shape):
            raise WeightLoadRecordingError("view copy broadcasting is unsupported")
        owner = self._owner_for(destination)
        if owner is None:
            return
        begin = _tensor_address(destination) - owner.begin
        cursor = 0
        fragments = []
        while cursor < source.numel():
            root, index, run = provenance.resolve(cursor)
            if root.unsupported_operation is not None:
                raise WeightLoadRecordingError(
                    f"{root.metadata.tensor_id} uses an unsupported loader "
                    f"operation: {root.unsupported_operation}"
                )
            run = min(run, source.numel() - cursor)
            # Split a flat run into maximal C-order boxes in the root's local
            # region. Large contiguous reshapes remain a single fragment.
            while run:
                coordinates = _unravel_index(index, root.local_shape)
                shape = [1] * len(root.local_shape)
                count = 1
                for dim in range(len(shape) - 1, -1, -1):
                    extent = min(root.local_shape[dim] - coordinates[dim], run // count)
                    shape[dim] = extent
                    count *= extent
                    if coordinates[dim] != 0 or extent != root.local_shape[dim]:
                        break
                fragments.append(
                    (
                        _replace_provenance_shape(
                            root,
                            global_offset=tuple(
                                a + b for a, b in zip(root.global_offset, coordinates)
                            ),
                            local_shape=tuple(shape),
                        ),
                        begin + cursor * destination.element_size(),
                    )
                )
                if len(fragments) > _MAX_VIEW_FRAGMENTS:
                    raise WeightLoadRecordingError(
                        f"{root.metadata.tensor_id} view exceeds "
                        f"{_MAX_VIEW_FRAGMENTS} fragments per copy"
                    )
                cursor += count
                index += count
                run -= count
        for fragment, byte_offset in fragments:
            self._append_event(
                provenance=fragment, owner=owner, byte_offset=byte_offset
            )

    def _append_event(
        self,
        *,
        provenance: _LogicalProvenance,
        owner: _Destination,
        byte_offset: int,
    ) -> None:
        expert_id = _CURRENT_EXPERT_ID.get()
        if expert_id is not None and not _names_an_expert_slot(
            provenance.metadata.tensor_id, expert_id
        ):
            # Fused shared experts reuse an out-of-range expert slot in the
            # generic MoE loader, but they remain replicated/shared tensors.
            expert_id = None

        shard_dims = tuple(
            index
            for index, (offset, local, global_) in enumerate(
                zip(
                    provenance.global_offset,
                    provenance.local_shape,
                    provenance.global_shape,
                )
            )
            if offset != 0 or local != global_
        )
        fingerprint = "sglang:recorded-copy:v1"
        if provenance.layout_ops:
            fingerprint = (
                "sglang:recorded-copy:" + ";".join(provenance.layout_ops) + ":v1"
            )
        self._events.append(
            RecordedWeightView(
                tensor_id=provenance.metadata.tensor_id,
                global_shape=provenance.global_shape,
                global_offset=provenance.global_offset,
                local_shape=provenance.local_shape,
                parameter=owner.parameter,
                parameter_names=owner.names,
                byte_offset=byte_offset,
                shard_dims=shard_dims,
                expert_id=expert_id,
                layout_fingerprint=fingerprint,
            )
        )

    def build_plan(self) -> WeightLoadPlan:
        if self._model is None or not (self._events or self._initializations):
            raise WeightLoadRecordingError(
                "native model.load_weights produced no recordable "
                "registered-tensor writes"
            )
        deduplicated: dict[tuple[Any, ...], RecordedWeightView] = {}
        for event in self._events:
            key = (
                id(event.parameter),
                event.byte_offset,
                event.nbytes,
                event.tensor_id,
                event.global_offset,
                event.local_shape,
            )
            deduplicated.setdefault(key, event)
        metadata = tuple(
            self._metadata[tensor_id] for tensor_id in sorted(self._metadata)
        )
        views = tuple(
            sorted(
                deduplicated.values(),
                key=lambda item: (
                    item.parameter_names,
                    item.byte_offset,
                    item.tensor_id,
                    item.global_offset,
                ),
            )
        )
        initializations = tuple(self._initializations)
        self._require_complete_coverage((*views, *initializations))
        return WeightLoadPlan(
            logical_weights=metadata, views=views, initializations=initializations
        )

    def _require_complete_coverage(
        self, views: Sequence[RecordedWeightView | RecordedWeightInitialization]
    ) -> None:
        """Reject a plan whose views do not tile every registered tensor.

        A write the dispatcher never saw (an attribute rebind, a direct kernel)
        leaves a hole here. Asserting in the layer that produces the plan keeps
        that guarantee with the plan itself rather than with one consumer.
        """
        covered: dict[int, list[tuple[int, int]]] = {}
        for view in views:
            covered.setdefault(id(view.parameter), []).append(
                (view.byte_offset, view.byte_offset + view.nbytes)
            )
        for name, tensor, is_parameter in _iter_registered_tensors(self._model):
            if id(tensor) in self._unsupported_destination_ids:
                continue
            canonical = self._destination_aliases.get(id(tensor), tensor)
            intervals = covered.get(id(canonical))
            if canonical is not tensor and (
                _tensor_geometry(tensor) != _tensor_geometry(canonical)
            ):
                intervals = None
            kind = "parameter" if is_parameter else "buffer"
            if not intervals:
                if not is_parameter:
                    continue
                raise WeightLoadRecordingError(
                    f"no recorded write covers {kind}: {name}"
                )
            cursor = 0
            for begin, end in sorted(intervals):
                if begin != cursor:
                    relation = "overlaps" if begin < cursor else "leaves a gap in"
                    raise WeightLoadRecordingError(
                        f"recorded writes {relation} {kind} {name} at byte "
                        f"{begin}; expected {cursor}"
                    )
                cursor = end
            nbytes = int(tensor.numel()) * int(tensor.element_size())
            if cursor != nbytes:
                raise WeightLoadRecordingError(
                    f"recorded writes do not cover complete {kind} {name}: "
                    f"covered={cursor}, nbytes={nbytes}"
                )


class WeightLoadCapture:
    def __init__(self) -> None:
        self._recorder = WeightLoadRecorder()

    def record_model_load(
        self,
        model: Any,
        weights: Any,
        *,
        load_weights_fn: Callable[[Any], Any] | None = None,
    ) -> Any:
        return self._recorder.record_model_load(
            model,
            weights,
            execute_writes=True,
            load_weights_fn=load_weights_fn,
        )

    @property
    def plan(self) -> WeightLoadPlan:
        return self._recorder.build_plan()


_MISSING_ATTRIBUTE = object()


def _restore_instance_attribute(instance: Any, name: str, previous: Any) -> None:
    if previous is _MISSING_ATTRIBUTE:
        delattr(instance, name)
    else:
        setattr(instance, name, previous)


@contextlib.contextmanager
def capture_weight_load_plan(loader: Any) -> Iterator[WeightLoadCapture]:
    """Record one daemon-owned loader instance without changing engine loaders.

    The generic model-loader module must not depend on weight-cache.  Instead,
    the source daemon installs a temporary override on its private loader
    instance.  The override preserves the loader's native quantization and
    post-processing path and intercepts only the model's ``load_weights`` call.
    Both instance attributes are restored before this context exits.
    """

    capture = WeightLoadCapture()
    load_and_postprocess = getattr(loader, "load_weights_and_postprocess", None)
    if not callable(load_and_postprocess):
        raise WeightLoadRecordingError(
            "weight-load recording requires a loader with load_weights_and_postprocess"
        )

    loader_override = vars(loader).get(
        "load_weights_and_postprocess", _MISSING_ATTRIBUTE
    )

    def recording_load_and_postprocess(
        model: Any, weights: Any, target_device: Any
    ) -> Any:
        native_load_weights = getattr(model, "load_weights", None)
        if not callable(native_load_weights):
            raise WeightLoadRecordingError(
                "weight-load recording requires model.load_weights"
            )
        model_override = vars(model).get("load_weights", _MISSING_ATTRIBUTE)

        def recording_load_weights(recorded_weights: Any) -> Any:
            return capture.record_model_load(
                model,
                recorded_weights,
                load_weights_fn=native_load_weights,
            )

        setattr(model, "load_weights", recording_load_weights)
        try:
            return load_and_postprocess(model, weights, target_device)
        finally:
            _restore_instance_attribute(model, "load_weights", model_override)

    setattr(
        loader,
        "load_weights_and_postprocess",
        recording_load_and_postprocess,
    )
    try:
        yield capture
    finally:
        _restore_instance_attribute(
            loader, "load_weights_and_postprocess", loader_override
        )


def record_target_weight_load_plan(
    model: Any, logical_weights: Sequence[LogicalWeightMetadata]
) -> WeightLoadPlan:
    recorder = WeightLoadRecorder()

    def meta_weights():
        for metadata in logical_weights:
            dtype = getattr(torch, metadata.dtype, None)
            if not isinstance(dtype, torch.dtype):
                raise WeightLoadRecordingError(
                    f"unsupported checkpoint dtype: {metadata.dtype}"
                )
            # Scalar loaders may call item() or branch on the checkpoint value.
            # Replay that exact value while continuing to suppress runtime writes.
            if metadata.scalar_value is None:
                tensor = torch.empty(metadata.shape, dtype=dtype, device="meta")
            else:
                tensor = torch.tensor(
                    metadata.scalar_value, dtype=dtype, device="cpu"
                ).reshape(metadata.shape)
            yield metadata.tensor_id, tensor

    runtime_tensor_state = {
        (is_parameter, name): _runtime_tensor_state(
            tensor, require_storage=is_parameter
        )
        for name, tensor, is_parameter in _iter_registered_tensors(model)
    }
    recorder.record_model_load(model, meta_weights(), execute_writes=False)
    replayed_tensor_state = {
        (is_parameter, name): _runtime_tensor_state(
            tensor, require_storage=is_parameter
        )
        for name, tensor, is_parameter in _iter_registered_tensors(model)
    }
    changed = next(
        (
            key
            for key in sorted(
                runtime_tensor_state.keys() | replayed_tensor_state.keys()
            )
            if runtime_tensor_state.get(key) != replayed_tensor_state.get(key)
        ),
        None,
    )
    if changed is not None:
        is_parameter, name = changed
        kind = "parameter" if is_parameter else "buffer"
        raise WeightLoadRecordingError(
            f"target layout replay mutated {kind} storage: {name}"
        )
    return recorder.build_plan()


def logical_weight_metadata_from_runtime_manifests(
    runtime_manifests: Sequence[Any],
) -> tuple[LogicalWeightMetadata, ...]:
    metadata: dict[str, LogicalWeightMetadata] = {}
    for manifest in runtime_manifests:
        load_tensors = (
            manifest.get("load_tensors", ())
            if isinstance(manifest, dict)
            else getattr(manifest, "load_tensors", ())
        )
        for item in load_tensors:
            if isinstance(item, dict):
                current = LogicalWeightMetadata(
                    tensor_id=str(item["tensor_id"]),
                    shape=tuple(int(value) for value in item["shape"]),
                    dtype=str(item["dtype"]),
                    itemsize=int(item["itemsize"]),
                    scalar_value=item.get("scalar_value"),
                )
            else:
                current = LogicalWeightMetadata(
                    tensor_id=str(item.tensor_id),
                    shape=tuple(int(value) for value in item.shape),
                    dtype=str(item.dtype),
                    itemsize=int(item.itemsize),
                    scalar_value=getattr(item, "scalar_value", None),
                )
            previous = metadata.setdefault(current.tensor_id, current)
            if previous != current:
                raise WeightLoadRecordingError(
                    "source ranks disagree on checkpoint tensor metadata: "
                    f"{current.tensor_id}"
                )
    if not metadata:
        raise WeightLoadRecordingError(
            "source runtime manifests contain no recorded load metadata"
        )
    return tuple(metadata[key] for key in sorted(metadata))
