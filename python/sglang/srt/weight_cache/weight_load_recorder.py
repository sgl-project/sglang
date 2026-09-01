# SPDX-License-Identifier: Apache-2.0
"""Record checkpoint-to-runtime weight placement from native weight loaders.

The recorder treats ``model.load_weights`` as the source of truth.  Source
daemons record the real load while allowing writes to execute.  Target daemons
replay the same checkpoint metadata with meta tensors and suppress registered
tensor writes.  In both modes the result is a model-independent set of logical
tensor boxes backed by final registered parameter or buffer storage.

Only byte-preserving layout operations are supported.  Anything that cannot be
lowered to a contiguous destination range fails explicitly instead of falling
back to model-specific semantics.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import inspect
import re
import weakref
from dataclasses import dataclass
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
class WeightLoadPlan:
    logical_weights: tuple[LogicalWeightMetadata, ...]
    views: tuple[RecordedWeightView, ...]

    def views_for_parameters(
        self, parameters: Sequence[Any]
    ) -> tuple[RecordedWeightView, ...]:
        parameter_ids = {id(parameter) for parameter in parameters}
        return tuple(view for view in self.views if id(view.parameter) in parameter_ids)


@dataclass(frozen=True)
class _LogicalProvenance:
    metadata: LogicalWeightMetadata
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    layout_ops: tuple[str, ...] = ()
    unsupported_operation: str | None = None

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
    provenance: _LogicalProvenance
    tensor_offset: tuple[int, ...]


@dataclass(frozen=True)
class _CompositeProvenance:
    tensor_shape: tuple[int, ...]
    pieces: tuple[_CompositePiece, ...]

    def unsupported(self, operation: str) -> _CompositeProvenance:
        return _CompositeProvenance(
            tensor_shape=self.tensor_shape,
            pieces=tuple(
                _CompositePiece(
                    provenance=piece.provenance.unsupported(operation),
                    tensor_offset=piece.tensor_offset,
                )
                for piece in self.pieces
            ),
        )


_TensorProvenance = _LogicalProvenance | _CompositeProvenance


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


def _contiguous_region_element_offset(
    *,
    full_shape: tuple[int, ...],
    offset: tuple[int, ...],
    shape: tuple[int, ...],
) -> int | None:
    if not (len(full_shape) == len(offset) == len(shape)):
        return None
    pivot = None
    for index, extent in enumerate(shape):
        if extent > 1:
            pivot = index
            break
    if pivot is None:
        pivot = len(shape) - 1
    if any(shape[index] != 1 for index in range(pivot)):
        return None
    if any(
        offset[index] != 0 or shape[index] != full_shape[index]
        for index in range(pivot + 1, len(shape))
    ):
        return None
    if any(
        begin < 0 or extent <= 0 or begin + extent > total
        for begin, extent, total in zip(offset, shape, full_shape)
    ):
        return None
    stride = 1
    element_offset = 0
    for dimension in range(len(full_shape) - 1, -1, -1):
        element_offset += offset[dimension] * stride
        stride *= full_shape[dimension]
    return element_offset


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

        if name == "aten::fill_":
            # default_weight_loader broadcasts single-element checkpoint tensors
            # with fill_(loaded_weight.item()), which passes a Python scalar and
            # so carries no provenance of its own.
            destination = args[0]
            self._recorder.record_scalar_fill(destination=destination)
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

        if name == "aten::cat":
            output_provenance = self._record_cat(
                args=args,
                kwargs=kwargs,
                result=result,
            )
            self._recorder.register_outputs(result, output_provenance)
            return result

        if any(
            isinstance(item, _CompositeProvenance)
            for _, item in provenance_inputs
        ):
            composites = {
                id(item): item
                for _, item in provenance_inputs
                if isinstance(item, _CompositeProvenance)
            }
            if len(composites) != 1 or len(provenance_inputs) != 1:
                output_provenance = next(iter(composites.values())).unsupported(
                    f"{name} combines a composite with another tensor"
                )
            else:
                input_tensor, composite = provenance_inputs[0]
                output_provenance = self._propagate_composite(
                    name=name,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    input_tensor=input_tensor,
                    provenance=composite,
                )
            self._recorder.register_outputs(result, output_provenance)
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
            output_provenances = (
                provenance.unsupported(f"{name}: {error}"),
            )

        outputs = tuple(_iter_tensors(result))
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
        if geometry_after != geometry_before:
            reason = f"geometry-changing in-place operation: {name}"
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
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
    ) -> _CompositeProvenance:
        inputs = tuple(_iter_tensors(args[0]))
        output = next(iter(_iter_tensors(result)))
        ndim = output.dim()
        dim = int(args[1] if len(args) > 1 else kwargs.get("dim", 0))
        if dim < 0:
            dim += ndim
        if not 0 <= dim < ndim:
            raise WeightLoadRecordingError(f"unsupported concat dimension: {dim}")

        pieces = []
        consumed = 0
        for index, tensor in enumerate(inputs):
            tracked = self._recorder.provenance_for(tensor)
            if tracked is None:
                if not pieces:
                    raise WeightLoadRecordingError(
                        "aten::cat mixes tracked checkpoint weights with an "
                        "untracked tensor"
                    )
                return _CompositeProvenance(
                    tensor_shape=tuple(int(value) for value in output.shape),
                    pieces=tuple(pieces),
                ).unsupported(
                    "aten::cat mixes tracked checkpoint weights with an untracked tensor"
                )
            if isinstance(tracked, _CompositeProvenance):
                for piece in tracked.pieces:
                    offset = list(piece.tensor_offset)
                    offset[dim] += consumed
                    pieces.append(
                        _CompositePiece(
                            provenance=piece.provenance,
                            tensor_offset=tuple(offset),
                        )
                    )
            else:
                shape = tuple(int(value) for value in tensor.shape)
                if len(shape) != ndim or shape != tracked.local_shape:
                    return _CompositeProvenance(
                        tensor_shape=tuple(int(value) for value in output.shape),
                        pieces=tuple(pieces)
                        or (
                            _CompositePiece(
                                provenance=tracked,
                                tensor_offset=(0,) * ndim,
                            ),
                        ),
                    ).unsupported("aten::cat after an unrepresented reshape")
                offset = [0] * ndim
                offset[dim] = consumed
                pieces.append(
                    _CompositePiece(
                        provenance=_replace_provenance_shape(
                            tracked,
                            layout_op=f"concat(dim={dim},index={index})",
                        ),
                        tensor_offset=tuple(offset),
                    )
                )
            consumed += int(tensor.shape[dim])
        return _CompositeProvenance(
            tensor_shape=tuple(int(value) for value in output.shape),
            pieces=tuple(pieces),
        )

    def _propagate_composite(
        self,
        *,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        input_tensor: torch.Tensor,
        provenance: _CompositeProvenance,
    ) -> _CompositeProvenance:
        output = next(iter(_iter_tensors(result)))
        output_shape = tuple(int(value) for value in output.shape)
        if name in self._TRANSPARENT_OPS:
            if name == "aten::_to_copy" and output.dtype != input_tensor.dtype:
                return provenance.unsupported("dtype-changing aten::_to_copy")
            return _CompositeProvenance(
                tensor_shape=output_shape,
                pieces=provenance.pieces,
            )

        if name not in ("aten::slice", "aten::narrow"):
            return provenance.unsupported(name)

        ndim = len(provenance.tensor_shape)
        if input_tensor.dim() != ndim:
            return provenance.unsupported(f"{name} after rank-changing view")
        dim = int(args[1])
        if dim < 0:
            dim += ndim
        if name == "aten::slice":
            start = 0 if len(args) < 3 or args[2] is None else int(args[2])
            end = (
                provenance.tensor_shape[dim]
                if len(args) < 4 or args[3] is None
                else int(args[3])
            )
            step = 1 if len(args) < 5 else int(args[4])
            if step != 1:
                return provenance.unsupported("slice step is not one")
        else:
            start = int(args[2])
            end = start + int(args[3])
        size = provenance.tensor_shape[dim]
        start = min(max(start, 0), size)
        end = min(max(end, start), size)

        pieces = []
        for piece in provenance.pieces:
            piece_start = piece.tensor_offset[dim]
            piece_end = piece_start + piece.provenance.local_shape[dim]
            intersection_start = max(start, piece_start)
            intersection_end = min(end, piece_end)
            if intersection_start >= intersection_end:
                continue
            logical_offset = list(piece.provenance.global_offset)
            logical_shape = list(piece.provenance.local_shape)
            logical_offset[dim] += intersection_start - piece_start
            logical_shape[dim] = intersection_end - intersection_start
            tensor_offset = list(piece.tensor_offset)
            tensor_offset[dim] = intersection_start - start
            pieces.append(
                _CompositePiece(
                    provenance=_replace_provenance_shape(
                        piece.provenance,
                        global_offset=tuple(logical_offset),
                        local_shape=tuple(logical_shape),
                    ),
                    tensor_offset=tuple(tensor_offset),
                )
            )
        if not pieces:
            return provenance.unsupported(f"{name} removed all tracked components")
        return _CompositeProvenance(
            tensor_shape=output_shape,
            pieces=tuple(pieces),
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

        if name in self._TRANSPARENT_OPS:
            if name == "aten::_to_copy":
                outputs = tuple(_iter_tensors(result))
                if outputs and outputs[0].dtype != input_tensor.dtype:
                    # A cast preserves the logical region: geometry is tracked in
                    # elements, and record_copy records the same layout op when a
                    # copy_ crosses dtypes. Keep both paths consistent.
                    return (
                        _replace_provenance_shape(
                            provenance,
                            layout_op=(
                                f"cast({_dtype_name(input_tensor.dtype)}"
                                f"->{_dtype_name(outputs[0].dtype)})"
                            ),
                        ),
                    )
            return (provenance,)

        if name in self._RESHAPE_OPS:
            if int(input_tensor.numel()) != prod(provenance.local_shape):
                return (provenance.unsupported(f"{name} after indexed view"),)
            return (provenance,)

        if name == "aten::slice":
            dim = int(args[1])
            ndim = len(provenance.local_shape)
            if dim < 0:
                dim += ndim
            if tuple(int(value) for value in input_tensor.shape) != tuple(
                provenance.local_shape
            ):
                return (
                    provenance.unsupported("slice after shape-changing view"),
                )
            start = 0 if len(args) < 3 or args[2] is None else int(args[2])
            end = (
                provenance.local_shape[dim]
                if len(args) < 4 or args[3] is None
                else int(args[3])
            )
            step = 1 if len(args) < 5 else int(args[4])
            if step != 1:
                return (provenance.unsupported("slice step is not one"),)
            size = provenance.local_shape[dim]
            start = min(max(start, 0), size)
            end = min(max(end, start), size)
            offset = list(provenance.global_offset)
            shape = list(provenance.local_shape)
            offset[dim] += start
            shape[dim] = end - start
            return (
                _replace_provenance_shape(
                    provenance,
                    global_offset=tuple(offset),
                    local_shape=tuple(shape),
                ),
            )

        if name == "aten::select":
            dim = int(args[1])
            ndim = len(provenance.local_shape)
            if dim < 0:
                dim += ndim
            if tuple(int(value) for value in input_tensor.shape) != tuple(
                provenance.local_shape
            ):
                return (
                    provenance.unsupported("select after shape-changing view"),
                )
            index = int(args[2])
            if index < 0:
                index += provenance.local_shape[dim]
            offset = list(provenance.global_offset)
            shape = list(provenance.local_shape)
            offset[dim] += index
            shape[dim] = 1
            return (
                _replace_provenance_shape(
                    provenance,
                    global_offset=tuple(offset),
                    local_shape=tuple(shape),
                ),
            )

        if name in ("aten::transpose", "aten::permute", "aten::t"):
            ndim = len(provenance.local_shape)
            if input_tensor.dim() != ndim:
                return (provenance.unsupported("permutation after rank-changing view"),)
            if tuple(provenance.local_shape) != tuple(int(v) for v in input_tensor.shape):
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

        if name in ("aten::split", "aten::split_with_sizes", "aten::unbind"):
            ndim = len(provenance.local_shape)
            if tuple(int(value) for value in input_tensor.shape) != tuple(
                provenance.local_shape
            ):
                return (
                    provenance.unsupported(f"{name} after shape-changing view"),
                )
            if name == "aten::unbind":
                # torch exposes unbind directly instead of decomposing it into
                # aten::select, so each output drops the unbound dimension.
                dim = int(args[1] if len(args) > 1 else kwargs.get("dim", 0))
                if dim < 0:
                    dim += ndim
                result_provenances = []
                for index in range(len(tuple(_iter_tensors(result)))):
                    offset = list(provenance.global_offset)
                    shape = list(provenance.local_shape)
                    offset[dim] += index
                    shape[dim] = 1
                    result_provenances.append(
                        _replace_provenance_shape(
                            provenance,
                            global_offset=tuple(offset),
                            local_shape=tuple(shape),
                        )
                    )
                return tuple(result_provenances)
            dim = int(args[2] if len(args) > 2 else kwargs.get("dim", 0))
            if dim < 0:
                dim += ndim
            outputs = tuple(_iter_tensors(result))
            result_provenances = []
            consumed = 0
            for output in outputs:
                extent = int(output.shape[dim])
                offset = list(provenance.global_offset)
                shape = list(provenance.local_shape)
                offset[dim] += consumed
                shape[dim] = extent
                result_provenances.append(
                    _replace_provenance_shape(
                        provenance,
                        global_offset=tuple(offset),
                        local_shape=tuple(shape),
                    )
                )
                consumed += extent
            return tuple(result_provenances)

        if name == "aten::narrow":
            # Most torch builds decompose Tensor.narrow into aten::slice.  Keep
            # this explicit path for backends that expose aten::narrow directly.
            dim, start, length = (int(args[1]), int(args[2]), int(args[3]))
            ndim = len(provenance.local_shape)
            if tuple(int(value) for value in input_tensor.shape) != tuple(
                provenance.local_shape
            ):
                return (
                    provenance.unsupported("narrow after shape-changing view"),
                )
            if dim < 0:
                dim += ndim
            offset = list(provenance.global_offset)
            shape = list(provenance.local_shape)
            offset[dim] += start
            shape[dim] = length
            return (
                _replace_provenance_shape(
                    provenance,
                    global_offset=tuple(offset),
                    local_shape=tuple(shape),
                ),
            )

        return (provenance.unsupported(name),)


class WeightLoadRecorder:
    def __init__(self) -> None:
        self._metadata: dict[str, LogicalWeightMetadata] = {}
        self._events: list[RecordedWeightView] = []
        self._provenance: dict[
            int, tuple[weakref.ReferenceType[torch.Tensor], _TensorProvenance]
        ] = {}
        self._destinations: dict[int, tuple[_Destination, ...]] = {}
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
            self._wrap_parameter_loaders(
                model, execute_writes=execute_writes
            ),
            self._propagate_dispatch_to_thread_pool(
                execute_writes=execute_writes
            ),
        ):
            token = _DISPATCH_ACTIVE.set(True)
            try:
                with _WeightLoadDispatchMode(
                    self, execute_writes=execute_writes
                ):
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
                with _WeightLoadDispatchMode(
                    self, execute_writes=execute_writes
                ):
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
            metadata = LogicalWeightMetadata(
                tensor_id=name,
                shape=tuple(int(value) for value in tensor.shape),
                dtype=_dtype_name(tensor.dtype),
                itemsize=int(tensor.element_size()),
            )
            previous = self._metadata.setdefault(name, metadata)
            if previous != metadata:
                raise WeightLoadRecordingError(
                    f"checkpoint tensor metadata changed during loading: {name}"
                )
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
            key: tuple(sorted(value, key=lambda item: (item.end - item.begin, item.names)))
            for key, value in by_storage.items()
        }
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

    def record_scalar_fill(self, *, destination: torch.Tensor) -> None:
        """Record a scalar broadcast into a single-element registered tensor.

        ``fill_`` receives a Python scalar, so the value itself proves nothing
        about origin. The logical weight in scope does: the loader is mid-way
        through one checkpoint tensor, and only a single-element logical weight
        can be broadcast this way, which makes the attribution unambiguous.
        """
        provenance = _CURRENT_LOGICAL.get()
        if provenance is None:
            return
        if prod(provenance.local_shape) != 1 or int(destination.numel()) != 1:
            return
        if provenance.unsupported_operation is not None:
            raise WeightLoadRecordingError(
                f"{provenance.metadata.tensor_id} uses an unsupported loader "
                f"operation: {provenance.unsupported_operation}"
            )
        owner = self._owner_for(destination)
        if owner is None:
            return
        self._append_event(
            provenance=provenance,
            owner=owner,
            byte_offset=_tensor_address(destination) - owner.begin,
        )

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
        if isinstance(provenance, _CompositeProvenance):
            self._record_composite_copy(
                destination=destination,
                source=source,
                provenance=provenance,
            )
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
        if destination.dtype != source.dtype:
            provenance = _replace_provenance_shape(
                provenance,
                layout_op=(
                    f"cast({_dtype_name(source.dtype)}->{_dtype_name(destination.dtype)})"
                ),
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

    def _record_composite_copy(
        self,
        *,
        destination: torch.Tensor,
        source: torch.Tensor,
        provenance: _CompositeProvenance,
    ) -> None:
        unsupported = next(
            (
                piece.provenance.unsupported_operation
                for piece in provenance.pieces
                if piece.provenance.unsupported_operation is not None
            ),
            None,
        )
        if unsupported is not None:
            raise WeightLoadRecordingError(
                "concatenated checkpoint weights use an unsupported loader "
                f"operation: {unsupported}"
            )
        if not destination.is_contiguous():
            raise WeightLoadRecordingError(
                "concatenated checkpoint weights write a non-contiguous target view"
            )
        if tuple(int(value) for value in source.shape) != provenance.tensor_shape:
            raise WeightLoadRecordingError(
                "concatenated checkpoint weight shape changed without a recorded operation"
            )
        if int(destination.numel()) != prod(provenance.tensor_shape):
            raise WeightLoadRecordingError(
                "concatenated checkpoint weights change tensor element count"
            )

        self._reject_unsupported_destination(destination)
        storage_id = _storage_id(destination)
        candidates = self._destinations.get(storage_id, ())
        begin = _tensor_address(destination)
        end = begin + int(destination.numel()) * int(destination.element_size())
        owner = next(
            (
                candidate
                for candidate in candidates
                if candidate.dtype == destination.dtype
                and candidate.begin <= begin
                and end <= candidate.end
            ),
            None,
        )
        if owner is None:
            self._reject_unsupported_destination(destination, storage_id)
            return

        for piece in provenance.pieces:
            element_offset = _contiguous_region_element_offset(
                full_shape=provenance.tensor_shape,
                offset=piece.tensor_offset,
                shape=piece.provenance.local_shape,
            )
            if element_offset is None:
                raise WeightLoadRecordingError(
                    f"{piece.provenance.metadata.tensor_id} occupies a non-contiguous "
                    "region after concatenation"
                )
            piece_provenance = piece.provenance
            if destination.dtype != source.dtype:
                piece_provenance = _replace_provenance_shape(
                    piece_provenance,
                    layout_op=(
                        f"cast({_dtype_name(source.dtype)}->"
                        f"{_dtype_name(destination.dtype)})"
                    ),
                )
            self._append_event(
                provenance=piece_provenance,
                owner=owner,
                byte_offset=(
                    begin
                    - owner.begin
                    + element_offset * int(destination.element_size())
                ),
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
                "sglang:recorded-copy:"
                + ";".join(provenance.layout_ops)
                + ":v1"
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
        if self._model is None or not self._events:
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
        used_ids = {event.tensor_id for event in deduplicated.values()}
        metadata = tuple(
            self._metadata[tensor_id] for tensor_id in sorted(used_ids)
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
        self._require_complete_coverage(views)
        return WeightLoadPlan(logical_weights=metadata, views=views)

    def _require_complete_coverage(
        self, views: tuple[RecordedWeightView, ...]
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
            intervals = covered.get(id(tensor))
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


def _restore_instance_attribute(
    instance: Any, name: str, previous: Any
) -> None:
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
            "weight-load recording requires a loader with "
            "load_weights_and_postprocess"
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
            tensor = torch.empty(metadata.shape, dtype=dtype, device="meta")
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
                )
            else:
                current = LogicalWeightMetadata(
                    tensor_id=str(item.tensor_id),
                    shape=tuple(int(value) for value in item.shape),
                    dtype=str(item.dtype),
                    itemsize=int(item.itemsize),
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
