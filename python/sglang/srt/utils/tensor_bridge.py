# Copied and adapted from: https://github.com/vllm-project/vllm-metal
# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between MLX and PyTorch on Apple silicon.

The MLX backend requires MLX >= 0.32 and PyTorch >= 2.13.  Ordinary
``torch_to_mlx`` conversion creates an independent MLX allocation.  The
zero-copy ``mlx_call`` helper is available for a complete MLX operation and
keeps all borrowed DLPack inputs alive until the result has been evaluated.
This lifetime boundary matters because MLX may donate a borrowed input buffer
to a lazy operation.

Bridge entry points are serialized because Torch and MLX use different stream
abstractions over the same Metal command queues.  The lock covers producer
fencing, MLX evaluation, and DLPack import.  It cannot cover arbitrary MPS
work outside the function, so callers must serialize any overlapping use or
mutation of source and returned MPS tensors as well.
"""

from __future__ import annotations

from functools import lru_cache, wraps
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

if TYPE_CHECKING:
    import mlx.core as mx


_BRIDGE_LOCK = RLock()


def _serialized_bridge(function: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize one complete Torch/MLX crossing, including result export."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _BRIDGE_LOCK:
            return function(*args, **kwargs)

    return wrapper


@lru_cache(maxsize=1)
def _mlx_core():
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError("The MLX tensor bridge requires MLX >= 0.32.0") from None
    return mx


def _get_torch_device() -> torch.device:
    """Get the PyTorch device for Metal/MPS.

    Returns:
        torch.device for MPS if available, else CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _torch_to_mlx(
    tensor: torch.Tensor,
    *,
    copy: bool,
    synchronize: bool = True,
) -> mx.array:
    """Convert one tensor, optionally borrowing its MPS allocation."""
    mx = _mlx_core()
    tensor = tensor.detach()

    if tensor.device.type == "mps":
        if synchronize:
            # Torch and MLX do not share stream state on Metal.
            torch.mps.synchronize()
        return mx.asarray(tensor, copy=copy)
    if tensor.device.type == "cpu":
        # CPU tensors always get MLX-owned storage.  In particular, do not
        # expose a NumPy/memoryview alias whose lifetime is controlled by the
        # caller.
        if tensor.dtype == torch.complex128:
            raise ValueError(
                "MLX 0.32 does not support complex128; convert the Torch tensor "
                "to complex64 explicitly"
            )
        # MLX 0.32 does not support float64 on its default Metal stream.  Keep
        # the dtype by constructing this uncommon CPU value on the CPU stream
        # instead of silently downcasting it to float32.
        if tensor.dtype == torch.float64:
            with mx.stream(mx.cpu):
                return mx.array(tensor, dtype=mx.float64)
        return mx.array(tensor)
    raise ValueError(
        f"The MLX tensor bridge supports CPU and MPS tensors, got {tensor.device}"
    )


class MlxTensorView:
    """A lifetime-bound, zero-copy MLX view of a Torch MPS tensor.

    The view deliberately retains a detached Torch tensor *and* the imported
    MLX array.  Holding only the array is insufficient: a later parameter
    replacement or garbage collection could invalidate the borrowed storage
    while MLX still has a lazy graph referring to it. This class is intended
    for immutable inference weights; construct a new view after replacing the
    source storage.
    """

    __slots__ = ("torch_tensor", "array")

    def __init__(self, tensor: torch.Tensor, *, synchronize: bool = True):
        with _BRIDGE_LOCK:
            owner = tensor.detach()
            if owner.device.type != "mps":
                raise ValueError(
                    f"MlxTensorView requires a Torch MPS tensor, got {owner.device}"
                )
            if synchronize:
                torch.mps.synchronize()
            self.torch_tensor = owner
            self.array = _torch_to_mlx(owner, copy=False, synchronize=False)

    @classmethod
    def _from_synchronized(cls, tensor: torch.Tensor) -> MlxTensorView:
        view = object.__new__(cls)
        owner = tensor.detach()
        if owner.device.type != "mps":
            raise ValueError(
                f"MlxTensorView requires a Torch MPS tensor, got {owner.device}"
            )
        view.torch_tensor = owner
        view.array = _torch_to_mlx(owner, copy=False, synchronize=False)
        return view

    def matches(self, tensor: torch.Tensor) -> bool:
        """Return whether ``tensor`` still refers to this borrowed storage."""
        owner = tensor.detach()
        return (
            owner.device == self.torch_tensor.device
            and owner.dtype == self.torch_tensor.dtype
            and owner.shape == self.torch_tensor.shape
            and owner.stride() == self.torch_tensor.stride()
            and owner.data_ptr() == self.torch_tensor.data_ptr()
        )


@_serialized_bridge
def borrow_torch_tensors(
    *tensors: torch.Tensor, synchronize: bool = True
) -> tuple[MlxTensorView, ...]:
    """Borrow one or more Torch MPS tensors, optionally synchronizing once.

    The returned views own the Torch tensor references for their entire
    lifetime.  No data copy is made.  Set ``synchronize=False`` only when a
    surrounding operation (such as :func:`mlx_call`) performs the producer
    barrier immediately before consuming the views.  This helper is
    intentionally separate from :func:`torch_to_mlx`, whose contract is an
    independent MLX copy.
    """
    detached = tuple(tensor.detach() for tensor in tensors)
    if any(tensor.device.type != "mps" for tensor in detached):
        devices = ", ".join(str(tensor.device) for tensor in detached)
        raise ValueError(f"borrow_torch_tensors requires MPS tensors, got {devices}")
    if synchronize and detached:
        torch.mps.synchronize()
    return tuple(MlxTensorView._from_synchronized(tensor) for tensor in detached)


@_serialized_bridge
def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to an independent MLX array.

    MPS inputs are copied inside the unified Metal device.  Use ``mlx_call``
    when a complete operation needs zero-copy MPS input imports; it owns the
    borrowed MLX arrays for the complete lazy operation.

    Args:
        tensor: PyTorch CPU or MPS tensor.

    Returns:
        MLX array with the same data
    """
    array = _torch_to_mlx(tensor, copy=True)
    if tensor.device.type == "mps":
        # Materialize the owned copy before the caller may mutate or release
        # the Torch source.
        _mlx_core().eval(array)
    return array


@_serialized_bridge
def mlx_call(
    operation: Callable[..., mx.array],
    *tensors: torch.Tensor | MlxTensorView,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:
    """Run one MLX operation with zero-copy Torch MPS input imports.

    The imported MLX arrays remain strongly referenced until
    :func:`mlx_to_torch` evaluates and exports ``operation``'s result.  Keep
    the operation inside this call; returning a lazy MLX result for later use
    or stashing a borrowed input through a callback side effect would escape
    the borrow scope.  The caller must also serialize any overlapping MPS work
    outside this function, including use or mutation of source and returned
    tensors.  The operation may allocate its own output normally.
    """
    mx = _mlx_core()
    target_device = _get_torch_device() if device is None else torch.device(device)
    if target_device.type not in {"cpu", "mps"}:
        raise ValueError(
            f"The MLX tensor bridge supports CPU and MPS targets, got {target_device}"
        )
    detached = tuple(
        tensor.detach() for tensor in tensors if isinstance(tensor, torch.Tensor)
    )
    if any(tensor.device.type == "mps" for tensor in detached) or any(
        isinstance(tensor, MlxTensorView) for tensor in tensors
    ):
        torch.mps.synchronize()
    borrowed: tuple[Any, ...] = tuple(
        (
            tensor.array
            if isinstance(tensor, MlxTensorView)
            else _torch_to_mlx(tensor.detach(), copy=False, synchronize=False)
        )
        for tensor in tensors
    )
    # MLX does not support float64 on the Metal stream.  Keep an explicitly
    # requested CPU call on the CPU stream when a borrowed input carries that
    # dtype; otherwise even constructing the lazy result would fail before the
    # export preparation below can move it.
    if target_device.type == "cpu" and any(
        array.dtype == mx.float64 for array in borrowed
    ):
        with mx.stream(mx.cpu):
            result = operation(*borrowed)
    else:
        result = operation(*borrowed)
    output = mlx_to_torch(result, device=target_device)
    # Keep the imported MLX objects (and any MlxTensorView Torch owners) alive
    # through lazy result evaluation and DLPack export.
    _ = borrowed
    return output


def _prepare_mlx_export(
    array: mx.array,
    target_device: torch.device,
    mx: Any,
) -> mx.array:
    """Prepare one lazy MLX result for the requested Torch target.

    This intentionally does not evaluate the result.  Callers which export
    several results should prepare every result first and then issue one
    shared ``mx.eval`` boundary.
    """
    if target_device.type not in {"cpu", "mps"}:
        raise ValueError(
            f"The MLX tensor bridge supports CPU and MPS targets, got {target_device}"
        )

    if target_device.type == "mps" and array.dtype == mx.float64:
        raise ValueError(
            "MLX float64 arrays cannot be exported to a Torch MPS tensor; "
            "use float32/bfloat16 or request device='cpu'"
        )

    return array


def _has_negative_stride(array: mx.array) -> bool:
    """Return whether an evaluated MLX array has a DLPack-incompatible view."""
    # PyTorch's DLPack importer aborts the process for negative strides.  MLX
    # exposes the evaluated layout through the buffer protocol, so inspect it
    # before handing the capsule to PyTorch.
    with memoryview(array) as view:
        return any(stride < 0 for stride in (view.strides or ()))


def _export_evaluated_mlx(
    array: mx.array,
    target_device: torch.device,
    mx: Any,
    *,
    materialize_negative: bool = True,
) -> torch.Tensor:
    """Export an already-evaluated MLX result through one DLPack capsule.

    Negative-stride results are materialized here as a safety fallback.  The
    normal (contiguous/positive-stride) path performs no copy and no extra
    evaluation; :func:`mlx_call_multi` batches any required materialization
    evaluations for all outputs together.
    """
    if materialize_negative and _has_negative_stride(array):
        materialize_stream = mx.cpu if target_device.type == "cpu" else mx.gpu
        array = mx.contiguous(array, stream=materialize_stream)
        mx.eval(array)

    if target_device.type == "cpu":
        # MLX owns CPU-accessible unified memory.  Request a CPU DLPack view
        # explicitly rather than importing on MPS and copying back.
        dlpack = array.__dlpack__(dl_device=(1, 0), copy=False)
        return torch.utils.dlpack.from_dlpack(dlpack)
    return torch.utils.dlpack.from_dlpack(array)


@_serialized_bridge
def mlx_call_multi(
    operation: Callable[..., tuple[mx.array, ...]],
    *tensors: torch.Tensor | MlxTensorView,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run one MLX operation and export all of its outputs as Torch tensors.

    ``operation`` must return a non-empty flat ``tuple`` or ``list`` of MLX
    arrays.  All Torch/MPS inputs are fenced once before import, and all
    ordinary outputs are evaluated with one ``mx.eval(*outputs)`` call before
    being exported through DLPack.  The imported arrays and detached Torch
    owners remain local until every output capsule has been consumed, which is
    required when MLX lazily donates a borrowed input buffer.  Callers must
    serialize any overlapping MPS work outside this function, including use
    or mutation of source and returned tensors.

    CPU targets retain the same float64 and negative-stride safeguards as
    :func:`mlx_to_torch`.  A negative-stride output necessarily needs one
    additional materialization evaluation; contiguous MPS model outputs take
    the single-evaluation, zero-copy path.

    """
    mx = _mlx_core()
    target_device = _get_torch_device() if device is None else torch.device(device)
    if target_device.type not in {"cpu", "mps"}:
        raise ValueError(
            f"The MLX tensor bridge supports CPU and MPS targets, got {target_device}"
        )

    detached = tuple(
        tensor.detach() for tensor in tensors if isinstance(tensor, torch.Tensor)
    )
    needs_mps_fence = any(tensor.device.type == "mps" for tensor in detached) or any(
        isinstance(tensor, MlxTensorView) for tensor in tensors
    )
    if needs_mps_fence:
        torch.mps.synchronize()

    borrowed: tuple[Any, ...] = tuple(
        (
            tensor.array
            if isinstance(tensor, MlxTensorView)
            else _torch_to_mlx(tensor.detach(), copy=False, synchronize=False)
        )
        for tensor in tensors
    )

    if target_device.type == "cpu" and any(
        array.dtype == mx.float64 for array in borrowed
    ):
        with mx.stream(mx.cpu):
            result = operation(*borrowed)
    else:
        result = operation(*borrowed)
    if not isinstance(result, (tuple, list)) or not result:
        raise TypeError(
            "mlx_call_multi operation must return a non-empty tuple or list of MLX arrays"
        )
    arrays = tuple(result)
    if any(not isinstance(array, mx.array) for array in arrays):
        raise TypeError("mlx_call_multi outputs must be MLX arrays")

    # Prepare all outputs before crossing the one shared MLX evaluation
    # boundary. This is the key difference from calling mlx_to_torch in a
    # loop, which would fence/evaluate every result separately.
    arrays = tuple(_prepare_mlx_export(array, target_device, mx) for array in arrays)
    mx.eval(*arrays)

    # DLPack cannot represent negative strides. Materialize all such outputs
    # together so even this safety path has one additional evaluation boundary
    # rather than one boundary per result.
    negative = tuple(_has_negative_stride(array) for array in arrays)
    if any(negative):
        materialized = []
        for array, needs_materialization in zip(arrays, negative):
            if needs_materialization:
                stream = mx.cpu if target_device.type == "cpu" else mx.gpu
                array = mx.contiguous(array, stream=stream)
            materialized.append(array)
        arrays = tuple(materialized)
        mx.eval(*(array for array, needs in zip(arrays, negative) if needs))

    outputs = tuple(
        _export_evaluated_mlx(array, target_device, mx, materialize_negative=False)
        for array in arrays
    )
    # Keep both borrowed MLX views and their Torch owners alive through the
    # final DLPack import.  (The local remains live until function return.)
    _ = borrowed
    return outputs


@_serialized_bridge
def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor.

    MLX arrays with PyTorch-compatible strides share their unified-memory
    allocation through DLPack, including explicit CPU views. Negative-stride
    views are materialized because PyTorch's DLPack importer cannot represent
    them safely. MLX is evaluated before the handoff because the frameworks do
    not share stream state. Only CPU and MPS targets are supported; other
    target devices are rejected.

    Args:
        array: MLX array
        device: Target PyTorch device (default: MPS if available)

    Returns:
        PyTorch tensor with the same data
    """
    mx = _mlx_core()
    target_device = _get_torch_device() if device is None else torch.device(device)
    array = _prepare_mlx_export(array, target_device, mx)
    mx.eval(array)
    return _export_evaluated_mlx(array, target_device, mx)


__all__ = [
    "MlxTensorView",
    "borrow_torch_tensors",
    "mlx_call",
    "mlx_call_multi",
    "mlx_to_torch",
    "torch_to_mlx",
]
