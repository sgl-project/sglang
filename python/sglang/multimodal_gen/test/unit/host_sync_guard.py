"""Guards that a hot path never blocks the host on the device.

A blocking device read (``item()``, ``float(t)``, ``if t:``, ``.cpu()``,
``nonzero``, boolean-mask indexing) or a pageable host-to-device copy
(``torch.tensor(..., device=cuda)``) stalls the host until the stream drains,
so it cannot enqueue the next step while the GPU runs this one.

CUDA tests wrap the hot path, after one warmup call, in ``no_cuda_sync``::

    with no_cuda_sync():
        stage._run_denoising_step(ctx, step, batch, server_args)

    # a sync the path keeps on purpose, named by the function that issues it
    with no_cuda_sync(allow=("stages/denoising.py:_run_denoising_loop",)):
        stage._run_denoising_loop(ctx, batch, server_args)

CPU tests replace device tensors (timesteps, sigmas, masks) with
``SyncOnRead`` and compare against a plain run::

    timesteps = torch.tensor([999.0, 500.0]).as_subclass(SyncOnRead)

``SyncOnRead`` catches host reads of device values only; host-to-device
uploads and per-step ``.cpu()`` copies need the CUDA guard.

CPU tests of code that builds its own tensors run it under
``forbid_host_syncs``, a dispatch mode that fails on the ops that would sync
on CUDA whatever tensor they read::

    with forbid_host_syncs():
        model(**inputs)
"""

import os
import traceback
import unittest
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp

_SYNC_WARNING = "called a synchronizing CUDA operation"
_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep


def _call_site(stack: traceback.StackSummary) -> traceback.StackSummary:
    """The stack up to its innermost frame outside torch, warnings and this guard."""
    for index in range(len(stack) - 1, -1, -1):
        filename = stack[index].filename
        if not (
            filename.startswith(_TORCH_DIR)
            or filename.endswith("warnings.py")
            or filename == __file__
        ):
            return stack[: index + 1]
    return stack


def _is_allowed(frame: traceback.FrameSummary, allow: Sequence[str]) -> bool:
    return any(
        frame.filename.endswith(path_suffix) and frame.name == function
        for path_suffix, function in (entry.rsplit(":", 1) for entry in allow)
    )


@contextmanager
def no_cuda_sync(allow: Sequence[str] = ()):
    """Fail if the body synchronizes the host with a CUDA device.

    Every implicit sync torch's sync debug mode reports, plus explicit
    ``torch.cuda.synchronize()`` calls, is collected with its Python stack; the
    body runs to completion and then the guard fails listing each sync whose
    call site, the function that issued it, matches no ``allow`` entry
    (``"<file suffix>:<function>"``; callees of that function stay guarded).
    Skips the test when CUDA is unavailable; restores the debug mode on exit.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no_cuda_sync needs a CUDA device")
    syncs: list[traceback.StackSummary] = []
    explicit_synchronize = torch.cuda.synchronize

    def synchronize(*args, **kwargs):
        syncs.append(traceback.extract_stack()[:-1])
        return explicit_synchronize(*args, **kwargs)

    with warnings.catch_warnings():
        # "warn" rather than "error": an error could be swallowed by a broad
        # except in the code under test, a warning always reaches this hook
        show_other = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if _SYNC_WARNING in str(message):
                syncs.append(traceback.extract_stack()[:-1])
            else:
                show_other(message, category, filename, lineno, file, line)

        warnings.filterwarnings("always", message=_SYNC_WARNING)
        warnings.showwarning = show
        previous_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with patch.object(torch.cuda, "synchronize", synchronize):
                yield
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)

    violations = [
        "".join(traceback.format_list(stack[-6:]))
        for stack in map(_call_site, syncs)
        if not _is_allowed(stack[-1], allow)
    ]
    if violations:
        raise AssertionError(
            f"{len(violations)} host-device sync(s) in a no-sync region:\n"
            + "\n".join(f"--- sync {i}:\n{v}" for i, v in enumerate(violations))
        )


def assert_no_cuda_sync(
    fn: Callable[..., Any], *args, allow: Sequence[str] = (), **kwargs
) -> Any:
    """Run ``fn(*args, **kwargs)`` under ``no_cuda_sync`` and return its result."""
    with no_cuda_sync(allow=allow):
        return fn(*args, **kwargs)


def _has_bool_index(index: Any) -> bool:
    indices = index if isinstance(index, tuple) else (index,)
    return any(
        isinstance(item, torch.Tensor) and item.dtype == torch.bool for item in indices
    )


# reads whose result is a host value, or whose output size depends on values
_HOST_READS = frozenset(
    {
        torch.Tensor.item,
        torch.Tensor.tolist,
        torch.Tensor.numpy,
        torch.Tensor.__array__,
        torch.Tensor.__bool__,
        torch.Tensor.__float__,
        torch.Tensor.__int__,
        torch.Tensor.__index__,
        torch.Tensor.__format__,
        torch.Tensor.nonzero,
        torch.nonzero,
        torch.argwhere,
        torch.masked_select,
        torch.Tensor.masked_select,
        torch.unique,
        torch.equal,
        torch.Tensor.equal,
    }
)


class SyncOnRead(torch.Tensor):
    """CPU stand-in for a device tensor: a host read of its values raises.

    Results of ops on it stay ``SyncOnRead``, ``.to(...)`` included, since a CPU
    test cannot tell a host copy from a same-device move; only ``.cpu()``, the
    explicit host copy, returns a plain tensor that reads freely. Compare
    results through ``.as_subclass(torch.Tensor)``.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func in _HOST_READS
            or (func is torch.where and len(args) + len(kwargs) == 1)
            or (
                func in (torch.Tensor.__getitem__, torch.Tensor.__setitem__)
                and _has_bool_index(args[1])
            )
        ):
            raise AssertionError(
                f"{func.__name__} reads a device tensor on the host, which syncs"
            )
        if func is torch.Tensor.cpu:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs).as_subclass(torch.Tensor)
        return super().__torch_function__(func, types, args, kwargs)


aten = torch.ops.aten

# a scalar read, or an output shape that depends on device values
_SYNCING_OPS = {
    aten._local_scalar_dense.default,
    aten.is_nonzero.default,
    aten.equal.default,
    aten.nonzero.default,
    aten.masked_select.default,
    aten.repeat_interleave.Tensor,
    aten._unique2.default,
    aten.unique_dim.default,
}
# boolean-mask indexing runs a nonzero to size its output
_INDEXING_OPS = {
    aten.index.Tensor,
    aten.index_put.default,
    aten.index_put_.default,
}


class forbid_host_syncs(TorchDispatchMode):
    """Raise when code under test would read device values on the host."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _SYNCING_OPS and not (
            func is aten.repeat_interleave.Tensor
            and kwargs.get("output_size") is not None
        ):
            raise AssertionError(f"{func} syncs the host with the device")
        if func in _INDEXING_OPS and any(
            index is not None and index.dtype in (torch.bool, torch.uint8)
            for index in args[1]
        ):
            raise AssertionError(f"boolean-mask {func} syncs the host with the device")
        return func(*args, **kwargs)


class RecordOps(TorchDispatchMode):
    """Records the aten ops the body runs, with their arguments."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[Any, tuple, dict]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    def casts(self, dtype: torch.dtype | None = None) -> int:
        """The number of dtype or device copies, only those to ``dtype`` if given."""
        return sum(
            func is aten._to_copy.default
            and (dtype is None or kwargs.get("dtype") is dtype)
            for func, _, kwargs in self.calls
        )

    def uploads(self) -> list[torch.Tensor]:
        """Tensors built from host data (``torch.tensor``, a list used as an index),
        each a blocking copy on CUDA; 0-dim ones are Python scalars that kernels
        read on the host, not uploads."""
        return [
            args[0]
            for func, args, _ in self.calls
            if func is aten.lift_fresh.default and args[0].dim() > 0
        ]


class CountedUploads(torch.Tensor):
    """A host tensor that counts its copies to the device in ``uploads``.

    The copy ``.to(...)`` returns is a plain tensor, so moving it again is not
    counted. Reset ``CountedUploads.uploads`` before the code under test.
    """

    uploads = 0

    def to(self, *args, **kwargs):
        CountedUploads.uploads += 1
        return super().to(*args, **kwargs).as_subclass(torch.Tensor)


def reference_custom_ops():
    """Run every CustomOp built in the body on its reference path: tiny test
    shapes miss the fused kernels' constraints."""
    return patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_native)


@contextmanager
def tiny_dit_runtime(compute_dtype: torch.dtype):
    """Build and run a tiny DiT in one process: SDPA attention computing in
    ``compute_dtype`` and reference CustomOps. Needs the
    ``single_process_model_parallel`` fixture."""
    with (
        patch.multiple(
            attention_layer,
            get_attn_backend=lambda *args, **kwargs: SDPABackend,
            get_compute_dtype=lambda *args, **kwargs: compute_dtype,
            _resolve_sp_attention_mode=lambda *args, **kwargs: ("ulysses", False),
            _maybe_install_backend_autotune=lambda *args, **kwargs: None,
        ),
        reference_custom_ops(),
    ):
        yield
