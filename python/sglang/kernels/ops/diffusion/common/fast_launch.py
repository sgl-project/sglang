# SPDX-License-Identifier: Apache-2.0
"""Replay a small Triton wrapper's launches for operands it has seen before.

A small kernel's wrapper costs more host time than its GPU time: operand checks,
stride math, the device guard and Triton's JIT dispatch take ~28 us per call against
a ~9 us launch (rmsnorm_scale, Z-Image 512^2, H200). All of it is fixed by the
operands' signature. ``@replay_launches`` runs the wrapper normally the first time it
sees a signature and records its Triton launches; later calls with that signature
allocate the outputs and relaunch the recorded compiled kernels on the new tensors.
Same kernels, grids and arguments, so results are bit-identical.

Contract for a decorated function: its GPU work is Triton launches whose tensor
arguments are its tensor inputs or the outputs it allocates with ``torch.empty*`` and
returns (a tensor, a tuple of tensors, or None), and its choices depend only on the
operands' shapes, strides, dtypes, devices, 16-byte alignment and scalar arguments.
Recording rejects what it can see breaking that (aten ops other than allocation and
views, reads of tensor values, launches on other tensors) and leaves such a signature
on the normal path.
Launches that bypass Triton's JIT (tvm-ffi, CuTe DSL) are invisible to it.

``SGLANG_DIFFUSION_TRITON_FAST_LAUNCH=0`` turns it off.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import threading
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from triton.runtime import driver
from triton.runtime.jit import JITFunction

try:
    from triton import knobs
    from triton.knobs import HookChain
except ImportError:
    # Vendor Triton builds may not expose the upstream launch-hook API.
    knobs = None

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get(
    "SGLANG_DIFFUSION_TRITON_FAST_LAUNCH", "1"
).strip().lower() not in {"0", "false", "off", "no"}

_aten = torch.ops.aten
_ALLOCATIONS = {
    _aten.empty,
    _aten.empty_like,
    _aten.empty_strided,
    _aten.new_empty,
    _aten.new_empty_strided,
}

_SCALARS = (bool, int, float, str, type(None), torch.dtype, torch.device)
_NORMAL = object()  # this signature always runs the wrapper
_NONE = object()  # this signature returns None without launching


class _Recording(threading.local):
    active = False


_recording = _Recording()
_record_lock = threading.Lock()


class _LaunchOnly(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.violation: str | None = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        # A decorated function may only allocate and take views.
        if (
            self.violation is None
            and func.overloadpacket not in _ALLOCATIONS
            and not _is_view(func, args, out)
        ):
            self.violation = str(func)
        return out


def _is_view(func, args, out) -> bool:
    """Whether ``func`` mutated nothing and returned only aliases of its inputs."""
    if func._schema.is_mutable:
        return False
    sources = {
        t.untyped_storage().data_ptr()
        for t in tree_leaves(args)
        if isinstance(t, torch.Tensor)
    }
    results = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor)]
    return bool(results) and all(
        t.untyped_storage().data_ptr() in sources for t in results
    )


def _signature(args: tuple, kwargs: dict) -> tuple[tuple | None, list[torch.Tensor]]:
    key, tensors = [], []
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
            # Triton specializes the compiled kernel on 16-byte alignment.
            key.append(
                (
                    value.shape,
                    value.stride(),
                    value.dtype,
                    value.device,
                    value.data_ptr() % 16,
                )
            )
        elif type(value) in _SCALARS and value == value:  # NaN never matches its key
            key.append((type(value), value))
        else:
            return None, tensors
    if kwargs:
        key.append(tuple(kwargs))
    return tuple(key), tensors


class _Plan:
    """Outputs to allocate and launches to replay for one signature."""

    def __init__(self, outputs: list, result: str, launches: list, device: int):
        self.outputs = outputs  # (size, stride, dtype, device), or an input index
        self.result = result  # "none", "tensor" or "tuple"
        # (kernel, grid, run, function, metadata, constant args, tensor slots)
        self.launches = launches
        self.device = device
        self.current_stream = driver.active.get_current_stream

    def replay(self, inputs: list[torch.Tensor]):
        outs = [
            (
                inputs[spec]
                if isinstance(spec, int)
                else torch.empty_strided(
                    spec[0], spec[1], dtype=spec[2], device=spec[3]
                )
            )
            for spec in self.outputs
        ]
        sources = (inputs, outs)
        stream = None if _launch_hooked() else self.current_stream(self.device)
        for compiled, grid, run, function, metadata, constants, slots in self.launches:
            values = list(constants)
            for position, source, index in slots:
                values[position] = sources[source][index]
            if stream is None:
                compiled[grid](*values)
            else:
                # Triton's runner, minus the hook metadata it builds per launch.
                run(*grid, stream, function, metadata, None, None, None, *values)
        if self.result == "tensor":
            return outs[0]
        return tuple(outs) if self.result == "tuple" else None


def _launch_hooked() -> bool:
    """A profiler such as proton hooks Triton launches; those go through the runner."""
    if knobs is None:
        # Keep the regular runner when its hooks cannot be inspected safely.
        return True
    return any(
        hook is not None and not (isinstance(hook, HookChain) and not hook.calls)
        for hook in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook)
    )


def _covers_storage(t: torch.Tensor) -> bool:
    extent = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    return (
        t.storage_offset() == 0
        and t.untyped_storage().nbytes() == extent * t.element_size()
    )


def _plan(out: Any, inputs: list[torch.Tensor], launches: list) -> Any:
    if out is None and not launches:
        return _NONE
    if out is None:
        result, outs = "none", []
    elif isinstance(out, tuple):
        result, outs = "tuple", list(out)
    else:
        result, outs = "tensor", [out]
    if not launches or not all(isinstance(o, torch.Tensor) for o in outs):
        return _NORMAL
    input_at: dict[int, list[int]] = {}
    for i, t in enumerate(inputs):
        input_at.setdefault(t.data_ptr(), []).append(i)
    output_at, specs = {}, []
    for j, o in enumerate(outs):
        aliased = [i for i in input_at.get(o.data_ptr(), ()) if inputs[i] is o]
        if aliased:
            specs.append(aliased[0])
        elif o.data_ptr() in input_at or o.data_ptr() in output_at:
            return _NORMAL
        elif not _covers_storage(o):
            return _NORMAL
        else:
            specs.append((o.shape, o.stride(), o.dtype, o.device))
        output_at[o.data_ptr()] = j
    devices, replays = set(), []
    for jit_fn, compiled, grid, args, kwargs, device in launches:
        names = jit_fn.arg_names
        values = list(args) + [
            kwargs[name] if name in kwargs else jit_fn.params[i].default
            for i, name in enumerate(names[len(args) :], start=len(args))
        ]
        # The launcher takes every parameter, constexprs included (Triton >= 3.3).
        if len(compiled.src.signature) != len(values):
            return _NORMAL
        if callable(grid):
            grid = grid(dict(zip(names, values)))
        constants, slots = [], []
        for position, value in enumerate(values):
            is_tensor = isinstance(value, torch.Tensor)
            constants.append(None if is_tensor else value)
            if not is_tensor:
                continue
            pointer = value.data_ptr()
            if pointer in output_at:
                slots.append((position, 1, output_at[pointer]))
            elif len(input_at.get(pointer, ())) == 1:
                slots.append((position, 0, input_at[pointer][0]))
            else:  # a temporary, or an ambiguous input
                return _NORMAL
        grid = tuple(grid) + (1,) * (3 - len(grid))
        launcher = (compiled.run, compiled.function, compiled.packed_metadata)
        replays.append((compiled, grid, *launcher, tuple(constants), tuple(slots)))
        devices.add(device)
    if len(devices) != 1:
        return _NORMAL
    return _Plan(specs, result, replays, devices.pop())


def _record(fn: Callable, args: tuple, kwargs: dict, inputs: list[torch.Tensor]):
    """Run ``fn`` and return its result and a plan, or None while another thread records."""
    if not _record_lock.acquire(blocking=False):
        return fn(*args, **kwargs), None
    launches = []
    thread = threading.get_ident()
    jit_run = JITFunction.run

    def run(self, *a, grid, warmup, **kw):
        compiled = jit_run(self, *a, grid=grid, warmup=warmup, **kw)
        if not warmup and threading.get_ident() == thread:
            launches.append(
                (self, compiled, grid, a, kw, torch.accelerator.current_device_index())
            )
        return compiled

    guard = _LaunchOnly()
    JITFunction.run = run
    _recording.active = True
    try:
        with guard:
            out = fn(*args, **kwargs)
    finally:
        JITFunction.run = jit_run
        _recording.active = False
        _record_lock.release()
    if guard.violation is not None:
        return out, _NORMAL
    try:
        return out, _plan(out, inputs, launches)
    except (AttributeError, TypeError) as exc:  # Triton's compiled-kernel API moved
        logger.warning("%s keeps its normal launch path: %s", fn.__qualname__, exc)
        return out, _NORMAL


def replay_launches(fn: Callable) -> Callable:
    """Decorate a Triton wrapper to replay its launches per operand signature."""
    plans: dict[tuple, Any] = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Checked first, so Dynamo traces ``fn`` without guarding on ``_ENABLED``.
        if torch.compiler.is_compiling() or not _ENABLED or _recording.active:
            return fn(*args, **kwargs)
        key, inputs = _signature(args, kwargs)
        if key is None:
            return fn(*args, **kwargs)
        plan = plans.get(key)
        if plan is None:
            out, plan = _record(fn, args, kwargs, inputs)
            if plan is not None:
                plans[key] = plan
            return out
        if plan is _NONE:
            return None
        if plan is _NORMAL or torch.accelerator.current_device_index() != plan.device:
            return fn(*args, **kwargs)
        return plan.replay(inputs)

    # Resolved annotations, so custom-op schema inference does not evaluate them here.
    wrapper.__signature__ = inspect.signature(fn, eval_str=True)
    return wrapper
