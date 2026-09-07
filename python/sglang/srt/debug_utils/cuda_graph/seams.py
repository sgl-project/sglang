# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The two `__init_subclass__` seams that arm the taps.

`cuda_graph_dump` needs four facts that only the graph machinery knows: when a
backend is being constructed, when a capture session is open, which shape is
being captured, and when a replay has just finished.  None of that is reachable
from an `nn.Module` forward hook, and none of it is worth threading through the
runners by hand.

Both seams are installed from `__init_subclass__` on the two abstract bases, so
a backend or a phase runner is covered the moment it is *defined* -- including
the accelerator ports and the speculative-decoding subclasses, in-tree or not.
Only methods found in `cls.__dict__` are wrapped: an override-free subclass
inherits its parent's already-wrapped method, so nothing is wrapped twice and
`super()` chains stay intact.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Optional

from sglang.srt.debug_utils.cuda_graph.state import cuda_graph_dump

_SEAM_MARK = "__cuda_graph_dump_seam__"

# `type(backend).__name__` -> the `--cuda-graph-backend` value it implements.
# A backend outside this table is tagged with its class name, which is still
# unambiguous; the table only exists to make the common frames readable.
_BACKEND_LABELS = {
    "FullCudaGraphBackend": "full",
    "BreakableCudaGraphBackend": "breakable",
    "TcPiecewiseCudaGraphBackend": "tc_piecewise",
}


def _wrap(cls: type, name: str, factory: Callable[[Callable], Callable]) -> None:
    """Replace `cls.<name>` with `factory(cls.<name>)`, at most once ever."""
    fn = cls.__dict__.get(name)
    if fn is None or getattr(fn, _SEAM_MARK, False):
        return
    wrapped = factory(fn)
    setattr(wrapped, _SEAM_MARK, True)
    setattr(cls, name, wrapped)


def _backend_label(backend: Any) -> str:
    name = type(backend).__name__
    return _BACKEND_LABELS.get(name, name)


def _first_arg(args: tuple, kwargs: dict, name: str) -> Any:
    """The value of a leading positional-or-keyword parameter, however passed."""
    return args[0] if args else kwargs.get(name)


# ------------------------------- backend seam -------------------------------


def install_backend_seam(cls: type) -> None:
    """Arm the capture-side scopes and the replay-side collect point."""
    _wrap(cls, "__init__", _seam_init)
    _wrap(cls, "capture_session", _seam_capture_session)
    _wrap(cls, "capture_one", _seam_capture_one)
    _wrap(cls, "replay", _seam_replay)
    _wrap(cls, "cleanup", _seam_cleanup)


def _seam_init(fn: Callable) -> Callable:
    """Backend construction: dummy-valued frames seen here are swallowed."""

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        cuda_graph_dump.configure()
        if not cuda_graph_dump.enabled:
            return fn(self, *args, **kwargs)
        with cuda_graph_dump.setup_scope():
            return fn(self, *args, **kwargs)

    return wrapped


def _seam_capture_session(fn: Callable) -> Callable:
    """Arms the T1 stream-capturing check for the duration of one session.

    All three in-tree backends implement this as a `@contextmanager`, so the
    wrapper has to be one too and has to re-yield the inner value rather than
    return the inner context manager.
    """

    @functools.wraps(fn)
    @contextmanager
    def wrapped(self, *args, **kwargs):
        cuda_graph_dump.configure()
        if not cuda_graph_dump.enabled:
            with fn(self, *args, **kwargs) as value:
                yield value
            return
        with cuda_graph_dump.capture_scope():
            with fn(self, *args, **kwargs) as value:
                yield value

    return wrapped


def _seam_capture_one(fn: Callable) -> Callable:
    """Publishes the shape being captured, and resets counters per forward.

    `shape_key` doubles as the shape token: it is a frozen dataclass, so it is
    hashable, and its `.size` is the padded row count `collect` needs.
    """

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not cuda_graph_dump.enabled:
            return fn(self, *args, **kwargs)
        shape_key = _first_arg(args, kwargs, "shape_key")
        args, kwargs = _wrap_forward_fn(args, kwargs)
        with cuda_graph_dump.shape_scope(shape_key):
            return fn(self, *args, **kwargs)

    return wrapped


def _wrap_forward_fn(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Substitute `forward_fn` with the occurrence-counter-resetting version.

    `breakable` and `tc_piecewise` both call `forward_fn()` twice per
    `capture_one`; resetting per `capture_one` instead would make the second
    pass allocate a second buffer for every name.
    """
    if len(args) >= 2:
        forward_fn = cuda_graph_dump.wrap_forward_fn(args[1])
        return (args[0], forward_fn) + args[2:], kwargs
    if "forward_fn" in kwargs:
        forward_fn = cuda_graph_dump.wrap_forward_fn(kwargs["forward_fn"])
        return args, {**kwargs, "forward_fn": forward_fn}
    return args, kwargs


def _seam_replay(fn: Callable) -> Callable:
    """The collect point: after `replay()` returns, outside the graph.

    Every recorded `copy_` has run by now, so the buffers hold *this* replay's
    values, and `dumper.dump` runs on the host exactly as it does in eager mode.
    """

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        output = fn(self, *args, **kwargs)
        if cuda_graph_dump.enabled:
            shape_key = _first_arg(args, kwargs, "shape_key")
            cuda_graph_dump.collect(
                shape_key,
                graph_backend=_backend_label(self),
                graph_size=getattr(shape_key, "size", None),
            )
        return output

    return wrapped


def _seam_cleanup(fn: Callable) -> Callable:
    """Reports coverage once the backend is torn down.

    Nothing in the tree calls `cleanup()` today; `state` also registers the
    same one-shot report with `atexit`, which is the path that actually fires.
    The report runs *after* the inner call and not in a `finally`, so a real
    teardown exception is never masked by a strict-mode coverage failure.
    """

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        cuda_graph_dump.report()
        return result

    return wrapped


# -------------------------------- runner seam -------------------------------


def install_runner_seam(cls: type) -> None:
    """Publish the replay-side runner and batch metadata around `execute`."""
    _wrap(cls, "execute", _seam_execute)


def _seam_execute(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not cuda_graph_dump.enabled:
            return fn(self, *args, **kwargs)
        forward_batch = _first_arg(args, kwargs, "forward_batch")
        # Replay-time eager frames (a `breakable` break body, or a module that
        # was never captured) number their occurrences from the same base as
        # the capture-side frames only if the counters restart here too.
        cuda_graph_dump.begin_forward()
        with cuda_graph_dump.runner_scope(self, **_replay_tags(self, forward_batch)):
            return fn(self, *args, **kwargs)

    return wrapped


def _replay_tags(runner: Any, forward_batch: Any) -> dict:
    """Metadata that belongs to *this* replay, read from the live batch."""
    mode = getattr(forward_batch, "forward_mode", None)
    return {
        # The class name, not a "prefill"/"decode" guess: it is the one label
        # that also separates the speculative draft and verify runners.
        "graph_runner": type(runner).__name__,
        "forward_mode": getattr(mode, "name", None),
        "pd_role": _pd_role(),
    }


def _pd_role() -> Optional[str]:
    """`prefill` / `decode` under PD disaggregation, None when co-located."""
    try:
        from sglang.srt.runtime_context import get_disagg

        mode = get_disagg().disaggregation_mode
    except Exception:  # no published context yet (unit tests, tooling)
        return None
    role = getattr(mode, "value", None) or getattr(mode, "name", None)
    role = str(role if role is not None else mode).lower()
    return None if role in ("null", "none") else role
