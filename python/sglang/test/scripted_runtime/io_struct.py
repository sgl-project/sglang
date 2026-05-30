"""Typed control messages for the scripted-runtime control plane.

These dataclasses are the wire format of the ZMQ ``PAIR`` socket that
connects the test process (:class:`ScriptedHttpServer`) to the
scheduler-subprocess :func:`router_script`. They are exchanged via
``send_pyobj`` / ``recv_pyobj`` (pickle), replacing the raw dicts that the
old ``multiprocessing.connection`` transport used.

Both ends import these from the same package, so the pickled qualified
names match across the process boundary. The style mirrors
:mod:`sglang.srt.managers.io_struct` (frozen + slots dataclasses).

Two directions:

- test process -> scheduler hook: :data:`ScriptedCommand`
  (:class:`RunScript`, :class:`Shutdown`).
- scheduler hook -> test process: :data:`ScriptedReply`
  (:class:`HookReady`, :class:`ScriptSucceeded`, :class:`ScriptFailed`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union


@dataclass(frozen=True, slots=True)
class RunScript:
    """Run the sub-script at ``fn_path`` with ``args`` after the context handle."""

    fn_path: str
    args: Tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class Shutdown:
    """Tell the router to return so the scheduler can tear down."""


@dataclass(frozen=True, slots=True)
class HookReady:
    """Sent once when the router connects, confirming the scheduler came up."""


@dataclass(frozen=True, slots=True)
class ScriptSucceeded:
    """A sub-script returned cleanly."""


@dataclass(frozen=True, slots=True)
class ScriptFailed:
    """A sub-script raised; carries the formatted traceback."""

    traceback: str


ScriptedCommand = Union[RunScript, Shutdown]
ScriptedReply = Union[HookReady, ScriptSucceeded, ScriptFailed]
