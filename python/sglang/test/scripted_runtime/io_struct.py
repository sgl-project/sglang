"""Typed control messages for the scripted-runtime control plane.

Most of these dataclasses are the wire format of the ZMQ ``PAIR`` socket
that connects the test process (:class:`ScriptedHttpServer`) to the
scheduler-subprocess dispatch loop owned by
:class:`ScriptedSchedulerHook`. They are exchanged via
``send_pyobj`` / ``recv_pyobj`` (pickle), replacing the raw dicts that the
old ``multiprocessing.connection`` transport used.

Both ends import these from the same package, so the pickled qualified
names match across the process boundary. The style mirrors
:mod:`sglang.srt.managers.io_struct` (frozen + slots dataclasses).

Two directions over the socket:

- test process -> scheduler hook: :data:`ScriptedCommand`
  (:class:`RunScript`, :class:`Shutdown`).
- scheduler hook -> test process: :data:`ScriptedReply`
  (:class:`HookReady`, :class:`ScriptSucceeded`, :class:`ScriptFailed`).

:class:`OutOfBandError` is the exception: it does **not** travel over the
socket. A fatal scheduler-side error tears the engine down (every rank
``sys.exit``s), so the socket is already gone by the time the caller
notices. The hook serializes it to JSON on the out-of-band error file
instead, and the caller deserializes it during shutdown.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Tuple, Union


@dataclass(frozen=True, slots=True)
class RunScript:
    """Run the sub-script at ``fn_path`` with ``args`` after the context handle."""

    fn_path: str
    args: Tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class Shutdown:
    """Tell the dispatch loop to return so the scheduler can tear down."""


@dataclass(frozen=True, slots=True)
class HookReady:
    """Sent once when the dispatch loop connects, confirming the scheduler came up."""


@dataclass(frozen=True, slots=True)
class ScriptSucceeded:
    """A sub-script returned cleanly."""


@dataclass(frozen=True, slots=True)
class ScriptFailed:
    """A sub-script raised; carries the formatted traceback."""

    traceback: str


@dataclass(frozen=True, slots=True)
class OutOfBandError:
    """A fatal scheduler-side error that tore the engine down.

    Unlike :class:`ScriptFailed` (in-band, leaves the engine alive), this is
    written to the out-of-band error file as JSON because the ZMQ socket is
    already gone once the engine ``sys.exit``s. Serialized / parsed via
    :meth:`to_json` / :meth:`from_json`.
    """

    traceback: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "OutOfBandError":
        return cls(**json.loads(text))


ScriptedCommand = Union[RunScript, Shutdown]
ScriptedReply = Union[HookReady, ScriptSucceeded, ScriptFailed]
