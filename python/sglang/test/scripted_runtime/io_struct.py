from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Tuple, Union


@dataclass(frozen=True, slots=True)
class RunScript:

    fn_path: str
    args: Tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass


@dataclass(frozen=True, slots=True)
class HookReady:
    pass


@dataclass(frozen=True, slots=True)
class ScriptSucceeded:
    pass


@dataclass(frozen=True, slots=True)
class ScriptFailed:

    traceback: str


@dataclass(frozen=True, slots=True)
class OutOfBandError:

    traceback: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, text: str) -> OutOfBandError:
        return cls(**json.loads(text))


ScriptedCommand = Union[RunScript, Shutdown]
ScriptedReply = Union[HookReady, ScriptSucceeded, ScriptFailed]
