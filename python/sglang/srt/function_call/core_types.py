from dataclasses import dataclass
from typing import Callable, List, Optional

from pydantic import BaseModel


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string
    # The tool_call_id string emitted by the model, captured verbatim.
    # Only populated by detectors whose downstream consumers need the exact
    # model-emitted id (e.g. RL training trajectories). Existing detectors
    # leave this as None and the serving layer falls back to its usual id
    # generation strategy.
    tool_call_id: Optional[str] = None


class StreamingParseResult(BaseModel):
    """Result of streaming incremental parsing."""

    normal_text: str = ""
    calls: List[ToolCallItem] = []


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


"""
Helper alias of function
Usually it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""
_GetInfoFunc = Callable[[str], StructureInfo]
