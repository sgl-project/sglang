from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from pydantic import BaseModel


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


class StreamingParseResult(BaseModel):
    """Result of streaming incremental parsing."""

    normal_text: str = ""
    calls: List[ToolCallItem] = []


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


class ToolCallProcessingResult(NamedTuple):
    """Result of processing tool calls in a response."""
    
    tool_calls: Optional[List[Any]]  # List of ToolCall objects or None if parsing failed
    remaining_text: str  # Text remaining after parsing tool calls
    finish_reason: Dict[str, Any]  # Updated finish reason dictionary


"""
Helper alias of function
Usually it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""
_GetInfoFunc = Callable[[str], StructureInfo]
