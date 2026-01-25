from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UniversalToolParserState(Enum):
    IDLE = auto()
    TOOL_START = auto()
    IN_TOOL_NAME = auto()
    TOOL_NAME_END = auto()
    PARAMETER_START = auto()
    IN_PARAMETER_NAME = auto()
    IN_PARAMETER_VALUE = auto()
    PARAMETER_END = auto()
    TOOL_END = auto()
    ERROR = auto()


class ParserEvent(BaseModel):
    event_type: str
    name: Optional[str] = None
    value: Optional[str] = None
    text_delta: Optional[str] = None


class ParseResult(BaseModel):
    state: UniversalToolParserState
    completed_tools: List[Dict[str, Any]]
    remaining: str
    events: List[ParserEvent] = []
    normal_text: str = ""
    error: Optional[str] = None


class XmlConfig(BaseModel):
    """Configuration for universal XML-like tool call parsing."""

    root_tag: Optional[str] = None
    tool_tag: str
    tool_name_tag: Optional[str] = None
    tool_name_attr: Optional[str] = None
    tool_name_tag_attr: Optional[str] = None
    param_tag: Optional[str] = None
    param_name_attr: Optional[str] = None
    param_key_tag: Optional[str] = None
    param_value_tag: Optional[str] = None
    attr_sep: str = '="'  # Separator for attributes, e.g. '=' for Qwen


class JsonConfig(BaseModel):
    """Configuration for universal JSON tool call parsing."""

    prefix: Optional[str] = None
    suffix: Optional[str] = None
    is_array: bool = True
    is_markdown: bool = False
    name_prefix: Optional[str] = None
    name_suffix: Optional[str] = None


class ToolParserStateMachine(ABC):
    @abstractmethod
    def parse(self, data: str) -> ParseResult:
        """
        Parse the incoming data stream.
        Returns a ParseResult containing the new state and any completed tools.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the state machine to its initial state."""
        pass

    def _ends_with_partial(self, buffer: str, tokens: List[Optional[str]]) -> int:
        max_partial = 0
        for token in tokens:
            if not token:
                continue
            for i in range(1, min(len(buffer), len(token)) + 1):
                if token.startswith(buffer[-i:]):
                    max_partial = max(max_partial, i)
        return max_partial
