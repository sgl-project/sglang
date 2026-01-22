from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from sglang.srt.function_call.core_types import ToolCallItem


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


class ParseResult(BaseModel):
    state: UniversalToolParserState
    completed_tools: List[Dict[str, Any]]
    remaining: str
    streaming_calls: List[ToolCallItem] = []
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


class ToolParserStateMachine(ABC):
    @abstractmethod
    def parse(self, data: str) -> ParseResult:
        """
        Parse the incoming data stream.
        Returns a ParseResult containing the new state and any completed tools.
        """
        pass
