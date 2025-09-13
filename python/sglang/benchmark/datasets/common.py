from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerBase


@dataclass
class DatasetRow:
    """Represents a single data point for a benchmark request."""

    prompt: str
    prompt_len: int
    output_len: int
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None


@dataclass
class RequestFuncInput:
    """Input parameters for a request function."""

    prompt: Any
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: Optional[str]
    image_data: Optional[List[str]]
    extra_request_body: Dict[str, Any]
    timestamp: Optional[float] = None


@dataclass
class RequestFuncOutput:
    """Output results from a request function."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""

    @staticmethod
    def from_input(request_input: RequestFuncInput):
        """Initializes an output object from an input object."""
        return RequestFuncOutput(prompt_len=request_input.prompt_len)


class BaseDatasetLoader(ABC):

    def __init__(self, args: Namespace, tokenizer: PreTrainedTokenizerBase):
        self.args = args
        self.tokenizer = tokenizer

    @abstractmethod
    def load(self) -> List[DatasetRow]:
        pass
