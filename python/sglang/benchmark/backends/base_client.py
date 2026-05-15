from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class RequestFuncInput:
    prompt: Union[str, List[str], List[Dict[str, str]]]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: str
    image_data: Optional[List[str]]
    extra_request_body: Dict[str, Any]
    timestamp: Optional[float] = None
    routing_key: Optional[str] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    text_chunks: List[str] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    start_time: float = 0.0

    @staticmethod
    def init_new(request_func_input: RequestFuncInput):
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        return output


class BaseBackendClient(ABC):
    def __init__(self, args: Any):
        self.args = args

    @abstractmethod
    async def request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Any = None,
    ) -> RequestFuncOutput:
        raise NotImplementedError
