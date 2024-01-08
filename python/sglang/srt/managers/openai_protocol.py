from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class CompletionRequest:
    prompt: Union[str, List[Any]]
    model: str = "default"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 16
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
