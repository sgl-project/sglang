import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional

import dill
import torch


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = json.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    def to_str(self) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(self).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)
