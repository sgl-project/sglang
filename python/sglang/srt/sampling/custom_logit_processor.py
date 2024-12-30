import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import dill
from torch import Tensor


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    required_args: List[str]

    @abstractmethod
    def __call__(
        self,
        logits: Tensor,
        batch_mask: List[bool],
        custom_params: Dict[str, List[Any]],
    ) -> Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    def to_str(self) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(self).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        data = json.loads(json_str)
        return dill.loads(bytes.fromhex(data["callable"]))
