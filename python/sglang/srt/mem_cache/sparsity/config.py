"""Configuration for HBM-resident post-hoc KV sparsity."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KVSparsityConfig:
    """Resolved configuration shared by policy, controller, and adaptor."""

    policy: str = "streaming_llm"
    backend: str = "fa3"
    page_size: int = 1
    min_sparse_tokens: int = 2048
    start_layer: int = 0
    end_layer: int = -1
    policy_config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (("policy", self.policy), ("backend", self.backend)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.page_size, int) or isinstance(self.page_size, bool):
            raise TypeError("page_size must be an integer")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if (
            not isinstance(self.min_sparse_tokens, int)
            or isinstance(self.min_sparse_tokens, bool)
            or self.min_sparse_tokens < 0
        ):
            raise ValueError("min_sparse_tokens must be a non-negative integer")
        if not isinstance(self.start_layer, int) or isinstance(self.start_layer, bool):
            raise TypeError("start_layer must be an integer")
        if not isinstance(self.end_layer, int) or isinstance(self.end_layer, bool):
            raise TypeError("end_layer must be an integer")
        if self.start_layer < 0:
            raise ValueError("start_layer must be non-negative")
        if self.end_layer != -1 and self.end_layer <= self.start_layer:
            raise ValueError("end_layer must be -1 or greater than start_layer")
        if not isinstance(self.policy_config, dict):
            raise TypeError("policy_config must be a JSON object")
