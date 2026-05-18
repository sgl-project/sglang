from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_PROMPT_COLUMN_CANDIDATES = [
    "prompt",
    "messages",
    "chat",
    "conversations",
    "text",
    "question",
    "instruction",
    "input",
    "query",
]
DAPO_MATH_17K_DEFAULT_PROMPT_COLUMN = "prompt"
CODEFORCES_REQUIRED_COLUMNS = [
    "description",
    "input_format",
    "output_format",
    "time_limit",
    "memory_limit",
]
CODEFORCES_OPTIONAL_COLUMNS = [
    "title",
    "note",
    "examples",
    "input_mode",
    "interaction_format",
]
CODEFORCES_LANGUAGE_ALIASES = {
    "python": "Python 3",
    "py": "Python 3",
    "cpp": "C++17",
    "c++": "C++17",
}


@dataclass
class DecoupledSpecEndpointConfig:
    """Bind/connect endpoint config for one decoupled-spec instance."""

    bind_endpoint: str
    connect_endpoints: list[str]
    rank: int


@dataclass
class DecoupledSpecTopology:
    """Endpoint topology and actor handles for a decoupled-spec run."""

    drafter_configs: list[DecoupledSpecEndpointConfig]
    verifier_configs: list[DecoupledSpecEndpointConfig]
    draft_actors: list[Any] | None = None


@dataclass
class PromptSample:
    row_index: int
    prompt: str
    prompt_input_ids: list[int]
    prompt_tokens: int


@dataclass
class ModeMetrics:
    mode: str
    generation_time_s: float
    total_generated_tokens: int
    output_throughput_tok_per_s: float
    per_request: list[dict[str, Any]]
    avg_spec_accept_length: float | None = None
    avg_spec_accept_rate: float | None = None
    avg_spec_valid_accept_rate: float | None = None
    avg_spec_valid_accept_rate_by_position: list[float | None] | None = None
    total_spec_valid_draft_token_num: int = 0
    total_spec_valid_accept_token_num: int = 0
    total_spec_valid_draft_token_num_by_position: list[int] | None = None
    total_spec_valid_accept_token_num_by_position: list[int] | None = None

