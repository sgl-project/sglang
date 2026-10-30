"""Boundary between decode orchestration and source-specific execution.

Only plain event records, cost queries and source paths cross this boundary;
no native tensors, schedulers, transports or private SGLang objects escape it.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class CostMapping(Protocol):
    """Adapter-owned signatures and disjoint billing for each invocation.

    Resolve must validate runtime geometry/backend against expected signatures.
    Distinct layer/rank/shape variants use distinct keys. Event translation must
    describe the same query used at submission, for critical-path attribution.
    """

    def expected_shapes(self) -> Mapping[str, Mapping[str, Any]]: ...

    def resolve(self, kind: str, metadata: Mapping[str, Any]) -> list[str]: ...

    def event_call(self, event: Mapping[str, Any]) -> tuple[str, dict]: ...


class CostProvider(Protocol):
    synthetic: bool
    queries: list
    config: Mapping[str, Any]

    def quote(self, kind: str, **metadata: Any) -> int: ...

    def components(self, kind: str, **metadata: Any) -> list: ...

    def event_components(self, event: Mapping[str, Any]) -> list: ...


@dataclass(frozen=True)
class ExecutionTrace:
    events: list[dict]
    source_files: list[str]
    total_devices: int


class ExecutionAdapter(Protocol):
    def cost_mapping(self, config: Mapping[str, Any]) -> CostMapping: ...

    def verify(self) -> list[str]: ...

    def execute(self, provider: CostProvider) -> ExecutionTrace: ...
