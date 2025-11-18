from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Protocol, Sequence, Tuple


@dataclass(slots=True)
class AutoTuneConfig:
    model_path: str
    tensor_parallel_size: int
    expert_parallel_size: int
    dtype: str
    per_channel_quant: bool
    disable_shared_experts_fusion: bool
    batch_size: Optional[int]
    seed: int
    results_dir: Path


@dataclass(slots=True)
class ComponentResult:
    name: str
    status: str
    details: Optional[str]
    output_files: Tuple[Path, ...]
    metadata: Dict[str, Any]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "details": self.details,
            "outputs": [str(path) for path in self.output_files],
            "metadata": self.metadata,
        }


class ComponentTuner(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def run(self, config: AutoTuneConfig) -> ComponentResult: ...


SummaryMapping = MutableMapping[str, Any]
ComponentList = Sequence[str]
