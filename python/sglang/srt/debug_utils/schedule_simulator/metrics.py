from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class MetricRecorder(ABC):
    @abstractmethod
    def on_step_end(self, step: int, gpu_states: List[GPUState]) -> None: ...

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]: ...


class BalancednessRecorder(MetricRecorder):
    def __init__(self, name: str, value_fn: Callable[[GPUState], float]):
        self._name = name
        self._value_fn = value_fn
        self._history: List[float] = []

    def on_step_end(self, step: int, gpu_states: List[GPUState]) -> None:
        values = [self._value_fn(gpu) for gpu in gpu_states]
        max_val = max(values) if values else 0
        mean_val = sum(values) / len(values) if values else 0
        balancedness = mean_val / max_val if max_val > 0 else 1.0
        self._history.append(balancedness)

    def get_summary(self) -> Dict[str, Any]:
        if not self._history:
            return {f"{self._name}_mean": 0.0}
        return {
            f"{self._name}_mean": sum(self._history) / len(self._history),
            f"{self._name}_min": min(self._history),
            f"{self._name}_max": max(self._history),
        }


def BatchSizeBalancednessRecorder() -> BalancednessRecorder:
    return BalancednessRecorder("batch_size_balancedness", lambda gpu: gpu.batch_size())


def AttentionComputeBalancednessRecorder() -> BalancednessRecorder:
    return BalancednessRecorder(
        "attention_compute_balancedness", lambda gpu: gpu.total_attention_compute()
    )


class AvgBatchSizeRecorder(MetricRecorder):
    def __init__(self):
        self._total_running = 0
        self._num_records = 0

    def on_step_end(self, step: int, gpu_states: List[GPUState]) -> None:
        for gpu in gpu_states:
            self._total_running += gpu.batch_size()
            self._num_records += 1

    def get_summary(self) -> Dict[str, Any]:
        avg = self._total_running / self._num_records if self._num_records else 0.0
        return {"avg_batch_size": avg}
