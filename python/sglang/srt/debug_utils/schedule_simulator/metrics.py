from abc import ABC, abstractmethod
from typing import Any, Dict, List

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class MetricRecorder(ABC):
    @abstractmethod
    def on_step_end(self, step: int, gpu_states: List[GPUState]) -> None: ...

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]: ...


class BatchSizeBalancednessRecorder(MetricRecorder):
    def __init__(self):
        self._history: List[float] = []

    def on_step_end(self, step: int, gpu_states: List[GPUState]) -> None:
        batch_sizes = [gpu.batch_size() for gpu in gpu_states]
        max_bs = max(batch_sizes) if batch_sizes else 0
        mean_bs = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        balancedness = mean_bs / max_bs if max_bs > 0 else 1.0
        self._history.append(balancedness)

    def get_summary(self) -> Dict[str, Any]:
        if not self._history:
            return {"batch_size_balancedness_mean": 0.0}
        return {
            "batch_size_balancedness_mean": sum(self._history) / len(self._history),
            "batch_size_balancedness_min": min(self._history),
            "batch_size_balancedness_max": max(self._history),
        }


class AttentionBalancednessRecorder(MetricRecorder):
    def __init__(self):
        self._history: List[float] = []

    def on_step_end(self, step: int, gpu_states: List[GPUState]) -> None:
        seq_lens = [gpu.total_seq_len() for gpu in gpu_states]
        max_seq = max(seq_lens) if seq_lens else 0
        mean_seq = sum(seq_lens) / len(seq_lens) if seq_lens else 0
        balancedness = mean_seq / max_seq if max_seq > 0 else 1.0
        self._history.append(balancedness)

    def get_summary(self) -> Dict[str, Any]:
        if not self._history:
            return {"attention_balancedness_mean": 0.0}
        return {
            "attention_balancedness_mean": sum(self._history) / len(self._history),
            "attention_balancedness_min": min(self._history),
            "attention_balancedness_max": max(self._history),
        }
