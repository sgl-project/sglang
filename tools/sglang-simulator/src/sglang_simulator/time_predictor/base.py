from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.utils import get_logger

logger = get_logger("sgl_simulator")


@dataclass
class ScheduleRequest:
    extend_length: int = 0
    past_kv_length: int = 0
    # Real scheduler phase of this request ("extend"/"decode"), attached by
    # the scheduler-entry snapshot for phase-aware predictors. Optional so
    # every existing caller keeps working unchanged.
    forward_mode: str | None = None


@dataclass
class ScheduleBatch:
    reqs: list[ScheduleRequest] = field(default_factory=list)
    # Real scheduler forward mode of this step ("extend"/"decode"/"mixed"),
    # attached by the scheduler-entry snapshot for phase-aware predictors.
    forward_mode: str | None = None

    def __repr__(self) -> str:
        return f"batch_size={len(self.reqs)},reqs={[(req.extend_length, req.past_kv_length) for req in self.reqs]}"

    def __eq__(self, batch: "ScheduleBatch"):
        if self.forward_mode != batch.forward_mode:
            return False
        if self.batch_size != batch.batch_size:
            return False

        req1, req2 = [], []
        for idx in range(self.batch_size):
            req1.append(
                (
                    self.reqs[idx].extend_length,
                    self.reqs[idx].past_kv_length,
                    self.reqs[idx].forward_mode or "",
                )
            )
            req2.append(
                (
                    batch.reqs[idx].extend_length,
                    batch.reqs[idx].past_kv_length,
                    batch.reqs[idx].forward_mode or "",
                )
            )

        return sorted(req1) == sorted(req2)

    def request_info(self) -> list[list[int, int]]:
        # The request information organized in the format `(input_len, past_kv_len)`
        return [[req.extend_length, req.past_kv_length] for req in self.reqs]

    @property
    def num_context_tokens(self) -> int:
        return sum(req.extend_length for req in self.reqs)

    @property
    def total_past_kv_length(self) -> int:
        return sum(req.past_kv_length for req in self.reqs)

    @property
    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def is_prefill(self) -> bool:
        return not self.is_decode()

    def is_decode(self) -> bool:
        if self.forward_mode is not None:
            return self.forward_mode == "decode"
        for req in self.reqs:
            if req.extend_length > 1:
                return False
        return True

    @property
    def num_ctx_requests(self) -> int:
        return self.batch_size if self.is_prefill() else 0

    @property
    def num_gen_requests(self) -> int:
        return self.batch_size if self.is_decode() else 0


class InferTimePredictor(ABC):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        *args,
        **kwargs,
    ):
        self.model: ModelInfo = model
        self.hw: AcceleratorInfo = hw
        self.config: SchedulerConfig = config

    @abstractmethod
    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        # Return the inference time in seconds. Return a negative value if an exception occurs (e.g., out of memory).
        pass

    def get_metrics(self) -> dict:
        """Return predictor-specific metrics for the current profile interval."""
        return {}

    def reset_metrics(self) -> None:
        """Reset predictor-specific metrics after a profile flush."""
        return None
