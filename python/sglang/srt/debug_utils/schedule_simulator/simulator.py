from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState, StepRecord
from sglang.srt.debug_utils.schedule_simulator.metrics import MetricRecorder
from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy
from sglang.srt.debug_utils.schedule_simulator.schedulers.base import SchedulerPolicy

_LOG_ID_LIMIT = 5


def _format_ids(requests: List[SimRequest]) -> str:
    if not requests:
        return "-"
    ids = ",".join(r.request_id for r in requests[:_LOG_ID_LIMIT])
    if len(requests) > _LOG_ID_LIMIT:
        ids += f"...+{len(requests) - _LOG_ID_LIMIT}"
    return ids


@dataclass
class SimulationResult:
    step_records: List[StepRecord]
    summary: Dict[str, Any]


class Simulator:
    def __init__(
        self,
        num_gpus: int,
        router: RouterPolicy,
        scheduler: SchedulerPolicy,
        recorders: Optional[List[MetricRecorder]] = None,
        log_level: int = 0,
        max_total_tokens: int = 100000,
    ):
        self.num_gpus = num_gpus
        self.router = router
        self.scheduler = scheduler
        self.recorders = recorders or []
        self.log_level = log_level
        self.max_total_tokens = max_total_tokens
        self.gpu_states: List[GPUState] = []
        self.step = 0

    def run(self, requests: List[SimRequest]) -> SimulationResult:
        self.gpu_states = [
            GPUState(gpu_id=i, max_total_tokens=self.max_total_tokens)
            for i in range(self.num_gpus)
        ]
        self.step = 0
        step_records: List[StepRecord] = []
        incoming_requests = list(requests)

        while self._has_work(incoming_requests):
            self._route_requests(incoming_requests)
            incoming_requests.clear()
            self._schedule_all_gpus()
            self._execute_step()
            step_records.extend(
                gpu.get_step_record(self.step) for gpu in self.gpu_states
            )
            self._log_step()
            self._record_metrics()
            self.step += 1

        return SimulationResult(step_records=step_records, summary=self._get_summary())

    def _has_work(self, incoming_requests: List[SimRequest]) -> bool:
        return bool(incoming_requests) or any(
            gpu.pending_requests or gpu.running_requests for gpu in self.gpu_states
        )

    def _route_requests(self, incoming_requests: List[SimRequest]) -> None:
        for req in incoming_requests:
            gpu_id = self.router.route(req, self.gpu_states)
            self.gpu_states[gpu_id].pending_requests.append(req)

    def _schedule_all_gpus(self) -> None:
        for gpu in self.gpu_states:
            self.scheduler.schedule(gpu)
            assert gpu.is_valid(), (
                f"GPU{gpu.gpu_id} invalid after scheduling "
                f"({gpu.total_seq_len()=}, {gpu.max_total_tokens=})"
            )

    def _execute_step(self) -> None:
        for gpu in self.gpu_states:
            gpu.execute_step()

    def _log_step(self) -> None:
        if self.log_level == 0:
            return
        parts = [f"step={self.step:<4}"]
        for gpu in self.gpu_states:
            r, q = len(gpu.running_requests), len(gpu.pending_requests)
            if self.log_level == 1:
                parts.append(f"GPU{gpu.gpu_id}[R={r:<3} Q={q:<3}]")
            else:
                run_ids = _format_ids(gpu.running_requests)
                queue_ids = _format_ids(gpu.pending_requests)
                parts.append(f"GPU{gpu.gpu_id}[R={r}:{run_ids} Q={q}:{queue_ids}]")
        print(" | ".join(parts))

    def _record_metrics(self) -> None:
        for recorder in self.recorders:
            recorder.on_step_end(self.step, self.gpu_states)

    def _get_summary(self) -> Dict[str, Any]:
        summary = {}
        for recorder in self.recorders:
            summary.update(recorder.get_summary())
        return summary
