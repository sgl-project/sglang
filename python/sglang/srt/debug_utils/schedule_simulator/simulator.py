from typing import Any, Dict, List, Optional

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState
from sglang.srt.debug_utils.schedule_simulator.metrics import MetricRecorder
from sglang.srt.debug_utils.schedule_simulator.request import RequestStage, SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy
from sglang.srt.debug_utils.schedule_simulator.schedulers.base import SchedulerPolicy


class Simulator:
    def __init__(
        self,
        num_gpus: int,
        router: RouterPolicy,
        scheduler: SchedulerPolicy,
        recorders: Optional[List[MetricRecorder]] = None,
    ):
        self.num_gpus = num_gpus
        self.router = router
        self.scheduler = scheduler
        self.recorders = recorders or []
        self.gpu_states: List[GPUState] = []

    def run(self, requests: List[SimRequest]) -> Dict[str, Any]:
        self.gpu_states = [GPUState(gpu_id=i) for i in range(self.num_gpus)]
        incoming_requests = list(requests)
        step = 0

        while self._has_work(incoming_requests):
            self._route_requests(incoming_requests)
            incoming_requests.clear()

            self._schedule_all_gpus()
            self._execute_step()
            self._record_metrics(step)

            step += 1

        return self._get_summary()

    def _has_work(self, incoming_requests: List[SimRequest]) -> bool:
        if incoming_requests:
            return True
        for gpu in self.gpu_states:
            if gpu.pending_requests or gpu.running_requests:
                return True
        return False

    def _route_requests(self, incoming_requests: List[SimRequest]) -> None:
        for req in incoming_requests:
            gpu_id = self.router.route(req, self.gpu_states)
            self.gpu_states[gpu_id].pending_requests.append(req)

    def _schedule_all_gpus(self) -> None:
        for gpu in self.gpu_states:
            decision = self.scheduler.schedule(gpu)

            for req in decision.to_preempt:
                if req in gpu.running_requests:
                    gpu.running_requests.remove(req)
                    gpu.pending_requests.append(req)

            for req in decision.to_run:
                if req in gpu.pending_requests:
                    gpu.pending_requests.remove(req)
                    gpu.running_requests.append(req)

    def _execute_step(self) -> None:
        for gpu in self.gpu_states:
            finished = []
            for req in gpu.running_requests:
                if req.stage == RequestStage.PREFILL:
                    req.stage = RequestStage.DECODE
                else:
                    req.decoded_tokens += 1

                if req.is_finished():
                    finished.append(req)

            for req in finished:
                gpu.running_requests.remove(req)

    def _record_metrics(self, step: int) -> None:
        for recorder in self.recorders:
            recorder.on_step_end(step, self.gpu_states)

    def _get_summary(self) -> Dict[str, Any]:
        summary = {}
        for recorder in self.recorders:
            summary.update(recorder.get_summary())
        return summary

