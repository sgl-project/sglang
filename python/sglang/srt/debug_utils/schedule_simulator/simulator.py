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
        log_level: int = 0,
    ):
        self.num_gpus = num_gpus
        self.router = router
        self.scheduler = scheduler
        self.recorders = recorders or []
        self.gpu_states: List[GPUState] = []
        self.log_level = log_level

    def run(self, requests: List[SimRequest]) -> Dict[str, Any]:
        self.gpu_states = [GPUState(gpu_id=i) for i in range(self.num_gpus)]
        incoming_requests = list(requests)
        step = 0

        while self._has_work(incoming_requests):
            self._route_requests(incoming_requests)
            incoming_requests.clear()

            self._schedule_all_gpus()
            self._execute_step()
            self._log_step(step)
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
                assert req in gpu.running_requests
                gpu.running_requests.remove(req)
                gpu.pending_requests.append(req)

            for req in decision.to_run:
                assert req in gpu.pending_requests
                gpu.pending_requests.remove(req)
                gpu.running_requests.append(req)

    def _execute_step(self) -> None:
        for gpu in self.gpu_states:
            finished = []
            for req in gpu.running_requests:
                # Prefill is instant, immediately transition to decode
                if req.stage == RequestStage.PREFILL:
                    req.stage = RequestStage.DECODE

                req.decoded_tokens += 1

                if req.is_finished():
                    finished.append(req)

            for req in finished:
                gpu.running_requests.remove(req)

    def _log_step(self, step: int) -> None:
        if self.log_level == 0:
            return

        parts = [f"step={step:<4}"]

        for gpu in self.gpu_states:
            run_count = len(gpu.running_requests)
            queue_count = len(gpu.pending_requests)

            if self.log_level == 1:
                parts.append(f"GPU{gpu.gpu_id}[R={run_count:<3} Q={queue_count:<3}]")
            else:
                run_ids = ",".join(r.request_id for r in gpu.running_requests[:5])
                if len(gpu.running_requests) > 5:
                    run_ids += f"...+{len(gpu.running_requests)-5}"
                queue_ids = ",".join(r.request_id for r in gpu.pending_requests[:3])
                if len(gpu.pending_requests) > 3:
                    queue_ids += f"...+{len(gpu.pending_requests)-3}"
                parts.append(f"GPU{gpu.gpu_id}[R={run_count}:{run_ids or'-'} Q={queue_count}:{queue_ids or'-'}]")

        print(" | ".join(parts))

    def _record_metrics(self, step: int) -> None:
        for recorder in self.recorders:
            recorder.on_step_end(step, self.gpu_states)

    def _get_summary(self) -> Dict[str, Any]:
        summary = {}
        for recorder in self.recorders:
            summary.update(recorder.get_summary())
        return summary
