from __future__ import annotations

from typing import Any, Callable, List

import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed import get_world_group
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class SchedulerPostTrainingMixin:
    def _run_on_all_ranks(
        self, worker_op: Callable[[Any], tuple[bool, str]], req: Any
    ) -> OutputBatch:
        """Every rank joins the all-gather, even one whose op raised, and the
        request succeeds only if every rank succeeded."""
        try:
            own_result = worker_op(req)
        except Exception as e:
            own_result = False, f"{type(e).__name__}: {e}"

        world = get_world_group()
        all_results = [None] * world.world_size
        dist.all_gather_object(all_results, own_result, group=world.cpu_group)

        failures = [
            f"Rank {rank}: {message}"
            for rank, (success, message) in enumerate(all_results)
            if not success
        ]
        if failures:
            return OutputBatch(
                output={"success": False, "message": failures[0]}, error=failures[0]
            )
        _, message = own_result
        return OutputBatch(output={"success": True, "message": message})

    def _handle_init_weights_update_group(self, reqs: List[Any]) -> OutputBatch:
        return self._run_on_all_ranks(self.worker.init_weights_update_group, reqs[0])

    def _handle_update_weights_from_distributed(self, reqs: List[Any]) -> OutputBatch:
        return self._run_on_all_ranks(
            self.worker.update_weights_from_distributed, reqs[0]
        )

    def _handle_destroy_weights_update_group(self, reqs: List[Any]) -> OutputBatch:
        return self._run_on_all_ranks(self.worker.destroy_weights_update_group, reqs[0])

    def _handle_update_weights_from_disk(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        success, message = self.worker.update_weights_from_disk(
            model_path=req.model_path,
            flush_cache=req.flush_cache,
            target_modules=req.target_modules,
        )
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )

    def _handle_update_weights_from_tensor(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        success, message = self.worker.update_weights_from_tensor(req)
        if self.server_args.tp_size > 1:
            import torch

            torch.distributed.barrier(group=self.worker.tp_cpu_group)
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )

    def _handle_update_weights_from_tensor_checker(
        self, reqs: List[Any]
    ) -> OutputBatch:
        req = reqs[0]
        success, message = self.worker.update_weights_from_tensor_checker(req)
        return OutputBatch(
            output={"success": success, "message": message},
            error=None if success else message,
        )

    def _handle_get_weights_checksum(self, reqs: List[Any]) -> OutputBatch:
        req = reqs[0]
        checksums = self.worker.get_weights_checksum(module_names=req.module_names)
        return OutputBatch(output=checksums)
