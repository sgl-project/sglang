from __future__ import annotations

from typing import Any, List

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class SchedulerPostTrainingMixin:
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
