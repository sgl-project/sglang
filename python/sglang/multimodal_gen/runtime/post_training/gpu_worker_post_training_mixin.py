from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.loader.weights_updater import WeightsUpdater

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker


class GPUWorkerPostTrainingMixin:
    def update_weights_from_disk(
        self: GPUWorker,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
    ) -> tuple[bool, str]:
        if not self.pipeline:
            return False, "Pipeline is not initialized"

        updater = WeightsUpdater(self.pipeline)
        success, message = updater.update_weights_from_disk(
            model_path,
            flush_cache=flush_cache,
            target_modules=target_modules,
        )
        if success:
            self.server_args.model_path = model_path
            self.pipeline.model_path = model_path
        return success, message
