from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.distributed import get_tp_rank
from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum
from sglang.multimodal_gen.runtime.loader.weights_updater import WeightsUpdater
from sglang.multimodal_gen.runtime.loader.weights_updater import get_updatable_modules
from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
    iter_materialized_weights,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
        UpdateWeightFromTensorReqInput,
    )
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

    def update_weights_from_tensor(
        self: GPUWorker,
        req: UpdateWeightFromTensorReqInput,
    ) -> tuple[bool, str]:
        if not self.pipeline:
            return False, "Pipeline is not initialized"

        payloads = req.serialized_named_tensors
        if not payloads:
            return False, "serialized_named_tensors is required"

        tp_world_size = self.server_args.tp_size
        if len(payloads) not in (1, tp_world_size):
            return (
                False,
                "serialized_named_tensors size must be 1 or tp_size "
                f"({tp_world_size}), got {len(payloads)}",
            )

        payload_idx = get_tp_rank() if len(payloads) == tp_world_size else 0

        monkey_patch_torch_reductions()
        try:
            named_tensors = MultiprocessingSerializer.deserialize(payloads[payload_idx])
        except Exception as e:
            return False, f"Failed to deserialize serialized_named_tensors: {e}"

        updater = WeightsUpdater(self.pipeline)
        return updater.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=req.load_format,
            target_modules=req.target_modules,
        )

    def get_weights_checksum(
        self: GPUWorker, module_names: list[str] | None = None
    ) -> dict[str, str]:
        if not self.pipeline:
            return {"error": "Pipeline is not initialized"}

        all_modules = get_updatable_modules(self.pipeline)
        names = module_names if module_names is not None else list(all_modules.keys())

        checksums: dict[str, str] = {}
        for name in names:
            module = all_modules.get(name)
            if module is None:
                checksums[name] = "not_found"
                continue
            checksums[name] = compute_weights_checksum(
                iter_materialized_weights(module)
            )
        return checksums
