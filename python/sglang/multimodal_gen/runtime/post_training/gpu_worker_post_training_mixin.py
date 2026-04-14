from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.distributed import get_tp_rank, get_tp_world_size
from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum
from sglang.multimodal_gen.runtime.loader.weights_updater import WeightsUpdater
from sglang.multimodal_gen.runtime.utils.update_weight_from_tensor_checker import (
    UpdateWeightFromTensorChecker,
)
from sglang.multimodal_gen.runtime.loader.weights_updater import get_updatable_modules
from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
    iter_materialized_weights,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
        UpdateWeightFromTensorCheckerReqInput,
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

        payload, error = self._select_rank_scoped_payload(
            payloads=req.serialized_named_tensors,
            field_name="serialized_named_tensors",
        )
        if error is not None:
            return False, error

        monkey_patch_torch_reductions()
        try:
            named_tensors = MultiprocessingSerializer.deserialize(payload)
        except Exception as e:
            return False, f"Failed to deserialize serialized_named_tensors: {e}"

        updater = WeightsUpdater(self.pipeline)
        return updater.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=req.load_format,
            target_modules=req.target_modules,
        )

    def update_weights_from_tensor_checker(
        self: GPUWorker,
        req: UpdateWeightFromTensorCheckerReqInput,
    ) -> tuple[bool, str]:
        if not self.pipeline:
            return False, "Pipeline is not initialized"

        expected_transformer_sha256, error = self._select_rank_scoped_payload(
            payloads=req.expected_transformer_sha256,
            field_name="expected_transformer_sha256",
        )
        if error is not None:
            return False, error

        checker = UpdateWeightFromTensorChecker(self.pipeline)
        return checker.verify_across_tp(
            expected_transformer_sha256=expected_transformer_sha256,
            tp_rank=get_tp_rank(),
            tp_world_size=get_tp_world_size(),
            tp_cpu_group=self.tp_cpu_group,
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

    def _select_rank_scoped_payload(
        self: GPUWorker,
        payloads: list,
        field_name: str,
    ) -> tuple[object | None, str | None]:
        if not isinstance(payloads, list):
            return None, f"{field_name} must be a list"
        if not payloads:
            return None, f"{field_name} is required"

        tp_world_size = get_tp_world_size()
        if len(payloads) not in (1, tp_world_size):
            return (
                None,
                f"{field_name} size must be 1 or tp_size ({tp_world_size}), "
                f"got {len(payloads)}",
            )

        payload_idx = get_tp_rank() if len(payloads) == tp_world_size else 0
        return payloads[payload_idx], None
