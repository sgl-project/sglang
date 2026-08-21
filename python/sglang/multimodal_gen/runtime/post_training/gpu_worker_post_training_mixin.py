from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.distributed import (
    get_tp_rank,
    get_tp_world_size,
    get_world_group,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    iter_materialized_weights,
)
from sglang.multimodal_gen.runtime.post_training.tensor_update_checker import (
    TensorUpdateChecker,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    WeightsUpdater,
    get_updatable_modules,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
        UpdateWeightFromTensorCheckerReqInput,
        UpdateWeightFromTensorReqInput,
    )


def _normalize_gpu_uuid(uuid: str) -> str:
    # NVML prefixes uuids with "GPU-"/"MIG-"; torch device properties do not.
    return uuid.removeprefix("MIG-").removeprefix("GPU-").lower()


class GPUWorkerPostTrainingMixin:
    def update_weights_from_disk(
        self,
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
        self,
        req: UpdateWeightFromTensorReqInput,
    ) -> tuple[bool, str]:
        if not self.pipeline:
            return False, "Pipeline is not initialized"

        payload, error = self._select_own_gpu_payload(
            payloads=req.serialized_named_tensors,
            payload_gpu_uuids=req.payload_gpu_uuids,
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
            weight_update_mode=req.weight_update_mode,
            lora_alpha=req.lora_alpha,
            lora_rank=req.lora_rank,
        )

    def update_weights_from_tensor_checker(
        self,
        req: UpdateWeightFromTensorCheckerReqInput,
    ) -> tuple[bool, str]:
        if not self.pipeline:
            return False, "Pipeline is not initialized"

        checker = TensorUpdateChecker(self.pipeline)
        result = checker.verify_across_tp(
            target_module=req.target_module,
            expected_named_tensors_sha256=req.expected_named_tensors_sha256,
            tp_rank=get_tp_rank(),
            tp_world_size=get_tp_world_size(),
            tp_cpu_group=self.tp_cpu_group,
            tp_root_rank=self.tp_group.first_rank,
        )
        if self.sp_group.world_size == 1:
            return result

        import torch

        is_sp_root = self.sp_group.rank_in_group == 0
        gathered_results = [None] * self.sp_group.world_size if is_sp_root else None
        torch.distributed.gather_object(
            result,
            gathered_results,
            dst=self.sp_group.first_rank,
            group=self.sp_cpu_group,
        )

        final_result = None
        if is_sp_root:
            failures = [
                (rank, message)
                for rank, (success, message) in enumerate(gathered_results)
                if not success
            ]
            if failures:
                rank, message = failures[0]
                if len(failures) == 1:
                    final_result = (False, f"SP rank {rank}: {message}")
                else:
                    final_result = (
                        False,
                        f"{len(failures)} SP ranks failed update_weight_from_tensor_checker; "
                        f"first failure on rank {rank}: {message}",
                    )
            else:
                final_result = result

        final_result_holder = [final_result]
        torch.distributed.broadcast_object_list(
            final_result_holder,
            src=self.sp_group.first_rank,
            group=self.sp_cpu_group,
        )
        return final_result_holder[0]

    def get_weights_checksum(
        self, module_names: list[str] | None = None
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

    def _select_own_gpu_payload(
        self,
        payloads: list,
        payload_gpu_uuids: list[str] | None,
    ) -> tuple[object | None, str | None]:
        if not isinstance(payloads, list):
            return None, "serialized_named_tensors must be a list"
        if not payloads:
            return None, "serialized_named_tensors is required"

        if payload_gpu_uuids is None:
            # Unlabeled fallback: world rank equals tp rank for tp-only runs.
            if len(payloads) == 1:
                return payloads[0], None
            world_group = get_world_group()
            if len(payloads) != world_group.world_size:
                return None, (
                    f"serialized_named_tensors size must be 1 or world_size "
                    f"({world_group.world_size}), got {len(payloads)}"
                )
            return payloads[world_group.rank_in_group], None

        if len(payload_gpu_uuids) != len(payloads):
            return None, (
                f"payload_gpu_uuids needs one entry per payload, "
                f"got {len(payload_gpu_uuids)} for {len(payloads)} payloads"
            )

        own_gpu_uuid = _normalize_gpu_uuid(
            current_platform.get_device_uuid(self.local_rank)
        )
        normalized_uuids = [_normalize_gpu_uuid(uuid) for uuid in payload_gpu_uuids]
        if own_gpu_uuid not in normalized_uuids:
            return None, (
                f"no payload was exported from this worker's GPU {own_gpu_uuid}, "
                f"got {payload_gpu_uuids}"
            )
        return payloads[normalized_uuids.index(own_gpu_uuid)], None
