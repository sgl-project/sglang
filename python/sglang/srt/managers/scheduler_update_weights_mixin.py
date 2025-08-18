import logging
from typing import Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            flush_cache_success = self.flush_cache()
            assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        success, message = self.tp_worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE]

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            # if self.server_args.attention_backend == "ascend":
            #     self.stashed_kv_dynamic_state = []
            #     self.k_storage_size = self.tp_worker.worker.model_runner.token_to_kv_pool.k_buffer.untyped_storage().size()
            #     self.v_storage_size = self.tp_worker.worker.model_runner.token_to_kv_pool.v_buffer.untyped_storage().size()
            #     self.tp_worker.worker.model_runner.token_to_kv_pool.k_buffer.untyped_storage().resize_(0)
            #     self.tp_worker.worker.model_runner.token_to_kv_pool.v_buffer.untyped_storage().resize_(0)

            #     self.stashed_kv_dynamic_state = [(self.tp_worker.worker.model_runner.token_to_kv_pool.k_buffer, self.k_storage_size), (self.tp_worker.worker.model_runner.token_to_kv_pool.v_buffer, self.v_storage_size)]
            # else:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()
            print(f"release kv 111111111111", flush=True)
        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)

            # if self.server_args.attention_backend == "ascend":
            #     self.stashed_model_dynamic_state = []
            #     for name, param in self.tp_worker.worker.model_runner.model.named_parameters():
            #         storage_size = param.untyped_storage().size()
            #         param.untyped_storage().resize_(0)

            #         self.stashed_model_dynamic_state.append((name, param, storage_size))
            # else:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)
            print(f"release model 111111111111", flush=True)

        torch.npu.empty_cache()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE]

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            # if self.server_args.attention_backend == "ascend":
            #     for name, param, storage_size in self.stashed_model_dynamic_state:
            #         param.untyped_storage().resize_(storage_size)
            # else:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)

            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            # if self.server_args.attention_backend == "ascend":
            #     for param, storage_size in self.stashed_kv_dynamic_state:
            #         param.untyped_storage().resize_(storage_size)
            # else:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
        torch.npu.empty_cache()

        return ResumeMemoryOccupationReqOutput()

    def save_remote_model(self, params):
        url = params["url"]

        worker = self.tp_worker.worker

        worker.model_runner.save_remote_model(url)

    def save_sharded_model(self, params):
        worker = self.tp_worker.worker

        worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
