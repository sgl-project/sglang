from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from sglang.srt.managers.tp_worker import BaseTpWorker

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        GetWeightsByNameReqInput,
        InitWeightsSendGroupForRemoteInstanceReqInput,
        InitWeightsUpdateGroupReqInput,
        LoadLoRAAdapterFromTensorsReqInput,
        LoadLoRAAdapterReqInput,
        SendWeightsToRemoteInstanceReqInput,
        UnloadLoRAAdapterReqInput,
        UpdateWeightFromDiskReqInput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromIPCReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner


class BaseDraftWorker(ABC):
    @abstractmethod
    def draft(self):
        pass

    @abstractmethod
    def draft_extend(self):
        pass


class BaseSpecWorker(BaseTpWorker):
    """Base class for speculative workers that wrap a target TP worker.

    Implements the full BaseTpWorker API by delegating to target_worker,
    while providing spec-specific hooks for adaptive/overlap scheduling.
    """

    @property
    @abstractmethod
    def target_worker(self) -> "TpModelWorker":
        """The target (verifier) model worker."""
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> BaseDraftWorker:
        """The draft (generator) model worker."""
        pass

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """Attn backends touched by spec_v2 forward; OR-ed by decide_needs_cpu_seq_lens.
        Default returns target only; subclasses extend with draft backends."""
        return (self.target_worker.model_runner.attn_backend,)

    @abstractmethod
    def clear_cache_pool(self):
        """Clear cache pools. Default no-op since pools are typically shared."""
        pass

    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        """
        pass

    # --- BaseTpWorker delegation ---

    @property
    def model_runner(self) -> "ModelRunner":
        """Return target model runner for unified TP worker interface."""
        return self.target_worker.model_runner

    def forward_batch_generation(self, *args, **kwargs) -> "GenerationBatchResult":
        """Must be overridden by subclass with spec-specific logic."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward_batch_generation"
        )

    def forward_batch_embedding(self, batch: "ScheduleBatch"):
        """Delegate to target worker for embedding inference."""
        return self.target_worker.forward_batch_embedding(batch)

    def get_worker_info(self):
        """Delegate to target worker."""
        return self.target_worker.get_worker_info()

    def get_pad_input_ids_func(self):
        """Delegate to target worker."""
        return self.target_worker.get_pad_input_ids_func()

    def get_memory_pool(self):
        """Delegate to target worker."""
        return self.target_worker.get_memory_pool()

    def update_weights_from_disk(self, recv_req: "UpdateWeightFromDiskReqInput"):
        """Delegate to target worker."""
        return self.target_worker.update_weights_from_disk(recv_req)

    def init_weights_update_group(self, recv_req: "InitWeightsUpdateGroupReqInput"):
        """Delegate to target worker."""
        return self.target_worker.init_weights_update_group(recv_req)

    def destroy_weights_update_group(
        self, recv_req: "DestroyWeightsUpdateGroupReqInput"
    ):
        """Delegate to target worker."""
        return self.target_worker.destroy_weights_update_group(recv_req)

    def init_weights_send_group_for_remote_instance(
        self, recv_req: "InitWeightsSendGroupForRemoteInstanceReqInput"
    ):
        """Delegate to target worker."""
        return self.target_worker.init_weights_send_group_for_remote_instance(recv_req)

    def send_weights_to_remote_instance(
        self, recv_req: "SendWeightsToRemoteInstanceReqInput"
    ):
        """Delegate to target worker."""
        return self.target_worker.send_weights_to_remote_instance(recv_req)

    def update_weights_from_distributed(
        self, recv_req: "UpdateWeightsFromDistributedReqInput"
    ):
        """Delegate to target worker."""
        return self.target_worker.update_weights_from_distributed(recv_req)

    def update_weights_from_tensor(self, recv_req: "UpdateWeightsFromTensorReqInput"):
        """Delegate to target worker."""
        return self.target_worker.update_weights_from_tensor(recv_req)

    def update_weights_from_ipc(self, recv_req: "UpdateWeightsFromIPCReqInput"):
        """Delegate to target worker."""
        return self.target_worker.update_weights_from_ipc(recv_req)

    def get_weights_by_name(self, recv_req: "GetWeightsByNameReqInput"):
        """Delegate to target worker."""
        return self.target_worker.get_weights_by_name(recv_req)

    def load_lora_adapter(self, recv_req: "LoadLoRAAdapterReqInput"):
        """Delegate to target worker."""
        return self.target_worker.load_lora_adapter(recv_req)

    def unload_lora_adapter(self, recv_req: "UnloadLoRAAdapterReqInput"):
        """Delegate to target worker."""
        return self.target_worker.unload_lora_adapter(recv_req)

    def load_lora_adapter_from_tensors(
        self, recv_req: "LoadLoRAAdapterFromTensorsReqInput"
    ):
        """Delegate to target worker."""
        return self.target_worker.load_lora_adapter_from_tensors(recv_req)

    def check_weights(self, action):
        """Delegate to target worker."""
        return self.target_worker.check_weights(action)

    def save_remote_model(self, params):
        """Save target and draft models when a draft URL is provided."""
        self.target_worker.save_remote_model(params)

        draft_runners = self._iter_draft_model_runners()
        if not draft_runners:
            return

        draft_url = params.get("draft_url", None)
        assert (
            draft_url is not None
        ), "draft_url must be provided when draft model is enabled"

        for draft_runner in draft_runners:
            draft_runner.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        """Delegate to target worker."""
        return self.target_worker.save_sharded_model(params)

    def _iter_draft_model_runners(self):
        draft_worker = self.draft_worker
        draft_runner_list = getattr(draft_worker, "draft_runner_list", None)
        if isinstance(draft_runner_list, (list, tuple)):
            return tuple(draft_runner_list)

        draft_runner = getattr(draft_worker, "draft_runner", None)
        if draft_runner is not None:
            return (draft_runner,)

        draft_model_runner = getattr(draft_worker, "model_runner", None)
        if draft_model_runner is not None:
            return (draft_model_runner,)

        return ()

    # --- Common properties for rank/device/capacity info ---

    @property
    def model_config(self) -> "ModelConfig":
        """Return target model config."""
        return self.target_worker.model_config

    @property
    def tokenizer(self):
        """Return target tokenizer."""
        return self.target_worker.tokenizer

    @property
    def processor(self):
        """Return target processor."""
        return self.target_worker.processor

    @property
    def device(self):
        """Return target device."""
        return self.target_worker.device

    @property
    def tp_rank(self) -> int:
        """Return target TP rank."""
        return self.target_worker.tp_rank

    @property
    def pp_rank(self) -> int:
        """Return target PP rank."""
        return self.target_worker.pp_rank

    @property
    def dp_rank(self) -> Optional[int]:
        """Return target DP rank."""
        return self.target_worker.dp_rank

    @property
    def gpu_id(self) -> int:
        """Return target GPU ID."""
        return self.target_worker.gpu_id

    @property
    def max_total_num_tokens(self) -> int:
        """Return target max total tokens."""
        return self.target_worker.max_total_num_tokens

    @property
    def max_prefill_tokens(self) -> int:
        """Return target max prefill tokens."""
        return self.target_worker.max_prefill_tokens

    @property
    def max_running_requests(self) -> int:
        """Return target max running requests."""
        return self.target_worker.max_running_requests

    @property
    def max_queued_requests(self) -> Optional[int]:
        """Return target max queued requests."""
        return self.target_worker.max_queued_requests

    @property
    def max_req_len(self) -> int:
        """Return target max request length."""
        return self.target_worker.max_req_len

    @property
    def max_req_input_len(self) -> int:
        """Return target max request input length."""
        return self.target_worker.max_req_input_len

    @property
    def random_seed(self) -> int:
        """Return target random seed."""
        return self.target_worker.random_seed

    @property
    def sliding_window_size(self) -> Optional[int]:
        """Return target sliding window size."""
        return self.target_worker.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        """Return target is_hybrid_swa status."""
        return self.target_worker.is_hybrid_swa

    def get_tokens_per_layer_info(self):
        """Delegate to target worker."""
        return self.target_worker.get_tokens_per_layer_info()
