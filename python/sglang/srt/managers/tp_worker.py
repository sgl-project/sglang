# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import torch

from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
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
from sglang.srt.managers.worker_contract import (
    AttentionRequirements,
    KVCacheLayout,
    UnsupportedWorkerOperation,
    WorkerMemoryUsage,
    WorkerPoolState,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import get_exec, get_model, get_schedule, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

if TYPE_CHECKING:
    from sglang.srt.layers.pooler import EmbeddingPoolerOutput
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.io_struct import ElasticScaleUpdateReq
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

logger = logging.getLogger(__name__)


class BaseTpWorker(ABC):
    """Backend-neutral operations used by generic framework code."""

    @abstractmethod
    def forward_batch_generation(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init: Optional[bool] = None,
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ) -> GenerationBatchResult:
        ...

    def forward_batch_embedding(
        self, batch: ScheduleBatch
    ) -> Tuple[EmbeddingPoolerOutput, bool]:
        raise UnsupportedWorkerOperation(
            "embedding forward", type(self).__name__
        )

    def forward_batch_split_prefill(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        raise UnsupportedWorkerOperation("split prefill", type(self).__name__)

    @abstractmethod
    def get_worker_info(self):
        ...

    @abstractmethod
    def get_kv_cache_layout(self) -> KVCacheLayout:
        """Return scheduler-visible cache layout without exposing the runner."""
        ...

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.get_kv_cache_layout().sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        return self.get_kv_cache_layout().is_hybrid_swa

    def get_tokens_per_layer_info(self):
        layout = self.get_kv_cache_layout()
        return layout.full_tokens_per_layer, layout.swa_tokens_per_layer

    def get_pad_input_ids_func(self) -> Optional[Callable[..., Any]]:
        return None

    @abstractmethod
    def get_memory_pool_state(self) -> WorkerPoolState:
        """Return initialized scheduler bookkeeping pools."""
        ...

    def get_memory_pool(
        self,
    ) -> Tuple[
        Optional[ReqToTokenPool],
        Optional[BaseTokenToKVPoolAllocator],
    ]:
        state = self.get_memory_pool_state()
        return (
            state.req_to_token_pool,
            state.token_to_kv_pool_allocator,
        )

    @abstractmethod
    def ensure_memory_pool(self) -> WorkerPoolState:
        """Initialize scheduler pools if needed and return their resolved state."""
        ...

    @abstractmethod
    def alloc_memory_pool(
        self,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ) -> None:
        ...

    def init_attention_backends(self) -> None:
        """Initialize backend-specific attention state, if any."""
        return None

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True) -> None:
        """Initialize backend graph capture, if supported."""
        return None

    def on_radix_cache_initialized(self, cache: BasePrefixCache) -> None:
        """Notify worker-owned observers that the framework cache is ready."""
        return None

    def finalize_graph_capture(self) -> None:
        """Run backend-owned post-capture finalization."""
        return None

    def configure_hisparse_coordinator(
        self, forward_stream: Any
    ) -> Optional[HiSparseCoordinator]:
        """Configure and return the framework HiSparse coordinator, if present."""
        raise UnsupportedWorkerOperation("HiSparse", type(self).__name__)

    def get_attention_requirements(self) -> AttentionRequirements:
        """Return backend attention requirements used by the scheduler."""
        return AttentionRequirements(needs_cpu_seq_lens=True)

    def apply_war_barrier(
        self,
        schedule_stream: Any,
        forward_stream: Any,
        *,
        force_coarse: bool,
    ) -> None:
        """Wait until the previous forward has finished shared-buffer reads."""
        schedule_stream.wait_stream(forward_stream)

    def prepare_ngram_embedding(
        self,
        batch: Optional[ScheduleBatch],
        *,
        chunked_req: Optional[Req],
    ) -> Optional[ScheduleBatch]:
        """Attach backend-owned n-gram state needed by the next forward."""
        return batch

    def init_lora_overlap_loader(self) -> None:
        """Initialize backend-owned state for overlapping LoRA loads."""
        raise UnsupportedWorkerOperation(
            "overlapping LoRA loading", type(self).__name__
        )

    def try_overlap_load_lora(
        self,
        lora_id: Optional[str],
        running_loras: set[Optional[str]],
    ) -> bool:
        """Try to make one adapter available without exposing its manager."""
        raise UnsupportedWorkerOperation(
            "overlapping LoRA loading", type(self).__name__
        )

    def validate_lora_batch(self, lora_ids: set[Optional[str]]) -> bool:
        """Validate a candidate batch against backend LoRA capacity."""
        raise UnsupportedWorkerOperation("LoRA batching", type(self).__name__)

    def take_pending_elastic_scale_update(
        self,
    ) -> Optional[ElasticScaleUpdateReq]:
        """Atomically return and clear a pending elastic-scale update."""
        return None

    def get_memory_usage(self) -> WorkerMemoryUsage:
        """Return backend-reported weight and graph memory usage."""
        return WorkerMemoryUsage(weight_gb=None, graph_gb=None)

    def register_hicache_layer_transfer_counter(
        self, counter: LayerDoneCounter
    ) -> None:
        """Register optional HiCache transfer bookkeeping."""
        raise UnsupportedWorkerOperation("HiCache", type(self).__name__)

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        raise UnsupportedWorkerOperation("disk weight update", type(self).__name__)

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        raise UnsupportedWorkerOperation(
            "weight update group initialization", type(self).__name__
        )

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        raise UnsupportedWorkerOperation(
            "weight update group destruction", type(self).__name__
        )

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        raise UnsupportedWorkerOperation(
            "remote weight send group initialization", type(self).__name__
        )

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        raise UnsupportedWorkerOperation(
            "remote weight transfer", type(self).__name__
        )

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        raise UnsupportedWorkerOperation(
            "distributed weight update", type(self).__name__
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        raise UnsupportedWorkerOperation("tensor weight update", type(self).__name__)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        raise UnsupportedWorkerOperation("IPC weight update", type(self).__name__)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        raise UnsupportedWorkerOperation("weight export", type(self).__name__)

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        raise UnsupportedWorkerOperation("LoRA loading", type(self).__name__)

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        raise UnsupportedWorkerOperation("LoRA unloading", type(self).__name__)

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ):
        raise UnsupportedWorkerOperation(
            "LoRA tensor loading", type(self).__name__
        )


class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        is_multi_layer_eagle: bool = False,
        context_length: Optional[int] = None,
    ):
        # Parse args
        self.server_args = server_args
        self.ps = ps
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.is_draft_worker = is_draft_worker
        self.is_multi_layer_eagle = is_multi_layer_eagle
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        # Draft worker: target's resolved MemoryPoolConfig (forwarded to ModelRunner).
        self.memory_pool_config = memory_pool_config
        # Draft worker: target's effective context length; the draft runs at
        # absolute target positions. None keeps server_args.context_length.
        self.context_length = context_length

        # MTP model runners
        self.model_runner_list: List[ModelRunner] = []

        self._init_model_config()
        self._init_model_runner()

        if is_multi_layer_eagle:
            self._init_multi_layer_eagle_model_runners()

        self._init_dllm_algorithm()

        if server_args.skip_tokenizer_init or self.is_draft_worker:
            # A draft worker's tokenizer would only duplicate the target's:
            # tokenizer_path always points at the target model.
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                    model_name=server_args.model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Sync random seed across TP workers.
        # Elastic joiners cannot enter the launch-time WORLD broadcast.
        if server_args.is_ep_joiner:
            self.random_seed = server_args.random_seed
        else:
            self.random_seed = broadcast_pyobj(
                [server_args.random_seed],
                self.ps.tp_size * self.ps.pp_rank + self.ps.tp_rank,
                self.world_group.cpu_group,
                src=self.world_group.ranks[0],
            )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = server_args.speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None

    def alloc_memory_pool(
        self,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ) -> None:
        """Allocate KV cache pools only (no backends or cuda graphs)."""
        if req_to_token_pool is not None:
            self.req_to_token_pool = req_to_token_pool
            self.model_runner.req_to_token_pool = req_to_token_pool
        if token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.model_runner.alloc_memory_pool(memory_pool_config)
        for mr in self.model_runner_list[1:]:
            mr.req_to_token_pool = self.req_to_token_pool
            mr.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
            mr.alloc_memory_pool(memory_pool_config)

        # Validation
        assert self.model_runner.max_running_requests > 0, "max_running_request is zero"
        max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.effective_max_total_num_tokens - 1,
        )
        assert max_req_len > 0, "Memory pool size is too small"

    def init_attention_backends(self) -> None:
        """Initialize attention backends for all model runners."""
        self.model_runner.init_attention_backends()
        for mr in self.model_runner_list[1:]:
            mr.init_attention_backends()

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True) -> None:
        """Capture cuda graphs for all model runners."""
        self.model_runner.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )
        for mr in self.model_runner_list[1:]:
            mr.init_cuda_graphs(capture_decode_cuda_graph=capture_decode_cuda_graph)

    def _init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=(
                get_model().model_path
                if not self.is_draft_worker
                else get_spec().speculative_draft_model_path
            ),
            model_revision=(
                get_model().revision
                if not self.is_draft_worker
                else get_spec().speculative_draft_model_revision
            ),
            is_draft_model=self.is_draft_worker,
            context_length=self.context_length,
        )

    def _init_model_runner(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )

    def _init_multi_layer_eagle_model_runners(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self.model_runner_list.append(self.model_runner)
        for i in range(1, get_spec().speculative_num_steps):
            self.model_runner_list.append(
                ModelRunner(
                    model_config=self.model_config,
                    mem_fraction_static=get_schedule().mem_fraction_static,
                    gpu_id=self.gpu_id,
                    ps=self.ps,
                    nccl_port=self.nccl_port,
                    server_args=self.server_args,
                    is_draft_worker=self.is_draft_worker,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    memory_pool_config=self.memory_pool_config,
                    draft_model_idx=i,
                )
            )

    def _init_dllm_algorithm(self):
        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        if get_exec().dllm.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)
        else:
            self.dllm_algorithm = None

    @property
    def model_runner(self) -> ModelRunner:
        return self._model_runner

    @property
    def war_fastpath_runner(self):
        # Compatibility surface for scheduler/spec workers. This is deliberately
        # not part of BaseTpWorker.
        return self._model_runner

    def get_kv_cache_layout(self) -> KVCacheLayout:
        runner = self._model_runner
        is_hybrid_swa = runner.is_hybrid_swa
        return KVCacheLayout(
            is_hybrid_swa=is_hybrid_swa,
            prefill_aware_swa=getattr(runner, "prefill_aware_swa", False),
            sliding_window_size=getattr(runner, "sliding_window_size", None),
            full_tokens_per_layer=(
                getattr(runner, "full_max_total_num_tokens", None)
                if is_hybrid_swa
                else None
            ),
            swa_tokens_per_layer=(
                getattr(runner, "swa_max_total_num_tokens", None)
                if is_hybrid_swa
                else None
            ),
        )

    def get_pad_input_ids_func(self) -> Optional[Callable[..., Any]]:
        return getattr(self._model_runner.model, "pad_input_ids", None)

    def get_memory_pool_state(self) -> WorkerPoolState:
        runner = self._model_runner
        if (
            runner.req_to_token_pool is None
            or runner.token_to_kv_pool_allocator is None
        ):
            raise RuntimeError("Worker memory pool has not been initialized.")
        return WorkerPoolState(
            config=runner.memory_pool_config,
            req_to_token_pool=runner.req_to_token_pool,
            token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        )

    def get_memory_pool(
        self,
    ) -> Tuple[
        Optional[ReqToTokenPool],
        Optional[BaseTokenToKVPoolAllocator],
    ]:
        # Compatibility for speculative workers that inspect the target before
        # scheduler-owned pool initialization.
        return (
            self._model_runner.req_to_token_pool,
            self._model_runner.token_to_kv_pool_allocator,
        )

    def ensure_memory_pool(self) -> WorkerPoolState:
        runner = self._model_runner
        if (
            runner.req_to_token_pool is None
            or runner.token_to_kv_pool_allocator is None
        ):
            self.alloc_memory_pool()
        return self.get_memory_pool_state()

    def on_radix_cache_initialized(self, cache: BasePrefixCache) -> None:
        if (
            manager := getattr(self._model_runner, "canary_manager", None)
        ) is not None:
            manager.attach_radix_cache(cache)

    def finalize_graph_capture(self) -> None:
        runner = self._model_runner
        if runner.token_to_kv_pool.post_capture_active:
            runner.post_capture_resize_kv_pool()

    def configure_hisparse_coordinator(
        self, forward_stream: Any
    ) -> Optional[HiSparseCoordinator]:
        coordinator = self._model_runner.hisparse_coordinator
        if coordinator is not None:
            coordinator.set_decode_producer_stream(forward_stream)
        return coordinator

    def get_attention_requirements(self) -> AttentionRequirements:
        backend = getattr(self._model_runner, "attn_backend", None)
        return AttentionRequirements(
            needs_cpu_seq_lens=(
                True
                if backend is None
                else getattr(backend, "needs_cpu_seq_lens", True)
            )
        )

    def apply_war_barrier(
        self,
        schedule_stream: Any,
        forward_stream: Any,
        *,
        force_coarse: bool,
    ) -> None:
        runner = self._model_runner
        event = runner.war_fastpath_read_done_event
        runner.war_fastpath_read_done_event = None
        if event is not None and not force_coarse:
            schedule_stream.wait_event(event)
        else:
            schedule_stream.wait_stream(forward_stream)

    def prepare_ngram_embedding(
        self,
        batch: Optional[ScheduleBatch],
        *,
        chunked_req: Optional[Req],
    ) -> Optional[ScheduleBatch]:
        return self._model_runner.ngram_embedding_manager.prepare_for_forward(
            batch, chunked_req=chunked_req
        )

    def init_lora_overlap_loader(self) -> None:
        from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader

        self._lora_overlap_loader = LoRAOverlapLoader(
            self._model_runner.lora_manager
        )

    def try_overlap_load_lora(
        self,
        lora_id: Optional[str],
        running_loras: set[Optional[str]],
    ) -> bool:
        loader = getattr(self, "_lora_overlap_loader", None)
        if loader is None:
            raise RuntimeError("LoRA overlap loading has not been initialized.")
        return loader.try_overlap_load_lora(lora_id, running_loras)

    def validate_lora_batch(self, lora_ids: set[Optional[str]]) -> bool:
        return self._model_runner.lora_manager.validate_lora_batch(lora_ids)

    def take_pending_elastic_scale_update(
        self,
    ) -> Optional[ElasticScaleUpdateReq]:
        runner = self._model_runner
        pending = runner._pending_elastic_scale_update
        runner._pending_elastic_scale_update = None
        return pending

    def get_memory_usage(self) -> WorkerMemoryUsage:
        return WorkerMemoryUsage(
            weight_gb=getattr(self._model_runner, "weight_load_mem_usage", None),
            graph_gb=getattr(self._model_runner, "graph_mem_usage", None),
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = (
            self._model_runner.weight_updater.update_weights_from_disk(
                recv_req.model_path,
                recv_req.load_format,
                recapture_cuda_graph=recv_req.recapture_cuda_graph,
            )
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = (
            self._model_runner.weight_updater.init_weights_update_group(
                recv_req.master_address,
                recv_req.master_port,
                recv_req.rank_offset,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = (
            self._model_runner.weight_updater.destroy_weights_update_group(
                recv_req.group_name,
            )
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        success, message = (
            self._model_runner.weight_exporter.init_weights_send_group_for_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_rank,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        success, message = (
            self._model_runner.weight_exporter.send_weights_to_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_name,
            )
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = (
            self._model_runner.weight_updater.update_weights_from_distributed(
                recv_req.names,
                recv_req.dtypes,
                recv_req.shapes,
                recv_req.group_name,
                recv_req.load_format,
            )
        )
        return success, message

    def _deserialize_own_rank(self, serialized_named_tensors):
        """Each rank deserializes only its own payload (index ps.tp_rank);
        deserializing another rank's copy would break producer-side CUDA-IPC
        refcounting."""
        monkey_patch_torch_reductions()
        return MultiprocessingSerializer.deserialize(
            serialized_named_tensors[self.ps.tp_rank]
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = (
            self._model_runner.weight_updater.update_weights_from_tensor(
                named_tensors=self._deserialize_own_rank(
                    recv_req.serialized_named_tensors
                ),
                load_format=recv_req.load_format,
            )
        )
        return success, message

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self._model_runner.weight_updater.update_weights_from_ipc(
            recv_req
        )
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self._model_runner.weight_exporter.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self._model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self._model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ):
        # The LoRA code handles TP sharding internally using slice_lora_a_weights
        # and slice_lora_b_weights methods (see lora/layers.py and mem_pool.py).
        data = self._deserialize_own_rank(recv_req.serialized_named_tensors)
        if recv_req.load_format == "flattened_bucket":
            bucket = FlattenedTensorBucket(
                flattened_tensor=data["flattened_tensor"],
                metadata=data["metadata"],
            )
            tensors = dict(bucket.reconstruct_tensors())
        else:
            tensors = data
        if recv_req.expected_checksums is not None:
            import hashlib

            exp = recv_req.expected_checksums
            mismatch, missing = [], []
            for name, want in exp.items():
                if name not in tensors:
                    missing.append(name)
                    continue
                got = hashlib.sha256(
                    tensors[name]
                    .detach()
                    .cpu()
                    .contiguous()
                    .flatten()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                ).hexdigest()
                if got != want:
                    mismatch.append(name)
            extra = [n for n in tensors if n not in exp]
            if mismatch or missing or extra:
                raise RuntimeError(
                    f"[LORA-CHECK] rank{self.ps.tp_rank} adapter sync MISMATCH of {len(exp)} expected: "
                    f"{len(mismatch)} value-diff {mismatch[:5]}, {len(missing)} missing {missing[:5]}, "
                    f"{len(extra)} extra {extra[:5]}"
                )
            logger.info(
                f"[LORA-CHECK] rank{self.ps.tp_rank} adapter sync OK: {len(exp)}/{len(exp)} tensors match (sha256)"
            )
        result = self._model_runner.load_lora_adapter_from_tensors(
            recv_req.to_ref(),
            tensors,
            recv_req.config_dict,
            recv_req.added_tokens_config,
        )
        return result

    def forward_batch_embedding(
        self, batch: ScheduleBatch
    ) -> Tuple[EmbeddingPoolerOutput, bool]:
        forward_batch = ForwardBatch.init_new(
            batch,
            self._model_runner,
            return_hidden_states_before_norm=False,
        )
        output = self._model_runner.forward(forward_batch)
        return output.logits_output, output.can_run_graph

    def register_hicache_layer_transfer_counter(
        self, counter: LayerDoneCounter
    ) -> None:
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def register_hisparse_coordinator(self, coordinator):
        self.model_runner.hisparse_coordinator = coordinator

    def get_worker_info(self):
        max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.effective_max_total_num_tokens - 1,
        )
        return (
            self.model_runner.max_total_num_tokens,
            get_schedule().max_prefill_tokens,
            self.model_runner.max_running_requests,
            get_schedule().max_queued_requests,
            max_req_len,
            max_req_len - 5,
            self.random_seed,
            self.device,
            self.model_runner.forward_stream,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def is_dllm(self):
        return self.dllm_algorithm is not None

    def _forward_batch_generation_dllm(
        self,
        forward_batch: ForwardBatch,
        batch: Optional[ScheduleBatch] = None,
    ) -> GenerationBatchResult:
        algo_states = None
        if self.dllm_algorithm.fdfo and batch is not None:
            algo_states = [req.dllm_algo_state for req in batch.reqs]

        (
            logits_output,
            next_token_ids,
            accept_length_per_req_cpu,
            dllm_algo_state,
            can_run_cuda_graph,
        ) = self.dllm_algorithm.run(self.model_runner, forward_batch, algo_states)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            dllm_algo_state=dllm_algo_state,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_batch_generation(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ) -> GenerationBatchResult:
        # Get forward batch from schedule batch
        if batch is not None:
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(batch.hicache_consumer_index)

            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                capture_hidden_mode=capture_hidden_mode,
                return_hidden_states_before_norm=False,
            )
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            assert forward_batch is not None
            assert (
                capture_hidden_mode is None
            ), "capture_hidden_mode override requires a ScheduleBatch input"

        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        if self.is_dllm():
            return self._forward_batch_generation_dllm(forward_batch, batch)

        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
                routed_experts_output=out.routed_experts_output,
                indexer_topk_output=out.indexer_topk_output,
            )

            if is_verify:
                # Skip sampling; spec_v2 worker fires its own publish post-verify.
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and forward_batch.sampling_info.grammars is not None
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not forward_batch.is_prefill_only:
                # For normal requests, sample the next token ids.
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(forward_batch.seq_lens),
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
                if (
                    forward_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    self.model_runner.compute_logprobs_only(
                        logits_output, forward_batch
                    )

            return batch_result
        else:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(
                batch,
                self.model_runner,
                return_hidden_states_before_norm=False,
            )
            batch.split_forward_batch = forward_batch

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output:
            next_token_ids = self.model_runner.sample(
                logits_output, batch.split_forward_batch
            )
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
