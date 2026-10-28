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
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import torch

from sglang.srt.beam_search.logits_capture import capture_pre_sample_logits
from sglang.srt.configs.draft_parallel import DraftParallelLayout
from sglang.srt.configs.load_config import _DEFAULT_LOAD_GROUP, LoadGroup
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    SendWeightsToRemoteInstanceReqInput,
    UnloadLoRAAdapterReqInput,
)
from sglang.srt.managers.overlap_utils import RelayPayload, resolve_forward_inputs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.page_interleave import page_interleave_shard_size
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.model_executor.graph_memory_usage import (
    merge_graph_memory_usage,
    merge_graph_time_usage,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.forward_batch_view import (
    derive_batch,
    forward_sampling_info,
    worker_forward_batch,
)
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.scheduler_components.pending_steps import PendingStep
    from sglang.srt.model_executor.encoder_swa_replay import FoldedExtend
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

logger = logging.getLogger(__name__)


class BaseTpWorker(ABC):
    # The relay the scheduler hands its model worker: a forward resolves its
    # input tokens from it and commits its sampled tokens to it. None for a
    # worker another worker drives (a speculative worker's target).
    future_map: Optional[FutureMap] = None
    # Whether a forward also publishes its rows' new lengths (overlap).
    publishes_new_seq_lens: bool = False

    def attach_future_map(self, future_map: FutureMap, *, enable_overlap: bool) -> None:
        """Make this worker the scheduler's model worker over `future_map`."""
        self.future_map = future_map
        self.publishes_new_seq_lens = enable_overlap

    def forward_batch_generation(
        self,
        step: ScheduleBatch,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        pending_grammar: Optional[Iterable[PendingStep]] = None,
    ) -> GenerationBatchResult:
        """Run one scheduled step, with its device commit. `step` is read
        only: the forward runs on a view of it. A step without speculation runs
        its grammar in its own host commit, so `pending_grammar` is not read."""
        forward_batch = self.forward_input(step)
        result = self.forward_model(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        return self.commit_forward(step, result)

    def forward_input(self, step: ScheduleBatch) -> ScheduleBatch:
        """The batch this worker forwards for `step`: a view of it with its
        input ids resolved from FutureMap."""
        forward_batch = worker_forward_batch(step)
        resolve_forward_inputs(forward_batch, self.future_map)
        hisparse_coordinator = self.model_runner.hisparse_coordinator
        if hisparse_coordinator is not None and step.forward_mode.is_decode():
            # Device commit of a HiSparse decode step's buffer maps, at its
            # forward entry on the forward stream: after the previous forward
            # read them.
            hisparse_coordinator.map_last_loc_to_buffer(
                step.seq_lens,
                step.out_cache_loc,
                step.req_pool_indices,
                step.seq_lens_cpu,
                step.req_pool_indices_cpu,
            )
        return forward_batch

    def commit_forward(
        self, step: ScheduleBatch, result: GenerationBatchResult
    ) -> GenerationBatchResult:
        """The device commit of a forward without speculation: publish the
        rows' new lengths (under overlap) and stash the sampled tokens for the
        next step's input; a delayed sample stashes them once it runs."""
        if self.publishes_new_seq_lens:
            self.future_map.publish(step.req_pool_indices, step.seq_lens + 1)
        if result.delay_sample_func is None:
            self._stash_sampled_tokens(step, result)
            return result
        sample = result.delay_sample_func

        def sample_and_stash():
            sampled = sample()
            self._stash_sampled_tokens(step, result)
            return sampled

        result.delay_sample_func = sample_and_stash
        return result

    def _stash_sampled_tokens(
        self, step: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        """Stash a step's sampled tokens for the next step's
        resolve_forward_inputs."""
        if not result.has_sampled_token_ids:
            return
        payload = RelayPayload(bonus_tokens=result.next_token_ids)
        future_indices = step.req_pool_indices
        if step.beam_tail is not None:
            # The worker sliced the tail off before sampling, so sampled tokens
            # cover only the reqs-aligned rows; the coordinator relays the rest.
            future_indices = future_indices[: step.beam_tail.num_base_rows]
        self.future_map.stash(future_indices, payload)

    @abstractmethod
    def forward_model(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> GenerationBatchResult:
        """The model forward of a forward batch, without a device commit."""

    def prepare_decode_step(self, step: ScheduleBatch) -> None:
        """The worker's part of building a decode step; none without
        speculation."""

    @property
    @abstractmethod
    def model_runner(self) -> ModelRunner:
        pass

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """No-op mirror of BaseSpecWorker's hook: PP+spec non-last stages
        process relayed spec results through a plain worker."""

    @property
    def last_shared_read_runner(self):
        # The runner that runs the step's LAST shared-buffer-reading phase --
        # it owns the read-done event the scheduler's WAR barrier waits on.
        # For a plain worker that's its own runner.
        return self.model_runner

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        return self.model_runner.is_hybrid_swa

    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    @property
    def graph_memory_usage(self) -> dict[str, float]:
        runners = self.model_runner_list or [self.model_runner]
        return merge_graph_memory_usage(
            *(runner.graph_memory_usage for runner in runners)
        )

    @property
    def graph_time_usage(self) -> dict[str, float]:
        runners = self.model_runner_list or [self.model_runner]
        return merge_graph_time_usage(*(runner.graph_time_usage for runner in runners))

    @property
    def weight_load_time(self) -> float:
        runners = self.model_runner_list or [self.model_runner]
        return sum(runner.weight_load_time for runner in runners)

    @property
    def preloaded_weights_bytes(self) -> int:
        runners = self.model_runner_list or [self.model_runner]
        return sum(runner.preloaded_weights_bytes for runner in runners)

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_memory_pool(self) -> Tuple[ReqToTokenPool, BaseTokenToKVPoolAllocator]:
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.weight_updater.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = (
            self.model_runner.weight_updater.destroy_weights_update_group(
                recv_req.group_name,
            )
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        success, message = (
            self.model_runner.weight_exporter.init_weights_send_group_for_remote_instance(
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
            self.model_runner.weight_exporter.send_weights_to_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_name,
            )
        )
        return success, message

    def deserialize_own_rank(self, serialized_named_tensors):
        """Each rank deserializes only its own payload (index tp_rank);
        deserializing another rank's copy would break producer-side CUDA-IPC
        refcounting."""
        monkey_patch_torch_reductions()
        return MultiprocessingSerializer.deserialize(
            serialized_named_tensors[get_parallel().tp_rank]
        )

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.weight_exporter.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ):
        # The LoRA code handles TP sharding internally using slice_lora_a_weights
        # and slice_lora_b_weights methods (see lora/layers.py and mem_pool.py).
        data = self.deserialize_own_rank(recv_req.serialized_named_tensors)
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
                    f"[LORA-CHECK] rank{get_parallel().tp_rank} adapter sync MISMATCH of {len(exp)} expected: "
                    f"{len(mismatch)} value-diff {mismatch[:5]}, {len(missing)} missing {missing[:5]}, "
                    f"{len(extra)} extra {extra[:5]}"
                )
            logger.info(
                f"[LORA-CHECK] rank{get_parallel().tp_rank} adapter sync OK: {len(exp)}/{len(exp)} tensors match (sha256)"
            )
        result = self.model_runner.load_lora_adapter_from_tensors(
            recv_req.to_ref(),
            tensors,
            recv_req.config_dict,
            recv_req.added_tokens_config,
        )
        return result

    def forward_batch_embedding(self, batch: ScheduleBatch):
        forward_batch = ForwardBatch.init_new(
            batch,
            self.model_runner,
            return_hidden_states_before_norm=False,
        )
        output = self.model_runner.forward(forward_batch)
        return output.logits_output, output.can_run_graph


class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        is_multi_layer_eagle: bool = False,
        context_length: Optional[int] = None,
        draft_attention_backend: Optional[str] = None,
        random_seed: Optional[int] = None,
        load_group: LoadGroup = _DEFAULT_LOAD_GROUP,
    ):
        # Parse args
        self.server_args = server_args
        self.load_group = load_group
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
        # Draft worker: the attention backend the algorithm resolved for it.
        self.draft_attention_backend = draft_attention_backend

        # MTP model runners
        self.model_runner_list: List[ModelRunner] = []

        self._init_model_config()
        self._init_model_runner()

        if is_multi_layer_eagle:
            self._init_multi_layer_eagle_model_runners()

        if get_serving().skip_tokenizer_init or self.is_draft_worker:
            # A draft worker's tokenizer would only duplicate the target's:
            # tokenizer_path always points at the target model.
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                    model_name=get_model().model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    get_serving().tokenizer_path,
                    tokenizer_mode=get_serving().tokenizer_mode,
                    trust_remote_code=get_model().trust_remote_code,
                    revision=get_model().revision,
                    tokenizer_backend=get_serving().tokenizer_backend,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_parallel().pp_group
        self.world_group = get_parallel().world_group

        # Sync random seed across TP workers.
        # Elastic joiners and last-stage-only draft workers cannot enter the WORLD
        # broadcast, so they reuse the target's already-broadcast seed.
        if random_seed is not None:
            self.random_seed = random_seed
        elif get_exec().moe.is_ep_joiner:
            self.random_seed = get_device().random_seed
        elif (
            envs.SGLANG_ENABLE_PP_SPEC.get()
            and is_draft_worker
            and get_parallel().pp_size > 1
        ):
            # PP+spec: the draft worker exists only on the last PP stage, so a
            # world-group broadcast here would deadlock (first-stage ranks never
            # join). Sync within the stage's TP group instead — that is exactly
            # the set of ranks holding a draft worker. The draft worker is
            # constructed with pp_rank=0, so derive the caller's global rank
            # from the TP group rather than tp_size * pp_rank + tp_rank.
            tp_group = get_parallel().tp_group
            self.random_seed = broadcast_pyobj(
                [get_device().random_seed],
                tp_group.ranks[tp_group.rank_in_group],
                tp_group.cpu_group,
                src=tp_group.ranks[0],
            )[0]
        else:
            self.random_seed = broadcast_pyobj(
                [get_device().random_seed],
                get_parallel().tp_size * get_parallel().pp_rank
                + get_parallel().tp_rank,
                self.world_group.cpu_group,
                src=self.world_group.ranks[0],
            )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not get_schedule().disable_overlap_schedule
        self.enable_spec = get_spec().speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None

    def alloc_memory_pool(
        self,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        *,
        draft_parallel_layout: DraftParallelLayout | None = None,
    ):
        """Allocate KV cache pools only (no backends or cuda graphs)."""
        if req_to_token_pool is not None:
            self.req_to_token_pool = req_to_token_pool
            self.model_runner.req_to_token_pool = req_to_token_pool
        if token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.model_runner.alloc_memory_pool(
            memory_pool_config, draft_parallel_layout=draft_parallel_layout
        )
        for mr in self.model_runner_list[1:]:
            mr.req_to_token_pool = self.req_to_token_pool
            mr.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
            mr.alloc_memory_pool(
                memory_pool_config, draft_parallel_layout=draft_parallel_layout
            )

        # Validation
        assert self.model_runner.max_running_requests > 0, "max_running_request is zero"
        max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.effective_logical_max_total_num_tokens - 1,
        )
        assert max_req_len > 0, "Memory pool size is too small"

    def init_attention_backends(self):
        """Initialize attention backends for all model runners."""
        self.model_runner.init_attention_backends()
        for mr in self.model_runner_list[1:]:
            mr.init_attention_backends()

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        """Capture cuda graphs for all model runners."""
        self.model_runner.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )
        for mr in self.model_runner_list[1:]:
            mr.init_cuda_graphs(capture_decode_cuda_graph=capture_decode_cuda_graph)

    def ensure_decode_cuda_graphs(self, capture_bs: Optional[List[int]] = None):
        """Idempotently capture decode cuda graphs for all model runners (used
        for the on-flip capture during a runtime PD role switch)."""
        self.model_runner.ensure_decode_cuda_graphs(capture_bs)
        for mr in self.model_runner_list[1:]:
            mr.ensure_decode_cuda_graphs(capture_bs)

    def get_decode_cuda_graph_bs(self) -> List[int]:
        """Decode bs captured as CUDA graphs (empty on a not-yet-flipped prefill,
        or on a runner that never allocates a KV pool, e.g. the MLX stub)."""
        return list(getattr(self.model_runner, "decode_cuda_graph_capture_bs", []))

    def start_startup_weight_load(self) -> None:
        """Start deferred checkpoint prefetching for all model runners."""
        self.model_runner.start_startup_weight_load()
        for mr in self.model_runner_list[1:]:
            mr.start_startup_weight_load()

    def finalize_startup_weight_load(self) -> None:
        """Commit deferred startup weights for all model runners."""
        self.model_runner.finalize_startup_weight_load()
        for mr in self.model_runner_list[1:]:
            mr.finalize_startup_weight_load()

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
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_attention_backend=self.draft_attention_backend,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
            load_group=self.load_group,
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
                    nccl_port=self.nccl_port,
                    server_args=self.server_args,
                    is_draft_worker=self.is_draft_worker,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    memory_pool_config=self.memory_pool_config,
                    draft_attention_backend=self.draft_attention_backend,
                    draft_model_idx=i,
                    load_group=self.load_group,
                )
            )

    @property
    def model_runner(self) -> ModelRunner:
        return self._model_runner

    def weight_update_runners(self) -> List[Tuple[str, ModelRunner]]:
        """(role, runner) pairs weight ops apply to; the target worker owns one."""
        return [("target", self._model_runner)]

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def register_hisparse_coordinator(self, coordinator):
        self.model_runner.hisparse_coordinator = coordinator

    def get_worker_info(self):
        # The runner already reports logical DCP capacity.
        kv_capacity = (
            self.model_runner.effective_logical_max_total_num_tokens
            * page_interleave_shard_size(self.model_runner.token_to_kv_pool_allocator)
        )
        # Not an error: a deployment that never sends max-length requests may
        # want a small pool. Checked here because this is the final served value.
        if kv_capacity < self.model_config.context_len:
            logger.warning(
                "KV pool holds %d tokens, fewer than --context-length %d. One "
                "full-length request cannot fit, so long requests will be "
                "refused or truncated at serving time. Lower --context-length, "
                "raise --mem-fraction-static, or free memory taken before the "
                "pool is sized.",
                kv_capacity,
                self.model_config.context_len,
            )
        max_req_len = min(
            self.model_config.context_len - 1,
            kv_capacity - 1,
        )
        max_req_input_len = max_req_len - 5
        return (
            self.model_runner.logical_max_total_num_tokens,
            get_schedule().max_prefill_tokens,
            self.model_runner.max_running_requests,
            get_schedule().max_queued_requests,
            max_req_len,
            max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.forward_stream,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def _maybe_finalize_elastic_cuda_graph_scale(self) -> None:
        if self.model_runner._elastic_cuda_graph_enabled():
            self.model_runner.maybe_join_ep_ranks()

    def init_forward_batch(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ) -> Tuple[Optional[ScheduleBatch], ForwardBatch]:
        """The forward batch of a step (and the step as the forward reads it),
        or the forward batch given."""
        batch, forward_batch, _ = self._init_forward_batch(
            batch,
            forward_batch,
            skip_attn_backend_init,
            capture_hidden_mode=capture_hidden_mode,
        )
        return batch, forward_batch

    def _init_forward_batch(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch],
        skip_attn_backend_init: Optional[bool],
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode],
    ) -> Tuple[Optional[ScheduleBatch], ForwardBatch, Optional[FoldedExtend]]:
        """`init_forward_batch`, plus the encoder-SWA replay rows folded into
        the forward, which the forward drops from its logits."""
        # Get forward batch from schedule batch
        folded = None
        if batch is not None:
            if self.enable_overlap:
                # The batch is read-only: the forward fills a sampling info of
                # its own, with the penalties accumulated (the next step's
                # schedule updates the batch's while this forward runs).
                batch = derive_batch(
                    batch, sampling_info=forward_sampling_info(batch.sampling_info)
                )
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(batch.hicache_consumer_index)

            if get_exec().features.enable_encoder_swa_bounded_replay:
                from sglang.srt.model_executor.encoder_swa_replay import (
                    apply_folded_extend,
                    fold_encoder_swa_replay,
                    run_encoder_swa_replay,
                )

                if self.model_runner.attn_backend.folds_encoder_swa_replay:
                    # Hits extend from their replay start, rebuilding the window in-pass.
                    folded = fold_encoder_swa_replay(self, batch)
                else:
                    # Replay reads restored main/indexer KV before the normal extend.
                    run_encoder_swa_replay(self, batch)

            forward_batch = ForwardBatch.init_new(
                batch if folded is None else folded.batch,
                self.model_runner,
                capture_hidden_mode=capture_hidden_mode,
                return_hidden_states_before_norm=False,
            )
            if folded is not None:
                apply_folded_extend(folded, forward_batch)
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            assert forward_batch is not None
            assert capture_hidden_mode is None, (
                "capture_hidden_mode override requires a ScheduleBatch input"
            )

        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        return batch, forward_batch, folded

    def prepare_decode_step(self, step: ScheduleBatch) -> None:
        """A speculative server's stage before the last under pipeline
        parallelism runs this worker without a draft worker: its decode step
        still takes the EAGLE preparation; the launch builds its verify input
        from the relay."""
        if not self.enable_spec:
            return
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode
        from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

        spec_prepare_for_decode(step)
        eagle_prepare_for_decode(step)

    def forward_model(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
        return_kv_loc_plan: bool = False,
    ) -> GenerationBatchResult:
        batch, forward_batch, folded = self._init_forward_batch(
            batch,
            forward_batch,
            skip_attn_backend_init,
            capture_hidden_mode=capture_hidden_mode,
        )

        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            if folded is not None:
                from sglang.srt.model_executor.encoder_swa_replay import (
                    drop_folded_rows,
                )

                tail = self.model_runner.attn_backend.tail_forward_metadata
                drop_folded_rows(
                    logits_output=logits_output,
                    folded=folded,
                    tail_lens=(
                        None
                        if tail is None
                        else tail.late_layer_tail.extend_seq_lens_cpu
                    ),
                )
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
                routed_experts_output=out.routed_experts_output,
                indexer_topk_output=out.indexer_topk_output,
                kv_loc_plan=(
                    None
                    if not return_kv_loc_plan
                    # A folded forward also wrote the replay rows; the caller's
                    # plan covers the batch it scheduled.
                    else forward_batch.kv_loc_plan
                    if folded is None
                    else self.model_runner.kv_index_translator.own_plan(batch)
                ),
            )
            if (
                batch is not None
                and (observer := self.model_runner.forward_observer) is not None
            ):
                batch_result.forward_auxiliary_output = observer.after_forward(
                    batch,
                    forward_batch,
                    logits_output,
                    can_run_graph=can_run_cuda_graph,
                )

            capture_pre_sample_logits(batch, forward_batch, logits_output)

            if is_verify:
                # Skip sampling; spec_v2 worker fires its own publish post-verify.
                return batch_result

            # Delay sampling only for normal generation requests.
            # Keep the existing grammar behavior unchanged.
            if (
                self.enable_overlap
                and not self.enable_spec
                and (
                    forward_batch.sampling_info.grammars is not None
                    or (
                        envs.SGLANG_ENABLE_DELAY_SAMPLE.get()
                        and not forward_batch.is_prefill_only
                    )
                )
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    self._maybe_finalize_elastic_cuda_graph_scale()
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

            self._maybe_finalize_elastic_cuda_graph_scale()
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
        self._maybe_finalize_elastic_cuda_graph_scale()
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
