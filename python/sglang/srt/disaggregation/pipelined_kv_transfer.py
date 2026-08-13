from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.disaggregation.base.conn import BaseKVManager, BufferType, StateType
from sglang.srt.disaggregation.utils import (
    MetadataBuffers,
    TransferBackend,
    get_dsv4_c128_state_indices,
    is_dsv4_c128_online_enabled,
)
from sglang.srt.environ import envs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True)
class LayerPipelinedTransferPlan:
    group_size: int
    layer_ids_by_group: List[List[int]]
    req_page_indices_list: List[np.ndarray]
    req_state_indices_list: List[List]


@dataclass(kw_only=True, frozen=True, slots=True)
class LayerPipelinedKVTransferAdapter:
    """Translate scheduler layer groups into layer-pipelined transfers."""

    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    kv_manager: BaseKVManager
    server_args: ServerArgs
    ps: ParallelState
    model_config: ModelConfig
    tp_worker: BaseTpWorker
    model_worker: object
    req_to_token_pool: ReqToTokenPool
    metadata_buffers: MetadataBuffers
    transfer_backend: TransferBackend
    sliding_window_size: Optional[int] = None
    enable_staging: bool = False
    _config_incompatibility_reason: Optional[str] = field(
        init=False, repr=False, compare=False
    )
    _draft_layer_ids: List[int] = field(init=False, repr=False, compare=False)
    _skip_dsa_state_layer_ids: Optional[set[int]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_draft_layer_ids",
            sorted(self.kv_manager.kv_args.draft_buffer_handles),
        )
        object.__setattr__(
            self,
            "_skip_dsa_state_layer_ids",
            (
                {
                    layer_id
                    for layer_id in range(self.model_config.num_hidden_layers)
                    if dsa_layer_skips_topk(self.model_config.hf_text_config, layer_id)
                }
                if is_deepseek_dsa(self.model_config.hf_text_config)
                else None
            ),
        )
        reason = self._get_config_incompatibility_reason()
        object.__setattr__(self, "_config_incompatibility_reason", reason)
        if reason is not None:
            logger.info("Layer-pipelined KV transfer is disabled: %s", reason)
        else:
            logger.info(
                "Layer-pipelined KV transfer is enabled (individual batches may "
                "still fall back to the regular path)"
            )

    def plan_batch(self, batch: ScheduleBatch) -> Optional[LayerPipelinedTransferPlan]:
        """Return a layer-pipelined transfer plan, or None for the normal path."""
        if self._config_incompatibility_reason is not None:
            return None
        if not self._is_batch_compatible(batch):
            return None

        group_size = (
            envs.SGLANG_PIPELINE_GROUP_SIZE.get()
            if envs.SGLANG_PIPELINE_GROUP_SIZE.is_set()
            else self._get_adaptive_group_size(batch)
        )
        if group_size <= 0:
            return None
        return self._build_transfer_plan(
            batch,
            num_layers=self.model_config.num_hidden_layers,
            group_size=group_size,
        )

    def run_batch(
        self,
        batch: ScheduleBatch,
        plan: LayerPipelinedTransferPlan,
        defer_sample: bool = False,
        on_publish=None,
    ) -> GenerationBatchResult:
        """Run split prefill and enqueue each completed layer group."""
        forward_batch = self.model_worker.forward_batch_generation_split_prefill_init(
            batch
        )
        for req, kv_indices, state_indices in zip(
            batch.reqs, plan.req_page_indices_list, plan.req_state_indices_list
        ):
            if req.inflight_middle_chunks > 0:
                continue
            req.disagg_kv_sender.prepare_layer_pipelined_transfer(
                kv_indices,
                state_indices,
                final_state_indices=getattr(req, "non_pipelined_state_indices", None),
                skip_dsa_state_layer_ids=self._skip_dsa_state_layer_ids,
            )

        logits_output = None
        for group_id, layer_ids in enumerate(plan.layer_ids_by_group):
            logits, ready_event = (
                self.model_worker.forward_batch_generation_split_prefill(
                    forward_batch, forward_count=len(layer_ids)
                )
            )
            if logits is not None:
                logits_output = logits

            for req in batch.reqs:
                if req.inflight_middle_chunks > 0:
                    continue
                req.disagg_kv_sender.send_layers(
                    layer_ids,
                    ready_event=ready_event,
                )
                if group_id == len(plan.layer_ids_by_group) - 1:
                    req.ready_for_pipelined_transfer_finalize = True
                    req.start_send_idx = min(
                        req.extend_range.end, len(req.origin_input_ids)
                    )

        assert logits_output is not None, (
            "forward_batch_generation_split_prefill should return logits after "
            "the last layer"
        )
        result = self.model_worker.forward_batch_generation_split_prefill_finalize(
            batch,
            logits_output,
            forward_batch,
            defer_sample=defer_sample,
            on_publish=on_publish,
        )

        if self._draft_layer_ids:
            # Draft prefill is not split, so all draft buffers become ready together.
            draft_ready_event = torch.cuda.Event()
            draft_ready_event.record()
            for req in batch.reqs:
                if req.inflight_middle_chunks > 0:
                    continue
                req.disagg_kv_sender.send_layers(
                    self._draft_layer_ids,
                    is_draft=True,
                    ready_event=draft_ready_event,
                )

        return result

    def finalize_pipelined_transfer(self, req: Req) -> bool:
        """Send final metadata for a request handled by the pipelined path."""
        if not getattr(req, "ready_for_pipelined_transfer_finalize", False):
            return False

        self.metadata_buffers.set_buf(req)
        metadata_ready_event = torch.cuda.Event()
        metadata_ready_event.record()
        req.disagg_kv_sender.send_final_metadata(metadata_ready_event)
        return True

    def _get_config_incompatibility_reason(self) -> Optional[str]:
        if not envs.SGLANG_ENABLE_PIPELINED_KV_TRANSFER.get():
            return "SGLANG_ENABLE_PIPELINED_KV_TRANSFER is not enabled"
        spec_algo_str = self.server_args.speculative_algorithm
        if spec_algo_str:
            spec_algo = SpeculativeAlgorithm.from_string(spec_algo_str)
            if not (spec_algo.is_eagle() or spec_algo.is_dspark()):
                return (
                    f"speculative algorithm {spec_algo_str!r} is unsupported "
                    "(only EAGLE family and DSPARK support pipelined KV transfer)"
                )
        if (
            self.ps.attn_dp_size > 1
            and self.server_args.cuda_graph_config.prefill.backend != Backend.DISABLED
        ):
            return (
                "prefill CUDA graph is incompatible with layer-pipelined KV transfer "
                "when DP attention is enabled; set "
                "--cuda-graph-backend-prefill disabled to use layer-pipelined KV "
                "transfer with DP attention"
            )
        if self.transfer_backend not in (
            TransferBackend.MOONCAKE,
            TransferBackend.FAKE,
        ):
            return f"transfer backend {self.transfer_backend} is unsupported"
        if getattr(self.server_args, "enable_hisparse", False):
            return "prefill-side HiSparse is unsupported"
        if self.ps.pp_size > 1:
            return "pipeline parallelism is unsupported"
        if self.ps.attn_cp_size > 1:
            return "context parallelism is unsupported"
        if getattr(self.ps, "dcp_size", 1) > 1:
            return "decode context parallelism is unsupported"
        if not callable(
            getattr(self.tp_worker.model_runner.model, "forward_split_prefill", None)
        ):
            return "the model does not implement forward_split_prefill"
        split_methods = (
            "forward_batch_generation_split_prefill_init",
            "forward_batch_generation_split_prefill",
            "forward_batch_generation_split_prefill_finalize",
        )
        if any(
            not callable(getattr(self.model_worker, method, None))
            for method in split_methods
        ):
            return "the model worker does not support layer-pipelined split prefill"
        if getattr(self.server_args, "expert_distribution_recorder_mode", None):
            return "expert distribution recording is unsupported"
        if getattr(self.server_args, "enable_return_routed_experts", False):
            return "returning routed experts is unsupported"
        if getattr(self.server_args, "enable_return_indexer_topk", False):
            return "returning indexer top-k results is unsupported"
        if getattr(self.server_args, "elastic_ep_backend", None):
            return "elastic EP is unsupported"
        return None

    def _is_batch_compatible(self, batch: ScheduleBatch) -> bool:
        if not batch.reqs:
            return False
        if all(req.inflight_middle_chunks > 0 for req in batch.reqs):
            return False
        if batch.forward_mode.is_split_prefill():
            return False
        if getattr(batch, "tbo_split_seq_index", None) is not None:
            return False
        if any(
            req.pending_bootstrap
            or req.multimodal_inputs is not None
            or req.input_embeds is not None
            or req.positional_embed_overrides is not None
            for req in batch.reqs
        ):
            return False
        return not self._batch_needs_staging_for_heterogeneous_tp(batch)

    def _batch_needs_staging_for_heterogeneous_tp(self, batch: ScheduleBatch) -> bool:
        if not self.enable_staging:
            return False
        transfer_infos = getattr(self.kv_manager, "transfer_infos", {})
        decode_kv_args_table = getattr(self.kv_manager, "decode_kv_args_table", {})
        attn_tp_size = getattr(self.kv_manager, "attn_tp_size", self.ps.attn_tp_size)
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            for transfer_info in transfer_infos.get(room, {}).values():
                if transfer_info.is_dummy:
                    continue
                register_info = decode_kv_args_table.get(
                    transfer_info.mooncake_session_id
                )
                if (
                    register_info is not None
                    and register_info.dst_attn_tp_size != attn_tp_size
                ):
                    return True
        return False

    def _get_adaptive_group_size(self, batch: ScheduleBatch) -> int:
        num_layers = self.model_config.num_hidden_layers
        if num_layers <= 4:
            return 0

        min_tokens = max(1, envs.SGLANG_PIPELINE_MIN_TOKENS.get())
        # Middle chunks stay on the regular path, so only count tokens that will
        # actually participate in pipelined KV transfer for the threshold.
        # Counting all tokens may enable pipelining when the actual transfer is
        # too small, causing a performance regression.
        pipelined_transfer_tokens = sum(
            req.extend_range.length
            for req in batch.reqs
            if req.inflight_middle_chunks <= 0
        )
        if pipelined_transfer_tokens < min_tokens:
            return 0

        avg_tokens = sum(req.extend_range.length for req in batch.reqs) // len(
            batch.reqs
        )

        # Adaptive group size via a continuous formula:
        #   target_iters = clamp(MAX - progress * (MAX - MIN), MIN, MAX)
        #   progress = (avg_tokens - min_tokens) / (sat_tokens - min_tokens)
        #
        # Pipeline total time:
        #   Good bandwidth (T < C): total = C + T/N (last transfer is exposed)
        #   Poor bandwidth (T > C): total = C/N + T (first compute is exposed)
        #
        # Short prompts have a high T/C ratio because attention is O(n^2) while
        # transfer is O(n), so more groups reduce exposed T/N or C/N. For long
        # prompts, compute dominates and fewer groups avoid unnecessary overhead.
        max_iters = envs.SGLANG_PIPELINE_MAX_ITERS.get()
        min_iters = envs.SGLANG_PIPELINE_MIN_ITERS.get()
        if min_iters > max_iters:
            min_iters, max_iters = max_iters, min_iters
        sat_tokens = min_tokens * max(1.01, envs.SGLANG_PIPELINE_SAT_MULTIPLIER.get())
        progress = min(
            1.0,
            max(0.0, (avg_tokens - min_tokens) / (sat_tokens - min_tokens)),
        )
        target_iters = max(
            min_iters, round(max_iters - progress * (max_iters - min_iters))
        )
        return max(1, num_layers // target_iters)

    def _build_transfer_plan(
        self, batch: ScheduleBatch, num_layers: int, group_size: int
    ) -> LayerPipelinedTransferPlan:
        state_types = tuple(getattr(self.kv_manager.kv_args, "state_types", ()))
        pipelined_component_ids = self._get_pipelined_component_ids(
            num_layers, state_types
        )
        req_page_indices_list, req_state_indices_list = (
            self._get_buffer_transfer_indices(
                batch, state_types, pipelined_component_ids
            )
        )
        layer_ids_by_group = [
            list(range(group_start, min(group_start + group_size, num_layers)))
            for group_start in range(0, num_layers, group_size)
        ]
        return LayerPipelinedTransferPlan(
            group_size=group_size,
            layer_ids_by_group=layer_ids_by_group,
            req_page_indices_list=req_page_indices_list,
            req_state_indices_list=req_state_indices_list,
        )

    def _get_pipelined_component_ids(
        self, num_layers: int, state_types: Tuple[StateType, ...]
    ) -> set[int]:
        component_ids = set()
        for layer_id in range(num_layers):
            handles = self.kv_manager.kv_args.target_buffer_handles.get(layer_id)
            if handles is None:
                continue
            for buffer_type, component_id in zip(
                handles.buffer_types, handles.state_component_ids
            ):
                if buffer_type != BufferType.STATE or component_id is None:
                    continue
                if (
                    state_types[component_id] == StateType.DSA
                    and self._skip_dsa_state_layer_ids
                    and layer_id in self._skip_dsa_state_layer_ids
                ):
                    continue
                component_ids.add(component_id)

        for handles in self.kv_manager.kv_args.draft_buffer_handles.values():
            for buffer_type, component_id in zip(
                handles.buffer_types, handles.state_component_ids
            ):
                if buffer_type == BufferType.STATE and component_id is not None:
                    component_ids.add(component_id)
        return component_ids

    def _get_buffer_transfer_indices(
        self,
        batch: ScheduleBatch,
        state_types: Tuple[StateType, ...],
        pipelined_component_ids: set[int],
    ) -> Tuple[List[np.ndarray], List[List]]:
        page_size = self.token_to_kv_pool_allocator.page_size
        token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        req_page_indices_list = []
        req_state_indices_list = []
        for req in batch.reqs:
            req.ready_for_pipelined_transfer_finalize = False
            seq_len = min(req.extend_range.end, len(req.origin_input_ids))
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.start_send_idx : seq_len
            ]
            req_page_indices_list.append(kv_to_page_indices(kv_indices, page_size))

            req_state_indices = []
            for state_type in state_types:
                if state_type == StateType.MAMBA:
                    mamba_indices = (
                        self.req_to_token_pool.req_index_to_mamba_index_mapping[
                            req.req_pool_idx
                        ]
                    )
                    mamba_indices = self.req_to_token_pool.translate_mamba_indices(
                        mamba_indices
                    )
                    req_state_indices.append([mamba_indices.cpu().numpy()])
                elif state_type == StateType.SWA:
                    window_start = max(0, seq_len - self.sliding_window_size)
                    window_start = (window_start // page_size) * page_size
                    window_kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, window_start:seq_len
                    ]
                    window_kv_indices = (
                        self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                            window_kv_indices
                        )
                    )
                    req_state_indices.append(
                        kv_to_page_indices(window_kv_indices, page_size)
                    )
                elif state_type in (StateType.DSA, StateType.MINIMAX_INDEX_K):
                    full_kv_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :seq_len
                    ]
                    req_state_indices.append(
                        kv_to_page_indices(full_kv_indices, page_size)
                    )
                elif state_type == StateType.SWA_RING:
                    window_start = max(0, seq_len - token_to_kv_pool.unified_swa_window)
                    positions = np.arange(window_start, seq_len, dtype=np.int64)
                    ring_rows = (
                        int(req.req_pool_idx) * token_to_kv_pool.unified_swa_ring_size
                        + positions % token_to_kv_pool.unified_swa_ring_size
                    )
                    req_state_indices.append(ring_rows.astype(np.int32))
                elif state_type == StateType.C128_STATE:
                    online = is_dsv4_c128_online_enabled()
                    ring_size = 1 if online else token_to_kv_pool.get_ring_size(128)
                    req_state_indices.append(
                        get_dsv4_c128_state_indices(
                            int(req.req_pool_idx),
                            len(req.origin_input_ids),
                            online=online,
                            ring_size=ring_size,
                        )
                    )
                else:
                    req_state_indices.append(None)
            req_state_indices_list.append(req_state_indices)
            non_pipelined = [
                None if i in pipelined_component_ids else indices
                for i, indices in enumerate(req_state_indices)
            ]
            req.non_pipelined_state_indices = (
                non_pipelined
                if any(indices is not None for indices in non_pipelined)
                else None
            )
        return req_page_indices_list, req_state_indices_list
