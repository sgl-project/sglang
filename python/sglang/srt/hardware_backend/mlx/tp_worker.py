"""MLX-specific TpModelWorker subclass for Apple Silicon.

Routes forward passes through the MLX model runner, bypassing PyTorch
MPS.  A lightweight stub provides scheduler bookkeeping; the actual
KV data lives in MlxKVPool.
"""

import logging
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)


class MlxTpModelWorker(TpModelWorker):
    """A tensor parallel model worker that routes inference through MLX.

    Inherits from TpModelWorker for scheduler integration, but replaces
    the standard ModelRunner with MlxModelRunnerStub (no PyTorch weights,
    zero-memory KV cache) and delegates all forward passes to a native
    MlxModelRunner.
    """

    def _init_model_runner(self):
        """Create MLX runner first (auto-sizes pool), then stub with matching size."""
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        logger.info("Initializing MlxModelRunner for end-to-end MLX inference")
        init_kwargs = dict(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
            disable_radix_cache=self.server_args.disable_radix_cache,
            mem_fraction_static=self.server_args.mem_fraction_static,
        )
        if self.server_args.max_total_tokens is not None:
            init_kwargs["pool_size"] = self.server_args.max_total_tokens
        self._mlx_runner = MlxModelRunner(**init_kwargs)

        self._model_runner = MlxModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            mlx_pool_size=self._mlx_runner.pool_size,
        )

        self._mlx_active_rids: set[str] = set()
        self._mlx_pool_initialized = False

    def get_pad_input_ids_func(self):
        """Override since the stub ModelRunner has no real model."""
        return None

    def _ensure_mlx_pool_initialized(self):
        """Lazily initialize the MlxKVPool after the stub's pools are ready."""
        if not self._mlx_pool_initialized:
            self._mlx_runner.init_kv_pool(self._model_runner.req_to_token_pool)
            self._mlx_pool_initialized = True

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        """Override to route through MLX model runner."""
        if model_worker_batch is not None:
            self._ensure_mlx_pool_initialized()
            return self._forward_batch_generation_mlx(model_worker_batch)

        # Fallback to standard path for None batches
        return super().forward_batch_generation(
            model_worker_batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
        )

    def _forward_batch_generation_mlx(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """Run forward pass through the MLX model runner (greedy only)."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        forward_mode = model_worker_batch.forward_mode
        reqs = model_worker_batch.reqs

        if forward_mode.is_idle():
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        # Auto-cleanup: remove MLX state for requests no longer in the batch.
        current_rids = {req.rid for req in reqs}
        if forward_mode.is_decode():
            stale_rids = self._mlx_active_rids - current_rids
            for rid in stale_rids:
                self._mlx_runner.remove_request(rid)
            self._mlx_active_rids = current_rids
        else:
            self._mlx_active_rids |= current_rids

        next_token_ids_list = []

        if forward_mode.is_extend():
            # Ensure pool is up-to-date before PoolBackedCache reads it
            # for prefix-cached prefills.  Only runs on extend batches.
            self._mlx_runner.flush_all_decode_kv()
            input_ids_cpu = model_worker_batch.input_ids.cpu().tolist()
            out_cache_loc_cpu = model_worker_batch.out_cache_loc.cpu().tolist()
            extend_seq_lens = model_worker_batch.extend_seq_lens

            offset = 0  # into input_ids_cpu
            slot_offset = 0  # into out_cache_loc_cpu
            prefill_rids = []
            extend_rids = []
            decode_rids = []

            for i, req in enumerate(reqs):
                seq_len = extend_seq_lens[i]
                req_token_ids = input_ids_cpu[offset : offset + seq_len]
                req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
                offset += seq_len
                slot_offset += seq_len

                if self._mlx_runner.has_request(req.rid):
                    if seq_len > 1:
                        # Chunked prefill continuation
                        next_token = self._mlx_runner.extend(
                            req.rid, req_token_ids, req_new_slots
                        )
                        extend_rids.append((req.rid, next_token))
                    else:
                        # MIXED mode: single-token decode
                        decode_rids.append(req.rid)
                else:
                    # New prefill
                    prefix_slot_ids = req.prefix_indices.tolist()
                    full_token_ids = list(req.fill_ids)
                    next_token = self._mlx_runner.prefill(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                    )
                    prefill_rids.append((req.rid, next_token))

            # Batch decode all existing requests at once
            if decode_rids:
                decode_results = self._mlx_runner.decode_batch(decode_rids)
                decode_map = dict(zip(decode_rids, decode_results))
            else:
                decode_map = {}

            prefill_map = dict(prefill_rids)
            extend_map = dict(extend_rids)

            for req in reqs:
                if req.rid in decode_map:
                    next_token_ids_list.append(decode_map[req.rid])
                elif req.rid in extend_map:
                    next_token_ids_list.append(extend_map[req.rid])
                else:
                    next_token_ids_list.append(prefill_map[req.rid])

        elif forward_mode.is_decode():
            req_ids = [req.rid for req in reqs]
            next_token_ids_list = self._mlx_runner.decode_batch(req_ids)

        else:
            raise ValueError(
                f"MLX runner does not support forward mode: {forward_mode}"
            )

        next_token_ids = torch.tensor(
            next_token_ids_list, dtype=torch.long, device="cpu"
        )

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
