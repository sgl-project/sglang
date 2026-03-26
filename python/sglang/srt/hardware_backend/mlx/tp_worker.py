"""MLX-specific TpModelWorker subclass for Apple Silicon.

Overrides the standard TpModelWorker to route forward passes through
the native MLX model runner, avoiding PyTorch MPS entirely for inference.

PyTorch model weights are never loaded.  A lightweight ModelRunner stub
(MlxModelRunnerStub) provides only the minimal bookkeeping structures
(req_to_token_pool, token_to_kv_pool_allocator with a zero-memory
dummy KV cache) that the SGLang scheduler expects.  The actual KV cache
is managed internally by the MLX model runner.
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
        """Override to use a lightweight ModelRunner that skips weight loading."""
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

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
        )

        # Initialize the MLX model runner (loads weights via MLX, not PyTorch)
        logger.info("Initializing MlxModelRunner for end-to-end MLX inference")
        self._mlx_runner = MlxModelRunner(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
        )
        self._mlx_active_rids: set[str] = set()

    def get_pad_input_ids_func(self):
        """Override since the stub ModelRunner has no real model."""
        return None

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
        """Run forward pass through the MLX model runner.

        Bypasses the standard ModelRunner forward+sample and uses native MLX
        inference for the entire model. Only supports greedy sampling.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        forward_mode = model_worker_batch.forward_mode
        reqs = model_worker_batch.reqs

        if forward_mode.is_idle():
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        # Auto-cleanup: remove MLX state for requests no longer in the batch
        current_rids = {req.rid for req in reqs}
        stale_rids = self._mlx_active_rids - current_rids
        for rid in stale_rids:
            self._mlx_runner.remove_request(rid)
        self._mlx_active_rids = current_rids

        next_token_ids_list = []

        if forward_mode.is_extend():
            # Prefill (or MIXED): extract per-request tokens from concatenated input_ids
            input_ids_cpu = model_worker_batch.input_ids.cpu().tolist()
            extend_seq_lens = model_worker_batch.extend_seq_lens
            offset = 0
            prefill_rids = []
            decode_rids = []
            for i, req in enumerate(reqs):
                seq_len = extend_seq_lens[i]
                req_token_ids = input_ids_cpu[offset : offset + seq_len]
                offset += seq_len
                if req.rid in self._mlx_runner._request_states:
                    # MIXED mode: this request already has MLX state, decode it
                    decode_rids.append(req.rid)
                else:
                    # Prefill: new request
                    next_token = self._mlx_runner.prefill(req.rid, req_token_ids)
                    prefill_rids.append((req.rid, next_token))

            # Batch decode all existing requests at once
            if decode_rids:
                decode_results = self._mlx_runner.decode_batch(decode_rids)
                decode_map = dict(zip(decode_rids, decode_results))
            else:
                decode_map = {}

            prefill_map = dict(prefill_rids)

            # Reassemble in original request order
            for req in reqs:
                if req.rid in decode_map:
                    next_token_ids_list.append(decode_map[req.rid])
                else:
                    next_token_ids_list.append(prefill_map[req.rid])

        elif forward_mode.is_decode():
            # Decode: batch decode all requests
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
