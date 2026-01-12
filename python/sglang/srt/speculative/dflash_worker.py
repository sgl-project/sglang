# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DFlash speculative decoding worker with torch-based non-causal attention."""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput

logger = logging.getLogger(__name__)


class DFlashWorker(TpModelWorker):
    """DFlash speculative decoding worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.block_size = server_args.speculative_dflash_block_size
        server_args.context_length = target_worker.model_runner.model_config.context_len
        server_args.disable_cuda_graph = True

        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        super().__init__(
            server_args=server_args, gpu_id=gpu_id, tp_rank=tp_rank, pp_rank=0,
            dp_rank=dp_rank, moe_ep_rank=moe_ep_rank, nccl_port=nccl_port,
            is_draft_worker=True, req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.draft_model_runner.model.set_embed_and_head(embed, head)

        target_model = target_worker.model_runner.model
        self.target_layer_ids = self.draft_model_runner.model.target_layer_ids
        if hasattr(target_model, "set_eagle3_layers_to_capture"):
            target_model.set_eagle3_layers_to_capture(self.target_layer_ids)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        self.mask_token_id = tokenizer.mask_token_id

        self._request_state: Dict[str, dict] = {}
        logger.info(f"DFlashWorker: block_size={self.block_size}, mask_id={self.mask_token_id}")

    @property
    def draft_model_runner(self):
        return self.model_runner

    def clear_cache_pool(self):
        self._request_state.clear()

    def _get_request_state(self, rid: str) -> dict:
        if rid not in self._request_state:
            self._request_state[rid] = {"target_hidden": None, "accumulated_hidden": None, "start": 0, "verified_id": None}
        return self._request_state[rid]

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        return self._forward_extend(batch) if batch.forward_mode.is_extend() else self._forward_decode(batch)

    def _forward_extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Prefill: capture hidden states from target model."""
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        hidden_states = batch_result.logits_output.hidden_states
        next_token_ids = batch_result.next_token_ids
        token_offset = 0

        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            new_token_count = batch.extend_lens[i] if hasattr(batch, "extend_lens") else len(req.origin_input_ids)
            state["start"] = len(req.origin_input_ids)

            if hidden_states is not None:
                new_hidden = hidden_states[token_offset:token_offset + new_token_count] if hidden_states.dim() == 2 else hidden_states[:, token_offset:token_offset + new_token_count]
                if new_hidden.dim() == 2:
                    new_hidden = new_hidden.unsqueeze(0)
                state["target_hidden"] = state["accumulated_hidden"] = new_hidden.clone()
                state["verified_id"] = next_token_ids[i].clone()
                token_offset += new_token_count

        batch.spec_info = DFlashDraftInput(hidden_states=None, verified_id=next_token_ids.clone(), block_size=self.block_size)
        return batch_result

    def _forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Decode: draft forward -> verify -> accept tokens."""
        spec_info = batch.spec_info
        if spec_info is None or not isinstance(spec_info, DFlashDraftInput):
            return self._fallback_decode(batch)

        # Draft forward
        draft_tokens, positions = self._draft_forward(batch, spec_info.verified_id)

        # Verify with target
        verify_input = DFlashVerifyInput(draft_token=draft_tokens, positions=positions, block_size=self.block_size)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        verify_input.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch, is_verify=True)

        logits_output, new_verified_id, num_accepted = verify_input.verify(batch, batch_result.logits_output, self.page_size)

        # Update per-request state
        accept_length_list = verify_input.accept_length.cpu().tolist()
        hidden_offset = 0
        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            acc_len = accept_length_list[i]
            state["start"] += acc_len + 1

            if logits_output.hidden_states is not None:
                num_tokens = acc_len + 1
                req_hidden = logits_output.hidden_states[hidden_offset:hidden_offset + num_tokens] if logits_output.hidden_states.dim() == 2 else logits_output.hidden_states[:, hidden_offset:hidden_offset + num_tokens]
                hidden_offset += num_tokens
                if req_hidden.dim() == 2:
                    req_hidden = req_hidden.unsqueeze(0)
                old = state.get("accumulated_hidden")
                state["accumulated_hidden"] = torch.cat([old, req_hidden], dim=1) if old is not None else req_hidden.clone()
                state["target_hidden"] = state["accumulated_hidden"]

            state["verified_id"] = new_verified_id[i].clone()

        batch.spec_info = DFlashDraftInput(hidden_states=None, verified_id=new_verified_id.clone(), block_size=self.block_size)
        batch.forward_mode = ForwardMode.DECODE
        self._cleanup_finished_requests(batch)

        return GenerationBatchResult(
            logits_output=logits_output, next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted, can_run_cuda_graph=batch_result.can_run_cuda_graph,
            accept_lens=verify_input.accept_length,
        )

    def _draft_forward(self, batch: ScheduleBatch, all_verified_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect per-request data and run batched draft forward."""
        target_hiddens, verified_ids, ctx_lens = [], [], []

        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            target_hidden = state.get("target_hidden")
            if target_hidden is None:
                continue

            req_verified_id = state.get("verified_id")
            if req_verified_id is None:
                req_verified_id = all_verified_id[i] if all_verified_id.dim() > 0 and all_verified_id.shape[0] > i else all_verified_id.flatten()[0]

            target_hiddens.append(target_hidden)
            verified_ids.append(req_verified_id)
            ctx_lens.append(target_hidden.shape[1] if target_hidden.dim() == 3 else target_hidden.shape[0])

        if not target_hiddens:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)

        return self._batched_draft_forward(batch, target_hiddens, verified_ids, ctx_lens)

    def _batched_draft_forward(
        self, batch: ScheduleBatch, target_hiddens: List[torch.Tensor],
        verified_ids: List[torch.Tensor], ctx_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process each request through draft model layers with non-causal attention."""
        bs, block_size = len(target_hiddens), self.block_size
        max_ctx_len = max(ctx_lens)
        hidden_size = self.draft_model_runner.model.config.hidden_size
        input_dim = target_hiddens[0].shape[-1]

        # Pad target_hiddens
        padded = torch.zeros(bs, max_ctx_len, input_dim, dtype=torch.bfloat16, device=self.device)
        for i, (th, cl) in enumerate(zip(target_hiddens, ctx_lens)):
            padded[i, :cl] = th[:cl] if th.dim() == 2 else th[0, :cl]

        # Stack verified_ids
        verified_ids_tensor = torch.stack([
            v if isinstance(v, torch.Tensor) and v.dim() == 0 else (v[0] if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device))
            for v in verified_ids
        ]).to(torch.long, device=self.device)

        # Create noise tokens with verified_id at position 0
        block_input_ids = torch.full((bs, block_size), self.mask_token_id, dtype=torch.long, device=self.device)
        block_input_ids[:, 0] = verified_ids_tensor

        # Get embeddings and project target hidden
        noise_embedding = self.draft_model_runner.model.model.embed_tokens(block_input_ids)
        if padded.shape[-1] != hidden_size:
            padded = self.draft_model_runner.model.model.fc(padded)
        padded = self.draft_model_runner.model.model.hidden_norm(padded)

        # Process each request
        all_draft_hidden = []
        for i in range(bs):
            ctx_len = ctx_lens[i]
            combined = torch.cat([padded[i, :ctx_len], noise_embedding[i]], dim=0)
            positions = torch.arange(ctx_len + block_size, device=self.device)

            hidden = combined
            for layer in self.draft_model_runner.model.model.layers:
                hidden = layer(positions, hidden, forward_batch=None, ctx_len=ctx_len)

            all_draft_hidden.append(self.draft_model_runner.model.model.norm(hidden[ctx_len:]))

        # Get draft tokens
        draft_hidden = torch.stack(all_draft_hidden, dim=0)
        draft_logits = torch.matmul(draft_hidden[:, 1:], self.draft_model_runner.model.lm_head.weight.t())
        block_input_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)

        # Flatten output
        draft_tokens = block_input_ids.flatten()
        positions = torch.cat([torch.arange(batch.seq_lens[i].item(), batch.seq_lens[i].item() + block_size, device=self.device) for i in range(bs)])
        return draft_tokens, positions

    def _fallback_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        from sglang.srt.mem_cache.common import alloc_for_decode
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
        batch.input_ids, batch.output_ids = batch.output_ids, None
        batch.seq_lens, batch.seq_lens_cpu = batch.seq_lens + 1, batch.seq_lens_cpu + 1
        batch.seq_lens_sum += len(batch.reqs)
        for req in batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1
        return self.target_worker.forward_batch_generation(batch.get_model_worker_batch())

    def _cleanup_finished_requests(self, batch: ScheduleBatch):
        active = {req.rid for req in batch.reqs}
        for rid in [r for r in self._request_state if r not in active]:
            del self._request_state[rid]
