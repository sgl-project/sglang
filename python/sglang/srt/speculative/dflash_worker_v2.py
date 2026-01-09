# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License")

"""DFlash speculative decoding worker V2 with CUDA graph and overlap support.

This module provides the V2 implementation of DFlash with:
- CUDA graph capture for draft model forward
- Spec V2 overlap (draft and verify preparation run on separate streams)
- Pre-allocated static buffers for maximum performance
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.dflash_draft_cuda_graph_runner import (
    DFlashDraftCudaGraphRunner,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.models.dflash_static_cache import DFlashStaticKVCache
from sglang.srt.models.qwen3_dflash import Qwen3ForCausalLMDFlash
from sglang.srt.utils.common import get_available_gpu_memory

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache

logger = logging.getLogger(__name__)


def _get_plan_stream(device: str) -> Tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    """Get plan stream for overlapped execution."""
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.cuda.Stream()
        plan_stream_ctx = torch.cuda.stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class DFlashDraftWorkerV2(BaseDraftWorker):
    """
    DFlash draft worker V2 with CUDA graph support.
    
    This worker manages the draft model forward pass and can use
    CUDA graphs for reduced kernel launch overhead.
    """
    
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
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_runner.model_config
        self.tp_rank = tp_rank
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.block_size = server_args.speculative_dflash_block_size
        
        # Load draft model
        logger.info(f"Loading DFlash draft model from {server_args.speculative_draft_model_path}")
        self.draft_model = Qwen3ForCausalLMDFlash.from_pretrained(
            server_args.speculative_draft_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(self.device).eval()
        
        self.target_layer_ids = self.draft_model.target_layer_ids
        logger.info(f"DFlash V2 target layer IDs: {self.target_layer_ids}")
        
        # Get embeddings from target model
        target_model = target_worker.model_runner.model
        if hasattr(target_model, 'model') and hasattr(target_model.model, 'embed_tokens'):
            self.embed_tokens = target_model.model.embed_tokens
        else:
            self.embed_tokens = target_model.get_input_embeddings()
        
        if hasattr(target_model, 'lm_head') and hasattr(target_model.lm_head, 'weight'):
            self.lm_head_weight = target_model.lm_head.weight
        else:
            self.lm_head_weight = self.embed_tokens.weight
        
        # Configure target model to capture hidden states
        if hasattr(target_model, 'set_eagle3_layers_to_capture'):
            target_model.set_eagle3_layers_to_capture(self.target_layer_ids)
        
        # Get mask token ID
        self.mask_token_id = getattr(self.draft_model.config, 'mask_token_id', 151669)
        
        # Static KV cache for CUDA graph compatibility
        draft_config = self.draft_model.config
        self.static_kv_cache = DFlashStaticKVCache(
            num_layers=draft_config.num_hidden_layers,
            num_kv_heads=draft_config.num_key_value_heads,
            head_dim=getattr(draft_config, "head_dim", draft_config.hidden_size // draft_config.num_attention_heads),
            max_seq_len=2048,  # Configurable
            max_batch_size=32,  # Configurable
            device=self.device,
            dtype=torch.bfloat16,
        )
        
        # Initialize CUDA graph runner
        self.cuda_graph_runner: Optional[DFlashDraftCudaGraphRunner] = None
        
        # Per-request state
        self._request_state: Dict[str, dict] = {}
    
    def init_cuda_graphs(self):
        """Capture CUDA graphs for draft model."""
        if self.server_args.disable_cuda_graph:
            logger.info("CUDA graphs disabled for DFlash draft model")
            return
        
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, 0)
        logger.info(
            f"Capture DFlash draft CUDA graph begin. avail mem={before_mem:.2f} GB"
        )
        
        try:
            self.cuda_graph_runner = DFlashDraftCudaGraphRunner(self)
            after_mem = get_available_gpu_memory(self.device, 0)
            logger.info(
                f"Capture DFlash draft CUDA graph end. "
                f"Time: {time.perf_counter() - tic:.2f}s, "
                f"mem usage: {before_mem - after_mem:.2f} GB"
            )
        except Exception as e:
            logger.warning(f"Failed to capture DFlash CUDA graphs: {e}")
            self.cuda_graph_runner = None
    
    def _get_request_state(self, request_id: str) -> dict:
        """Get or create per-request state."""
        if request_id not in self._request_state:
            self._request_state[request_id] = {
                'target_hidden': None,
                'accumulated_hidden': None,
                'draft_pos': 0,
                'start': 0,
                'verified_id': None,
                'slot_idx': self.static_kv_cache.allocate_slot(request_id),
            }
        return self._request_state[request_id]
    
    def _cleanup_request(self, request_id: str):
        """Clean up state for a finished request."""
        if request_id in self._request_state:
            del self._request_state[request_id]
        self.static_kv_cache.free_slot(request_id)
    
    def draft(self, batch: ModelWorkerBatch) -> DFlashVerifyInput:
        """
        Generate draft tokens.
        
        Uses CUDA graph if available and batch size is supported.
        """
        draft_input: DFlashDraftInput = batch.spec_info
        bs = len(batch.reqs) if hasattr(batch, 'reqs') else batch.batch_size()
        
        # Check if we can use CUDA graph
        can_cuda_graph = (
            self.cuda_graph_runner is not None
            and self.cuda_graph_runner.can_run(bs)
        )
        
        if can_cuda_graph:
            return self._draft_with_cuda_graph(batch, draft_input)
        else:
            return self._draft_eager(batch, draft_input)
    
    def _draft_with_cuda_graph(
        self,
        batch: ModelWorkerBatch,
        draft_input: DFlashDraftInput,
    ) -> DFlashVerifyInput:
        """Draft using CUDA graph replay."""
        bs = len(batch.reqs) if hasattr(batch, 'reqs') else batch.batch_size()
        
        # Prepare inputs
        target_hidden_list = []
        noise_embedding_list = []
        position_ids_list = []
        ctx_lens = []
        
        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            target_hidden = state.get('target_hidden')
            verified_id = state.get('verified_id', draft_input.verified_id[i])
            
            if target_hidden is None:
                # Fallback to eager if state not ready
                return self._draft_eager(batch, draft_input)
            
            ctx_len = target_hidden.shape[1]
            cache_len = self.static_kv_cache.get_seq_length(req.rid)
            
            # Build noise embedding
            block_output_ids = torch.full(
                (1, self.block_size),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device
            )
            block_output_ids[0, 0] = verified_id
            noise_emb = self.embed_tokens(block_output_ids)
            
            # Build position IDs
            total_len = ctx_len + self.block_size
            pos_ids = torch.arange(
                cache_len, cache_len + total_len,
                device=self.device
            ).unsqueeze(0)
            
            target_hidden_list.append(target_hidden)
            noise_embedding_list.append(noise_emb)
            position_ids_list.append(pos_ids)
            ctx_lens.append(ctx_len)
        
        # Pad to same length and batch
        max_ctx = max(ctx_lens)
        target_hidden_batched = torch.zeros(
            (bs, max_ctx, target_hidden_list[0].shape[2]),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for i, th in enumerate(target_hidden_list):
            target_hidden_batched[i, :th.shape[1]] = th.squeeze(0)
        
        noise_embedding_batched = torch.cat(noise_embedding_list, dim=0)
        
        max_pos = max(p.shape[1] for p in position_ids_list)
        position_ids_batched = torch.zeros(
            (bs, max_pos), dtype=torch.int64, device=self.device
        )
        for i, p in enumerate(position_ids_list):
            position_ids_batched[i, :p.shape[1]] = p.squeeze(0)
        
        ctx_lens_tensor = torch.tensor(ctx_lens, dtype=torch.int32, device=self.device)
        
        # Replay CUDA graph
        output = self.cuda_graph_runner.replay(
            target_hidden=target_hidden_batched,
            noise_embedding=noise_embedding_batched,
            position_ids=position_ids_batched,
            ctx_lens=ctx_lens_tensor,
        )
        
        # Convert output to draft tokens
        draft_tokens = self._output_to_tokens(output)
        
        return DFlashVerifyInput(
            draft_token=draft_tokens.flatten(),
            positions=position_ids_batched[:, -self.block_size:].flatten(),
            block_size=self.block_size,
        )
    
    def _draft_eager(
        self,
        batch: ModelWorkerBatch,
        draft_input: DFlashDraftInput,
    ) -> DFlashVerifyInput:
        """Draft using eager execution (fallback)."""
        bs = len(batch.reqs) if hasattr(batch, 'reqs') else batch.batch_size()
        
        all_draft_tokens = []
        all_positions = []
        
        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            target_hidden = state.get('target_hidden')
            
            if target_hidden is None:
                # Return empty verify input
                return DFlashVerifyInput(
                    draft_token=torch.zeros(
                        bs * self.block_size, dtype=torch.long, device=self.device
                    ),
                    positions=torch.zeros(
                        bs * self.block_size, dtype=torch.long, device=self.device
                    ),
                    block_size=self.block_size,
                )
            
            verified_id = state.get('verified_id')
            if verified_id is None:
                # Fallback to draft_input.verified_id
                verified_id = draft_input.verified_id[i] if draft_input.verified_id is not None else 0
            ctx_len = target_hidden.shape[1]
            cache_len = self.static_kv_cache.get_seq_length(req.rid)
            
            # Build noise embedding
            block_output_ids = torch.full(
                (1, self.block_size),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device
            )
            # Handle scalar vs tensor verified_id
            if isinstance(verified_id, torch.Tensor):
                block_output_ids[0, 0] = verified_id.item() if verified_id.numel() == 1 else verified_id
            else:
                block_output_ids[0, 0] = verified_id
            noise_emb = self.embed_tokens(block_output_ids)
            
            # Build position IDs
            pos_ids = torch.arange(
                cache_len, cache_len + ctx_len + self.block_size,
                device=self.device
            ).unsqueeze(0)
            
            # Forward
            output = self.draft_model(
                target_hidden=target_hidden,
                noise_embedding=noise_emb,
                position_ids=pos_ids,
                use_cache=True,
                is_causal=False,
                static_kv_cache=self.static_kv_cache,
                request_id=req.rid,
                use_flash_attention=True,
            )
            
            # Get draft tokens
            draft_tokens = self._output_to_tokens(output)
            
            all_draft_tokens.append(draft_tokens.flatten())
            all_positions.append(pos_ids[0, -self.block_size:])
        
        return DFlashVerifyInput(
            draft_token=torch.cat(all_draft_tokens),
            positions=torch.cat(all_positions),
            block_size=self.block_size,
        )
    
    def _output_to_tokens(self, output: torch.Tensor) -> torch.Tensor:
        """Convert model output to draft tokens."""
        # output: [bs, block_size, hidden]
        logits = torch.matmul(output, self.lm_head_weight.T)
        return logits.argmax(dim=-1)
    
    def draft_extend(
        self,
        batch: ModelWorkerBatch,
        hidden_states: torch.Tensor,
    ) -> DFlashDraftInput:
        """
        Extend draft after prefill.
        
        Initializes KV cache and prepares for decode phase.
        """
        # Store hidden states for each request
        # hidden_states can be:
        # - [total_tokens, hidden] (SGLang format, flattened across batch)
        # - [bs, seq, hidden] (standard format)
        # We need [1, seq, hidden] per request for the draft model
        
        if hidden_states.dim() == 2:
            # Flattened format [total_tokens, hidden]
            # Need to split by request using seq_lens
            token_offset = 0
            for i, req in enumerate(batch.reqs):
                state = self._get_request_state(req.rid)
                # Get sequence length for this request
                if hasattr(batch, 'seq_lens') and batch.seq_lens is not None:
                    seq_len = batch.seq_lens[i].item() if batch.seq_lens.dim() > 0 else batch.seq_lens.item()
                else:
                    # Fallback: assume equal distribution
                    seq_len = hidden_states.shape[0] // len(batch.reqs)
                
                # Extract this request's hidden states and reshape to 3D
                req_hidden = hidden_states[token_offset:token_offset + seq_len]
                state['target_hidden'] = req_hidden.unsqueeze(0)  # [1, seq_len, hidden]
                state['draft_pos'] = 0
                token_offset += seq_len
        else:
            # Already 3D [bs, seq, hidden]
            for i, req in enumerate(batch.reqs):
                state = self._get_request_state(req.rid)
                state['target_hidden'] = hidden_states[i:i+1]  # [1, seq, hidden]
                state['draft_pos'] = 0
        
        # For prefill, new_seq_lens is the current batch seq_lens
        new_seq_lens = batch.seq_lens if hasattr(batch, 'seq_lens') and batch.seq_lens is not None else None
        
        return DFlashDraftInput(
            hidden_states=hidden_states,
            verified_id=torch.zeros(len(batch.reqs), dtype=torch.long, device=self.device),
            block_size=self.block_size,
            new_seq_lens=new_seq_lens,
        )


class DFlashWorkerV2(BaseSpecWorker):
    """
    DFlash speculative decoding worker V2.
    
    Features:
    - CUDA graph support for draft model
    - Spec V2 overlap (draft and verify preparation on separate streams)
    - Pre-allocated static buffers
    """
    
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
        self.tp_rank = tp_rank
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.block_size = server_args.speculative_dflash_block_size
        
        self._target_worker = target_worker
        
        # Memory pools (shared with target)
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        
        # Initialize draft worker
        self._draft_worker = DFlashDraftWorkerV2(
            server_args, gpu_id, tp_rank, dp_rank,
            moe_ep_rank, nccl_port, target_worker
        )
        
        # Plan stream for overlapped execution
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)
        
        # Per-request state (similar to draft worker but for orchestration)
        self._request_state: Dict[str, dict] = {}
    
    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker
    
    @property
    def draft_worker(self) -> DFlashDraftWorkerV2:
        return self._draft_worker
    
    def init_cuda_graphs(self):
        """Initialize CUDA graphs for draft model."""
        self._draft_worker.init_cuda_graphs()
    
    def clear_cache_pool(self):
        """Clear cache pools (shared with target, cleared by scheduler)."""
        pass
    
    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """
        Main entry point with spec V2 overlap.
        """
        if model_worker_batch.forward_mode.is_extend():
            return self._forward_prefill(model_worker_batch)
        else:
            return self._forward_decode_v2(model_worker_batch)
    
    def _forward_prefill(
        self,
        batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """Prefill phase: Target forward → Draft extend."""
        # Target prefill
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_output = self.target_worker.forward_batch_generation(batch)
        
        # Draft extend (prepare for decode)
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch_output.next_draft_input = self.draft_worker.draft_extend(
            batch,
            batch_output.logits_output.hidden_states,
        )
        
        return batch_output
    
    def _forward_decode_v2(
        self,
        batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """
        Decode with overlapped draft and verify planning.
        
        Timeline:
        1. Draft forward (compute stream)
        2. Plan verify (plan stream, overlapped with draft)
        3. Sync streams
        4. Target verify (compute stream)
        5. Sample and accept
        6. Prepare next iteration
        """
        # Step 1: Draft forward
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInput(
                hidden_states=None,
                verified_id=torch.zeros(
                    len(batch.reqs), dtype=torch.long, device=self.device
                ),
                block_size=self.block_size,
                new_seq_lens=batch.seq_lens if hasattr(batch, 'seq_lens') else None,
            )
        
        verify_input: DFlashVerifyInput = self.draft_worker.draft(batch)
        
        # Step 2: Plan verify (overlapped on plan stream)
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                self._prepare_verify(batch, verify_input)
            )
        
        # Step 3: Sync streams
        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)
        
        # Step 4: Target verify
        batch.spec_info = verify_input
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output
        
        # Step 5: Sample and accept
        predict, accept_length, accept_index = verify_input.verify(
            batch, logits_output
        )
        
        # Step 6: Prepare next iteration
        batch_output = self._update_state_after_verify(
            batch,
            forward_batch_output,
            predict,
            accept_length,
            accept_index,
        )
        
        return batch_output
    
    def _prepare_verify(
        self,
        batch: ModelWorkerBatch,
        verify_input: DFlashVerifyInput,
    ) -> Tuple[ForwardBatch, bool]:
        """
        Prepare for verification (can run on plan stream).
        """
        if not batch.forward_mode.is_idle():
            # Prepare cache locations using V2 method (works with ModelWorkerBatch)
            verify_input.prepare_for_verify_v2(
                batch,
                self.target_worker.model_runner.token_to_kv_pool_allocator,
                self.target_worker.model_runner.req_to_token_pool,
            )
        
        # Create forward batch
        batch.forward_mode = (
            ForwardMode.IDLE if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(
            batch, self.target_worker.model_runner
        )
        
        # Check CUDA graph compatibility
        can_run_cuda_graph = bool(
            self.target_worker.model_runner.graph_runner
            and self.target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        
        if can_run_cuda_graph:
            self.target_worker.model_runner.graph_runner.replay_prepare(
                verify_forward_batch
            )
        else:
            if not batch.forward_mode.is_idle():
                self.target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )
        
        return verify_forward_batch, can_run_cuda_graph
    
    def _update_state_after_verify(
        self,
        batch: ModelWorkerBatch,
        forward_batch_output,
        predict: torch.Tensor,
        accept_length: torch.Tensor,
        accept_index: torch.Tensor,
    ) -> GenerationBatchResult:
        """Update state after verification and prepare next iteration."""
        # Calculate new sequence lengths after accepting tokens
        new_seq_lens = batch.seq_lens + accept_length
        
        # Record CUDA event for synchronization
        verify_done = torch.cuda.Event()
        verify_done.record()
        
        # Update KV cache lengths based on accepted tokens
        for i, req in enumerate(batch.reqs):
            accepted = accept_length[i].item()
            
            # Crop draft KV cache to accepted length
            self.draft_worker.static_kv_cache.crop(
                req.rid,
                self.draft_worker.static_kv_cache.get_seq_length(req.rid) + accepted
            )
            
            # Update verified_id in request state
            state = self.draft_worker._get_request_state(req.rid)
            state['verified_id'] = predict[i]
        
        # Prepare next draft input with new_seq_lens and verify_done for overlap scheduling
        forward_batch_output.next_draft_input = DFlashDraftInput(
            hidden_states=forward_batch_output.logits_output.hidden_states,
            verified_id=predict,
            block_size=self.block_size,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
        )
        
        return GenerationBatchResult(
            logits_output=forward_batch_output.logits_output,
            next_token_ids=predict,
            next_draft_input=forward_batch_output.next_draft_input,
        )



