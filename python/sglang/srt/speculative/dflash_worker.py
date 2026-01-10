# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License")

"""DFlash speculative decoding worker using TpModelWorker infrastructure.

DFlash draft model generates a full block of tokens in one forward pass.
Uses torch-based non-causal attention for parallel block drafting.
Inherits from TpModelWorker for proper model loading and memory management.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache

logger = logging.getLogger(__name__)


class DFlashWorker(TpModelWorker):
    """DFlash speculative decoding worker inheriting from TpModelWorker.

    This enables:
    - Proper model loading via ModelRunner
    - Shared memory pools with target model
    - Tensor parallelism support
    - Hidden state caching via radix cache
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
        # Parse arguments
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.block_size = server_args.speculative_dflash_block_size

        # Override context length to match target model
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture CUDA graphs in super().__init__() - will be done later
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share memory pools with target worker
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # DFlash draft model uses ENCODER_ONLY attention (non-causal, no incremental KV cache)
        # Set a minimal draft_runner_cache_size to avoid OOM
        # We only need enough for one batch: max_bs * (max_ctx_len + block_size)
        max_bs = min(32, server_args.max_running_requests or 256)
        max_seq_len = min(4096, server_args.context_length or 32768) + self.block_size
        minimal_cache_size = max_bs * max_seq_len
        server_args.draft_runner_cache_size = minimal_cache_size
        logger.info(
            f"DFlash draft cache size: {minimal_cache_size} tokens (bs={max_bs}, seq_len={max_seq_len})"
        )

        # Initialize TpModelWorker with draft model
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=0,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )

        # Get embeddings and lm_head from target model and share with draft model
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Configure target model to capture multi-layer hidden states
        target_model = target_worker.model_runner.model
        self.target_layer_ids = self.draft_model_runner.model.target_layer_ids
        if hasattr(target_model, "set_eagle3_layers_to_capture"):
            logger.info(f"DFlash target_layer_ids: {self.target_layer_ids}")
            target_model.set_eagle3_layers_to_capture(self.target_layer_ids)

        # Get mask token ID
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        self.mask_token_id = tokenizer.mask_token_id
        logger.info(f"DFlash mask_token_id: {self.mask_token_id}")

        # Restore CUDA graph setting and initialize attention backend + CUDA graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.init_attention_backend()
        self.init_cuda_graphs()

        # Per-request state for hidden states and metadata
        self._request_state: Dict[str, dict] = {}

        # Radix cache integration
        self.radix_cache: Optional["RadixCache"] = None
        self.unified_hidden_cache_enabled = False

        logger.info(f"DFlashWorker initialized: block_size={self.block_size}")

    @property
    def draft_model_runner(self):
        """Alias for model_runner to match Eagle pattern."""
        return self.model_runner

    def init_attention_backend(self):
        """Initialize attention backend for draft model.

        DFlash uses torch-based non-causal attention, not the standard
        attention backend. This method is a no-op but kept for interface
        consistency with other speculative workers.
        """
        # DFlash uses direct torch attention in _batched_draft_forward()
        # No attention backend initialization needed
        pass

    def init_cuda_graphs(self):
        """Initialize CUDA graphs for draft model.

        DFlash draft model uses eager execution due to variable-length
        attention patterns. CUDA graphs are not captured for the draft model.
        """
        self.cuda_graph_runner = None
        logger.info("DFlash draft model: using eager mode")

    def clear_cache_pool(self):
        """Clear all per-request state."""
        self._request_state.clear()

    def set_radix_cache(self, radix_cache: "RadixCache", token_to_kv_pool=None):
        """Set up radix cache integration for hidden state prefix sharing."""
        self.radix_cache = radix_cache

        num_selected_layers = len(self.target_layer_ids)
        hidden_size = self.draft_model_runner.model.config.hidden_size

        # Check if hidden buffer is already enabled (to avoid duplicate allocation)
        if token_to_kv_pool is not None and hasattr(
            token_to_kv_pool, "enable_dflash_hidden_buffer"
        ):
            # Check if already enabled by checking for hidden_buffer attribute
            if (
                not hasattr(token_to_kv_pool, "hidden_buffer")
                or token_to_kv_pool.hidden_buffer is None
            ):
                token_to_kv_pool.enable_dflash_hidden_buffer(
                    hidden_size=hidden_size,
                    num_target_layers=num_selected_layers,
                )
                logger.info(
                    f"DFlash unified hidden cache enabled: {num_selected_layers} layers"
                )
            else:
                logger.info("DFlash hidden buffer already enabled, skipping")
            self.unified_hidden_cache_enabled = True
        else:
            self.unified_hidden_cache_enabled = False

    def _get_request_state(self, rid: str) -> dict:
        """Get or create per-request state."""
        if rid not in self._request_state:
            self._request_state[rid] = {
                "target_hidden": None,
                "accumulated_hidden": None,
                "start": 0,
                "verified_id": None,
                "noise_cache_locs": None,
            }
        return self._request_state[rid]

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Main entry point for forward pass."""
        if batch.forward_mode.is_extend():
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    def _forward_extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Prefill phase - capture hidden states from target model."""
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        logits_output = batch_result.logits_output
        next_token_ids = batch_result.next_token_ids
        hidden_states = logits_output.hidden_states

        # Store hidden states per request
        token_offset = 0

        for i, req in enumerate(batch.reqs):
            req_len = len(req.origin_input_ids)
            state = self._get_request_state(req.rid)

            new_token_count = (
                batch.extend_lens[i] if hasattr(batch, "extend_lens") else req_len
            )
            cached_count = (
                req_len - new_token_count if hasattr(batch, "extend_lens") else 0
            )

            state["start"] = req_len

            if hidden_states is not None:
                # Extract NEW tokens' hidden states
                if hidden_states.dim() == 2:
                    new_hidden = hidden_states[
                        token_offset : token_offset + new_token_count, :
                    ]
                else:
                    new_hidden = hidden_states[
                        :, token_offset : token_offset + new_token_count, :
                    ]

                if new_hidden.dim() == 2:
                    new_hidden = new_hidden.unsqueeze(0)

                # Try to get cached hidden states from radix cache
                if cached_count > 0 and self.unified_hidden_cache_enabled:
                    token_to_kv_pool = self.target_worker.model_runner.token_to_kv_pool
                    if (
                        hasattr(req, "hidden_indices")
                        and req.hidden_indices is not None
                    ):
                        prefix_locs = req.hidden_indices[:cached_count]
                        cached_hidden = token_to_kv_pool.get_all_hidden_states(
                            prefix_locs
                        )
                        if cached_hidden is not None:
                            cached_hidden = cached_hidden.unsqueeze(0)
                            new_hidden = torch.cat([cached_hidden, new_hidden], dim=1)

                state["target_hidden"] = new_hidden.clone()
                state["accumulated_hidden"] = new_hidden.clone()

                # Store hidden states in KV pool
                if self.unified_hidden_cache_enabled:
                    token_to_kv_pool = self.target_worker.model_runner.token_to_kv_pool
                    if (
                        hasattr(batch, "out_cache_loc")
                        and batch.out_cache_loc is not None
                    ):
                        req_start = (
                            sum(batch.extend_lens[:i])
                            if hasattr(batch, "extend_lens")
                            else token_offset
                        )
                        req_end = req_start + new_token_count
                        req_cache_loc = batch.out_cache_loc[req_start:req_end]
                        hidden_to_store = new_hidden.squeeze(0)[-new_token_count:]
                        if hidden_to_store.shape[0] == req_cache_loc.shape[0]:
                            token_to_kv_pool.set_all_hidden_states(
                                req_cache_loc, hidden_to_store
                            )
                            # Update req.hidden_indices for radix cache insertion
                            # This ensures cache_finished_req() will store hidden indices in the tree
                            if (
                                not hasattr(req, "hidden_indices")
                                or req.hidden_indices is None
                            ):
                                req.hidden_indices = req_cache_loc.clone()
                            else:
                                req.hidden_indices = torch.cat(
                                    [req.hidden_indices, req_cache_loc]
                                )

                state["verified_id"] = next_token_ids[i].clone()
                token_offset += new_token_count

        batch.spec_info = DFlashDraftInput(
            hidden_states=None,
            verified_id=next_token_ids.clone(),
            block_size=self.block_size,
        )

        return batch_result

    def _forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Decode phase with DFlash speculation."""
        spec_info = batch.spec_info

        if spec_info is None or not isinstance(spec_info, DFlashDraftInput):
            logger.warning(
                "[DFLASH] No DFlashDraftInput found, running normal target decode"
            )
            return self._fallback_decode(batch)

        block_size = self.block_size
        all_verified_id = spec_info.verified_id
        bs = len(batch.reqs)

        # ===== Step 1: Draft forward =====
        draft_tokens, positions = self._draft_forward(batch, all_verified_id)

        # ===== Step 2: Verify with target model =====
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens,
            positions=positions,
            block_size=block_size,
        )

        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        verify_input.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )

        logits_output = batch_result.logits_output

        # ===== Step 3: Verify and accept tokens =====
        logits_output, new_verified_id, num_accepted = verify_input.verify(
            batch, logits_output, self.page_size
        )

        # ===== Step 4: Update state and evict rejected cache =====
        accept_length_list = verify_input.accept_length.cpu().tolist()
        new_hidden_states = logits_output.hidden_states

        # CRITICAL FIX: After _filter_logits(), hidden_states are FILTERED to only accepted tokens.
        # The shape is [total_accepted, hidden], NOT [bs * block_size, hidden].
        # We need to track cumulative offset based on actual accepted tokens per request.
        hidden_offset = 0

        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            acc_len = accept_length_list[i]

            # Update positions
            state["start"] += acc_len + 1

            # Evict rejected noise positions from cache
            self._evict_rejected_cache(batch, req, state, acc_len)

            # Update hidden states
            if new_hidden_states is not None:
                num_tokens = acc_len + 1
                # Use cumulative offset, not i * block_size (hidden states are filtered!)
                start_idx = hidden_offset
                if new_hidden_states.dim() == 2:
                    req_hidden = new_hidden_states[
                        start_idx : start_idx + num_tokens, :
                    ]
                else:
                    req_hidden = new_hidden_states[
                        :, start_idx : start_idx + num_tokens, :
                    ]
                # Update offset for next request
                hidden_offset += num_tokens
                if req_hidden.dim() == 2:
                    req_hidden = req_hidden.unsqueeze(0)

                old_hidden = state.get("accumulated_hidden")
                if old_hidden is not None:
                    state["accumulated_hidden"] = torch.cat(
                        [old_hidden, req_hidden], dim=1
                    )
                else:
                    state["accumulated_hidden"] = req_hidden.clone()

                state["target_hidden"] = state["accumulated_hidden"]

            state["verified_id"] = new_verified_id[i].clone()

        # Create new spec_info for next iteration
        batch.spec_info = DFlashDraftInput(
            hidden_states=None,
            verified_id=new_verified_id.clone(),
            block_size=self.block_size,
        )

        batch.forward_mode = ForwardMode.DECODE

        # Cleanup finished requests
        self._cleanup_finished_requests(batch)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted,
            can_run_cuda_graph=False,
            accept_lens=verify_input.accept_length,
        )

    def _draft_forward(
        self,
        batch: ScheduleBatch,
        all_verified_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft forward pass using the draft model.

        Collects target hidden states and verified IDs per request,
        then runs batched draft forward through model layers.
        """
        bs = len(batch.reqs)
        block_size = self.block_size

        # Collect per-request data
        target_hiddens = []
        verified_ids = []
        ctx_lens = []

        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            target_hidden = state.get("target_hidden")

            if target_hidden is None:
                logger.warning(f"[DFLASH] No target_hidden for request {req.rid}")
                continue

            req_verified_id = state.get("verified_id")
            if req_verified_id is None:
                if all_verified_id.dim() > 0 and all_verified_id.shape[0] > i:
                    req_verified_id = all_verified_id[i]
                else:
                    req_verified_id = all_verified_id.flatten()[0]

            target_hiddens.append(target_hidden)
            verified_ids.append(req_verified_id)
            ctx_len = (
                target_hidden.shape[1]
                if target_hidden.dim() == 3
                else target_hidden.shape[0]
            )
            ctx_lens.append(ctx_len)

        if not target_hiddens:
            # No valid requests
            return torch.tensor([], device=self.device), torch.tensor(
                [], device=self.device
            )

        # Use batched forward through model runner
        return self._batched_draft_forward(
            batch, target_hiddens, verified_ids, ctx_lens
        )

    def _batched_draft_forward(
        self,
        batch: ScheduleBatch,
        target_hiddens: List[torch.Tensor],
        verified_ids: List[torch.Tensor],
        ctx_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched draft forward using torch-based attention.

        Processes each request through the draft model layers with
        non-causal attention (Q from noise, K/V from full sequence).
        """
        bs = len(target_hiddens)
        block_size = self.block_size
        max_ctx_len = max(ctx_lens)
        hidden_size = self.draft_model_runner.model.config.hidden_size

        # Get input dimension (may be num_layers * hidden_size before FC projection)
        first_th = target_hiddens[0]
        input_dim = first_th.shape[-1]

        # Pad and stack target_hiddens to [bs, max_ctx_len, input_dim]
        padded_target_hiddens = torch.zeros(
            bs, max_ctx_len, input_dim, dtype=torch.bfloat16, device=self.device
        )
        ctx_lens_tensor = torch.tensor(ctx_lens, dtype=torch.long, device=self.device)

        for i, (th, cl) in enumerate(zip(target_hiddens, ctx_lens)):
            if th.dim() == 2:
                padded_target_hiddens[i, :cl, :] = th[:cl, :]
            else:
                padded_target_hiddens[i, :cl, :] = th[0, :cl, :]

        # Stack verified_ids to [bs]
        verified_ids_tensor = torch.stack(
            [
                (
                    v
                    if isinstance(v, torch.Tensor) and v.dim() == 0
                    else (
                        v[0]
                        if isinstance(v, torch.Tensor)
                        else torch.tensor(v, device=self.device)
                    )
                )
                for v in verified_ids
            ]
        ).to(dtype=torch.long, device=self.device)

        # Create noise input_ids [bs, block_size]
        block_input_ids = torch.full(
            (bs, block_size), self.mask_token_id, dtype=torch.long, device=self.device
        )
        block_input_ids[:, 0] = verified_ids_tensor

        # Get embedding from draft model
        noise_embedding = self.draft_model_runner.model.model.embed_tokens(
            block_input_ids
        )

        # Project and normalize target hidden ONCE at model level (matching original DFlash)
        if padded_target_hiddens.shape[-1] != hidden_size:
            padded_target_hiddens = self.draft_model_runner.model.model.fc(
                padded_target_hiddens
            )
        padded_target_hiddens = self.draft_model_runner.model.model.hidden_norm(
            padded_target_hiddens
        )

        # Process each request through model layers
        # Using torch fallback attention (forward_batch=None) which correctly implements
        # DFlash pattern: Q from noise only, K/V from full sequence
        all_draft_hidden = []

        for i in range(bs):
            ctx_len = ctx_lens[i]
            total_len = ctx_len + block_size

            # Get valid target hidden [ctx_len, hidden] - already normalized
            th = padded_target_hiddens[i, :ctx_len, :]

            # Get noise embedding [block_size, hidden]
            ne = noise_embedding[i]

            # Concatenate [total_len, hidden]
            combined = torch.cat([th, ne], dim=0)

            # Create positions [total_len]
            positions = torch.arange(total_len, device=self.device)

            # Forward through model layers with torch fallback (forward_batch=None)
            hidden = combined
            for layer in self.draft_model_runner.model.model.layers:
                hidden = layer(positions, hidden, forward_batch=None, ctx_len=ctx_len)

            # Extract noise portion and apply final norm
            noise_hidden = hidden[ctx_len:]
            noise_hidden = self.draft_model_runner.model.model.norm(noise_hidden)
            all_draft_hidden.append(noise_hidden)

        # Stack to [bs, block_size, hidden]
        draft_hidden = torch.stack(all_draft_hidden, dim=0)

        # Get draft logits and tokens [bs, block_size-1]
        draft_hidden_for_logits = draft_hidden[:, 1:, :]  # [bs, block_size-1, hidden]
        lm_head_weight = self.draft_model_runner.model.lm_head.weight
        draft_logits = torch.matmul(draft_hidden_for_logits, lm_head_weight.t())
        block_input_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)

        # Flatten for output
        all_draft_tokens = []
        all_positions = []
        for i in range(bs):
            all_draft_tokens.append(block_input_ids[i])
            current_seq_len = batch.seq_lens[i].item()
            all_positions.append(
                torch.arange(
                    current_seq_len, current_seq_len + block_size, device=self.device
                )
            )

        draft_tokens = torch.cat(all_draft_tokens, dim=0)
        positions = torch.cat(all_positions, dim=0)
        return draft_tokens, positions

    def _evict_rejected_cache(
        self,
        batch: ScheduleBatch,
        req,
        state: dict,
        accept_length: int,
    ):
        """Evict rejected noise positions from cache after verification."""
        noise_cache_locs = state.get("noise_cache_locs")
        if noise_cache_locs is None:
            return

        rejected_count = self.block_size - accept_length - 1
        if rejected_count > 0:
            # Free rejected positions
            rejected_locs = noise_cache_locs[accept_length + 1 :]
            if len(rejected_locs) > 0:
                batch.token_to_kv_pool_allocator.free(rejected_locs)

        # Clear tracked noise positions
        state["noise_cache_locs"] = None

    def _fallback_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Fallback decode without speculation."""
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
        batch.input_ids = batch.output_ids
        batch.output_ids = None
        bs = len(batch.reqs)
        batch.seq_lens = batch.seq_lens + 1
        batch.seq_lens_cpu = batch.seq_lens_cpu + 1
        batch.seq_lens_sum += bs
        for req in batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1
        model_worker_batch = batch.get_model_worker_batch()
        return self.target_worker.forward_batch_generation(model_worker_batch)

    def _cleanup_finished_requests(self, batch: ScheduleBatch):
        """Clean up state for finished requests."""
        active_rids = {req.rid for req in batch.reqs}
        finished_rids = [rid for rid in self._request_state if rid not in active_rids]
        for rid in finished_rids:
            del self._request_state[rid]
