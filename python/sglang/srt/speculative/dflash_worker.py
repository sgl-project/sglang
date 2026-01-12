# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License")

"""DFlash speculative decoding worker using TpModelWorker infrastructure.

DFlash draft model generates a full block of tokens in one forward pass.
Supports two attention modes:
1. Torch fallback: Custom torch-based non-causal attention (original)
2. RadixAttention: Uses optimized backends with incremental KV caching
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache

logger = logging.getLogger(__name__)


class DFlashWorker(TpModelWorker):

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

        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share allocator with target worker
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
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

        When use_radix_attention is True, initializes the FlashAttention
        backend for RadixAttention. Otherwise uses torch fallback.
        """
        # Check if we should use RadixAttention mode (enabled by default)
        self.use_radix_attention = getattr(
            self.server_args, "dflash_use_radix_attention", True
        )

        if self.use_radix_attention:
            try:
                from sglang.srt.layers.attention.flashattention_backend import (
                    FlashAttentionBackend,
                )

                # Create FlashAttention backend for DRAFT_EXTEND_V2 mode with ENCODER_ONLY attention
                self.draft_attn_backend = FlashAttentionBackend(
                    model_runner=self.draft_model_runner,
                    skip_prefill=False,
                )
                self.draft_model_runner.attn_backend = self.draft_attn_backend
                logger.info(
                    "DFlash: RadixAttention mode enabled with FlashAttention backend (DRAFT_EXTEND_V2 + ENCODER_ONLY)"
                )
            except Exception as e:
                logger.warning(
                    f"DFlash: Failed to initialize FlashAttention backend: {e}"
                )
                logger.info("DFlash: Falling back to torch attention")
                self.use_radix_attention = False
                self.draft_attn_backend = None
        else:
            logger.info("DFlash: Using torch fallback attention")
            self.draft_attn_backend = None

    def init_cuda_graphs(self):
        """Initialize CUDA graphs for draft model using DRAFT_EXTEND_V2 mode.

        Uses DRAFT_EXTEND_V2 mode with ENCODER_ONLY attention type for
        non-causal attention.
        """
        self.cuda_graph_runner = None

        if not self.use_radix_attention:
            logger.info("DFlash draft model: using eager mode (torch fallback)")
            return

        if self.server_args.disable_cuda_graph:
            logger.info("DFlash draft model: CUDA graph disabled, using eager mode")
            return

        try:
            from sglang.srt.speculative.dflash_draft_cuda_graph_runner import (
                DFlashDraftCudaGraphRunner,
            )

            self.cuda_graph_runner = DFlashDraftCudaGraphRunner(self)
            logger.info(
                "DFlash draft model: CUDA graph initialized (DRAFT_EXTEND_V2 mode)"
            )
        except Exception as e:
            raise RuntimeError(f"DFlash CUDA graph init failed: {e}")

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
                "kv_cache_locs": None,  # Cache locations for projected K/V
                "cached_len": 0,  # Number of tokens cached in draft KV
            }
        return self._request_state[rid]

    def _cache_hidden_to_kv_batched(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
    ):
        """
        Project hidden states to K/V and cache them in draft model's KV pool.

        Args:
            hidden_states: Hidden states from target model [num_tokens, hidden_dim]
                          (already projected through fc and hidden_norm)
            positions: Position IDs for RoPE [num_tokens]
            cache_locs: Cache locations in KV pool [num_tokens]
        """
        if not self.use_radix_attention:
            return

        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool

        # Project hidden states to K/V for each layer
        for layer_idx, layer in enumerate(self.draft_model_runner.model.model.layers):
            attn = layer.self_attn

            # Use the project_hidden_to_kv method from the attention layer
            k, v = attn.project_hidden_to_kv(hidden_states, positions)

            # Store in KV cache at the specified locations
            # Use direct indexing to bypass JIT kernel which has compatibility issues
            # with the K/V tensor layout from project_hidden_to_kv
            k_buffer, v_buffer = token_to_kv_pool.get_kv_buffer(attn.layer_id)
            k_buffer[cache_locs] = k
            v_buffer[cache_locs] = v

    def _cache_hidden_to_kv_per_request(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        req_idx: int,
        num_tokens: int,
        token_offset: int,
    ):
        """
        Project and cache hidden states for a single request.

        Args:
            batch: ScheduleBatch with cache allocation info
            hidden_states: Hidden states [total_batch_tokens, hidden_dim]
            req_idx: Request index in batch
            num_tokens: Number of tokens for this request
            token_offset: Offset into hidden_states for this request
        """
        if not self.use_radix_attention:
            return

        req = batch.reqs[req_idx]
        state = self._get_request_state(req.rid)

        # Get cache locations for this request from req_to_token_pool
        # These are the same locations used by the attention backend
        req_pool_idx = req.req_pool_idx
        prefix_len = getattr(req, "extend_prefix_len", 0)

        # For prefill, cache locations are in out_cache_loc
        if hasattr(batch, "out_cache_loc") and batch.out_cache_loc is not None:
            # Calculate the start and end indices for this request's cache locations
            if hasattr(batch, "extend_lens"):
                req_start = sum(batch.extend_lens[:req_idx])
            else:
                req_start = token_offset

            cache_locs = batch.out_cache_loc[req_start : req_start + num_tokens]

            # Get positions: these are the token positions in the sequence
            # For extend mode, positions start at prefix_len
            positions = torch.arange(
                prefix_len,
                prefix_len + num_tokens,
                device=self.device,
                dtype=torch.int64,
            )

            # Extract hidden states for this request
            if hidden_states.dim() == 2:
                req_hidden = hidden_states[token_offset : token_offset + num_tokens]
            else:
                req_hidden = hidden_states[0, token_offset : token_offset + num_tokens]

            # Project and normalize (if not already done)
            hidden_size = self.draft_model_runner.model.config.hidden_size
            if req_hidden.shape[-1] != hidden_size:
                req_hidden = self.draft_model_runner.model.model.fc(req_hidden)
            req_hidden = self.draft_model_runner.model.model.hidden_norm(req_hidden)

            # Cache K/V for all layers
            self._cache_hidden_to_kv_batched(req_hidden, positions, cache_locs)

            # Track cache locations and cached length for this request
            if state["kv_cache_locs"] is None:
                state["kv_cache_locs"] = cache_locs.clone()
            else:
                state["kv_cache_locs"] = torch.cat([state["kv_cache_locs"], cache_locs])
            state["cached_len"] = prefix_len + num_tokens

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Main entry point for forward pass."""
        if batch.forward_mode.is_extend():
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    def _forward_extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Prefill phase - capture hidden states from target model and cache K/V."""
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        logits_output = batch_result.logits_output
        next_token_ids = batch_result.next_token_ids
        hidden_states = logits_output.hidden_states

        # Store hidden states per request and cache K/V for RadixAttention mode
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

                # Store hidden states in KV pool (for unified hidden cache)
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
                            if (
                                not hasattr(req, "hidden_indices")
                                or req.hidden_indices is None
                            ):
                                req.hidden_indices = req_cache_loc.clone()
                            else:
                                req.hidden_indices = torch.cat(
                                    [req.hidden_indices, req_cache_loc]
                                )

                # Cache projected K/V for RadixAttention mode
                # Projects target hidden states to K/V and caches them at the same
                # locations as the target KV cache (shared allocator pattern)
                if self.use_radix_attention and hidden_states is not None:
                    self._cache_hidden_to_kv_per_request(
                        batch, hidden_states, i, new_token_count, token_offset
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
        can_run_cuda_graph = batch_result.can_run_cuda_graph

        # ===== Step 3: Verify and accept tokens =====
        logits_output, new_verified_id, num_accepted = verify_input.verify(
            batch, logits_output, self.page_size
        )

        # ===== Step 4: Update state and evict rejected cache =====
        accept_length_list = verify_input.accept_length.cpu().tolist()
        new_hidden_states = logits_output.hidden_states

        hidden_offset = 0

        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            acc_len = accept_length_list[i]

            # Update positions
            old_start = state["start"]
            state["start"] += acc_len + 1

            # Evict rejected noise positions from cache
            self._evict_rejected_cache(batch, req, state, acc_len)

            # Update hidden states and incrementally cache K/V
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

                # Incrementally cache K/V for newly accepted tokens (RadixAttention mode)
                # This ensures the draft model's KV cache is updated with newly accepted tokens
                if self.use_radix_attention and num_tokens > 0:
                    self._cache_incremental_kv(
                        batch, req, state, req_hidden.squeeze(0), old_start, num_tokens
                    )

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
            can_run_cuda_graph=can_run_cuda_graph,
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

        # Try RadixAttention eager mode
        if self.use_radix_attention:
            # CUDA graph support is currently disabled, use eager mode
            if self.cuda_graph_runner is not None:
                return self._draft_forward_with_cuda_graph(
                    batch, verified_ids, ctx_lens
                )
            else:
                return self._draft_forward_radix_eager_simple(
                    batch, verified_ids, ctx_lens
                )

        # Use batched forward through model runner (torch fallback)
        return self._batched_draft_forward(
            batch, target_hiddens, verified_ids, ctx_lens
        )

    def _draft_forward_radix_eager_simple(
        self,
        batch: ScheduleBatch,
        verified_ids: List[torch.Tensor],
        ctx_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager mode draft forward using RadixAttention.

        Uses the cached K/V from projected target hidden states.
        Q comes from noise tokens, K/V comes from cached context.
        """
        bs = len(verified_ids)
        block_size = self.block_size
        num_tokens = bs * block_size

        # Stack verified_ids to tensor
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

        # Create input_ids: [verified_id, mask, mask, ..., mask] for each request
        input_ids = torch.full(
            (bs, block_size), self.mask_token_id, dtype=torch.long, device=self.device
        )
        input_ids[:, 0] = verified_ids_tensor

        # Build positions for noise tokens (starting from current seq_len)
        all_positions = []
        for i in range(bs):
            current_seq_len = batch.seq_lens[i].item()
            all_positions.append(
                torch.arange(
                    current_seq_len, current_seq_len + block_size, device=self.device
                )
            )
        positions = torch.cat(all_positions, dim=0)

        # Allocate cache locations for noise tokens from shared allocator
        out_cache_loc = self.token_to_kv_pool_allocator.alloc(num_tokens)
        if out_cache_loc is None:
            logger.warning("DFlash: Failed to allocate cache, falling back to torch")
            target_hiddens = []
            for req in batch.reqs:
                state = self._get_request_state(req.rid)
                target_hidden = state.get("target_hidden")
                if target_hidden is not None:
                    target_hiddens.append(target_hidden)
            return self._batched_draft_forward(
                batch, target_hiddens, verified_ids, ctx_lens
            )

        # Build seq_lens for extend: context_len + block_size
        # For RadixAttention extend mode, seq_lens = total sequence length
        seq_lens = batch.seq_lens[:bs] + block_size
        seq_lens_cpu = seq_lens.cpu()

        # Build extend_seq_lens (number of new tokens) and extend_prefix_lens (cached)
        extend_seq_lens = torch.full(
            (bs,), block_size, dtype=torch.int32, device=self.device
        )
        extend_prefix_lens = batch.seq_lens[:bs].to(dtype=torch.int32)

        # Compute extend_start_loc (cumulative start positions for each request's new tokens)
        extend_start_loc = torch.zeros(bs, dtype=torch.int32, device=self.device)
        extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)

        # Embed noise tokens
        noise_embeds = self.draft_model_runner.model.model.embed_tokens(input_ids)
        hidden_states = noise_embeds.view(num_tokens, -1)

        # Create spec_info for DRAFT_EXTEND_V2 mode
        spec_info = DFlashDraftInput(
            hidden_states=None,
            verified_id=verified_ids_tensor,
            block_size=block_size,
            ctx_lens=torch.tensor(ctx_lens, dtype=torch.int32, device=self.device),
            num_tokens_per_batch=block_size,
        )

        # Use DRAFT_EXTEND_V2 mode with ENCODER_ONLY attention type for non-causal attention
        # This matches the original DFlash pattern: Q from noise, K/V from [context, noise]
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=bs,
            input_ids=input_ids.flatten(),
            req_pool_indices=batch.req_pool_indices[:bs],
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens.cpu().tolist(),
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens.cpu().tolist(),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            req_to_token_pool=self.draft_model_runner.req_to_token_pool,
            token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
            attn_backend=self.draft_attn_backend,
            return_logprob=False,
            positions=positions,
            spec_info=spec_info,
        )

        # Update req_to_token mapping so attention can find noise K/V
        # Maps positions [prefix_len, prefix_len + block_size) to out_cache_loc for each request
        assign_req_to_token_pool_func(
            batch.req_pool_indices[:bs],
            self.draft_model_runner.req_to_token_pool.req_to_token,
            batch.seq_lens[:bs],  # start_offset = current seq_lens (prefix)
            seq_lens,  # end_offset = seq_lens + block_size
            out_cache_loc,
            bs,
        )

        # Initialize attention backend metadata
        if self.draft_attn_backend is not None:
            self.draft_attn_backend.init_forward_metadata(forward_batch)

        # Process through model layers with RadixAttention
        for layer in self.draft_model_runner.model.model.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                ctx_len=0,  # Context is pre-cached
            )

        # Apply final norm
        hidden_states = self.draft_model_runner.model.model.norm(hidden_states)

        # Compute logits
        hidden_states = hidden_states.view(bs, block_size, -1)
        lm_head_weight = self.draft_model_runner.model.lm_head.weight
        logits = torch.matmul(hidden_states[:, 1:, :], lm_head_weight.t())

        # Get draft tokens
        input_ids[:, 1:] = torch.argmax(logits, dim=-1)

        # Free the allocated cache locations (noise tokens are temporary)
        self.token_to_kv_pool_allocator.free(out_cache_loc)

        return input_ids.flatten(), positions

    def _draft_forward_with_cuda_graph(
        self,
        batch: ScheduleBatch,
        verified_ids: List[torch.Tensor],
        ctx_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draft forward using CUDA graph runner with RadixAttention.

        K/V is already cached from projected hidden states.
        """
        bs = len(verified_ids)
        block_size = self.block_size

        # Stack verified_ids to tensor
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

        ctx_lens_tensor = torch.tensor(ctx_lens, dtype=torch.int32, device=self.device)

        # Build positions for noise tokens
        all_positions = []
        for i in range(bs):
            current_seq_len = batch.seq_lens[i].item()
            all_positions.append(
                torch.arange(
                    current_seq_len, current_seq_len + block_size, device=self.device
                )
            )
        positions = torch.cat(all_positions, dim=0)

        # Allocate cache for noise tokens from DRAFT model's allocator
        num_tokens = bs * block_size
        draft_allocator = self.draft_model_runner.token_to_kv_pool_allocator
        out_cache_loc = draft_allocator.alloc(num_tokens)
        if out_cache_loc is None:
            raise RuntimeError(
                f"DFlash: Failed to allocate {num_tokens} tokens from draft KV cache."
            )

        try:
            # Try CUDA graph replay
            if self.cuda_graph_runner is not None and self.cuda_graph_runner.can_run(
                bs
            ):
                draft_tokens, out_positions = self.cuda_graph_runner.replay(
                    batch,
                    verified_ids_tensor,
                    ctx_lens_tensor,
                    positions,
                    out_cache_loc,
                )
                return draft_tokens, out_positions

            # Fall back to eager mode
            logger.debug("DFlash: CUDA graph not available, falling back to eager mode")
            return self._draft_forward_radix_eager(
                batch, verified_ids_tensor, ctx_lens_tensor, positions, out_cache_loc
            )
        finally:
            # Free noise token cache locations after draft forward completes
            # The K/V was only needed during attention, not stored permanently
            draft_allocator.free(out_cache_loc)

    def _draft_forward_radix_eager(
        self,
        batch: ScheduleBatch,
        verified_ids: torch.Tensor,
        ctx_lens: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager mode draft forward with RadixAttention.

        Used when CUDA graph is not available.
        """
        bs = verified_ids.shape[0]
        block_size = self.block_size
        num_tokens = bs * block_size

        # Create input_ids for noise tokens
        input_ids = torch.full(
            (bs, block_size), self.mask_token_id, dtype=torch.long, device=self.device
        )
        input_ids[:, 0] = verified_ids

        # Embed noise tokens
        noise_embeds = self.draft_model_runner.model.model.embed_tokens(input_ids)
        hidden_states = noise_embeds.view(num_tokens, -1)

        # Build seq_lens: context_len + block_size (total sequence length)
        seq_lens = batch.seq_lens[:bs] + block_size
        seq_lens_cpu = seq_lens.cpu()

        # Build spec info for DFlash
        spec_info = DFlashDraftInput(
            hidden_states=None,
            verified_id=verified_ids,
            block_size=block_size,
            ctx_lens=ctx_lens,
        )

        # Use DECODE mode with spec_info - triggers topk > 1 path in FlashAttention
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids.flatten(),
            req_pool_indices=batch.req_pool_indices[:bs],
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            req_to_token_pool=self.draft_model_runner.req_to_token_pool,
            token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
            attn_backend=self.draft_attn_backend,
            return_logprob=False,
            positions=positions,
            spec_algorithm=self.draft_model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        # Update req_to_token mapping so attention can find noise K/V
        assign_req_to_token_pool_func(
            batch.req_pool_indices[:bs],
            self.draft_model_runner.req_to_token_pool.req_to_token,
            batch.seq_lens[:bs],
            seq_lens,
            out_cache_loc,
            bs,
        )

        # Initialize attention backend metadata
        if self.draft_attn_backend is not None:
            self.draft_attn_backend.init_forward_metadata(forward_batch)

        # Process through model layers with RadixAttention
        for layer in self.draft_model_runner.model.model.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                ctx_len=0,  # Context is pre-cached
            )

        # Apply final norm
        hidden_states = self.draft_model_runner.model.model.norm(hidden_states)

        # Compute logits
        hidden_states = hidden_states.view(bs, block_size, -1)
        lm_head_weight = self.draft_model_runner.model.lm_head.weight
        logits = torch.matmul(hidden_states[:, 1:, :], lm_head_weight.t())

        # Get draft tokens
        input_ids[:, 1:] = torch.argmax(logits, dim=-1)

        return input_ids.flatten(), positions

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

    def _cache_incremental_kv(
        self,
        batch: ScheduleBatch,
        req,
        state: dict,
        hidden_states: torch.Tensor,
        old_start: int,
        num_tokens: int,
    ):
        """
        Incrementally cache K/V for newly accepted tokens.

        This is called after verification to project and cache only the
        hidden states of newly accepted tokens, avoiding recomputation
        of the entire context.

        Args:
            batch: ScheduleBatch
            req: Request object
            state: Per-request state dict
            hidden_states: Hidden states for accepted tokens [num_tokens, hidden_dim]
            old_start: Start position before this decode round
            num_tokens: Number of newly accepted tokens
        """
        if not self.use_radix_attention:
            return

        # Get cache locations for the newly accepted tokens
        # These should have been allocated during verify
        req_pool_idx = req.req_pool_idx
        req_to_token_pool = batch.req_to_token_pool

        # Get the token indices for the new positions
        new_positions = torch.arange(
            old_start, old_start + num_tokens, device=self.device
        )
        cache_locs = req_to_token_pool.req_to_token[
            req_pool_idx, old_start : old_start + num_tokens
        ]

        # Project and normalize hidden states
        hidden_size = self.draft_model_runner.model.config.hidden_size
        if hidden_states.shape[-1] != hidden_size:
            hidden_states = self.draft_model_runner.model.model.fc(hidden_states)
        hidden_states = self.draft_model_runner.model.model.hidden_norm(hidden_states)

        # Cache K/V for all layers
        self._cache_hidden_to_kv_batched(hidden_states, new_positions, cache_locs)

        # Update tracked cache locations and cached length
        if state["kv_cache_locs"] is None:
            state["kv_cache_locs"] = cache_locs.clone()
        else:
            state["kv_cache_locs"] = torch.cat([state["kv_cache_locs"], cache_locs])
        state["cached_len"] = old_start + num_tokens

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
