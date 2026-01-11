# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License")

"""DFlash draft model CUDA graph runner using RadixAttention.

This module implements CUDA graph capture and replay for DFlash draft model
using the attention backend's native CUDA graph support. Unlike the torch
fallback mode, this uses RadixAttention with cached K/V for efficient
batched inference.

Key features:
- Uses attention backend's CUDA graph infrastructure
- Supports variable batch sizes with padding
- Incrementally cached K/V from projected hidden states
- Non-causal attention for DFlash parallel block drafting
"""

from __future__ import annotations

import bisect
import logging
from typing import TYPE_CHECKING, Callable, Tuple

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.dflash_info import DFlashDraftInput
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.dflash_worker import DFlashWorker

logger = logging.getLogger(__name__)


class DFlashDraftCudaGraphRunner:
    """CUDA graph runner for DFlash draft model with RadixAttention.

    This runner captures CUDA graphs for the draft forward pass using
    RadixAttention and the standard attention backend infrastructure.
    """

    def __init__(self, dflash_worker: DFlashWorker):
        # Parse args
        self.dflash_worker = dflash_worker
        self.model_runner = model_runner = dflash_worker.draft_model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.block_size = dflash_worker.block_size
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)

        # DFlash specific: num_tokens = bs * block_size
        self.num_tokens_per_bs = self.block_size
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        # Initialize attention backend for CUDA graph
        if (
            hasattr(self.model_runner, "attn_backend")
            and self.model_runner.attn_backend is not None
        ):
            self.model_runner.attn_backend.init_cuda_graph_state(
                self.max_bs, self.max_num_token
            )
            self.seq_len_fill_value = (
                self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
            )
        else:
            # Fallback for when attention backend is not available
            logger.warning("DFlash CUDA graph: attention backend not available")
            self.seq_len_fill_value = 1

        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )
        self.extend_seq_lens_cpu = [self.block_size] * self.max_bs
        self.extend_prefix_lens_cpu = [self.seq_len_fill_value] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        # Graph inputs - pre-allocated buffers
        with torch.device(model_runner.device):
            # Input IDs for noise tokens (mask tokens)
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)

            # Request pool indices
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)

            # Output cache locations for noise tokens
            self.out_cache_loc = torch.zeros(
                (self.max_num_token,), dtype=self._cache_loc_dtype()
            )

            # Positions for noise tokens
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)

            # Sequence lengths (context length for each request)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )

            # Extend sequence lengths (block_size for each request)
            self.extend_seq_lens = torch.full(
                (self.max_bs,), self.block_size, dtype=torch.int32
            )

            # Extend prefix lengths (context length for each request)
            self.extend_prefix_lens = torch.zeros((self.max_bs,), dtype=torch.int32)

            # Context lengths per request (for DFlash pattern)
            self.ctx_lens = torch.zeros((self.max_bs,), dtype=torch.int32)

            # Verified IDs from previous round
            self.verified_ids = torch.zeros((self.max_bs,), dtype=torch.int64)

            # Global tokens for DP
            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    self.global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    self.global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    self.global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                self.global_num_tokens_gpu = None
                self.global_num_tokens_for_logprob_gpu = None

        # Capture CUDA graphs
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"DFlash capture CUDA graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _cache_loc_dtype(self):
        return torch.int64

    def can_run(self, batch_size: int) -> bool:
        """Check if CUDA graph can be used for this batch size."""
        if not self.dflash_worker.use_radix_attention:
            return False

        is_bs_supported = (
            batch_size in self.graphs
            if self.disable_padding
            else batch_size <= self.max_bs
        )

        return is_bs_supported

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self, forward_batch):
        self.graphs[self.bs].replay()

    def capture(self):
        """Capture CUDA graphs for all batch sizes."""
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        """Capture CUDA graph for a specific batch size."""
        graph = self._create_graph()
        stream = self.stream
        num_tokens = num_seqs * self.num_tokens_per_bs

        # Graph inputs for this batch size
        req_pool_indices = self.req_pool_indices[:num_seqs]
        seq_lens = self.seq_lens[:num_seqs]
        seq_lens_cpu = self.seq_lens_cpu[:num_seqs]
        extend_seq_lens = self.extend_seq_lens[:num_seqs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:num_seqs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        ctx_lens = self.ctx_lens[:num_seqs]
        verified_ids = self.verified_ids[:num_seqs]
        input_ids = self.input_ids[:num_tokens]

        # DP buffer setup
        if self.require_mlp_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_num_tokens = self.global_num_tokens_gpu
            global_dp_buffer_len = num_tokens * self.dp_size
            global_num_tokens_for_logprob = self.global_num_tokens_for_logprob_gpu
        elif self.require_attn_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_num_tokens = self.global_num_tokens_gpu
            global_dp_buffer_len = num_tokens
            global_num_tokens_for_logprob = self.global_num_tokens_for_logprob_gpu
        else:
            global_num_tokens = None
            global_dp_buffer_len = None
            global_num_tokens_for_logprob = None

        # Extend prefix lens (context length for each request, to be filled at capture time)
        extend_prefix_lens = self.extend_prefix_lens[:num_seqs]
        extend_prefix_lens_cpu = self.extend_prefix_lens_cpu[:num_seqs]

        # Create spec_info with num_tokens_per_batch for DRAFT_EXTEND_V2 CUDA graph
        spec_info = DFlashDraftInput(
            hidden_states=None,
            verified_id=verified_ids,
            block_size=self.block_size,
            ctx_lens=ctx_lens,
            num_tokens_per_batch=self.block_size,
        )

        # Forward batch for capture - use DRAFT_EXTEND_V2 mode with ENCODER_ONLY attention
        # This matches the original DFlash pattern: Q from noise, K/V from [context, noise]
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=num_seqs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,  # Required for RadixAttention
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_info=spec_info,
        )

        # Initialize attention backend for CUDA graph capture
        if (
            hasattr(self.model_runner, "attn_backend")
            and self.model_runner.attn_backend is not None
        ):
            self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
                bs=num_seqs,
                num_tokens=num_tokens,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                spec_info=spec_info,
            )

        # Run once function for capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                (
                    forward_batch.dp_padding_mode.is_max_len()
                    if forward_batch.dp_padding_mode
                    else False
                ),
            )
            set_is_extend_in_batch(False)

            # Run draft forward with RadixAttention
            ret = self._draft_forward_radix(forward_batch)
            return ret

        self.deepep_adapter.capture(is_extend_in_batch=False)

        # Warmup runs
        self._capture_init(run_once)

        # Capture graph
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def _draft_forward_radix(
        self, forward_batch: ForwardBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draft forward using RadixAttention with pre-cached K/V.

        The context K/V has been pre-cached from projected hidden states.
        This function only processes noise tokens through the model.
        """
        bs = forward_batch.batch_size
        block_size = self.block_size
        num_tokens = bs * block_size

        # Get input_ids for noise tokens (mask tokens with first token as verified_id)
        input_ids = forward_batch.input_ids[:num_tokens]

        # Get positions for noise tokens
        positions = forward_batch.positions[:num_tokens]

        # Embed noise tokens
        noise_embeds = self.model_runner.model.model.embed_tokens(
            input_ids.view(bs, block_size)
        )

        # Process through model layers using RadixAttention
        # The context K/V is automatically read from cache by attention backend
        hidden_states = noise_embeds.view(num_tokens, -1)

        for layer in self.model_runner.model.model.layers:
            # Forward with RadixAttention mode (ctx_len=0 means all context is cached)
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                ctx_len=0,  # Context is pre-cached
            )

        # Apply final norm
        hidden_states = self.model_runner.model.model.norm(hidden_states)

        # Compute logits
        hidden_states = hidden_states.view(bs, block_size, -1)
        lm_head_weight = self.model_runner.model.lm_head.weight
        logits = torch.matmul(hidden_states[:, 1:, :], lm_head_weight.t())

        # Get draft tokens
        draft_tokens = input_ids.view(bs, block_size).clone()
        draft_tokens[:, 1:] = torch.argmax(logits, dim=-1)

        return draft_tokens.flatten(), positions

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        """Post-process output for raw batch size (remove padding)."""
        draft_tokens, positions = out
        raw_num_token = raw_bs * self.num_tokens_per_bs
        return draft_tokens[:raw_num_token], positions[:raw_num_token]

    def replay(
        self,
        batch,  # ScheduleBatch
        verified_ids: torch.Tensor,
        ctx_lens: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replay CUDA graph for draft forward.

        Args:
            batch: ScheduleBatch with batch metadata
            verified_ids: Verified token IDs from previous round [bs]
            ctx_lens: Context lengths for each request [bs]
            positions: Pre-computed positions for noise tokens [bs * block_size]
            out_cache_loc: Pre-allocated cache locations [bs * block_size]

        Returns:
            Tuple of (draft_tokens, positions)
        """
        self.deepep_adapter.replay()

        raw_bs = verified_ids.shape[0]
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Determine batch size with padding
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]

        if bs != raw_bs:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()
            self.positions.zero_()
            self.input_ids.fill_(self.dflash_worker.mask_token_id)

        num_tokens = bs * self.num_tokens_per_bs

        # Compute extended seq_lens (context + noise tokens)
        # This is needed because the draft model attends to [context, noise]
        extended_seq_lens = batch.seq_lens[:raw_bs] + self.block_size

        # Copy inputs to graph buffers
        # IMPORTANT: seq_lens must include noise tokens for correct attention!
        self.seq_lens[:raw_bs].copy_(extended_seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(out_cache_loc)
        self.positions[:raw_num_token].copy_(positions)
        self.req_pool_indices[:raw_bs].copy_(batch.req_pool_indices[:raw_bs])
        self.verified_ids[:raw_bs].copy_(verified_ids)
        self.ctx_lens[:raw_bs].copy_(ctx_lens)

        # Build input_ids: [mask_token_id] * block_size with first token = verified_id
        for i in range(raw_bs):
            start_idx = i * self.block_size
            self.input_ids[start_idx] = verified_ids[i]
            self.input_ids[start_idx + 1 : start_idx + self.block_size] = (
                self.dflash_worker.mask_token_id
            )

        # DP buffer update
        if self.require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * self.num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs * self.num_tokens_per_bs)

        # Get seq_lens_cpu
        seq_lens_cpu = None
        if hasattr(batch, "seq_lens_cpu") and batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(batch.seq_lens_cpu[:raw_bs])
            seq_lens_cpu = self.seq_lens_cpu[:bs]

        # Update req_to_token mapping so attention can find noise K/V
        # This must be done OUTSIDE the cuda graph (not captured)
        assign_req_to_token_pool_func(
            batch.req_pool_indices[:raw_bs],
            self.model_runner.req_to_token_pool.req_to_token,
            batch.seq_lens[:raw_bs],  # start_offset = current seq_lens (prefix)
            extended_seq_lens,  # end_offset = seq_lens + block_size
            out_cache_loc,
            raw_bs,
        )

        # Update extend_prefix_lens for replay (context length per request)
        self.extend_prefix_lens[:raw_bs].copy_(batch.seq_lens[:raw_bs])

        # Create spec_info for DRAFT_EXTEND_V2 CUDA graph replay
        spec_info = DFlashDraftInput(
            hidden_states=None,
            verified_id=verified_ids,
            block_size=self.block_size,
            ctx_lens=ctx_lens,
            num_tokens_per_batch=self.block_size,
        )

        # Update attention backend metadata for replay
        # self.seq_lens already has the extended values (context + noise)
        # Create extended seq_lens_cpu to match
        extended_seq_lens_cpu = self.seq_lens[:bs].cpu()

        if (
            hasattr(self.model_runner, "attn_backend")
            and self.model_runner.attn_backend is not None
        ):
            self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
                bs=bs,
                req_pool_indices=self.req_pool_indices[:bs],
                seq_lens=self.seq_lens[:bs],
                seq_lens_sum=self.seq_lens[:bs].sum().item(),
                encoder_lens=None,
                forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                spec_info=spec_info,
                seq_lens_cpu=extended_seq_lens_cpu,
            )

        self.raw_bs = raw_bs
        self.bs = bs

        # Replay graph
        self._replay(None)
        out = self.output_buffers[bs]

        # Remove padding if needed
        if bs != raw_bs:
            out = self._postprocess_output_to_raw_bs(out, raw_bs)

        return out
