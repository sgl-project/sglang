# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License")

"""CUDA Graph Runner for DFlash draft model.

Captures and replays CUDA graphs for the DFlash attention pattern,
which requires special handling for the dual-input attention.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner, model_capture_mode
from sglang.srt.utils.common import get_available_gpu_memory

if TYPE_CHECKING:
    from sglang.srt.speculative.dflash_worker import DFlashWorker

logger = logging.getLogger(__name__)


class DFlashDraftCudaGraphRunner:
    """CUDA Graph runner for DFlash draft model.
    
    Preallocates buffers and captures CUDA graphs for the DFlash attention
    pattern, handling variable context lengths through padding and masking.
    """
    
    def __init__(
        self,
        dflash_worker: "DFlashWorker",
        max_bs: int = 32,
        max_ctx_len: int = 4096,
    ):
        self.dflash_worker = dflash_worker
        self.device = dflash_worker.device
        self.block_size = dflash_worker.block_size
        self.max_bs = max_bs
        self.max_ctx_len = max_ctx_len
        self.max_total_len = max_ctx_len + self.block_size
        
        # Get model config
        self.draft_model = dflash_worker.draft_model
        self.hidden_size = self.draft_model.model.hidden_size
        self.num_layers = len(self.draft_model.model.layers)
        
        # Get attention config from first layer
        first_layer = self.draft_model.model.layers[0]
        self.num_heads = first_layer.self_attn.num_heads
        self.num_kv_heads = first_layer.self_attn.num_kv_heads
        self.head_dim = first_layer.self_attn.head_dim
        self.q_size = first_layer.self_attn.q_size
        self.kv_size = first_layer.self_attn.kv_size
        self.scaling = first_layer.self_attn.scaling
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # Input dimension (num_target_layers * hidden_size before FC projection)
        self.num_target_layers = len(self.draft_model.model.target_layer_ids)
        self.input_dim = self.num_target_layers * self.hidden_size
        
        # CUDA graph storage
        self.graphs = {}  # batch_size -> CUDAGraph
        self.stream = torch.cuda.Stream()
        
        # Batch size list for capture
        self.capture_bs_list = self._get_capture_bs_list()
        
        # Initialize buffers and capture graphs
        self._init_buffers()
        self._capture_graphs()
    
    def _get_capture_bs_list(self) -> List[int]:
        """Get list of batch sizes to capture."""
        bs_list = []
        bs = 1
        while bs <= self.max_bs:
            bs_list.append(bs)
            if bs < 4:
                bs += 1
            elif bs < 16:
                bs += 2
            else:
                bs += 8
        return bs_list
    
    def _init_buffers(self):
        """Initialize preallocated buffers for CUDA graph capture."""
        bs = self.max_bs
        block_size = self.block_size
        max_ctx_len = self.max_ctx_len
        max_total_len = self.max_total_len
        hidden_size = self.hidden_size
        
        dtype = torch.bfloat16
        
        # Input buffers (target_hidden has input_dim = num_layers * hidden_size before FC)
        self.target_hidden_buffer = torch.zeros(
            bs, max_ctx_len, self.input_dim, dtype=dtype, device=self.device
        )
        self.noise_embedding_buffer = torch.zeros(
            bs, block_size, hidden_size, dtype=dtype, device=self.device
        )
        self.ctx_lens_buffer = torch.zeros(bs, dtype=torch.long, device=self.device)
        self.verified_ids_buffer = torch.zeros(bs, dtype=torch.long, device=self.device)
        
        # Block input IDs buffer
        self.block_input_ids_buffer = torch.zeros(
            bs, block_size, dtype=torch.long, device=self.device
        )
        
        # Attention mask buffer
        self.attn_mask_buffer = torch.zeros(
            bs, 1, block_size, max_total_len, dtype=dtype, device=self.device
        )
        
        # Per-layer intermediate buffers
        # These are reused across layers
        self.noise_hidden_buffer = torch.zeros(
            bs, block_size, hidden_size, dtype=dtype, device=self.device
        )
        
        # QKV buffers
        self.q_buffer = torch.zeros(
            bs, self.num_heads, block_size, self.head_dim, dtype=dtype, device=self.device
        )
        self.k_buffer = torch.zeros(
            bs, self.num_heads, max_total_len, self.head_dim, dtype=dtype, device=self.device
        )
        self.v_buffer = torch.zeros(
            bs, self.num_heads, max_total_len, self.head_dim, dtype=dtype, device=self.device
        )
        
        # Attention output buffer
        self.attn_weights_buffer = torch.zeros(
            bs, self.num_heads, block_size, max_total_len, dtype=dtype, device=self.device
        )
        self.attn_output_buffer = torch.zeros(
            bs, block_size, hidden_size, dtype=dtype, device=self.device
        )
        
        # Output buffer
        self.draft_hidden_buffer = torch.zeros(
            bs, block_size, hidden_size, dtype=dtype, device=self.device
        )
        self.draft_logits_buffer = torch.zeros(
            bs, block_size - 1, self.draft_model.config.vocab_size,
            dtype=torch.float, device=self.device
        )
        
        # Position buffers (precomputed)
        self.q_positions = torch.arange(
            max_ctx_len, max_total_len, device=self.device
        )
        self.k_positions = torch.arange(max_total_len, device=self.device)
        
        logger.info(
            f"DFlash CUDA graph buffers initialized: max_bs={bs}, "
            f"max_ctx_len={max_ctx_len}, block_size={block_size}"
        )
    
    def _capture_graphs(self):
        """Capture CUDA graphs for all batch sizes."""
        logger.info(f"Capturing DFlash CUDA graphs for batch sizes: {self.capture_bs_list}")
        
        with model_capture_mode():
            for bs in self.capture_bs_list:
                self._capture_one_batch_size(bs)
        
        logger.info(f"Captured {len(self.graphs)} DFlash CUDA graphs")
    
    def _capture_one_batch_size(self, bs: int):
        """Capture CUDA graph for a specific batch size."""
        # Set up input slices
        target_hidden = self.target_hidden_buffer[:bs]
        noise_embedding = self.noise_embedding_buffer[:bs]
        ctx_lens = self.ctx_lens_buffer[:bs]
        attn_mask = self.attn_mask_buffer[:bs]
        
        # Warmup runs
        for _ in range(2):
            torch.cuda.synchronize()
            self._forward_for_capture(bs, target_hidden, noise_embedding, ctx_lens, attn_mask)
        
        # Capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self._forward_for_capture(bs, target_hidden, noise_embedding, ctx_lens, attn_mask)
        
        self.graphs[bs] = graph
    
    def _forward_for_capture(
        self,
        bs: int,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        ctx_lens: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        """Forward pass for CUDA graph capture.
        
        Uses preallocated buffers and avoids dynamic allocations.
        """
        block_size = self.block_size
        max_ctx_len = self.max_ctx_len
        max_total_len = self.max_total_len
        
        # Project and normalize target hidden (in-place where possible)
        if target_hidden.shape[-1] != self.hidden_size:
            target_hidden = self.draft_model.model.fc(target_hidden)
        target_hidden = self.draft_model.model.hidden_norm(target_hidden)
        
        # Working buffer for noise hidden
        noise_hidden = noise_embedding.clone()
        
        # Process layers
        for layer_idx, layer in enumerate(self.draft_model.model.layers):
            noise_hidden = self._layer_forward_for_capture(
                layer, bs, noise_hidden, target_hidden, attn_mask
            )
        
        # Final norm
        noise_hidden_flat = noise_hidden.reshape(-1, self.hidden_size)
        draft_hidden = self.draft_model.model.norm(noise_hidden_flat)
        self.draft_hidden_buffer[:bs] = draft_hidden.view(bs, block_size, -1)
    
    def _layer_forward_for_capture(
        self,
        layer,
        bs: int,
        noise_hidden: torch.Tensor,  # [bs, block_size, hidden]
        target_hidden: torch.Tensor,  # [bs, max_ctx_len, hidden]
        attn_mask: torch.Tensor,  # [bs, 1, block_size, max_total_len]
    ) -> torch.Tensor:
        """Single layer forward for CUDA graph capture."""
        block_size = self.block_size
        max_ctx_len = self.max_ctx_len
        max_total_len = self.max_total_len
        
        # Residual
        residual = noise_hidden
        
        # LayerNorm on noise
        noise_norm = layer.input_layernorm(noise_hidden.reshape(-1, self.hidden_size))
        noise_norm = noise_norm.view(bs, block_size, -1)
        
        # QKV projection for noise
        qkv_noise, _ = layer.self_attn.qkv_proj(noise_norm.reshape(-1, self.hidden_size))
        qkv_noise = qkv_noise.view(bs, block_size, -1)
        q, k_noise, v_noise = qkv_noise.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # QKV projection for target
        qkv_target, _ = layer.self_attn.qkv_proj(target_hidden.reshape(-1, self.hidden_size))
        qkv_target = qkv_target.view(bs, max_ctx_len, -1)
        _, k_target, v_target = qkv_target.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Concatenate K/V
        k = torch.cat([k_target, k_noise], dim=1)
        v = torch.cat([v_target, v_noise], dim=1)
        
        # Reshape
        q = q.view(bs, block_size, self.num_heads, self.head_dim)
        k = k.view(bs, max_total_len, self.num_kv_heads, self.head_dim)
        v = v.view(bs, max_total_len, self.num_kv_heads, self.head_dim)
        
        # Q/K normalization
        q = layer.self_attn.q_norm(q.reshape(-1, self.head_dim)).view(bs, block_size, self.num_heads, self.head_dim)
        k = layer.self_attn.k_norm(k.reshape(-1, self.head_dim)).view(bs, max_total_len, self.num_kv_heads, self.head_dim)
        
        # Transpose
        q = q.transpose(1, 2)  # [bs, num_heads, block_size, head_dim]
        k = k.transpose(1, 2)  # [bs, num_kv_heads, max_total_len, head_dim]
        v = v.transpose(1, 2)
        
        # Rotary embedding
        cos_sin_cache = layer.self_attn.rotary_emb.cos_sin_cache
        cos_sin_q = cos_sin_cache.index_select(0, self.q_positions)
        cos_q, sin_q = cos_sin_q.chunk(2, dim=-1)
        cos_sin_k = cos_sin_cache.index_select(0, self.k_positions)
        cos_k, sin_k = cos_sin_k.chunk(2, dim=-1)
        
        # Apply rotary
        q1, q2 = q.chunk(2, dim=-1)
        cos_q_bc = cos_q.unsqueeze(0).unsqueeze(0)
        sin_q_bc = sin_q.unsqueeze(0).unsqueeze(0)
        q = torch.cat([q1 * cos_q_bc - q2 * sin_q_bc, q2 * cos_q_bc + q1 * sin_q_bc], dim=-1)
        
        k1, k2 = k.chunk(2, dim=-1)
        cos_k_bc = cos_k.unsqueeze(0).unsqueeze(0)
        sin_k_bc = sin_k.unsqueeze(0).unsqueeze(0)
        k = torch.cat([k1 * cos_k_bc - k2 * sin_k_bc, k2 * cos_k_bc + k1 * sin_k_bc], dim=-1)
        
        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attn_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # Ensure dtype consistency (softmax may produce float32)
        if attn_weights.dtype != v.dtype:
            attn_weights = attn_weights.to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bs * block_size, -1)
        attn_output, _ = layer.self_attn.o_proj(attn_output)
        attn_output = attn_output.view(bs, block_size, -1)
        
        # Residual + MLP
        noise_hidden = residual + attn_output
        residual = noise_hidden
        mlp_input = layer.post_attention_layernorm(noise_hidden.reshape(-1, self.hidden_size))
        mlp_output = layer.mlp(mlp_input)
        noise_hidden = residual + mlp_output.view(bs, block_size, -1)
        
        return noise_hidden
    
    def can_run(self, batch_size: int, max_ctx_len: int) -> bool:
        """Check if CUDA graph can be used for given batch size and context length."""
        return (
            batch_size <= self.max_bs and
            max_ctx_len <= self.max_ctx_len and
            batch_size in self.graphs
        )
    
    def run(
        self,
        batch,  # ScheduleBatch
        target_hiddens: List[torch.Tensor],
        verified_ids: List[torch.Tensor],
        ctx_lens: List[int],
    ) -> tuple:
        """Run CUDA graph for draft forward.
        
        Copies inputs to preallocated buffers, replays the graph,
        and returns the output tokens and positions.
        """
        bs = len(target_hiddens)
        max_ctx_len = max(ctx_lens)
        block_size = self.block_size
        
        # Copy inputs to buffers
        self.target_hidden_buffer.zero_()
        for i, (th, cl) in enumerate(zip(target_hiddens, ctx_lens)):
            if th.dim() == 2:
                self.target_hidden_buffer[i, :cl, :] = th[:cl, :]
            else:
                self.target_hidden_buffer[i, :cl, :] = th[0, :cl, :]
        
        # Copy verified IDs
        for i, vid in enumerate(verified_ids):
            if isinstance(vid, torch.Tensor):
                self.verified_ids_buffer[i] = vid.item() if vid.dim() == 0 else vid[0].item()
            else:
                self.verified_ids_buffer[i] = vid
        
        # Set up block input IDs
        self.block_input_ids_buffer[:bs] = self.dflash_worker.mask_token_id
        self.block_input_ids_buffer[:bs, 0] = self.verified_ids_buffer[:bs]
        
        # Get noise embeddings
        self.noise_embedding_buffer[:bs] = self.dflash_worker.embed_tokens(
            self.block_input_ids_buffer[:bs]
        )
        
        # Set up context lengths
        self.ctx_lens_buffer[:bs] = torch.tensor(ctx_lens, dtype=torch.long, device=self.device)
        
        # Build attention mask
        self.attn_mask_buffer.zero_()
        for i, cl in enumerate(ctx_lens):
            # Mask out padding positions
            self.attn_mask_buffer[i, :, :, cl:self.max_ctx_len] = float('-inf')
        
        # Update position buffers for correct rotary embeddings
        # The graph was captured with fixed-size position buffers. We update VALUES
        # while keeping the same buffer sizes.
        #
        # Q positions: noise tokens start at actual ctx_len
        # Shape must stay [block_size], just update values
        self.q_positions[:] = torch.arange(
            max_ctx_len, max_ctx_len + self.block_size, device=self.device
        )
        
        # K positions: [0..actual_total_len-1] for real tokens, rest are padding
        # The padding positions get masked out by attention mask (-inf), so their
        # rotary values don't matter. We fill them with 0 for simplicity.
        actual_total_len = max_ctx_len + self.block_size
        self.k_positions.zero_()  # Clear first
        self.k_positions[:actual_total_len] = torch.arange(
            actual_total_len, device=self.device
        )
        
        # Replay graph
        graph = self.graphs[bs]
        graph.replay()
        
        # Get logits and tokens
        draft_hidden = self.draft_hidden_buffer[:bs, 1:, :]  # Skip first position
        draft_logits = torch.matmul(draft_hidden, self.dflash_worker.lm_head_weight.t())
        self.block_input_ids_buffer[:bs, 1:] = torch.argmax(draft_logits, dim=-1)
        
        # Build output
        all_draft_tokens = []
        all_positions = []
        for i in range(bs):
            all_draft_tokens.append(self.block_input_ids_buffer[i])
            current_seq_len = batch.seq_lens[i].item()
            all_positions.append(torch.arange(
                current_seq_len, current_seq_len + block_size, device=self.device
            ))
        
        draft_tokens = torch.cat(all_draft_tokens, dim=0)
        positions = torch.cat(all_positions, dim=0)
        return draft_tokens, positions

