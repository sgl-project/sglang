# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License")

"""DFlash speculative decoding worker with RadixAttention integration.

DFlash draft model generates a full block of tokens in one forward pass.
Uses native SGLang RadixAttention with ENCODER_ONLY for non-causal attention.
Supports native paged KV cache with eviction of noise positions after verification.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

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
from sglang.srt.models.dflash import RMSNorm3D, build_target_layer_ids
from sglang.srt.models.qwen3_dflash import Qwen3ForCausalLMDFlash
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_draft_cuda_graph_runner import DFlashDraftCudaGraphRunner
from sglang.srt.utils.common import get_available_gpu_memory

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache

logger = logging.getLogger(__name__)


class DFlashWorker:
    """DFlash speculative decoding worker with RadixAttention."""

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
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.model_config = target_worker.model_runner.model_config
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.block_size = server_args.speculative_dflash_block_size
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self.server_args = server_args
        self.gpu_id = gpu_id

        # Load DFlash draft model
        logger.info(f"Loading DFlash draft model from {server_args.speculative_draft_model_path}")
        self._load_draft_model(server_args)
        
        # Get embeddings and lm_head from target model
        target_model = target_worker.model_runner.model
        if hasattr(target_model, 'model') and hasattr(target_model.model, 'embed_tokens'):
            self.embed_tokens = target_model.model.embed_tokens
        else:
            self.embed_tokens = target_model.get_input_embeddings()
            
        if hasattr(target_model, 'lm_head') and hasattr(target_model.lm_head, 'weight'):
            self.lm_head_weight = target_model.lm_head.weight
        else:
            self.lm_head_weight = self.embed_tokens.weight

        # Configure target model to capture multi-layer hidden states
        if hasattr(target_model, 'set_eagle3_layers_to_capture'):
            logger.info(f"DFlash target_layer_ids: {self.target_layer_ids}")
            target_model.set_eagle3_layers_to_capture(self.target_layer_ids)
            # #region agent log
            import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "O", "location": "dflash_worker.py:80", "message": "Configured target model for aux_hidden_states", "data": {"target_layer_ids": self.target_layer_ids, "capture_aux_hidden_states": getattr(target_model, 'capture_aux_hidden_states', None), "layers_to_capture": getattr(target_model.model, 'layers_to_capture', None) if hasattr(target_model, 'model') else None}, "timestamp": __import__('time').time()}) + '\n')
            # #endregion

        # Per-request state for hidden states and metadata
        self._request_state: Dict[str, dict] = {}
        
        # Radix cache integration
        self.radix_cache: Optional["RadixCache"] = None
        self.unified_hidden_cache_enabled = False
        
        # Pre-allocate position buffer
        self.max_seq_len = self.model_config.context_len
        
        # Get mask token ID
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)
        if tokenizer.mask_token_id is None:
            tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        self.mask_token_id = tokenizer.mask_token_id
        logger.info(f"DFlash mask_token_id: {self.mask_token_id}")

        logger.info(f"DFlashWorker initialized: block_size={self.block_size}")

    def _load_draft_model(self, server_args: ServerArgs):
        """Load DFlash draft model with weight mapping."""
        import glob
        import json
        import os

        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        model_path = server_args.speculative_draft_model_path
        
        # Download from HuggingFace Hub if not a local path
        if not os.path.isdir(model_path):
            model_path = snapshot_download(
                repo_id=model_path,
                allow_patterns=["*.safetensors", "*.json", "*.bin"],
            )
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Create config object
        class Config:
            pass
        config = Config()
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        # Ensure required attributes
        if not hasattr(config, "rms_norm_eps"):
            config.rms_norm_eps = 1e-6
        if not hasattr(config, "attention_bias"):
            config.attention_bias = True
        
        # Create model
        self.draft_model = Qwen3ForCausalLMDFlash(config)
        
        # Load weights
        weight_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(os.path.join(model_path, "*.bin"))
        
        if weight_files:
            weights = []
            for wf in weight_files:
                if wf.endswith(".safetensors"):
                    file_weights = load_file(wf)
                else:
                    file_weights = torch.load(wf, map_location="cpu")
                weights.extend(file_weights.items())
            
            self.draft_model.load_weights(weights)
        
        self.draft_model = self.draft_model.to(self.device).to(torch.bfloat16).eval()
        self.target_layer_ids = self.draft_model.target_layer_ids
        logger.info(f"DFlash draft model loaded: {config.num_hidden_layers} layers")
        
        # CUDA graph runner (initialized in init_cuda_graphs)
        self.cuda_graph_runner = None

    def clear_cache_pool(self):
        """Clear all per-request state."""
        self._request_state.clear()
    
    def init_cuda_graphs(self):
        """Initialize CUDA graphs for draft model."""
        if self.server_args.disable_cuda_graph:
            logger.info("DFlash CUDA graphs disabled by server args")
            return
        
        import time
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory("cuda", self.gpu_id)
        logger.info(
            f"Capturing DFlash CUDA graphs. avail mem={before_mem:.2f} GB"
        )
        
        try:
            # Use conservative settings to fit in available memory
            # Position buffers are updated dynamically before each graph replay
            max_bs = min(8, self.server_args.max_running_requests or 256)
            max_ctx_len = min(2048, self.server_args.context_length or 4096)
            
            self.cuda_graph_runner = DFlashDraftCudaGraphRunner(
                dflash_worker=self,
                max_bs=max_bs,
                max_ctx_len=max_ctx_len,
            )
            
            after_mem = get_available_gpu_memory("cuda", self.gpu_id)
            logger.info(
                f"Captured DFlash CUDA graphs. Time: {time.perf_counter() - tic:.2f}s, "
                f"mem usage: {before_mem - after_mem:.2f} GB, avail: {after_mem:.2f} GB"
            )
        except Exception as e:
            logger.warning(f"Failed to capture DFlash CUDA graphs: {e}")
            self.cuda_graph_runner = None
    
    def set_radix_cache(self, radix_cache: "RadixCache", token_to_kv_pool=None):
        """Set up radix cache integration for hidden state prefix sharing."""
        self.radix_cache = radix_cache
        
        num_selected_layers = len(self.target_layer_ids)
        hidden_size = self.draft_model.config.hidden_size
        
        if token_to_kv_pool is not None and hasattr(token_to_kv_pool, 'enable_dflash_hidden_buffer'):
            token_to_kv_pool.enable_dflash_hidden_buffer(
                hidden_size=hidden_size,
                num_target_layers=num_selected_layers,
            )
            self.unified_hidden_cache_enabled = True
            logger.info(f"DFlash unified hidden cache enabled: {num_selected_layers} layers")
        else:
            self.unified_hidden_cache_enabled = False

    def _get_request_state(self, rid: str) -> dict:
        """Get or create per-request state."""
        if rid not in self._request_state:
            self._request_state[rid] = {
                'target_hidden': None,
                'accumulated_hidden': None,
                'start': 0,
                'verified_id': None,
                'noise_cache_locs': None,
            }
        return self._request_state[rid]

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
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
        
        # #region agent log
        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "P", "location": "dflash_worker.py:225", "message": "Prefill hidden states", "data": {"has_hidden_states": hidden_states is not None, "hidden_states_shape": list(hidden_states.shape) if hidden_states is not None else None, "hidden_states_dim": hidden_states.shape[-1] if hidden_states is not None else None, "expected_dim": 5 * 2560}, "timestamp": __import__('time').time()}) + '\n')
        # #endregion
        
        # Store hidden states per request
        token_offset = 0
        
        for i, req in enumerate(batch.reqs):
            req_len = len(req.origin_input_ids)
            state = self._get_request_state(req.rid)
            
            new_token_count = batch.extend_lens[i] if hasattr(batch, 'extend_lens') else req_len
            cached_count = req_len - new_token_count if hasattr(batch, 'extend_lens') else 0
            
            state['start'] = req_len
            
            if hidden_states is not None:
                # Extract NEW tokens' hidden states
                if hidden_states.dim() == 2:
                    new_hidden = hidden_states[token_offset:token_offset + new_token_count, :]
                else:
                    new_hidden = hidden_states[:, token_offset:token_offset + new_token_count, :]
                
                if new_hidden.dim() == 2:
                    new_hidden = new_hidden.unsqueeze(0)
                
                # Try to get cached hidden states
                if cached_count > 0 and self.unified_hidden_cache_enabled:
                    token_to_kv_pool = self.model_runner.token_to_kv_pool
                    if hasattr(req, 'prefix_indices') and req.prefix_indices is not None:
                        prefix_locs = req.prefix_indices[:cached_count]
                        cached_hidden = token_to_kv_pool.get_all_hidden_states(prefix_locs)
                        if cached_hidden is not None:
                            cached_hidden = cached_hidden.unsqueeze(0)
                            new_hidden = torch.cat([cached_hidden, new_hidden], dim=1)
                
                state['target_hidden'] = new_hidden.clone()
                state['accumulated_hidden'] = new_hidden.clone()
                
                # Store hidden states in KV pool
                if self.unified_hidden_cache_enabled:
                    token_to_kv_pool = self.model_runner.token_to_kv_pool
                    if hasattr(batch, 'out_cache_loc') and batch.out_cache_loc is not None:
                        req_start = sum(batch.extend_lens[:i]) if hasattr(batch, 'extend_lens') else token_offset
                        req_end = req_start + new_token_count
                        req_cache_loc = batch.out_cache_loc[req_start:req_end]
                        hidden_to_store = new_hidden.squeeze(0)[-new_token_count:]
                        if hidden_to_store.shape[0] == req_cache_loc.shape[0]:
                            token_to_kv_pool.set_all_hidden_states(req_cache_loc, hidden_to_store)
                
                state['verified_id'] = next_token_ids[i].clone()
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
            logger.warning("[DFLASH] No DFlashDraftInput found, running normal target decode")
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
        
        # #region agent log
        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "O", "location": "dflash_worker.py:323", "message": "Hidden states from verification", "data": {"has_hidden_states": new_hidden_states is not None, "hidden_states_shape": list(new_hidden_states.shape) if new_hidden_states is not None else None, "hidden_states_dim": new_hidden_states.shape[-1] if new_hidden_states is not None else None, "expected_dim": 5 * 2560}, "timestamp": __import__('time').time()}) + '\n')
        # #endregion
        
        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            acc_len = accept_length_list[i]
            
            # Update positions
            state['start'] += acc_len + 1
            
            # Evict rejected noise positions from cache
            self._evict_rejected_cache(batch, req, state, acc_len)
            
            # Update hidden states
            if new_hidden_states is not None:
                num_tokens = acc_len + 1
                start_idx = i * block_size
                if new_hidden_states.dim() == 2:
                    req_hidden = new_hidden_states[start_idx:start_idx + num_tokens, :]
                else:
                    req_hidden = new_hidden_states[:, start_idx:start_idx + num_tokens, :]
                if req_hidden.dim() == 2:
                    req_hidden = req_hidden.unsqueeze(0)
                
                old_hidden = state.get('accumulated_hidden')
                if old_hidden is not None:
                    state['accumulated_hidden'] = torch.cat([old_hidden, req_hidden], dim=1)
                else:
                    state['accumulated_hidden'] = req_hidden.clone()
                
                state['target_hidden'] = state['accumulated_hidden']
            
            state['verified_id'] = new_verified_id[i].clone()
        
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
    ) -> tuple:
        """Draft forward pass using RadixAttention model."""
        bs = len(batch.reqs)
        block_size = self.block_size
        
        # Collect per-request data
        target_hiddens = []
        verified_ids = []
        ctx_lens = []
        
        for i, req in enumerate(batch.reqs):
            state = self._get_request_state(req.rid)
            target_hidden = state.get('target_hidden')
            
            if target_hidden is None:
                logger.warning(f"[DFLASH] No target_hidden for request {req.rid}")
                continue
            
            req_verified_id = state.get('verified_id')
            if req_verified_id is None:
                if all_verified_id.dim() > 0 and all_verified_id.shape[0] > i:
                    req_verified_id = all_verified_id[i]
                else:
                    req_verified_id = all_verified_id.flatten()[0]
            
            target_hiddens.append(target_hidden)
            verified_ids.append(req_verified_id)
            ctx_len = target_hidden.shape[1] if target_hidden.dim() == 3 else target_hidden.shape[0]
            ctx_lens.append(ctx_len)
        
        if not target_hiddens:
            # No valid requests
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Use sequential forward (known working path) while debugging batched
        return self._sequential_draft_forward(batch, target_hiddens, verified_ids, ctx_lens)

    def _cuda_graph_draft_forward(
        self,
        batch: ScheduleBatch,
        target_hiddens: List[torch.Tensor],
        verified_ids: List[torch.Tensor],
        ctx_lens: List[int],
    ) -> tuple:
        """Draft forward using CUDA graph."""
        return self.cuda_graph_runner.run(
            batch, target_hiddens, verified_ids, ctx_lens
        )

    def _batched_draft_forward(
        self,
        batch: ScheduleBatch,
        target_hiddens: List[torch.Tensor],
        verified_ids: List[torch.Tensor],
        ctx_lens: List[int],
    ) -> tuple:
        """Batched draft forward with padding for variable context lengths.
        
        This is the CUDA-graph-compatible version that processes all requests
        in parallel using padded tensors.
        """
        bs = len(target_hiddens)
        block_size = self.block_size
        max_ctx_len = max(ctx_lens)
        
        # Get input dimension (may be num_layers * hidden_size before FC projection)
        first_th = target_hiddens[0]
        input_dim = first_th.shape[-1]
        
        # Pad and stack target_hiddens to [bs, max_ctx_len, input_dim]
        padded_target_hiddens = torch.zeros(
            bs, max_ctx_len, input_dim,
            dtype=torch.bfloat16, device=self.device
        )
        ctx_lens_tensor = torch.tensor(ctx_lens, dtype=torch.long, device=self.device)
        
        for i, (th, cl) in enumerate(zip(target_hiddens, ctx_lens)):
            if th.dim() == 2:
                padded_target_hiddens[i, :cl, :] = th[:cl, :]
            else:
                padded_target_hiddens[i, :cl, :] = th[0, :cl, :]
        
        # Stack verified_ids to [bs]
        verified_ids_tensor = torch.stack([
            v if isinstance(v, torch.Tensor) and v.dim() == 0 
            else (v[0] if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device))
            for v in verified_ids
        ]).to(dtype=torch.long, device=self.device)
        
        # Create noise embeddings [bs, block_size, hidden]
        block_input_ids = torch.full(
            (bs, block_size), self.mask_token_id, 
            dtype=torch.long, device=self.device
        )
        block_input_ids[:, 0] = verified_ids_tensor
        noise_embedding = self.embed_tokens(block_input_ids)  # [bs, block_size, hidden]
        
        # Run batched attention forward
        draft_hidden = self._dflash_attention_forward(
            padded_target_hiddens, noise_embedding, ctx_lens_tensor
        )
        
        # Get draft logits and tokens [bs, block_size-1]
        draft_hidden_for_logits = draft_hidden[:, 1:, :]  # [bs, block_size-1, hidden]
        draft_logits = torch.matmul(draft_hidden_for_logits, self.lm_head_weight.t())
        block_input_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)
        
        # Flatten for output
        all_draft_tokens = []
        all_positions = []
        for i in range(bs):
            all_draft_tokens.append(block_input_ids[i])
            current_seq_len = batch.seq_lens[i].item()
            all_positions.append(torch.arange(
                current_seq_len, current_seq_len + block_size, device=self.device
            ))
        
        draft_tokens = torch.cat(all_draft_tokens, dim=0)
        positions = torch.cat(all_positions, dim=0)
        return draft_tokens, positions

    def _dflash_attention_forward(
        self,
        target_hidden: torch.Tensor,  # [bs, max_ctx_len, hidden]
        noise_embedding: torch.Tensor,  # [bs, block_size, hidden]
        ctx_lens: torch.Tensor,  # [bs]
    ) -> torch.Tensor:
        """DFlash attention forward pass.
        
        Implements the DFlash-specific attention pattern:
        - Q: from noise only
        - K/V: from concat(target_hidden, noise)
        - Non-causal attention with position-based masking
        
        Args:
            target_hidden: Padded target hidden states [bs, max_ctx_len, hidden]
            noise_embedding: Noise embeddings [bs, block_size, hidden]
            ctx_lens: Actual context lengths per request [bs]
            
        Returns:
            Draft hidden states [bs, block_size, hidden]
        """
        bs = target_hidden.shape[0]
        max_ctx_len = target_hidden.shape[1]
        block_size = noise_embedding.shape[1]
        max_total_len = max_ctx_len + block_size
        hidden_size = self.draft_model.model.hidden_size
        
        # Project and normalize target hidden
        if target_hidden.shape[-1] != hidden_size:
            target_hidden = self.draft_model.model.fc(target_hidden)
        target_hidden = self.draft_model.model.hidden_norm(target_hidden)
        
        # Build attention mask for variable context lengths [bs, 1, block_size, max_total_len]
        # Mask out padding positions in target_hidden
        attn_mask = torch.zeros(
            bs, 1, block_size, max_total_len,
            dtype=torch.bfloat16, device=self.device
        )
        for i in range(bs):
            cl = ctx_lens[i].item()
            # Mask out padding (positions >= ctx_len in target region)
            attn_mask[i, :, :, cl:max_ctx_len] = float('-inf')
        
        # Current noise hidden states [bs, block_size, hidden]
        noise_hidden = noise_embedding
        
        # Process through layers
        for layer_idx, layer in enumerate(self.draft_model.model.layers):
            noise_hidden = self._dflash_layer_forward(
                layer, layer_idx, noise_hidden, target_hidden, 
                ctx_lens, max_ctx_len, max_total_len, attn_mask
            )
        
        # Final norm (RMSNorm expects 2D input)
        bs, seq_len, hidden = noise_hidden.shape
        noise_hidden_2d = noise_hidden.view(bs * seq_len, hidden)
        draft_hidden_2d = self.draft_model.model.norm(noise_hidden_2d)
        draft_hidden = draft_hidden_2d.view(bs, seq_len, hidden)
        return draft_hidden

    def _dflash_layer_forward(
        self,
        layer,
        layer_idx: int,
        noise_hidden: torch.Tensor,  # [bs, block_size, hidden]
        target_hidden: torch.Tensor,  # [bs, max_ctx_len, hidden]
        ctx_lens: torch.Tensor,  # [bs]
        max_ctx_len: int,
        max_total_len: int,
        attn_mask: torch.Tensor,  # [bs, 1, block_size, max_total_len]
    ) -> torch.Tensor:
        """Single layer forward for DFlash attention."""
        bs = noise_hidden.shape[0]
        block_size = noise_hidden.shape[1]
        
        # Residual
        residual = noise_hidden
        
        # LayerNorm on noise only (target already normalized by hidden_norm)
        noise_norm = layer.input_layernorm(noise_hidden.reshape(-1, noise_hidden.shape[-1]))
        noise_norm = noise_norm.view(bs, block_size, -1)
        
        # QKV projection
        num_heads = layer.self_attn.num_heads
        num_kv_heads = layer.self_attn.num_kv_heads
        head_dim = layer.self_attn.head_dim
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        
        # Q from noise [bs, block_size, q_size]
        noise_norm_flat = noise_norm.reshape(-1, noise_norm.shape[-1])
        qkv_noise, _ = layer.self_attn.qkv_proj(noise_norm_flat)
        qkv_noise = qkv_noise.view(bs, block_size, -1)
        q, k_noise, v_noise = qkv_noise.split([q_size, kv_size, kv_size], dim=-1)
        
        # K/V from target [bs, max_ctx_len, kv_size]
        target_flat = target_hidden.reshape(-1, target_hidden.shape[-1])
        qkv_target, _ = layer.self_attn.qkv_proj(target_flat)
        qkv_target = qkv_target.view(bs, max_ctx_len, -1)
        _, k_target, v_target = qkv_target.split([q_size, kv_size, kv_size], dim=-1)
        
        # Concatenate K/V: [bs, max_total_len, kv_size]
        k = torch.cat([k_target, k_noise], dim=1)
        v = torch.cat([v_target, v_noise], dim=1)
        
        # Reshape for attention: [bs, seq, num_heads, head_dim]
        q = q.view(bs, block_size, num_heads, head_dim)
        k = k.view(bs, max_total_len, num_kv_heads, head_dim)
        v = v.view(bs, max_total_len, num_kv_heads, head_dim)
        
        # Apply Q/K normalization (reshape for 2D RMSNorm)
        q_flat = q.reshape(-1, head_dim)
        k_flat = k.reshape(-1, head_dim)
        q_flat = layer.self_attn.q_norm(q_flat)
        k_flat = layer.self_attn.k_norm(k_flat)
        q = q_flat.view(bs, block_size, num_heads, head_dim)
        k = k_flat.view(bs, max_total_len, num_kv_heads, head_dim)
        
        # Transpose to [bs, heads, seq, dim]
        q = q.transpose(1, 2)  # [bs, num_heads, block_size, head_dim]
        k = k.transpose(1, 2)  # [bs, num_kv_heads, max_total_len, head_dim]
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings with different positions
        # Q: positions [max_ctx_len, max_ctx_len+block_size)
        # K: positions [0, max_total_len)
        cos_sin_cache = layer.self_attn.rotary_emb.cos_sin_cache
        if cos_sin_cache.dtype != torch.float32:
            layer.self_attn.rotary_emb.cos_sin_cache = cos_sin_cache.to(torch.float32)
            cos_sin_cache = layer.self_attn.rotary_emb.cos_sin_cache
        
        # Positions for Q (noise region)
        q_positions = torch.arange(max_ctx_len, max_ctx_len + block_size, device=self.device)
        cos_sin_q = cos_sin_cache.index_select(0, q_positions)
        cos_q, sin_q = cos_sin_q.chunk(2, dim=-1)
        
        # Positions for K (full sequence)
        k_positions = torch.arange(max_total_len, device=self.device)
        cos_sin_k = cos_sin_cache.index_select(0, k_positions)
        cos_k, sin_k = cos_sin_k.chunk(2, dim=-1)
        
        # Apply rotary: split head_dim into halves
        q1, q2 = q.chunk(2, dim=-1)
        cos_q_bc = cos_q.unsqueeze(0).unsqueeze(0)  # [1, 1, block_size, rotary_dim]
        sin_q_bc = sin_q.unsqueeze(0).unsqueeze(0)
        q = torch.cat([q1 * cos_q_bc - q2 * sin_q_bc, q2 * cos_q_bc + q1 * sin_q_bc], dim=-1)
        
        k1, k2 = k.chunk(2, dim=-1)
        cos_k_bc = cos_k.unsqueeze(0).unsqueeze(0)  # [1, 1, max_total_len, rotary_dim]
        sin_k_bc = sin_k.unsqueeze(0).unsqueeze(0)
        k = torch.cat([k1 * cos_k_bc - k2 * sin_k_bc, k2 * cos_k_bc + k1 * sin_k_bc], dim=-1)
        
        # Expand KV for GQA
        num_kv_groups = num_heads // num_kv_heads
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)
        
        # Attention with mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * layer.self_attn.scaling
        attn_weights = attn_weights + attn_mask  # Apply padding mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # Ensure dtype consistency (softmax may produce float32)
        if attn_weights.dtype != v.dtype:
            attn_weights = attn_weights.to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()  # [bs, block_size, num_heads, head_dim]
        attn_output = attn_output.view(bs * block_size, -1)
        attn_output, _ = layer.self_attn.o_proj(attn_output)
        attn_output = attn_output.view(bs, block_size, -1)
        
        # Residual
        noise_hidden = residual + attn_output
        
        # MLP
        residual = noise_hidden
        mlp_input = layer.post_attention_layernorm(noise_hidden.reshape(-1, noise_hidden.shape[-1]))
        mlp_output = layer.mlp(mlp_input)
        mlp_output = mlp_output.view(bs, block_size, -1)
        noise_hidden = residual + mlp_output
        
        return noise_hidden

    def _sequential_draft_forward(
        self,
        batch: ScheduleBatch,
        target_hiddens: List[torch.Tensor],
        verified_ids: List[torch.Tensor],
        ctx_lens: List[int],
    ) -> tuple:
        """Sequential draft forward (temporary until ForwardBatch integration is complete)."""
        bs = len(target_hiddens)
        block_size = self.block_size
        all_draft_tokens = []
        all_positions = []
        
        for i, (target_hidden, verified_id, ctx_len) in enumerate(zip(target_hiddens, verified_ids, ctx_lens)):
            req = batch.reqs[i]
            
            # Ensure 3D tensor
            if target_hidden.dim() == 2:
                target_hidden = target_hidden.unsqueeze(0)
            
            # Create noise embeddings with first token as verified
            block_output_ids = torch.full(
                (1, block_size), self.mask_token_id,
                dtype=torch.long, device=self.device
            )
            if isinstance(verified_id, torch.Tensor):
                block_output_ids[0, 0] = verified_id.item() if verified_id.dim() == 0 else verified_id[0].item()
            else:
                block_output_ids[0, 0] = verified_id
            
            noise_embedding = self.embed_tokens(block_output_ids)  # [1, block_size, hidden]
            
            # #region agent log
            if i == 0:
                import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "R", "location": "dflash_worker.py:458", "message": "Noise embedding", "data": {"noise_embedding_shape": list(noise_embedding.shape), "noise_embedding_norm": float(noise_embedding.norm().cpu()), "block_output_ids": block_output_ids.tolist(), "first_token_embed_norm": float(noise_embedding[0, 0].norm().cpu()), "mask_token_embed_norm": float(noise_embedding[0, 1].norm().cpu())}, "timestamp": __import__('time').time()}) + '\n')
            # #endregion
            
            # Project and normalize target hidden
            hidden_size = self.draft_model.model.hidden_size
            # #region agent log
            if i == 0:
                import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "N", "location": "dflash_worker.py:452", "message": "Before fc projection", "data": {"target_hidden_shape": list(target_hidden.shape), "hidden_size": hidden_size, "needs_projection": target_hidden.shape[-1] != hidden_size, "target_hidden_last_dim": target_hidden.shape[-1]}, "timestamp": __import__('time').time()}) + '\n')
            # #endregion
            if target_hidden.shape[-1] != hidden_size:
                target_hidden = self.draft_model.model.fc(target_hidden)
                # #region agent log
                if i == 0:
                    import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "N", "location": "dflash_worker.py:456", "message": "After fc projection", "data": {"target_hidden_shape": list(target_hidden.shape), "target_hidden_norm": float(target_hidden.norm().cpu())}, "timestamp": __import__('time').time()}) + '\n')
                # #endregion
            target_hidden = self.draft_model.model.hidden_norm(target_hidden)
            
            # #region agent log
            if i == 0:
                import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "Q", "location": "dflash_worker.py:474", "message": "DFlash input tensors (req 0)", "data": {"target_hidden_shape": list(target_hidden.shape), "noise_embedding_shape": list(noise_embedding.shape), "ctx_len": ctx_len, "block_size": block_size, "target_hidden_norm": float(target_hidden.norm().cpu()), "noise_embedding_norm": float(noise_embedding.norm().cpu())}, "timestamp": __import__('time').time()}) + '\n')
            # #endregion
            
            # ORIGINAL DFLASH ATTENTION PATTERN:
            # Q: from noise only
            # K/V: from concat(target_hidden, noise)
            
            total_len = ctx_len + block_size
            positions = torch.arange(total_len, device=self.device)
            noise_positions = positions[ctx_len:]  # Positions for Q (noise only)
            
            # Current hidden states for the noise part (used for residual connections)
            noise_hidden = noise_embedding.squeeze(0)  # [block_size, hidden]
            
            with torch.no_grad():
                for layer_idx, layer in enumerate(self.draft_model.model.layers):
                    # Residual connection for noise positions
                    residual = noise_hidden  # [block_size, hidden]
                    
                    # LayerNorm - ONLY applied to noise, NOT to target_hidden
                    # (target_hidden was already processed by hidden_norm at model level)
                    noise_norm = layer.input_layernorm(noise_hidden)  # [block_size, hidden]
                    target_for_kv = target_hidden.squeeze(0)  # [ctx_len, hidden] - NO layernorm!
                    
                    # === ORIGINAL DFLASH ATTENTION ===
                    # Q only from noise (block_size tokens)
                    qkv_noise, _ = layer.self_attn.qkv_proj(noise_norm)
                    q_noise, k_noise, v_noise = qkv_noise.split([
                        layer.self_attn.q_size,
                        layer.self_attn.kv_size,
                        layer.self_attn.kv_size
                    ], dim=-1)
                    
                    # K/V from target (ctx_len tokens) - NO layernorm on target
                    qkv_target, _ = layer.self_attn.qkv_proj(target_for_kv)
                    _, k_target, v_target = qkv_target.split([
                        layer.self_attn.q_size,
                        layer.self_attn.kv_size,
                        layer.self_attn.kv_size
                    ], dim=-1)
                    
                    # Concatenate K and V: [ctx_len + block_size, kv_size]
                    k = torch.cat([k_target, k_noise], dim=0)
                    v = torch.cat([v_target, v_noise], dim=0)
                    q = q_noise  # Q only from noise
                    
                    # Make contiguous
                    q = q.contiguous()
                    k = k.contiguous()
                    v = v.contiguous()
                    
                    # ============================================================
                    # MATCH ORIGINAL DFLASH ORDER: view -> norm -> transpose -> rotary
                    # ============================================================
                    
                    num_heads = layer.self_attn.num_heads
                    num_kv_heads = layer.self_attn.num_kv_heads
                    head_dim = layer.self_attn.head_dim
                    
                    # Step 1: View to 3D [seq, num_heads, head_dim]
                    q = q.view(block_size, num_heads, head_dim)
                    k = k.view(total_len, num_kv_heads, head_dim)
                    v = v.view(total_len, num_kv_heads, head_dim)
                    
                    # #region agent log - Compare step by step
                    if layer_idx == 0 and i == 0:
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "step1_view", "message": "After view to 3D", "data": {"q_shape": list(q.shape), "k_shape": list(k.shape), "v_shape": list(v.shape), "q_norm": float(q.norm().cpu()), "k_norm": float(k.norm().cpu())}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    
                    # Step 2: Apply Q/K normalization
                    # SGLang RMSNorm kernel requires 2D input, so reshape: [seq, heads, dim] -> [seq*heads, dim]
                    # Original: q = self.q_norm(q) where q is [B, seq, heads, dim], norm on last dim
                    q_2d = q.reshape(-1, head_dim)  # [seq*heads, dim]
                    k_2d = k.reshape(-1, head_dim)  # [kv_seq*kv_heads, dim]
                    q_2d = layer.self_attn.q_norm(q_2d)
                    k_2d = layer.self_attn.k_norm(k_2d)
                    q = q_2d.view(block_size, num_heads, head_dim)
                    k = k_2d.view(total_len, num_kv_heads, head_dim)
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "step2_norm", "message": "After Q/K norm (ORIGINAL: Q=378, K=210)", "data": {"q_shape": list(q.shape), "q_norm": float(q.norm().cpu()), "k_norm": float(k.norm().cpu()), "v_norm": float(v.norm().cpu()), "EXPECT_Q": 378.0, "EXPECT_K": 210.0, "EXPECT_V": 324.0}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    
                    # Step 3: Transpose to [heads, seq, head_dim] (BEFORE rotary, like original)
                    q = q.transpose(0, 1)  # [num_heads, block_size, head_dim]
                    k = k.transpose(0, 1)  # [num_kv_heads, total_len, head_dim]
                    v = v.transpose(0, 1)  # [num_kv_heads, total_len, head_dim]
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "step3_transpose", "message": "After transpose", "data": {"q_shape": list(q.shape), "k_shape": list(k.shape)}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    
                    # Step 4: Apply rotary embeddings (on transposed tensors)
                    if layer.self_attn.rotary_emb.cos_sin_cache.dtype != torch.float32:
                        layer.self_attn.rotary_emb.cos_sin_cache = layer.self_attn.rotary_emb.cos_sin_cache.to(torch.float32)
                    
                    cos_sin_cache = layer.self_attn.rotary_emb.cos_sin_cache
                    rotary_dim = cos_sin_cache.shape[-1] // 2
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        # Check actual rotary config
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "rotary_config", "message": "Rotary embedding config", "data": {"cos_sin_cache_shape": list(cos_sin_cache.shape), "computed_rotary_dim": rotary_dim, "head_dim": head_dim, "rotary_emb_rotary_dim": getattr(layer.self_attn.rotary_emb, 'rotary_dim', 'N/A'), "rotary_emb_head_size": getattr(layer.self_attn.rotary_emb, 'head_size', 'N/A')}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    
                    # Get cos/sin for Q (noise positions) and K (all positions)
                    cos_sin_q = cos_sin_cache.index_select(0, noise_positions)  # [block_size, rotary_dim*2]
                    cos_q, sin_q = cos_sin_q.chunk(2, dim=-1)  # [block_size, rotary_dim]
                    
                    cos_sin_k = cos_sin_cache.index_select(0, positions)  # [total_len, rotary_dim*2]
                    cos_k, sin_k = cos_sin_k.chunk(2, dim=-1)  # [total_len, rotary_dim]
                    
                    # SGLang-style rotary: cos/sin each have rotary_dim dims (64)
                    # Split q/k into two halves, apply rotation, recombine
                    # This is equivalent to HF's approach with duplicated freqs
                    
                    # Expand cos/sin for broadcasting: [1, seq, rotary_dim]
                    cos_q_exp = cos_q.unsqueeze(0).unsqueeze(-2)  # [1, block_size, 1, rotary_dim]
                    sin_q_exp = sin_q.unsqueeze(0).unsqueeze(-2)
                    cos_k_exp = cos_k.unsqueeze(0).unsqueeze(-2)  # [1, total_len, 1, rotary_dim]
                    sin_k_exp = sin_k.unsqueeze(0).unsqueeze(-2)
                    
                    # q is [num_heads, block_size, head_dim] -> need [num_heads, block_size, 1, head_dim]
                    # Actually simpler: just reshape for the rotation
                    
                    # Q: [num_heads, block_size, head_dim=128]
                    q1, q2 = torch.chunk(q, 2, dim=-1)  # Each [num_heads, block_size, 64]
                    # cos_q: [block_size, 64] -> [1, block_size, 64] for broadcast
                    cos_q_bc = cos_q.unsqueeze(0)  # [1, block_size, 64]
                    sin_q_bc = sin_q.unsqueeze(0)
                    q_o1 = q1 * cos_q_bc - q2 * sin_q_bc
                    q_o2 = q2 * cos_q_bc + q1 * sin_q_bc
                    q = torch.cat((q_o1, q_o2), dim=-1)  # [num_heads, block_size, 128]
                    
                    # K: [num_kv_heads, total_len, head_dim=128]
                    k1, k2 = torch.chunk(k, 2, dim=-1)  # Each [num_kv_heads, total_len, 64]
                    cos_k_bc = cos_k.unsqueeze(0)  # [1, total_len, 64]
                    sin_k_bc = sin_k.unsqueeze(0)
                    k_o1 = k1 * cos_k_bc - k2 * sin_k_bc
                    k_o2 = k2 * cos_k_bc + k1 * sin_k_bc
                    k = torch.cat((k_o1, k_o2), dim=-1)  # [num_kv_heads, total_len, 128]
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "step4_rotary", "message": "After rotary", "data": {"q_shape": list(q.shape), "k_shape": list(k.shape), "q_norm": float(q.norm().cpu()), "k_norm": float(k.norm().cpu()), "rotary_dim": rotary_dim}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    
                    # Expand KV for GQA
                    num_kv_groups = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
                    if num_kv_groups > 1:
                        k = k.repeat_interleave(num_kv_groups, dim=0)
                        v = v.repeat_interleave(num_kv_groups, dim=0)
                    
                    # Non-causal attention: Q [block_size] attends to K/V [total_len]
                    # Ensure all tensors have same dtype (rotary may have converted to float32)
                    compute_dtype = q.dtype
                    if v.dtype != compute_dtype:
                        v = v.to(compute_dtype)
                    
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * layer.self_attn.scaling
                    attn_weights = torch.softmax(attn_weights, dim=-1)
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        # Check attention distribution for first layer, first request
                        # attn_weights: [num_heads, block_size, total_len]
                        avg_attn = attn_weights.mean(dim=0)  # [block_size, total_len]
                        # Check where position 1 (second noise token) is attending
                        pos1_attn = avg_attn[1]  # [total_len]
                        max_attn_pos = pos1_attn.argmax().item()
                        max_attn_val = pos1_attn.max().item()
                        # Check attention to different regions
                        attn_to_target = pos1_attn[:ctx_len].sum().item()
                        attn_to_noise = pos1_attn[ctx_len:].sum().item()
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "S", "location": "dflash_worker.py:attn", "message": "Attention distribution (layer 0, pos 1)", "data": {"total_len": total_len, "ctx_len": ctx_len, "max_attn_pos": max_attn_pos, "max_attn_val": max_attn_val, "attn_to_target": attn_to_target, "attn_to_noise": attn_to_noise, "scaling": layer.self_attn.scaling, "q_norm": float(q.norm().cpu()), "k_norm": float(k.norm().cpu())}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    
                    attn_output = torch.matmul(attn_weights, v)
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "step6_attn_out", "message": "Attn output (ORIGINAL: 378)", "data": {"attn_output_shape": list(attn_output.shape), "attn_output_norm": float(attn_output.norm().cpu()), "EXPECT": 378.0}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    
                    # Reshape and project: [num_heads, block_size, head_dim] -> [block_size, hidden]
                    attn_output = attn_output.transpose(0, 1).contiguous().view(block_size, -1)
                    # Convert back to model dtype before o_proj
                    if attn_output.dtype != residual.dtype:
                        attn_output = attn_output.to(residual.dtype)
                    attn_output, _ = layer.self_attn.o_proj(attn_output)
                    
                    # Residual connection (noise only)
                    noise_hidden = residual + attn_output
                    
                    # MLP (noise only)
                    residual = noise_hidden
                    mlp_input = layer.post_attention_layernorm(noise_hidden)
                    mlp_output = layer.mlp(mlp_input)
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "step7_after_mlp", "message": "After MLP (noise_hidden + mlp_output)", "data": {"noise_hidden_pre_mlp_norm": float(residual.norm().cpu()), "mlp_output_norm": float(mlp_output.norm().cpu())}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                    noise_hidden = residual + mlp_output
                    
                    # #region agent log
                    if layer_idx == 0 and i == 0:
                        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "T", "location": "step8_layer_out", "message": "Layer 0 output (ORIGINAL: 8704)", "data": {"layer_output_norm": float(noise_hidden.norm().cpu()), "EXPECT": 8704.0}, "timestamp": __import__('time').time()}) + '\n')
                    # #endregion
                
                # Final norm
                draft_hidden = self.draft_model.model.norm(noise_hidden)  # [block_size, hidden]
            
            # #region agent log
            import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "Q", "location": "dflash_worker.py:574", "message": "Draft output (original DFlash attention)", "data": {"draft_hidden_shape": list(draft_hidden.shape), "block_size": block_size, "total_len": total_len, "ctx_len": ctx_len, "draft_hidden_norm": float(draft_hidden.norm().cpu())}, "timestamp": __import__('time').time()}) + '\n')
            # #endregion
            
            # Get draft logits
            draft_hidden_for_logits = draft_hidden[1:, :]  # Skip first position
            draft_logits = torch.matmul(draft_hidden_for_logits, self.lm_head_weight.t())
            
            # #region agent log
            import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "M", "location": "dflash_worker.py:549", "message": "Draft logits computed", "data": {"draft_logits_shape": list(draft_logits.shape), "draft_logits_max": float(draft_logits.max().cpu()), "draft_logits_min": float(draft_logits.min().cpu()), "lm_head_shape": list(self.lm_head_weight.shape)}, "timestamp": __import__('time').time()}) + '\n')
            # #endregion
            
            block_output_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)
            
            # #region agent log
            import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "ALL", "location": "dflash_worker.py:553", "message": "Draft tokens generated", "data": {"block_output_ids": block_output_ids.tolist(), "verified_id_used": verified_id.item() if isinstance(verified_id, torch.Tensor) else verified_id}, "timestamp": __import__('time').time()}) + '\n')
            # #endregion
            
            all_draft_tokens.append(block_output_ids.flatten())
            current_seq_len = batch.seq_lens[i].item()
            all_positions.append(torch.arange(
                current_seq_len, current_seq_len + block_size, device=self.device
            ))
        
        draft_tokens = torch.cat(all_draft_tokens, dim=0)
        positions = torch.cat(all_positions, dim=0)
        
        # #region agent log
        import json as _json; open('/sgl-workspace/.cursor/debug.log', 'a').write(_json.dumps({"hypothesisId": "ALL", "location": "dflash_worker.py:557", "message": "Sequential draft forward COMPLETE", "data": {"draft_tokens_shape": list(draft_tokens.shape), "positions_shape": list(positions.shape), "num_requests": len(batch.reqs)}, "timestamp": __import__('time').time()}) + '\n')
        # #endregion
        
        return draft_tokens, positions

    def _evict_rejected_cache(
        self,
        batch: ScheduleBatch,
        req,
        state: dict,
        accept_length: int,
    ):
        """Evict rejected noise positions from cache after verification."""
        noise_cache_locs = state.get('noise_cache_locs')
        if noise_cache_locs is None:
            return
        
        rejected_count = self.block_size - accept_length - 1
        if rejected_count > 0:
            # Free rejected positions
            rejected_locs = noise_cache_locs[accept_length + 1:]
            if len(rejected_locs) > 0:
                batch.token_to_kv_pool_allocator.free(rejected_locs)
        
        # Clear tracked noise positions
        state['noise_cache_locs'] = None

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
