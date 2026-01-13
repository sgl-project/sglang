# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DFlash speculative decoding worker - minimal implementation.

DFlash draft model does NOT use KV cache - it uses pure torch attention.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput

logger = logging.getLogger(__name__)


class DFlashWorker:
    """DFlash worker - loads draft model without KV cache."""

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
        self.device = server_args.device
        self.block_size = server_args.speculative_dflash_block_size
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.server_args = server_args
        self.is_draft_worker = True

        # Load draft model config and model directly
        draft_path = server_args.speculative_draft_model_path
        config = AutoConfig.from_pretrained(draft_path, trust_remote_code=True)

        # Import and instantiate DFlash model directly
        from sglang.srt.models.qwen3_dflash import DFlashDraftModel
        from sglang.srt.configs.model_config import ModelConfig
        self.model = DFlashDraftModel(config)
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=draft_path,
            model_revision=server_args.speculative_draft_model_revision,
        )

        # Load weights
        self._load_weights(draft_path)
        self.model = self.model.to(dtype=torch.bfloat16, device=self.device)
        self.model.eval()

        # Share embeddings and lm_head from target
        target_model = target_worker.model_runner.model
        self.model.set_embed_and_head(
            target_model.model.embed_tokens.weight,
            target_model.lm_head.weight,
        )

        # Configure target to capture multi-layer hidden states
        self.target_layer_ids = self.model.target_layer_ids
        if hasattr(target_model, "set_eagle3_layers_to_capture"):
            target_model.set_eagle3_layers_to_capture(self.target_layer_ids)
            logger.info(f"DFlash: target_layer_ids={self.target_layer_ids}")

        # Get mask token
        tok = AutoTokenizer.from_pretrained(server_args.model_path, trust_remote_code=True)
        if tok.mask_token_id is None:
            tok.add_special_tokens({"mask_token": "<|MASK|>"})
        self.mask_token_id = tok.mask_token_id

        # Per-request state: {rid: {"hidden": tensor, "verified_id": int, "pos": int}}
        self._state: Dict[str, dict] = {}
        logger.info(f"DFlashWorker: block_size={self.block_size}, mask_id={self.mask_token_id}")

    def _load_weights(self, model_path: str):
        """Load weights from HuggingFace hub or local path."""
        from huggingface_hub import hf_hub_download, HfFileSystem
        from safetensors import safe_open
        import os

        # Try to find safetensors files
        fs = HfFileSystem()
        try:
            files = fs.ls(model_path, detail=False)
            safetensor_files = [f for f in files if f.endswith('.safetensors')]
        except Exception:
            # Local path
            if os.path.isdir(model_path):
                safetensor_files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith('.safetensors')]
            else:
                safetensor_files = []

        if not safetensor_files:
            # Fall back to HF hub
            try:
                local_path = hf_hub_download(model_path, "model.safetensors")
                safetensor_files = [local_path]
            except Exception:
                pass

        # Load weights
        weights = []
        for sf_file in safetensor_files:
            # Get local path if it's a HF path
            if not os.path.exists(sf_file):
                try:
                    sf_file = hf_hub_download(model_path, os.path.basename(sf_file))
                except Exception:
                    continue
            with safe_open(sf_file, framework='pt') as f:
                for key in f.keys():
                    weights.append((key, f.get_tensor(key)))

        self.model.load_weights(weights)

    # Expose model_runner-like interface for compatibility
    @property
    def model_runner(self):
        """Fake model_runner for compatibility - delegates to target's pools."""
        target_runner = self.target_worker.model_runner
        class FakeModelRunner:
            def __init__(self, model, target):
                self.model = model
                # Share target's pools
                self.token_to_kv_pool = target.token_to_kv_pool
                self.token_to_kv_pool_allocator = target.token_to_kv_pool_allocator
                self.req_to_token_pool = target.req_to_token_pool
        return FakeModelRunner(self.model, target_runner)

    def get_memory_pool(self):
        """DFlash doesn't use memory pool - delegate to target."""
        return self.target_worker.get_memory_pool()

    def clear_cache_pool(self):
        self._state.clear()

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend():
            return self._prefill(batch)
        return self._decode(batch)

    def _prefill(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Prefill: run target and capture hidden states."""
        mwb = batch.get_model_worker_batch()
        mwb.capture_hidden_mode = CaptureHiddenMode.FULL
        result = self.target_worker.forward_batch_generation(mwb)

        # hidden_states is [total_tokens, num_layers * hidden_size]
        h = result.logits_output.hidden_states
        next_ids = result.next_token_ids
        
        if h is None:
            logger.warning("DFlash: hidden_states is None!")

        offset = 0
        for i, req in enumerate(batch.reqs):
            ext_len = batch.extend_lens[i] if hasattr(batch, "extend_lens") else len(req.origin_input_ids)
            self._state[req.rid] = {
                "hidden": h[offset:offset + ext_len].clone() if h is not None else None,
                "verified_id": next_ids[i].item(),
                "pos": len(req.origin_input_ids),
            }
            offset += ext_len

        batch.spec_info = DFlashDraftInput(
            hidden_states=None,
            verified_id=next_ids.clone(),
            block_size=self.block_size,
        )
        return result

    def _decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Decode: draft -> verify -> accept."""
        if not isinstance(batch.spec_info, DFlashDraftInput):
            return self._fallback(batch)

        # Draft tokens
        draft, positions = self._draft(batch)
        if draft.numel() == 0:
            return self._fallback(batch)

        # Verify
        verify = DFlashVerifyInput(draft, positions, self.block_size)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify
        verify.prepare_for_verify(batch, batch.page_size if hasattr(batch, 'page_size') else 1)

        mwb = batch.get_model_worker_batch()
        mwb.capture_hidden_mode = CaptureHiddenMode.FULL
        result = self.target_worker.forward_batch_generation(mwb, is_verify=True)

        logits, new_ids, accepted = verify.verify(batch, result.logits_output, 1)

        # Update state
        accept_lens = verify.accept_length.cpu().tolist()
        h = logits.hidden_states

        for i, req in enumerate(batch.reqs):
            s = self._state.get(req.rid)
            if s is None:
                continue
            acc = accept_lens[i]
            s["pos"] += acc + 1
            s["verified_id"] = new_ids[i].item()

            # Append new hidden states
            if h is not None:
                start = sum(accept_lens[:i]) + i
                end = start + acc + 1
                if end <= h.shape[0]:
                    new_h = h[start:end]
                    if s["hidden"] is not None:
                        s["hidden"] = torch.cat([s["hidden"], new_h], dim=0)
                    else:
                        s["hidden"] = new_h

        batch.spec_info = DFlashDraftInput(None, new_ids.clone(), self.block_size)
        batch.forward_mode = ForwardMode.DECODE
        self._cleanup(batch)

        return GenerationBatchResult(
            logits_output=logits,
            next_token_ids=new_ids,
            num_accepted_tokens=accepted,
            can_run_cuda_graph=result.can_run_cuda_graph,
            accept_lens=verify.accept_length,
        )

    def _draft(self, batch: ScheduleBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens."""
        all_tokens, all_pos = [], []

        for req in batch.reqs:
            s = self._state.get(req.rid)
            if s is None or s["hidden"] is None:
                continue

            target_hidden = s["hidden"]  # [ctx_len, num_layers * hidden]
            verified_id = s["verified_id"]
            ctx_len = target_hidden.shape[0]

            # Create noise tokens: [verified_id, mask, mask, ...]
            noise = torch.full((self.block_size,), self.mask_token_id, dtype=torch.long, device=self.device)
            noise[0] = verified_id

            # Positions for full sequence
            position_ids = torch.arange(ctx_len + self.block_size, device=self.device).unsqueeze(0)

            # Embed and forward
            noise_emb = self.model.embed_tokens(noise).unsqueeze(0)  # [1, block, hidden]
            target_hidden_batch = target_hidden.unsqueeze(0)  # [1, ctx, hidden*layers]

            with torch.no_grad():
                logits = self.model(noise_emb, target_hidden_batch, position_ids)

            # Sample: logits[0, 1:] predicts tokens 2-N
            draft_ids = torch.argmax(logits[0, 1:], dim=-1)
            full = torch.cat([noise[0:1], draft_ids])

            all_tokens.append(full)
            all_pos.append(torch.arange(s["pos"], s["pos"] + self.block_size, device=self.device))

        if not all_tokens:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)

        return torch.cat(all_tokens), torch.cat(all_pos)

    def _fallback(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Fallback to standard decode."""
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)
        batch.input_ids, batch.output_ids = batch.output_ids, None
        batch.seq_lens = batch.seq_lens + 1
        batch.seq_lens_cpu = batch.seq_lens_cpu + 1
        batch.seq_lens_sum += len(batch.reqs)
        for req in batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1
        return self.target_worker.forward_batch_generation(batch.get_model_worker_batch())

    def _cleanup(self, batch: ScheduleBatch):
        active = {req.rid for req in batch.reqs}
        for rid in list(self._state.keys()):
            if rid not in active:
                del self._state[rid]
