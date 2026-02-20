from typing import Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class HierarchyBlock(DllmAlgorithm):
    """Fast dLLM v2 hierarchical block decoding with token inheritance."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.sub_block_size = config.algorithm_config.get("sub_block_size", 8)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        self.debug = config.algorithm_config.get("debug", True)
        self.use_AR_for_first_token = config.algorithm_config.get("use_AR_for_first_token", True)
        
        self.last_inherited_token = None
        self.last_block_end_position = None
        self.tokenizer = None
    
    def _decode_tokens(self, token_ids, model_runner):
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                if hasattr(model_runner, 'model_config') and hasattr(model_runner.model_config, 'hf_config'):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_runner.model_config.hf_config._name_or_path,
                        trust_remote_code=True
                    )
            except Exception as e:
                logger.warning(f"[HierarchyBlock] Failed to load tokenizer: {e}")
                self.tokenizer = False
        
        if self.tokenizer and self.tokenizer is not False:
            try:
                return self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception as e:
                return f"<decode_error: {e}>"
        return None

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        total_len = len(forward_batch.input_ids)
        block_mask = forward_batch.input_ids == self.mask_id
        num_masked = block_mask.sum().item()
        num_sub_blocks = total_len // self.sub_block_size
        block_start = total_len - num_masked
        # num_sub_blocks = (num_masked + self.sub_block_size - 1) // self.sub_block_size

        # if self.debug:
        #     logger.info(f"[HierarchyBlock] total_len={total_len}, num_masked={num_masked}, "
        #                f"block_start={block_start}, num_sub_blocks={num_sub_blocks}, "
        #                f"use_AR_for_first={self.use_AR_for_first_token and block_start > 0}")
        #     logger.info(f"[HierarchyBlock] input_ids[:5]={forward_batch.input_ids[:5].tolist()}, "
        #                f"input_ids[-5:]={forward_batch.input_ids[-5:].tolist()}")
        #     if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
        #         positions_list = forward_batch.positions.tolist()
        #         logger.info(f"[HierarchyBlock] positions[:5]={positions_list[:5]}, "
        #                    f"positions[-5:]={positions_list[-5:]}")

        
        # Detect new request (positions start from 0) and clear inheritance
        is_new_request = False
        if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
            if forward_batch.positions[0] == 0:
                is_new_request = True
                # if self.last_inherited_token is not None and self.debug:
                #     logger.info(f"[HierarchyBlock] New request detected, clearing inheritance")
                self.last_inherited_token = None
                self.last_block_end_position = None
        
        # Handle token inheritance for all-mask blocks
        first_token_is_mask = forward_batch.input_ids[0] == self.mask_id
        if first_token_is_mask and self.last_inherited_token is not None and not is_new_request:
            forward_batch.input_ids[0] = self.last_inherited_token
        
        # input_text = self._decode_tokens(forward_batch.input_ids.tolist(), model_runner)
        # logger.info(f"Input text: {input_text}")
        
        #     if self.debug:
        #         logger.info(f"[HierarchyBlock] Inherited token: {self.last_inherited_token}")
        # elif self.debug and first_token_is_mask:
        #     logger.info(f"[HierarchyBlock] First mask, no inheritance")

        # AR mode should only be used when there's prompt before masks
        # For all-mask blocks with inheritance, first token is already filled, no need for AR
        # use_AR_mode = self.use_AR_for_first_token and block_start > 0
        
        # Find first mask position (only needed for AR mode)
        # first_mask_rel_pos = -1
        # if use_AR_mode:
        #     block_mask_full = forward_batch.input_ids[block_start:] == self.mask_id
        #     first_mask_rel_pos = block_mask_full.nonzero(as_tuple=True)[0][0].item() if block_mask_full.any() else -1
        
        # Process sub-blocks
        for sub_idx in range(num_sub_blocks):
            rel_start = sub_idx * self.sub_block_size
            rel_end = rel_start + self.sub_block_size
            # contains_first_mask = (use_AR_mode and first_mask_rel_pos >= 0 and 
            #                       first_mask_rel_pos >= rel_start and first_mask_rel_pos < rel_end)

            while True:
                sub_mask = forward_batch.input_ids[rel_start:rel_end] == self.mask_id
                if sub_mask.sum().item() == 0:
                    break

                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
                full_logits = logits_output.full_logits
                assert full_logits is not None

                # if self.debug:
                #     logger.info(f"[HierarchyBlock] sub_idx={sub_idx}, logits.shape={full_logits.shape}")
                
                # Token shift: [L0, L0, L1, ..., LN-2]
                if self.token_shift > 0:
                    shifted_full = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                else:
                    shifted_full = full_logits
                
                # Extract block and sub-block logits
                # logits_len = full_logits.shape[0]
                # if logits_len == total_len:
                #     shifted_block = shifted_full[block_start:]
                # elif logits_len >= num_masked:
                #     shifted_block = shifted_full[-num_masked:]
                # else:
                #     shifted_block = shifted_full
                
                sub_logits = shifted_full[rel_start:rel_end, :]
                
                # Compute predictions and confidence
                preds = sub_logits.argmax(dim=-1)
                probs = F.softmax(sub_logits, dim=-1)
                conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                conf = torch.where(sub_mask, conf, torch.tensor(-np.inf, device=conf.device))

                # Select unmask strategy
                # AR mode: only for first generation token after prompt
                # if contains_first_mask and forward_batch.input_ids[block_start + first_mask_rel_pos] == self.mask_id:
                #     # AR: unmask first generation token (after prompt) only
                #     unmask = torch.zeros_like(sub_mask, dtype=torch.bool)
                #     first_mask_idx_in_sub = first_mask_rel_pos - rel_start
                #     if 0 <= first_mask_idx_in_sub < len(sub_mask) and sub_mask[first_mask_idx_in_sub]:
                #         unmask[first_mask_idx_in_sub] = True
                #     # if self.debug:
                #     #     logger.info(f"[HierarchyBlock] AR unmask first generation token at abs_pos={block_start + first_mask_rel_pos}")
                # else:
                # Confidence-based unmask
                unmask = conf > self.threshold
                if unmask.sum().item() == 0:
                    unmask[conf.argmax()] = True
                unmask = unmask & sub_mask

                # Update tokens
                # abs_start = block_start + rel_start
                # abs_end = block_start + rel_end
                forward_batch.input_ids[rel_start:rel_end] = torch.where(
                    unmask, preds, forward_batch.input_ids[rel_start:rel_end]
                )

                # if self.debug:
                #     unmask_positions = [abs_start + i for i in unmask.nonzero(as_tuple=True)[0].tolist()]
                #     unmask_tokens = preds[unmask].tolist()
                #     logger.info(f"[HierarchyBlock] Unmasked {unmask.sum().item()} tokens at {unmask_positions}: {unmask_tokens}")
                    
                #     decoded_block = self._decode_tokens(forward_batch.input_ids[block_start:].tolist(), model_runner)
                #     if decoded_block:
                #         logger.info(f"[HierarchyBlock] Block: {repr(decoded_block)}")

        # Final forward to get inherited token for next block
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        full_logits = logits_output.full_logits
        
        if full_logits is not None:
            self.last_inherited_token = full_logits[-1].argmax().item()
            # if self.debug:
            #     logger.info(f"[HierarchyBlock] Stored inherited token: {self.last_inherited_token}")
        
        # Debug: print final block state
        # if self.debug:
        #     logger.info(f"[HierarchyBlock] ===== BLOCK COMPLETED =====")
        #     block_token_ids = forward_batch.input_ids[block_start:].tolist()
        #     decoded_block = self._decode_tokens(block_token_ids, model_runner)
        #     logger.info(f"[HierarchyBlock] Block IDs: {block_token_ids}")
        #     if decoded_block:
        #         logger.info(f"[HierarchyBlock] Block text: {repr(decoded_block)}")
            
        #     if block_start > 0:
        #         decoded_full = self._decode_tokens(forward_batch.input_ids.tolist(), model_runner)
        #         if decoded_full:
        #             logger.info(f"[HierarchyBlock] Full: {repr(decoded_full)}")
        
        # decoded_full = self._decode_tokens(forward_batch.input_ids.tolist(), model_runner)
        # logger.info(f"[HierarchyBlock] Decoded block: {decoded_full}")
        # Update position tracking
        if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
            self.last_block_end_position = forward_batch.positions[-1].item()

        return logits_output, forward_batch.input_ids[block_start:], can_run_cuda_graph


Algorithm = HierarchyBlock
