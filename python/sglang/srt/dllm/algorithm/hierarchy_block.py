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
    """
    Fast dLLM v2 hierarchical block decoding.
    
    Process sub-blocks sequentially: only unmask tokens in current sub-block.
    Token shift: [L0, L1, ..., LN-1] -> [L0, L0, L1, ..., LN-2]
    
    Key feature: Maintains inherited tokens across blocks to ensure correct
    prediction of each block's first token (like Fast dLLM prefill).
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.sub_block_size = config.algorithm_config.get("sub_block_size", 8)
        self.token_shift = config.algorithm_config.get("token_shift", 1)
        self.debug = config.algorithm_config.get("debug", True)
        
        # Maintain inherited first token from previous block
        self.last_inherited_token = None
        
        # Track last block end position for continuity checks
        self.last_block_end_position = None
        
        # Tokenizer for decoding (loaded lazily)
        self.tokenizer = None
    
    def _decode_tokens(self, token_ids, model_runner):
        """Helper method to decode token IDs to text."""
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
                self.tokenizer = False  # Mark as failed to avoid retry
        
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
        block_start = total_len - num_masked

        num_sub_blocks = (num_masked + self.sub_block_size - 1) // self.sub_block_size

        if self.debug:
            logger.info(f"[HierarchyBlock] total_len={total_len}, num_masked={num_masked}, "
                       f"block_start={block_start}, block_size={self.block_size}, "
                       f"sub_block_size={self.sub_block_size}, num_sub_blocks={num_sub_blocks}, "
                       f"token_shift={self.token_shift}")
            logger.info(f"[HierarchyBlock] input_ids[:5]={forward_batch.input_ids[:5].tolist()}, "
                       f"input_ids[-5:]={forward_batch.input_ids[-5:].tolist()}")
            if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
                positions_list = forward_batch.positions.tolist()
                logger.info(f"[HierarchyBlock] positions[:5]={positions_list[:5]}, "
                           f"positions[-5:]={positions_list[-5:]}")
                # Check position continuity for all-mask blocks
                if block_start == 0 and len(positions_list) > 0:
                    expected_start = self.last_block_end_position + 1 if self.last_block_end_position is not None else 0
                    actual_start = positions_list[0]
                    if expected_start != actual_start:
                        logger.warning(f"[HierarchyBlock] Position discontinuity detected! "
                                     f"Expected start={expected_start}, actual start={actual_start}. "
                                     f"This will cause incorrect generation!")

        # Step 1: Handle first token inheritance for all-mask blocks
        # If first token in input_ids is mask AND we have inherited token, replace it
        first_token_is_mask = forward_batch.input_ids[0] == self.mask_id
        
        if first_token_is_mask and self.last_inherited_token is not None:
            # All-mask block: use inherited token from previous block
            inherited_token = self.last_inherited_token
            # Replace the first mask token with inherited token
            # Note: block_start is the absolute position, for all-mask block it's 0
            forward_batch.input_ids[0] = inherited_token
            if self.debug:
                logger.info(f"[HierarchyBlock] INHERITED first token: position=0, "
                           f"token_id={inherited_token} (from previous block)")
                logger.info(f"[HierarchyBlock] After inheritance: input_ids[:5]={forward_batch.input_ids[:5].tolist()}")
        elif self.debug:
            if first_token_is_mask:
                logger.info(f"[HierarchyBlock] First token is MASK but no inherited token available (first block)")
            else:
                logger.info(f"[HierarchyBlock] First token is real token (prompt in block), "
                           f"token_id={forward_batch.input_ids[0].item()}")

        # Step 2: Process sub-blocks sequentially (remaining tokens)
        for sub_idx in range(num_sub_blocks):
            rel_start = sub_idx * self.sub_block_size
            rel_end = min(rel_start + self.sub_block_size, num_masked)

            while True:
                block_mask = forward_batch.input_ids[block_start:] == self.mask_id
                sub_mask = block_mask[rel_start:rel_end]
                if sub_mask.sum().item() == 0:
                    break

                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
                
                full_logits = logits_output.full_logits
                assert full_logits is not None
                if full_logits.dim() == 3:
                    full_logits = full_logits.reshape(-1, full_logits.shape[-1])

                logits_len = full_logits.shape[0]
                
                if self.debug:
                    logger.info(f"[HierarchyBlock] sub_idx={sub_idx}, rel_start={rel_start}, "
                               f"rel_end={rel_end}, full_logits.shape={full_logits.shape}, "
                               f"logits_len={logits_len}")
                
                # Extract block logits and apply token shift
                # For token shift: logits[i] predicts token[i+1]
                # Like Fast dLLM v2, we need to shift logits: [L0, L0, L1, ..., LN-2]
                # Note: full_logits is 2D [seq_len, vocab_size] after reshape
                
                if logits_len == total_len:
                    # We have full logits
                    block_logits = full_logits[block_start:]
                elif logits_len >= num_masked:
                    # We have enough logits for the block
                    block_logits = full_logits[-num_masked:]
                else:
                    block_logits = full_logits

                if self.debug:
                    logger.info(f"[HierarchyBlock] block_logits.shape={block_logits.shape} (before shift)")
                
                # Apply token shift: [L0, L0, L1, ..., LN-2]
                # This aligns logits[i] to predict token[i] instead of token[i+1]
                # block_logits is 2D: [seq_len, vocab_size]
                if self.token_shift > 0:
                    # Duplicate first logit and drop last logit
                    first_logit = block_logits[:1, :]  # [1, vocab]
                    rest_logits = block_logits[:-1, :]  # [N-1, vocab]
                    shifted = torch.cat([first_logit, rest_logits], dim=0)
                    
                    if self.debug:
                        logger.info(f"[HierarchyBlock] Token shift applied: [L0, L0, L1, ..., L{block_logits.shape[0]-2}], "
                                   f"shifted.shape={shifted.shape}")
                else:
                    shifted = block_logits
                
                # Take only current sub-block logits
                # shifted[i] predicts token[block_start + i]
                sub_logits = shifted[rel_start:rel_end, :]

                if self.debug:
                    logger.info(f"[HierarchyBlock] shifted.shape={shifted.shape}, "
                               f"sub_logits.shape={sub_logits.shape}, sub_mask.shape={sub_mask.shape}")
                
                # Compute predictions and confidence
                # sub_logits shape: [sub_block_len, vocab_size]
                preds = sub_logits.argmax(dim=-1)
                probs = F.softmax(sub_logits, dim=-1)
                conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                
                # Only consider masked positions
                conf = torch.where(sub_mask, conf, torch.tensor(-np.inf, device=conf.device))

                # Select tokens to unmask
                unmask = conf > self.threshold
                if unmask.sum().item() == 0:
                    unmask[conf.argmax()] = True
                unmask = unmask & sub_mask

                # Update input_ids
                abs_start = block_start + rel_start
                abs_end = block_start + rel_end
                forward_batch.input_ids[abs_start:abs_end] = torch.where(
                    unmask, preds, forward_batch.input_ids[abs_start:abs_end]
                )

                if self.debug:
                    # Show which positions are being unmasked and their predictions
                    unmask_positions_rel = unmask.nonzero(as_tuple=True)[0].tolist()
                    # Convert to absolute positions in input_ids
                    unmask_positions_abs = [abs_start + pos for pos in unmask_positions_rel]
                    unmask_tokens = preds[unmask].tolist()
                    logger.info(f"[HierarchyBlock] unmask_count={unmask.sum().item()}, "
                               f"positions(rel)={unmask_positions_rel}, positions(abs)={unmask_positions_abs}, "
                               f"token_ids={unmask_tokens}")
                    
                    # Decode current state (prompt + generation)
                    all_token_ids = forward_batch.input_ids.tolist()
                    decoded_all = self._decode_tokens(all_token_ids, model_runner)
                    if decoded_all:
                        # Also show just the block part for clarity
                        block_token_ids = forward_batch.input_ids[block_start:].tolist()
                        decoded_block = self._decode_tokens(block_token_ids, model_runner)
                        logger.info(f"[HierarchyBlock] Current block state: {repr(decoded_block)}")
                        if block_start > 0:
                            # Has prompt, show full sequence
                            logger.info(f"[HierarchyBlock] Full sequence (prompt+block): {repr(decoded_all)}")

        # Final forward
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        # Step 3: Compute and store inherited token for next block
        # Use the last token's logits to predict next block's first token
        full_logits = logits_output.full_logits
        if full_logits is not None:
            if full_logits.dim() == 3:
                full_logits = full_logits.reshape(-1, full_logits.shape[-1])
            
            # Last position's logits predict next token
            next_block_first_token = full_logits[-1].argmax().item()
            self.last_inherited_token = next_block_first_token
            
            if self.debug:
                logger.info(f"[HierarchyBlock] Stored inherited token for next block: "
                           f"token_id={next_block_first_token}")
        
        # Step 4: Decode and print block tokens after processing
        if self.debug:
            logger.info(f"[HierarchyBlock] ===== BLOCK COMPLETED =====")
            
            # Decode block part
            block_token_ids = forward_batch.input_ids[block_start:].tolist()
            decoded_block = self._decode_tokens(block_token_ids, model_runner)
            logger.info(f"[HierarchyBlock] Block token IDs: {block_token_ids}")
            if decoded_block:
                logger.info(f"[HierarchyBlock] Block decoded text: {repr(decoded_block)}")
            
            # Decode full sequence (prompt + block)
            if block_start > 0:
                all_token_ids = forward_batch.input_ids.tolist()
                prompt_token_ids = forward_batch.input_ids[:block_start].tolist()
                decoded_all = self._decode_tokens(all_token_ids, model_runner)
                decoded_prompt = self._decode_tokens(prompt_token_ids, model_runner)
                
                logger.info(f"[HierarchyBlock] Prompt token IDs: {prompt_token_ids}")
                if decoded_prompt:
                    logger.info(f"[HierarchyBlock] Prompt decoded text: {repr(decoded_prompt)}")
                if decoded_all:
                    logger.info(f"[HierarchyBlock] Full sequence (prompt+block): {repr(decoded_all)}")
        
        # Step 5: Update last block end position for next block continuity check
        if hasattr(forward_batch, 'positions') and forward_batch.positions is not None:
            self.last_block_end_position = forward_batch.positions[-1].item()
            if self.debug:
                logger.info(f"[HierarchyBlock] Updated last_block_end_position={self.last_block_end_position}")

        return logits_output, forward_batch.input_ids[block_start:], can_run_cuda_graph


Algorithm = HierarchyBlock