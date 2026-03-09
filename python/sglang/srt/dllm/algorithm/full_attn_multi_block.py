"""Full-attention multi-block decoding for bidirectional dLLMs (e.g. Dream).

Matches d3LLM's _sample_multi_block behavior: all max_new_tokens masks are
present from the start. Multiple blocks are decoded in parallel, with new
blocks activated based on previous block's decoding progress.

Key matching with d3LLM _sample_multi_block (no cache):
  - ALL mask positions up to rightmost active block are eligible for decoding
    (not just fully-activated blocks) — entropy threshold is the only gate
  - Forced progress only applies to the first fully-activated block
  - block_add_threshold=0.1 and decoded_token_threshold=0.95 are the defaults
    matching d3LLM's eval commands for parallel multi-block decoding
"""

from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class FullAttnMultiBlock(DllmAlgorithm):
    """Multi-block parallel decoding with entropy threshold for full-attention dLLMs.

    Config keys (from algorithm_config YAML):
      threshold               - entropy threshold for token acceptance  (default: 0.4)
      block_add_threshold     - prev block progress to add next block   (default: 0.1)
      decoded_token_threshold - prev progress for full activation       (default: 0.95)
      cache_delay_iter        - NFE delay before "cached" state         (default: 10000)
      refresh_interval        - periodic refresh interval               (default: 10000)
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.4)
        self.block_add_threshold = config.algorithm_config.get(
            "block_add_threshold", 0.1
        )
        self.decoded_token_threshold = config.algorithm_config.get(
            "decoded_token_threshold", 0.95
        )
        self.cache_delay_iter = config.algorithm_config.get("cache_delay_iter", 10000)
        self.refresh_interval = config.algorithm_config.get("refresh_interval", 10000)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        if batch_size == 0 or forward_batch.input_ids.numel() == 0:
            return LogitsProcessorOutput(next_token_logits=torch.empty(0)), [], True

        item_lens = forward_batch.extend_seq_lens_cpu
        offsets = [0]
        for l in item_lens:
            offsets.append(offsets[-1] + l)

        # Use origin_input_lens if available, otherwise fall back to mask counting
        origin_lens = forward_batch.dllm_origin_input_lens

        # Per-item setup: calculate start positions and mask counts
        start_list = []
        gen_lengths = []
        for i in range(batch_size):
            ids = forward_batch.input_ids[offsets[i] : offsets[i + 1]]
            n_masks = (ids == self.mask_id).sum().item()
            if origin_lens is not None:
                start_list.append(origin_lens[i])
                gen_lengths.append(item_lens[i] - origin_lens[i])
            else:
                start_list.append(item_lens[i] - n_masks)
                gen_lengths.append(n_masks)

        # Fast path: no masks - all tokens already decoded
        if (forward_batch.input_ids == self.mask_id).sum().item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            # Return the generated portion (from start_list[i] to end) for each item
            next_token_ids_list = [
                forward_batch.input_ids[offsets[i] + start_list[i] : offsets[i + 1]]
                for i in range(batch_size)
            ]
            return out.logits_output, next_token_ids_list, out.can_run_graph

        # Build block states per item (matches d3LLM block_states dict)
        # Block 0 = prompt (always complete). Gen blocks start from 1.
        item_blocks = []
        for i in range(batch_size):
            num_blocks = (gen_lengths[i] + self.block_size - 1) // self.block_size
            blocks = []
            for k in range(num_blocks):
                bs = start_list[i] + k * self.block_size
                be = min(bs + self.block_size, start_list[i] + gen_lengths[i])
                blocks.append(
                    {
                        "start": bs,
                        "end": be,
                        "total": be - bs,
                        "mask_count": be - bs,
                        "is_active": k == 0,  # block has been "added"
                        "is_fully_active": k
                        == 0,  # prev block progress >= decoded_token_threshold
                        "completed_at_nfe": None,
                        "is_cached": False,
                    }
                )
            item_blocks.append(blocks)

        nfe = 0
        logits_output = None
        can_run_cuda_graph = False
        max_nfe = sum(gen_lengths) + batch_size  # Upper bound on iterations

        while True:
            # Check termination: all masks decoded
            if (forward_batch.input_ids == self.mask_id).sum().item() == 0:
                break

            # Safety check: prevent infinite loop
            if nfe >= max_nfe:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            nfe += 1
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph

            for i in range(batch_size):
                s, e = offsets[i], offsets[i + 1]
                ids = forward_batch.input_ids[s:e]
                logits = logits_output.full_logits[s:e]
                blocks = item_blocks[i]

                # Update block activation based on previous block progress
                for k in range(1, len(blocks)):
                    prev = blocks[k - 1]
                    prev_progress = 1 - prev["mask_count"] / prev["total"]
                    if not blocks[k]["is_active"]:
                        if (
                            prev_progress >= self.block_add_threshold
                            or prev["mask_count"] == 0
                        ):
                            blocks[k]["is_active"] = True
                    if blocks[k]["is_active"] and not blocks[k]["is_fully_active"]:
                        if prev_progress >= self.decoded_token_threshold:
                            blocks[k]["is_fully_active"] = True

                # Find rightmost active block end (matches d3LLM: any block that
                # is_fully_active OR has mask_count > 0 determines active_end)
                active_end = 0
                for blk in blocks:
                    if blk["is_active"] and (
                        blk["is_fully_active"] or blk["mask_count"] > 0
                    ):
                        active_end = blk["end"]

                if active_end == 0:
                    continue

                # Build decode mask: ALL mask positions up to active_end
                # (matches d3LLM: mask_index_for_decode[:, active_end:] = 0)
                decode_mask = ids[:active_end].eq(self.mask_id)
                n_decode = decode_mask.sum().item()
                if n_decode == 0:
                    continue

                # Entropy-based denoising across all decodable positions
                mask_logits = logits[:active_end][decode_mask]
                probs = torch.softmax(mask_logits.float(), dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                x0 = torch.argmax(mask_logits, dim=-1)

                full_entropy = torch.full(
                    (active_end,), float("inf"), dtype=entropy.dtype, device=ids.device
                )
                full_x0 = ids[:active_end].clone()
                full_entropy[decode_mask] = entropy
                full_x0[decode_mask] = x0

                # Global entropy threshold (matches d3LLM)
                transfer = (full_entropy < self.threshold) & decode_mask

                # Per-block forced progress for first fully-activated block
                # (matches d3LLM: force even if other blocks accepted tokens)
                first_fa_blk = None
                for blk in blocks:
                    if blk["is_fully_active"] and blk["mask_count"] > 0:
                        first_fa_blk = blk
                        break
                if first_fa_blk is not None:
                    bs_, be_ = first_fa_blk["start"], min(
                        first_fa_blk["end"], active_end
                    )
                    blk_transfer = transfer[bs_:be_]
                    if not blk_transfer.any():
                        blk_mask = ids[bs_:be_].eq(self.mask_id)
                        blk_ent = full_entropy[bs_:be_].clone()
                        blk_ent[~blk_mask] = float("inf")
                        best = blk_ent.argmin()
                        transfer[bs_ + best] = True

                # Update ids in-place: expand transfer mask to full ids length
                full_transfer = torch.zeros(
                    len(ids), dtype=torch.bool, device=ids.device
                )
                full_transfer[:active_end] = transfer
                ids[full_transfer] = full_x0[transfer]

                # Update block mask counts and completion tracking
                for blk in blocks:
                    if blk["mask_count"] > 0:
                        new_count = (
                            ids[blk["start"] : blk["end"]].eq(self.mask_id).sum().item()
                        )
                        if new_count == 0 and blk["completed_at_nfe"] is None:
                            blk["completed_at_nfe"] = nfe
                        blk["mask_count"] = new_count

        # Final forward for fresh logits (needed by sglang framework)
        if (
            logits_output is None
            or (forward_batch.input_ids == self.mask_id).sum().item() > 0
        ):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            nfe += 1
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph

        next_token_ids_list = [
            forward_batch.input_ids[offsets[i] + start_list[i] : offsets[i + 1]]
            for i in range(batch_size)
        ]
        logits_output.customized_info = {"nfe": [nfe] * batch_size}
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = FullAttnMultiBlock
