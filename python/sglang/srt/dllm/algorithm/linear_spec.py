"""
LinearSpec algorithm for SGLang DLLM — diffusion model as speculator.

Implements the linear_spec_generate approach from Nemotron-Labs-Diffusion v2:
each block needs only 2 forward passes (1 bidirectional draft + 1 causal verify)
instead of FastDiffuser's iterative denoising.

Algorithm per block:
  1. DRAFT (bidirectional): forward [seed, mask, ..., mask] → draft tokens
  2. VERIFY (causal): forward [seed, draft_1, ..., draft_N] → AR tokens
  3. ACCEPT: c = consecutive shifted matches (draft[i] == ar[i-1]), output seed + c tokens
  4. FREE rejected KV positions

Key insight: in the diffusion model,
  - Bidirectional (draft) logits[i] predicts the token at position i
  - Causal (verify) logits[i] predicts the token at position i+1

So draft[i] and ar[i-1] both predict position i → comparison is shifted by 1.
The seed at position 0 provides the correct AR token (from prefix/prev block),
and ar[c] becomes the seed for the next block.

Usage:
  dllm_algorithm: LinearSpec
  dllm_algorithm_config:
    causal_context: true
    stats_file: null
"""

import json as _json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LinearSpec(DllmAlgorithm):
    """
    LinearSpec: diffusion model as speculator for speculative decoding.

    Each block is processed with exactly 2 forward passes:
      - Draft pass (bidirectional attention): generates candidate tokens
      - Verify pass (causal attention): produces AR-quality tokens
    The seed at position 0 is always correct (from prefix or prev block).
    Acceptance: longest consecutive prefix where draft[i] == ar[i-1] for i>=1.
    Output includes the seed, giving c+1 tokens total.
    """

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        cfg = config.algorithm_config
        self.causal_context: bool = config.causal_context
        self._seed_tokens: Dict[str, int] = {}
        self._eos_token_id: Optional[int] = None

        # Stats
        self._stats_file: Optional[str] = cfg.get("stats_file", None)
        self._stats_forward_passes: int = 0

        # Profiling accumulators (wall-clock with cuda sync, zero-overhead when disabled)
        self._profile: bool = cfg.get("profile", False)
        self._prof_n: int = 0
        self._prof_scheduler: float = 0.0  # between blocks (scheduler overhead)
        self._prof_pre_draft: float = 0.0  # setup inside run() before draft fwd
        self._prof_draft_fwd: float = 0.0
        self._prof_between_fwd: float = 0.0
        self._prof_verify_fwd: float = 0.0
        self._prof_accept: float = 0.0
        self._prof_total: float = 0.0
        self._prof_last_return: float = 0.0  # time.perf_counter at last return

        logger.info(
            "LinearSpec: block_size=%d  causal_context=%s",
            self.block_size,
            self.causal_context,
        )

    def _get_eos_id(self, model_runner: ModelRunner) -> Optional[int]:
        if self._eos_token_id is None:
            try:
                hf_cfg = model_runner.model_config.hf_config
                eos = getattr(hf_cfg, "eos_token_id", None)
                if isinstance(eos, list):
                    eos = eos[0]
                self._eos_token_id = int(eos) if eos is not None else None
            except (AttributeError, TypeError, ValueError, IndexError):
                self._eos_token_id = None
        return self._eos_token_id

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        if self._profile:
            _t_entry = time.perf_counter()

        batch_size = forward_batch.batch_size
        bs = self.block_size
        eos_id = self._get_eos_id(model_runner)

        # ----------------------------------------------------------------
        # Fast path: no masks → single forward (STAGING_PREFILL phase)
        # ----------------------------------------------------------------
        mask_index_all = forward_batch.input_ids == self.mask_id
        if not mask_index_all.any():
            if forward_batch.input_ids.numel() == 0:
                empty_logits = LogitsProcessorOutput(
                    next_token_logits=None, full_logits=None
                )
                return empty_logits, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)

            # Extract seed token from the prefill's last-position logits.
            # In causal mode, logits[-1] predicts the first generated token.
            logits_output = out.logits_output
            if logits_output.next_token_logits is not None:
                rids = forward_batch.rids
                seed_logits = logits_output.next_token_logits  # [batch, V]
                seed_logits_no_mask = seed_logits.clone()
                seed_logits_no_mask[:, self.mask_id] = -np.inf
                seeds = torch.argmax(seed_logits_no_mask, dim=-1)

                for b_idx in range(len(rids)):
                    rid = rids[b_idx]
                    self._seed_tokens[rid] = int(seeds[b_idx].item())

            return logits_output, [], out.can_run_graph

        # ----------------------------------------------------------------
        # Per-request: figure out gen_start (where masks begin in block)
        # ----------------------------------------------------------------
        start_list: List[int] = []
        for b in range(batch_size):
            b_start = b * bs
            b_end = b_start + bs
            n_masks = int(
                (forward_batch.input_ids[b_start:b_end] == self.mask_id).sum()
            )
            start_list.append(bs - n_masks)

        # ----------------------------------------------------------------
        # Seed token handling: inject at first mask position
        # ----------------------------------------------------------------
        rids = forward_batch.rids[:batch_size]
        has_seed = [False] * batch_size
        for b in range(batch_size):
            rid = rids[b]
            if rid in self._seed_tokens:
                seed = self._seed_tokens[rid]
                block_start = b * bs + start_list[b]
                if block_start < forward_batch.input_ids.numel():
                    forward_batch.input_ids[block_start] = seed
                    has_seed[b] = True

        # ----------------------------------------------------------------
        # DRAFT pass (bidirectional attention)
        # ----------------------------------------------------------------
        if self._profile:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
            if self._prof_last_return > 0:
                self._prof_scheduler += _t_entry - self._prof_last_return
            self._prof_pre_draft += _t0 - _t_entry

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        self._stats_forward_passes += 1
        draft_logits = out.logits_output.full_logits  # [B*bs, V]

        if self._profile:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        # Get draft tokens via argmax (exclude mask_id, in-place)
        draft_logits[:, self.mask_id] = -1e9
        draft_all = torch.argmax(draft_logits, dim=-1)  # [B*bs]

        # Replace mask positions with draft tokens (vectorized)
        mask_positions = forward_batch.input_ids == self.mask_id
        forward_batch.input_ids[mask_positions] = draft_all[mask_positions]

        if self._profile:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()

        # ----------------------------------------------------------------
        # VERIFY pass (causal attention)
        # ----------------------------------------------------------------
        forward_batch.dllm_causal_kv_update = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_causal_kv_update = False
        self._stats_forward_passes += 1
        verify_logits = out.logits_output.full_logits  # [B*bs, V]
        logits_output = out.logits_output
        can_run_graph = out.can_run_graph

        # Get AR tokens via argmax (exclude mask_id, in-place)
        verify_logits[:, self.mask_id] = -1e9
        ar_all = torch.argmax(verify_logits, dim=-1)  # [B*bs]

        if self._profile:
            torch.cuda.synchronize()
            _t3 = time.perf_counter()

        # ----------------------------------------------------------------
        # ACCEPT: shifted comparison — draft[i] vs ar[i-1]
        # Both predict position i (draft from bidirectional, ar from causal shift).
        # Output: [seed, ar[0], ar[1], ..., ar[c-1]] = c+1 tokens
        # ----------------------------------------------------------------
        next_token_ids_list: List[torch.Tensor] = []
        accepted_counts: List[int] = []

        for b in range(batch_size):
            b_start = b * bs
            gen_start = start_list[b]
            gen_len = bs - gen_start

            # Vectorized shifted consecutive match: draft[i] vs ar[i-1]
            if gen_len > 1:
                offset = b_start + gen_start
                draft_block = forward_batch.input_ids[offset + 1 : offset + gen_len]
                ar_prev = ar_all[offset : offset + gen_len - 1]
                matches = draft_block == ar_prev
                # cumprod of bools: 1,1,...,1,0,0,...  sum = count of leading matches
                c = int(matches.cumprod(0).sum().item())
            else:
                c = 0

            # accepted = 1 (seed) + c (matches) = c + 1
            accepted = c + 1

            # Build output: [seed, ar[0], ar[1], ..., ar[c-1]]
            seed_pos = b_start + gen_start
            seed_tensor = forward_batch.input_ids[seed_pos : seed_pos + 1]
            if c > 0:
                ar_slice = ar_all[seed_pos : seed_pos + c]
                output_tokens = torch.cat([seed_tensor, ar_slice])
            else:
                output_tokens = seed_tensor

            # EOS check: vectorized
            if eos_id is not None:
                eos_mask = output_tokens == eos_id
                if eos_mask.any():
                    eos_pos = int(eos_mask.to(torch.int32).argmax().item()) + 1
                    output_tokens = output_tokens[:eos_pos]
                    accepted = eos_pos

            next_token_ids_list.append(output_tokens)
            accepted_counts.append(accepted)

            # Seed for next block: ar[c] predicts position c+1
            rid = rids[b]
            ar_c_pos = b_start + gen_start + c
            if ar_c_pos < b_start + bs:
                self._seed_tokens[rid] = int(ar_all[ar_c_pos].item())
            else:
                self._seed_tokens[rid] = int(ar_all[b_start + bs - 1].item())

        # ----------------------------------------------------------------
        # Clean up seeds for requests that hit EOS
        # ----------------------------------------------------------------
        for b in range(batch_size):
            tokens = next_token_ids_list[b]
            if (
                eos_id is not None
                and len(tokens) > 0
                and int(tokens[-1].item()) == eos_id
            ):
                self._seed_tokens.pop(rids[b], None)

        # ----------------------------------------------------------------
        # Stats
        # ----------------------------------------------------------------
        total_accepted = sum(accepted_counts)
        avg_accepted = total_accepted / max(batch_size, 1)

        if self._stats_file:
            with open(self._stats_file, "a") as _sf:
                for b in range(batch_size):
                    tokens = int(next_token_ids_list[b].shape[0])
                    gen_len = bs - start_list[b]
                    _sf.write(
                        _json.dumps(
                            {
                                "forward_passes": 2,
                                "tokens": tokens,
                                "block_gen_positions": gen_len,
                                "acceptance_rate": (
                                    tokens / gen_len if gen_len > 0 else 0
                                ),
                            }
                        )
                        + "\n"
                    )

        logger.debug(
            "LinearSpec block done: fp=%d  avg_accepted=%.1f/%d  total_tokens=%d",
            self._stats_forward_passes,
            avg_accepted,
            bs,
            total_accepted,
        )

        if self._profile:
            _t4 = time.perf_counter()
            self._prof_draft_fwd += _t1 - _t0
            self._prof_between_fwd += _t2 - _t1
            self._prof_verify_fwd += _t3 - _t2
            self._prof_accept += _t4 - _t3
            self._prof_total += _t4 - _t0
            self._prof_n += 1
            if self._prof_n % 500 == 0:
                n = self._prof_n
                sched = self._prof_scheduler / max(n - 1, 1) * 1000
                total_wall = (self._prof_total + self._prof_scheduler) / n * 1000
                fwd_time = (self._prof_draft_fwd + self._prof_verify_fwd) / n * 1000
                logger.info(
                    "LinearSpec PROFILE (%d blocks, per-block avg): "
                    "scheduler=%.2fms  pre_draft=%.2fms  draft_fwd=%.2fms  "
                    "between_fwd=%.2fms  verify_fwd=%.2fms  accept=%.2fms | "
                    "total_wall=%.2fms  fwd_only=%.2fms  overhead=%.1f%%",
                    n,
                    sched,
                    self._prof_pre_draft / n * 1000,
                    self._prof_draft_fwd / n * 1000,
                    self._prof_between_fwd / n * 1000,
                    self._prof_verify_fwd / n * 1000,
                    self._prof_accept / n * 1000,
                    total_wall,
                    fwd_time,
                    (1 - fwd_time / total_wall) * 100 if total_wall > 0 else 0,
                )
            self._prof_last_return = time.perf_counter()

        return logits_output, next_token_ids_list, can_run_graph


Algorithm = LinearSpec
