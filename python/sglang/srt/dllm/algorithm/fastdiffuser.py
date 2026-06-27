"""FastDiffuser denoising for block-diffusion LMs."""

import dataclasses
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_count: int, steps: int) -> List[int]:
    base = mask_count // steps
    remainder = mask_count % steps
    return [base + (1 if i < remainder else 0) for i in range(steps)]


class FastDiffuser(DllmAlgorithm):
    """Fill a masked block over several bidirectional forwards."""

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        cfg = config.algorithm_config
        self.temperature: float = cfg.get("temperature", 0.0)
        self.threshold: Optional[float] = cfg.get("threshold", None)
        self.max_steps: int = config.max_steps
        self.causal_context: bool = config.causal_context
        self.tokens_per_step: Optional[int] = cfg.get("tokens_per_step", None)

        self._eos_token_id: Optional[int] = None

        self._stats_forward_passes: int = 0
        self._stats_tokens_generated: int = 0
        self._stats_cuda_graph: int = 0
        self._stats_eager: int = 0
        self._stats_file: Optional[str] = cfg.get("stats_file", None)

        logger.info(
            "FastDiffuser: block_size=%d  max_steps=%d  temperature=%s  "
            "threshold=%s",
            self.block_size,
            self.max_steps,
            self.temperature,
            self.threshold,
        )

    def _flush_stats(self) -> None:
        if not self._stats_file:
            return
        import json
        import os

        data = {
            "forward_passes": self._stats_forward_passes,
            "tokens_generated": self._stats_tokens_generated,
        }
        tmp = self._stats_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self._stats_file)

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

    def _compute_confidence(
        self,
        logits: torch.Tensor,  # [block_size, vocab]
        mask_index: torch.Tensor,  # [block_size] bool
        current_ids: torch.Tensor,  # [block_size] int
        eos_freeze: torch.Tensor,  # [block_size] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep mask_id out of committed tokens while matching HF confidence.
        logits_for_argmax = logits.clone()
        logits_for_argmax[:, self.mask_id] = -np.inf
        logits_with_noise = _add_gumbel_noise(logits_for_argmax, self.temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        x0_p = F.softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        x0 = torch.where(mask_index, x0, current_ids)
        active = mask_index & ~eos_freeze
        confidence = torch.where(active, x0_p, torch.full_like(x0_p, -np.inf))
        return x0, confidence

    def _compact_forward_batch(
        self,
        forward_batch: ForwardBatch,
        active_indices: List[int],
    ) -> ForwardBatch:
        bs = self.block_size
        n = len(active_indices)
        device = forward_batch.input_ids.device

        active_t = torch.tensor(active_indices, dtype=torch.long, device=device)
        active_t_cpu = torch.tensor(active_indices, dtype=torch.long)

        token_idx = torch.cat(
            [torch.arange(a * bs, (a + 1) * bs, device=device) for a in active_indices]
        )

        new_input_ids = forward_batch.input_ids[token_idx].clone()
        new_positions = forward_batch.positions[token_idx]
        new_out_cache_loc = forward_batch.out_cache_loc[token_idx]

        new_seq_lens = forward_batch.seq_lens[active_t]
        new_orig_seq_lens = (
            forward_batch.orig_seq_lens[active_t]
            if forward_batch.orig_seq_lens is not None
            else None
        )
        new_req_pool = forward_batch.req_pool_indices[active_t]
        new_ext_seq = forward_batch.extend_seq_lens[active_t]
        new_ext_pre = forward_batch.extend_prefix_lens[active_t]

        new_ext_start = torch.zeros(n, dtype=torch.long, device=device)
        new_ext_start[1:] = new_ext_seq[:-1].cumsum(0)

        new_ext_pre_cpu = (
            [forward_batch.extend_prefix_lens_cpu[a] for a in active_indices]
            if forward_batch.extend_prefix_lens_cpu is not None
            else None
        )
        new_ext_seq_cpu = (
            [forward_batch.extend_seq_lens_cpu[a] for a in active_indices]
            if forward_batch.extend_seq_lens_cpu is not None
            else None
        )
        new_seq_lens_cpu = (
            forward_batch.seq_lens_cpu[active_t_cpu]
            if forward_batch.seq_lens_cpu is not None
            else None
        )

        return dataclasses.replace(
            forward_batch,
            batch_size=n,
            input_ids=new_input_ids,
            req_pool_indices=new_req_pool,
            seq_lens=new_seq_lens,
            out_cache_loc=new_out_cache_loc,
            seq_lens_sum=int(new_seq_lens.sum()),
            orig_seq_lens=new_orig_seq_lens,
            positions=new_positions,
            extend_num_tokens=n * bs,
            extend_seq_lens=new_ext_seq,
            extend_prefix_lens=new_ext_pre,
            extend_start_loc=new_ext_start,
            extend_prefix_lens_cpu=new_ext_pre_cpu,
            extend_seq_lens_cpu=new_ext_seq_cpu,
            seq_lens_cpu=new_seq_lens_cpu,
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        eos_id = self._get_eos_id(model_runner)

        mask_index_all = forward_batch.input_ids == self.mask_id
        if not mask_index_all.any():
            if forward_batch.input_ids.numel() == 0:
                empty_logits = LogitsProcessorOutput(
                    next_token_logits=None, full_logits=None
                )
                return empty_logits, [], False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        start_list: List[int] = []
        for b in range(batch_size):
            bs, be = b * self.block_size, (b + 1) * self.block_size
            n_masks = int((forward_batch.input_ids[bs:be] == self.mask_id).sum())
            start_list.append(self.block_size - n_masks)

        finished = [False] * batch_size
        req_steps_active = [0] * batch_size

        for step in range(self.max_steps):
            active_indices = [b for b in range(batch_size) if not finished[b]]
            if not active_indices:
                break
            for b in active_indices:
                req_steps_active[b] += 1

            self._stats_forward_passes += 1
            if len(active_indices) == batch_size:
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            else:
                compact_fb = self._compact_forward_batch(forward_batch, active_indices)
                out = model_runner.forward(compact_fb, pp_proxy_tensors=None)

            logits_output, can_run_graph = out.logits_output, out.can_run_graph
            if can_run_graph:
                self._stats_cuda_graph += 1
            else:
                self._stats_eager += 1

            for compact_i, orig_b in enumerate(active_indices):
                lbs = compact_i * self.block_size
                lbe = lbs + self.block_size
                bs_o = orig_b * self.block_size
                be_o = bs_o + self.block_size

                block_ids = forward_batch.input_ids[bs_o:be_o]
                block_mask = block_ids == self.mask_id

                if not block_mask.any():
                    finished[orig_b] = True
                    continue

                block_logits = logits_output.full_logits[lbs:lbe]

                # Prompt EOS should not freeze generated positions.
                gen_start = start_list[orig_b]
                eos_freeze = torch.zeros_like(block_mask)
                if eos_id is not None:
                    placed_in_gen = ~block_mask
                    placed_in_gen[:gen_start] = False
                    eos_placed = placed_in_gen & (block_ids == eos_id)
                    if eos_placed.any():
                        first_eos = int(eos_placed.nonzero(as_tuple=True)[0][0])
                        eos_freeze[first_eos:] = True

                x0, confidence = self._compute_confidence(
                    block_logits, block_mask, block_ids, eos_freeze
                )

                remaining = int(block_mask.sum())
                if self.threshold is not None:
                    k = max(1, int((confidence >= self.threshold).sum()))
                else:
                    steps_left = self.max_steps - step
                    k = _get_num_transfer_tokens(remaining, steps_left)[0]
                k = min(k, remaining)

                _, top_idx = torch.topk(confidence, k=k)
                block_ids[top_idx] = x0[top_idx]

                if eos_id is not None:
                    gen_top_idx = top_idx[top_idx >= gen_start]
                    if len(gen_top_idx) and (block_ids[gen_top_idx] == eos_id).any():
                        gen_eos = (block_ids[gen_start:] == eos_id).nonzero(
                            as_tuple=True
                        )[0]
                        if len(gen_eos):
                            first_eos = gen_start + int(gen_eos[0])
                            still_mask = block_ids[first_eos:] == self.mask_id
                            block_ids[first_eos:][still_mask] = eos_id

                if not (block_ids == self.mask_id).any():
                    finished[orig_b] = True

        # Refresh KV/logits for the accepted block.
        self._stats_forward_passes += 1
        if self.causal_context:
            forward_batch.dllm_causal_kv_update = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if self.causal_context:
            forward_batch.dllm_causal_kv_update = False
        logits_output, can_run_graph = out.logits_output, out.can_run_graph
        if can_run_graph:
            self._stats_cuda_graph += 1
        else:
            self._stats_eager += 1

        # Threshold mode may leave masks; commit them before the next block.
        forced_any = False
        for b in range(batch_size):
            bs_o = b * self.block_size
            be_o = bs_o + self.block_size
            block_ids = forward_batch.input_ids[bs_o:be_o]
            remaining_mask = (block_ids == self.mask_id).clone()
            if not remaining_mask.any():
                continue
            block_logits = logits_output.full_logits[bs_o:be_o].clone()
            block_logits[:, self.mask_id] = -np.inf
            x0_final = torch.argmax(block_logits, dim=-1)
            if eos_id is not None:
                gen_start = start_list[b]
                placed_in_gen = ~remaining_mask
                placed_in_gen[:gen_start] = False
                eos_placed = placed_in_gen & (block_ids == eos_id)
                if eos_placed.any():
                    first_eos = int(eos_placed.nonzero(as_tuple=True)[0][0])
                    remaining_mask[first_eos:] = False
            if remaining_mask.any():
                block_ids[remaining_mask] = x0_final[remaining_mask]
                forced_any = True

        if forced_any:
            self._stats_forward_passes += 1
            if self.causal_context:
                forward_batch.dllm_causal_kv_update = True
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            if self.causal_context:
                forward_batch.dllm_causal_kv_update = False
            logits_output, can_run_graph = out.logits_output, out.can_run_graph
            if can_run_graph:
                self._stats_cuda_graph += 1
            else:
                self._stats_eager += 1

        token_grid = forward_batch.input_ids.reshape(batch_size, self.block_size)
        next_token_ids_list = [
            token_grid[b, start_list[b] :] for b in range(batch_size)
        ]

        if self._stats_file:
            import json as _json

            with open(self._stats_file, "a") as _sf:
                for b in range(batch_size):
                    fp = req_steps_active[b]
                    tokens = int(next_token_ids_list[b].shape[0])
                    self._stats_tokens_generated += tokens
                    tpfp = tokens / fp if fp > 0 else 0.0
                    _sf.write(
                        _json.dumps(
                            {
                                "forward_passes": fp,
                                "tokens": tokens,
                                "tokens_per_fp": round(tpfp, 4),
                            }
                        )
                        + "\n"
                    )

        logger.debug(
            "FastDiffuser block done: CG=%d  eager=%d  total_fp=%d",
            self._stats_cuda_graph,
            self._stats_eager,
            self._stats_forward_passes,
        )
        return logits_output, next_token_ids_list, can_run_graph


Algorithm = FastDiffuser
