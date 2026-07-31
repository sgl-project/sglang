"""CUDA-graphed ReplaySSM fold commit.

The post-verify mamba commit is ~16 eager launches (a dozen tiny index ops +
the fold kernel + two conv scatters) on the host seam right after
``eagle_sample``. This module fuses the index math into one Triton kernel and
replays the whole commit as one CUDA graph per batch size: per step the host
does four staging copies + one graph launch.

Padding-free by construction: graphs are captured per exact ``bs`` against
fixed staging buffers; pool tensors are address-stable, so capture is
allocation-free. Capture uses thread_local error mode so the overlap
scheduler's other streams are not poisoned mid-serving.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@triton.jit
def _fused_verify_commit_indices_kernel(
    accept_index,
    accept_lens,
    seq_lens,
    req_pool_indices,
    mamba_map,
    state_batch_indices_out,
    last_correct_out,
    track_steps_out,
    accept_width,
    draft_token_num,
    track_interval,
):
    i = tl.program_id(0)
    n = tl.load(accept_lens + i).to(tl.int64)
    off = i.to(tl.int64) * draft_token_num
    last = tl.load(accept_index + i * accept_width + n - 1) - off
    tl.store(last_correct_out + i, last)
    req = tl.load(req_pool_indices + i).to(tl.int64)
    slot = tl.load(mamba_map + req).to(tl.int64)
    tl.store(state_batch_indices_out + i, slot)

    pre = tl.load(seq_lens + i).to(tl.int64)
    post = pre + n
    crossed = (pre // track_interval) != (post // track_interval)
    point = (post // track_interval) * track_interval
    ith = tl.maximum(point - pre - 1, 0)
    cand = tl.load(accept_index + i * accept_width + ith) - off
    step = tl.where(crossed, cand, -1)
    tl.store(track_steps_out + i, step)


class ReplayssmCommitGraphRunner:
    """Per-(req_pool) runner; graphs keyed by batch size, lazily captured."""

    def __init__(
        self,
        *,
        req_pool,
        draft_token_num: int,
        accept_width: int,
        track_interval: int,
        max_bs: int,
        device: torch.device,
    ):
        self.req_pool = req_pool
        self.spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        self.mamba_map = req_pool.req_index_to_mamba_index_mapping
        self.draft_token_num = draft_token_num
        self.accept_width = accept_width
        self.track_interval = track_interval
        self.max_bs = max_bs
        self.device = device

        self.in_req = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.in_accept_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.in_accept_index = torch.zeros(
            max_bs * accept_width, dtype=torch.int64, device=device
        )
        self.in_seq_lens = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.in_track = torch.full((max_bs,), -1, dtype=torch.int64, device=device)

        self.out_state_idx = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.out_last_correct = torch.zeros(max_bs, dtype=torch.int64, device=device)
        self.out_track_steps = torch.zeros(max_bs, dtype=torch.int64, device=device)

        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}

    def _run_body(self, bs: int) -> None:
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_all_layers,
        )
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )

        _fused_verify_commit_indices_kernel[(bs,)](
            self.in_accept_index,
            self.in_accept_lens,
            self.in_seq_lens,
            self.in_req,
            self.mamba_map,
            self.out_state_idx,
            self.out_last_correct,
            self.out_track_steps,
            self.accept_width,
            self.draft_token_num,
            self.track_interval,
        )
        spec_state = self.spec_state
        commit_gdn_replayssm_fold_all_layers(
            checkpoint_state=spec_state.temporal,
            rawv_cache=spec_state.replayssm_rawv,
            rawk_cache=spec_state.replayssm_rawk,
            g_cache=spec_state.replayssm_g,
            beta_cache=spec_state.replayssm_beta,
            ssm_state_indices=self.out_state_idx[:bs],
            accept_lens=self.in_accept_lens[:bs],
            max_cache_len=spec_state.replayssm_rawv.shape[-2],
            num_k_heads=spec_state.replayssm_rawk.shape[2],
            mamba_track_indices=self.in_track[:bs],
            mamba_steps_to_track=self.out_track_steps[:bs],
            null_block_id=-1,
        )
        for conv_states, interm_conv in zip(
            spec_state.conv, spec_state.intermediate_conv_window
        ):
            fused_conv_window_scatter_with_mask(
                conv_states,
                interm_conv,
                self.out_state_idx[:bs],
                self.out_last_correct[:bs],
            )
            fused_conv_window_scatter_with_mask(
                conv_states,
                interm_conv,
                self.in_track[:bs],
                self.out_track_steps[:bs],
            )

    def commit(
        self,
        *,
        req_pool_indices: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        seq_lens: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
    ) -> bool:
        bs = accept_lens.shape[0]
        if bs > self.max_bs or accept_index.shape[1] != self.accept_width:
            return False

        self.in_req[:bs].copy_(req_pool_indices, non_blocking=True)
        self.in_accept_lens[:bs].copy_(accept_lens, non_blocking=True)
        self.in_accept_index[: bs * self.accept_width].copy_(
            accept_index.reshape(-1), non_blocking=True
        )
        self.in_seq_lens[:bs].copy_(seq_lens, non_blocking=True)
        if mamba_track_indices is not None:
            self.in_track[:bs].copy_(mamba_track_indices, non_blocking=True)
        else:
            self.in_track[:bs].fill_(-1)

        graph = self.graphs.get(bs)
        if graph is None:
            self._run_body(bs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self._run_body(bs)
            self.graphs[bs] = graph
            return True
        graph.replay()
        return True


_runners: Dict[int, ReplayssmCommitGraphRunner] = {}


def commit_via_cuda_graph(
    *,
    req_pool,
    batch,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
    track_interval: int,
    max_bs: int,
) -> bool:
    if envs.SGLANG_DISABLE_REPLAYSSM_COMMIT_GRAPH.get():
        return False
    key = id(req_pool)
    runner = _runners.get(key)
    if runner is None:
        try:
            runner = ReplayssmCommitGraphRunner(
                req_pool=req_pool,
                draft_token_num=draft_token_num,
                accept_width=accept_index.shape[1],
                track_interval=track_interval,
                max_bs=max_bs,
                device=accept_lens.device,
            )
        except Exception:
            logger.exception("replayssm commit graph init failed; staying eager")
            return False
        _runners[key] = runner
    try:
        return runner.commit(
            req_pool_indices=batch.req_pool_indices,
            accept_lens=accept_lens,
            accept_index=accept_index,
            seq_lens=batch.seq_lens,
            mamba_track_indices=batch.mamba_track_indices,
        )
    except Exception:
        logger.exception("replayssm commit graph failed; falling back to eager")
        _runners.pop(key, None)
        return False
