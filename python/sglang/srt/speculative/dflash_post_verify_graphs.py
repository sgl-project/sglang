# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from SGLang's eagle_post_verify_graphs.py (sgl-project/sglang #45474).
"""SGLang's post-verify CUDA graphs for the DFLASH verify
(SGLANG_DFLASH_GRAPH_POST_VERIFY), per batch size: one graph for acceptance and the
commit outputs, then, after the seq_lens publish, one for the target mamba state
commit (forked onto a side stream) beside the draft-KV append. Any batch a graph
cannot reproduce runs the worker's eager path."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.runtime_context import get_exec, get_parallel, mamba_track_grid
from sglang.srt.speculative import dflash_utils
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    SIMULATE_ACC_METHOD,
    sample_simulated_acc_len,
)
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = logging.getLogger(__name__)

# Eager rounds per key before its graphs are captured (JIT compiles, lazy buffers).
_NUM_WARMUP = 2
# Bound on graph sets (key x verify logits buffer); a batch past it runs eagerly.
_MAX_POST_VERIFY_STATES = 64


def post_verify_graphs_disabled_reason(worker: DFlashWorkerV2) -> Optional[str]:
    """Why the worker cannot run its post-verify section in graphs, or None."""
    parallel = get_parallel()
    if not is_cuda():
        return "needs CUDA"
    if parallel.attn_dp_enabled:
        # A key is captured at its first rounds, which DP ranks do not reach
        # together, and a capture ends in collectives over the TP group.
        return "DP attention"
    if parallel.pp_size > 1:
        return "pipeline parallelism"
    if worker.draft_parallel_layout.tp_group is not None:
        return "the draft runs on its own TP group"
    if worker.lilicorr is not None:
        return "LiLiCorr draft head"
    if worker._use_fused_kv_materialize and worker._fused_kv_helper is not None:
        return "fused draft KV materialization"
    if worker._need_mamba_verify_commit and _mamba_commit_backend(worker) is None:
        return "the target attention backend has no staged mamba commit"
    return None


def _mamba_commit_backend(worker: DFlashWorkerV2) -> Optional[Any]:
    """The target attention backend whose mamba commit takes staged state indices."""
    backend = worker.target_worker.model_runner.attn_backend
    inner = backend
    if hasattr(backend, "spec_attn_is_decode"):
        inner = (
            backend.decode_backend
            if backend.spec_attn_is_decode
            else backend.prefill_backend
        )
    if not callable(getattr(inner, "mtp_verify_state_indices", None)):
        return None
    return backend


class DFlashPostVerifyGraphs:
    """The DFLASH verify's post-section in SGLang's own CUDA graphs, one set per key:
    batch size, acceptance path, sampling mode, forced acceptance, mamba tracking
    and the verify's logits / hidden-state buffers.

    Inputs are staged into fixed buffers each round; outputs land in fixed buffers
    the worker copies into its two-slot ring, so they live as long as the eager
    path's. Forced acceptance lengths (SGLANG_SIMULATE_ACC_LEN) are drawn on the
    host from the same global generator as the eager path, once per round.
    """

    def __init__(self, worker: DFlashWorkerV2):
        self._worker = worker
        self._states: dict[tuple, SimpleNamespace] = {}
        self._pool = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._side_stream: Optional[torch.cuda.Stream] = None
        # Largest batch a verify graph replays: sizes the buffers graphs read in place.
        self._max_bs = max(get_exec().graph.cuda_graph_config.decode.bs)
        self._reserved = False

    def run(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: Any,
        can_run_cuda_graph: bool,
        draft_input: Any,
        candidates: torch.Tensor,
        prefix_lens: torch.Tensor,
        seq_lens_pre_verify: Optional[torch.Tensor],
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        on_publish: Optional[Callable[[torch.Tensor], None]],
    ) -> Optional[tuple]:
        """(commit_lens, bonus, out_tokens, new_seq_lens) after acceptance, the mamba
        commit and the draft-KV append ran in graphs (or their warmup rounds), or
        None when the worker's eager path must run this batch."""
        plan = self._plan(batch, logits_output, can_run_cuda_graph, draft_input)
        if plan is None:
            return None
        key, config = plan
        # The chain buffers' cache is keyed on the device index: the tensor's
        # device, not the worker's "cuda".
        device = candidates.device
        state = self._states.get(key)
        if state is None:
            if len(self._states) >= _MAX_POST_VERIFY_STATES:
                return None
            self._reserve_shared_buffers(device)
            state = self._states[key] = SimpleNamespace(
                warmups=0, slot=None, chain_buffers=self._chain_buffers(device)
            )
        elif state.slot is not None and state.chain_buffers is not (
            self._chain_buffers(device)
        ):
            # An eager batch past the reserved size regrew the chain buffers the
            # graphs read in place: re-warm and re-capture against the new ones.
            state.warmups, state.slot = 0, None
            state.chain_buffers = self._chain_buffers(device)

        inputs = self._inputs(
            batch,
            config,
            candidates=candidates,
            prefix_lens=prefix_lens,
            seq_lens_pre_verify=seq_lens_pre_verify,
            positions=positions,
            cache_loc=cache_loc,
        )
        slot = state.slot
        if slot is None:
            slot = state.slot = self._make_slot(batch, logits_output, config, inputs)
        elif not self._load(slot, inputs):
            return None
        if config.simulate:
            slot.staged["forced_commit_len"].fill_(
                sample_simulated_acc_len(
                    SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, config.block_size
                )
            )

        if slot.graphs is None and state.warmups < _NUM_WARMUP:
            state.warmups += 1
            self._accept(slot)
            result = self._worker._ring_post_verify_outputs(slot.outputs)
            if on_publish is not None:
                on_publish(result[3])
            self._commit(slot)
            return result
        if slot.graphs is None:
            self._capture(slot)
        accept_graph, commit_graph = slot.graphs
        accept_graph.replay()
        result = self._worker._ring_post_verify_outputs(slot.outputs)
        if on_publish is not None:
            on_publish(result[3])
        commit_graph.replay()
        return result

    def _plan(
        self,
        batch: ScheduleBatch,
        logits_output: Any,
        can_run_cuda_graph: bool,
        draft_input: Any,
    ) -> Optional[tuple[tuple, SimpleNamespace]]:
        """(key, static configuration) of a batch the graphs reproduce, else None.
        Every input is TP-replicated, so all ranks graph the same rounds."""
        worker = self._worker
        info = batch.sampling_info
        logits = logits_output.next_token_logits
        hidden = logits_output.hidden_states
        if (
            not can_run_cuda_graph
            or batch.forward_mode.is_idle()
            or batch.has_grammar
            or batch.return_logprob
            or logits is None
            or hidden is None
        ):
            return None
        if info is not None:
            penalizer = info.penalizer_orchestrator
            if (
                info.has_custom_logit_processor
                or getattr(info, "acc_linear_penalties", None) is not None
                or (penalizer is not None and penalizer.is_required)
                or getattr(info, "grammar_mask", None) is not None
                or getattr(info, "logit_bias", None) is not None
                or info.sampling_mask_batch_indices is not None
                or getattr(info, "sampling_seed", None) is not None
                or info.need_min_p_sampling
            ):
                return None

        bs = len(batch.seq_lens)
        block_size = int(worker.block_size)
        selector_sample = worker._selector_sample
        all_greedy = info is None or info.is_all_greedy
        if selector_sample is not None:
            # Only the draft graph's own outputs: their addresses are fixed.
            sampler = worker._draft_sampler
            if (
                getattr(sampler, "candidate_out", None) is None
                or selector_sample[0].data_ptr() != sampler.candidate_out.data_ptr()
                or selector_sample[1].data_ptr() != sampler.q_out.data_ptr()
            ):
                return None
            path = "selector"
        elif not all_greedy and dflash_utils.is_dflash_sampling_verify_available():
            # Its full-vocabulary matrices borrow the graph pool; stays eager.
            return None
        else:
            path = "greedy"

        need_top_k = bool(info is not None and info.need_top_k_sampling)
        need_top_p = bool(info is not None and info.need_top_p_sampling)
        max_top_k = getattr(draft_input, "max_top_k", None)
        uniform_top_k_value = getattr(draft_input, "uniform_top_k_value", None)
        if path == "selector" and need_top_k and max_top_k is None:
            # The sparse top-k path would read its width from the device.
            return None

        lp = worker._target_logits_processor
        deferred = lp is not None and lp.is_deferred_shard(logits)
        skip_logits = deferred and worker._can_skip_verify_logits(batch, info)
        track = (
            worker._need_mamba_verify_commit and batch.mamba_track_indices is not None
        )
        config = SimpleNamespace(
            bs=bs,
            block_size=block_size,
            path=path,
            sampled=path == "selector",
            need_top_k=need_top_k,
            need_top_p=need_top_p,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
            deferred=deferred,
            skip_logits=skip_logits,
            simulate=SIMULATE_ACC_LEN > 0,
            mamba=worker._need_mamba_verify_commit,
            track=track,
        )
        key = (
            bs,
            path,
            need_top_k,
            need_top_p,
            max_top_k,
            uniform_top_k_value,
            deferred,
            skip_logits,
            config.simulate,
            # The mamba commit reads the tracking interval on the host.
            mamba_track_grid(batch.tree_cache.page_size) if track else None,
            # The graphs read the verify's outputs in place.
            logits.data_ptr(),
            tuple(logits.shape),
            hidden.data_ptr(),
            tuple(hidden.shape),
        )
        return key, config

    def _inputs(
        self,
        batch: ScheduleBatch,
        config: SimpleNamespace,
        *,
        candidates: torch.Tensor,
        prefix_lens: torch.Tensor,
        seq_lens_pre_verify: Optional[torch.Tensor],
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """The round's inputs the graphs read, staged into fixed buffers."""
        inputs = {
            "candidates": candidates,
            "prefix_lens": prefix_lens,
            "positions": positions,
            "cache_loc": cache_loc,
        }
        if config.sampled:
            info = batch.sampling_info
            inputs["temperatures"] = info.temperatures
            if config.need_top_p:
                inputs["top_ps"] = info.top_ps
            if config.need_top_k:
                inputs["top_ks"] = info.top_ks
        if config.mamba:
            req_pool_indices = batch.req_pool_indices[: config.bs]
            inputs["seq_lens_pre_verify"] = seq_lens_pre_verify
            inputs["req_pool_indices"] = req_pool_indices
            inputs["state_indices"] = _mamba_commit_backend(
                self._worker
            ).mtp_verify_state_indices(config.bs, req_pool_indices)
            if config.track:
                inputs["track_indices"] = batch.mamba_track_indices
        return inputs

    def _make_slot(
        self,
        batch: ScheduleBatch,
        logits_output: Any,
        config: SimpleNamespace,
        inputs: dict[str, torch.Tensor],
    ) -> SimpleNamespace:
        bs, block_size = config.bs, config.block_size
        device = inputs["candidates"].device
        staged = {name: src.clone() for name, src in inputs.items()}
        if config.simulate:
            staged["forced_commit_len"] = torch.ones(
                (1,), dtype=torch.int64, device=device
            )
        sampling_info = None
        if config.sampled:
            # The fields acceptance reads, on the staged copies.
            sampling_info = SimpleNamespace(
                is_all_greedy=False,
                need_top_k_sampling=config.need_top_k,
                need_top_p_sampling=config.need_top_p,
                temperatures=staged["temperatures"],
                top_ps=staged.get("top_ps"),
                top_ks=staged.get("top_ks"),
            )
        elif batch.sampling_info is not None:
            sampling_info = SimpleNamespace(is_all_greedy=True)
        selector_sample = None
        if config.sampled:
            sampler = self._worker._draft_sampler
            selector_sample = (sampler.candidate_out[:bs], sampler.q_out[:bs])
        outputs = (
            torch.empty((bs,), dtype=torch.int32, device=device),  # accept_len
            torch.empty((bs,), dtype=torch.int32, device=device),  # commit_lens
            torch.empty((bs,), dtype=torch.int64, device=device),  # bonus
            torch.empty((bs, block_size), dtype=torch.int64, device=device),
            torch.empty((bs,), dtype=torch.int64, device=device),  # new_seq_lens
        )
        batch_view = SimpleNamespace(
            tree_cache=batch.tree_cache,
            req_pool_indices=staged.get("req_pool_indices"),
            mamba_track_indices=staged.get("track_indices"),
        )
        return SimpleNamespace(
            config=config,
            staged=staged,
            outputs=outputs,
            logits=logits_output.next_token_logits,
            hidden=logits_output.hidden_states,
            sampling_info=sampling_info,
            draft_input=SimpleNamespace(
                max_top_k=config.max_top_k,
                uniform_top_k_value=config.uniform_top_k_value,
            ),
            selector_sample=selector_sample,
            batch_view=batch_view,
            graphs=None,
        )

    @staticmethod
    def _load(slot: SimpleNamespace, inputs: dict[str, torch.Tensor]) -> bool:
        """Copy the round's inputs into the slot's staging buffers, one multi-tensor
        copy per dtype; False (eager round) on an unexpected shape."""
        staged = slot.staged
        groups: dict = {}
        for name, src in inputs.items():
            dst = staged[name]
            if dst.shape != src.shape or dst.dtype != src.dtype:
                return False
            dsts, srcs = groups.setdefault(src.dtype, ([], []))
            dsts.append(dst)
            srcs.append(src)
        for dsts, srcs in groups.values():
            torch._foreach_copy_(dsts, srcs, non_blocking=True)
        return True

    def _reserve_shared_buffers(self, device: torch.device) -> None:
        """Size the buffers acceptance reads in place for the largest graphed batch,
        once, before the first capture, so their addresses stay fixed."""
        if self._reserved:
            return
        self._reserved = True
        worker = self._worker
        dflash_utils._get_or_create_chain_verify_buffers(
            bs=self._max_bs, draft_token_num=int(worker.block_size), device=device
        )
        if worker.selector is not None:
            worker._reserve_selector_q_buffer(self._max_bs)

    def _chain_buffers(self, device: torch.device) -> Optional[dict]:
        return dflash_utils._DFLASH_CHAIN_VERIFY_BUFFERS.get(
            (device.index, int(self._worker.block_size))
        )

    def _accept(self, slot: SimpleNamespace) -> None:
        """Acceptance and the commit outputs into slot.outputs; recorded in the
        first graph."""
        worker, config, staged = self._worker, slot.config, slot.staged
        logits = slot.logits
        target_predict = None
        if config.deferred:
            lp = worker._target_logits_processor
            if config.skip_logits:
                target_predict = lp.deferred_argmax(logits)
                logits = None
            else:
                logits = lp.finish_deferred_gather(logits)
        worker._accept_block(
            candidates=staged["candidates"],
            next_token_logits=logits,
            sampling_info=slot.sampling_info,
            draft_input=slot.draft_input,
            prefix_lens=staged["prefix_lens"],
            bs=config.bs,
            target_predict=target_predict,
            selector_sample=slot.selector_sample,
            out=slot.outputs,
            forced_commit_len=staged.get("forced_commit_len"),
        )

    def _commit(self, slot: SimpleNamespace) -> None:
        """The target mamba state commit, forked onto a side stream, beside the
        draft-KV append; recorded in the second graph, which joins both."""
        worker, config, staged = self._worker, slot.config, slot.staged
        _, commit_lens, _, _, new_seq_lens = slot.outputs
        main = side = None
        if config.mamba:
            if self._side_stream is None:
                self._side_stream = torch.cuda.Stream()
            main, side = torch.cuda.current_stream(), self._side_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                worker._update_target_mamba_state_after_verify(
                    batch=slot.batch_view,
                    seq_lens_pre_verify=staged["seq_lens_pre_verify"],
                    seq_lens_post_verify=new_seq_lens,
                    commit_lens=commit_lens,
                    state_indices=staged["state_indices"],
                )
        cache_loc = staged["cache_loc"]
        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=slot.hidden,
            cache_loc=cache_loc,
            cache_loc_2d=cache_loc.view(config.bs, config.block_size),
            positions=staged["positions"],
            commit_lens=commit_lens,
        )
        if side is not None:
            main.wait_stream(side)

    def _capture(self, slot: SimpleNamespace) -> None:
        from sglang.srt.distributed.parallel_state import graph_capture

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            # Not the legacy default stream, which cannot capture.
            self._capture_stream = torch.cuda.Stream()
        accept_graph, commit_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with graph_capture(stream=self._capture_stream) as ctx:
            with torch.cuda.graph(
                accept_graph,
                pool=self._pool,
                stream=ctx.stream,
                capture_error_mode="thread_local",
            ):
                self._accept(slot)
            with torch.cuda.graph(
                commit_graph,
                pool=self._pool,
                stream=ctx.stream,
                capture_error_mode="thread_local",
            ):
                self._commit(slot)
        slot.graphs = (accept_graph, commit_graph)
        if get_parallel().tp_rank == 0:
            logger.info(
                "Captured the DFLASH post-verify graphs: bs=%d path=%s sampled=%s "
                "simulated_acceptance=%s mamba=%s",
                slot.config.bs,
                slot.config.path,
                slot.config.sampled,
                slot.config.simulate,
                slot.config.mamba,
            )
