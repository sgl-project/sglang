# SPDX-License-Identifier: Apache-2.0
"""SGLang's post-verify CUDA graphs for the EAGLE verify
(SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH): sampling and acceptance, the commit
outputs and, on a side stream, the mamba state commit, for every batch a graph can
reproduce; any other batch runs SGLang's eager path."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.kernels.ops.memory.small_field_copy import SmallFieldCopyPlan
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

# Eager rounds per post-verify key before its graphs are captured.
_NUM_WARMUP = 2
# XORed into the server seed so these draws differ from other samplers'.
_SIM_ACC_LEN_DOMAIN = 0x0F3D6B8C2A915E47
# Arbitrary bound on post-verify graph sets (key x verify logits buffer); a batch
# past it runs SGLang's eager path.
_MAX_POST_VERIFY_STATES = 64
# SGLANG_Q35MF_MKH_N3C: a round's inputs and its forced length staged in one launch.
_ONE_LAUNCH_LOAD = envs.SGLANG_Q35MF_MKH_N3C.get()


def simulated_acc_len_generator(seed: int) -> torch.Generator:
    """The forced acceptance lengths' generator (SGLANG_SIMULATE_ACC_LEN), seeded
    alike on every rank from the server seed."""
    return torch.Generator().manual_seed(
        (int(seed) ^ _SIM_ACC_LEN_DOMAIN) & 0x7FFFFFFFFFFFFFFF
    )


class EaglePostVerifyGraphs:
    """The verify's post-section in SGLang's own CUDA graphs, one set per key: batch
    size, sampling mode, forced acceptance, tracking and the verify logits buffer."""

    def __init__(self, *, target_worker: Any, simulated_acc_len_rng: torch.Generator):
        self._target_worker = target_worker
        self._graphs = _PostVerifyGraphs(
            body=self._body,
            commit=self._commit,
            sim_rng=simulated_acc_len_rng,
        )

    def run(
        self,
        *,
        batch: ScheduleBatch,
        verify_input: Any,
        logits_output: Any,
        can_run_cuda_graph: bool,
        max_top_k: Optional[int] = None,
    ) -> Optional[tuple]:
        """(predict, accept_lens, accept_index, new_seq_lens, bonus_tokens,
        draft-extend inputs) from the post-verify graphs, or None when SGLang's
        sampling and commit run eagerly. max_top_k: the compact top-k branch's
        support (compact_verify_max_top_k), None for the dense branch."""
        self.join()
        key = self._key(
            batch, verify_input, logits_output, can_run_cuda_graph, max_top_k
        )
        if key is None:
            return None
        return self._graphs.run(
            key=key,
            inputs=_post_verify_inputs(batch, verify_input),
            make_slot=lambda staged: self._make_slot(
                staged, batch, verify_input, logits_output, max_top_k
            ),
            sim_max_len=verify_input.max_tree_depth,
        )

    def join(self) -> None:
        """Order the current stream after the last side-stream mamba state commit; the
        worker calls it once the draft extend is enqueued."""
        self._graphs.join()

    def _key(
        self,
        batch: ScheduleBatch,
        verify_input: Any,
        logits_output: Any,
        can_run_cuda_graph: bool,
        max_top_k: Optional[int] = None,
    ) -> Optional[tuple]:
        from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

        info = batch.sampling_info
        watermark = self._target_worker.model_runner.watermark_state
        if (
            not can_run_cuda_graph
            or batch.forward_mode.is_idle()
            or verify_input.topk != 1
            or batch.has_grammar
            or batch.hisparse_coordinator is not None
            or verify_input.draft_probs is not None
            or info.acc_additive_penalties is not None
            or info.acc_scaling_penalties is not None
            or info.logit_bias is not None
            or info.sampling_seed is not None
            or info.need_min_p_sampling
            or (watermark is not None and info.has_watermark_candidates)
        ):
            return None
        return (
            len(batch.seq_lens),
            bool(info.is_all_greedy),
            bool(info.need_top_k_sampling),
            bool(info.need_top_p_sampling),
            max_top_k,
            SIMULATE_ACC_LEN > 0,
            batch.mamba_track_indices is not None,
            verify_input.draft_token_num,
            verify_input.max_tree_depth,
            # the graphs read the verify logits in place; a runner may own several
            logits_output.next_token_logits.data_ptr(),
        )

    def _make_slot(
        self,
        staged: dict[str, torch.Tensor],
        batch: ScheduleBatch,
        verify_input: Any,
        logits_output: Any,
        max_top_k: Optional[int] = None,
    ) -> SimpleNamespace:
        from sglang.srt.runtime_context import mamba_track_grid

        info = batch.sampling_info
        return SimpleNamespace(
            num_requests=len(batch.seq_lens),
            tokens_per_request=verify_input.draft_token_num,
            max_tree_depth=verify_input.max_tree_depth,
            is_all_greedy=bool(info.is_all_greedy),
            need_top_k=bool(info.need_top_k_sampling),
            need_top_p=bool(info.need_top_p_sampling),
            max_top_k=max_top_k,
            logits=logits_output.next_token_logits,
            staged=staged,
            device=batch.device,
            forward_mode=batch.forward_mode,
            tree_cache=batch.tree_cache,
            track_interval=(
                mamba_track_grid(batch.tree_cache.page_size)
                if batch.mamba_track_indices is not None
                else 0
            ),
        )

    def _body(self, slot: SimpleNamespace) -> tuple:
        """Sampling and acceptance, then the commit outputs; recorded in SGLang's
        post-verify graph."""
        from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs

        predict, accept_lens, accept_index = _sample(slot)
        new_seq_lens, bonus_tokens, commit_steps, draft_extend_inputs = (
            prepare_verify_commit_outputs(
                predict,
                accept_index,
                accept_lens,
                slot.staged["seq_lens"],
                num_draft_tokens=slot.tokens_per_request,
                mamba_track_interval=slot.track_interval,
            )
        )
        return (
            predict,
            accept_lens,
            accept_index,
            new_seq_lens,
            bonus_tokens,
            draft_extend_inputs,
            commit_steps,
        )

    def _commit(self, slot: SimpleNamespace, outputs: tuple) -> None:
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        # the batch fields the commit reads, on the slot's staged copies
        batch_view = SimpleNamespace(
            forward_mode=slot.forward_mode,
            seq_lens=slot.staged["seq_lens"],
            req_pool_indices=slot.staged["req_pool_indices"],
            mamba_track_indices=slot.staged.get("track_indices"),
            tree_cache=slot.tree_cache,
        )
        commit_mamba_states_after_verify(
            self._target_worker,
            batch_view,
            outputs[1],
            outputs[2],
            slot.tokens_per_request,
            prepared_step_indices=outputs[6],
        )


def _sample(slot: SimpleNamespace) -> tuple:
    """SGLang's eagle_sample over the slot's staged copies, with the injected
    acceptance length read on the device."""
    from sglang.srt.speculative.eagle_utils import eagle_sample

    staged = slot.staged
    sampling_info = SimpleNamespace(
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        has_watermark_candidates=False,
        is_all_greedy=slot.is_all_greedy,
        need_top_k_sampling=slot.need_top_k,
        need_top_p_sampling=slot.need_top_p,
        need_min_p_sampling=False,
        temperatures=staged.get("temperatures"),
        top_ks=staged.get("top_ks"),
        top_ps=staged.get("top_ps"),
    )
    batch_view = SimpleNamespace(
        device=slot.device,
        forward_mode=slot.forward_mode,
        seq_lens=staged["seq_lens"],
        req_pool_indices=staged["req_pool_indices"],
        sampling_info=sampling_info,
        return_logprob=False,
    )
    verify_view = SimpleNamespace(
        draft_token=staged["draft_tokens"],
        draft_token_num=slot.tokens_per_request,
        max_tree_depth=slot.max_tree_depth,
        retrieve_index=staged["retrieve_index"],
        retrieve_next_token=staged["retrieve_next_token"],
        retrieve_next_sibling=staged["retrieve_next_sibling"],
        tree_topk=1,
        draft_probs=None,
    )
    return eagle_sample(
        verify_view,
        batch_view,
        SimpleNamespace(next_token_logits=slot.logits),
        uno_target_max_top_k=slot.max_top_k,
        simulated_acc_len=staged.get("sim_accept_len"),
        sparse_bonus_argmax=slot.max_top_k is not None,
    )


class _PostVerifyGraphs:
    """SGLang-owned post-verify graphs: per key, two slots of fixed staging buffers
    and outputs alternate, so one round's outputs stay valid while the host consumes
    them; the mamba state commit replays on a side stream until join()."""

    def __init__(self, *, body: Callable, commit: Callable, sim_rng: torch.Generator):
        self._body, self._commit = body, commit
        # forced acceptance lengths: the worker's one stream, from which SGLang's eager
        # verify draws too; once per round, whatever else uses the global RNG
        self._sim_rng = sim_rng
        self._states: dict[tuple, SimpleNamespace] = {}
        self._pool = None
        self._side_stream = self._main_done = self._commit_done = None
        self._commit_pending = False
        # The host runs up to ~2 rounds ahead; a pinned slot is rewritten only after
        # the event behind its last copy has completed.
        self._sim_host: list[torch.Tensor] = []
        self._sim_done: list[Optional[torch.cuda.Event]] = []
        self._sim_turn = 0

    def run(
        self, *, key: tuple, inputs: dict, make_slot: Callable, sim_max_len: int
    ) -> Optional[tuple]:
        state = self._states.get(key)
        if state is None:
            if len(self._states) >= _MAX_POST_VERIFY_STATES:
                return None
            state = self._states[key] = SimpleNamespace(
                warmups=0, turn=0, slots=[None, None]
            )
        turn, state.turn = state.turn, state.turn ^ 1
        sim_len, sim_turn = self._draw_sim_len(sim_max_len)
        slot = state.slots[turn]
        if slot is None:
            staged = {name: src.clone() for name, src in inputs.items()}
            if sim_len is not None:
                staged["sim_accept_len"] = sim_len.to(
                    inputs["seq_lens"].device, copy=True
                )
            slot = state.slots[turn] = make_slot(staged)
            slot.staged, slot.graph, slot.commit_graph = staged, None, None
            slot.load_plan = (
                SmallFieldCopyPlan(allow_pinned_src=True) if _ONE_LAUNCH_LOAD else None
            )
        self._load(slot, inputs, sim_len)
        if sim_len is not None:
            self._record_sim_done(sim_turn)
        if slot.graph is None:
            if state.warmups < _NUM_WARMUP * 2:
                state.warmups += 1
                outputs = self._body(slot)
                self._commit(slot, outputs)
                return outputs[:6]
            self._capture(slot)
        slot.graph.replay()
        self._main_done.record()
        self._side_stream.wait_event(self._main_done)
        with torch.cuda.stream(self._side_stream):
            slot.commit_graph.replay()
        self._commit_done.record(self._side_stream)
        self._commit_pending = True
        return slot.outputs[:6]

    def join(self) -> None:
        if self._commit_pending:
            torch.cuda.current_stream().wait_event(self._commit_done)
            self._commit_pending = False

    @staticmethod
    def _load(slot: SimpleNamespace, inputs: dict, sim_len) -> None:
        staged = slot.staged
        if slot.load_plan is not None:
            dsts, srcs = [staged[name] for name in inputs], list(inputs.values())
            if sim_len is not None:
                dsts.append(staged["sim_accept_len"])
                srcs.append(sim_len)
            # one launch for the round's inputs and the pinned host draw
            if slot.load_plan(dsts, srcs):
                return
        # One multi-tensor copy per dtype; one copy per input costs ~4 us of GPU time.
        groups: dict = {}
        for name, src in inputs.items():
            dsts, srcs = groups.setdefault(src.dtype, ([], []))
            dsts.append(staged[name])
            srcs.append(src)
        for dsts, srcs in groups.values():
            torch._foreach_copy_(dsts, srcs, non_blocking=True)
        if sim_len is not None:
            staged["sim_accept_len"].copy_(sim_len, non_blocking=True)

    def _draw_sim_len(
        self, max_len: int
    ) -> tuple[Optional[torch.Tensor], Optional[int]]:
        from sglang.srt.speculative.spec_utils import (
            SIMULATE_ACC_LEN,
            SIMULATE_ACC_METHOD,
            sample_simulated_acc_len,
        )

        if SIMULATE_ACC_LEN <= 0:
            return None, None
        if not self._sim_host:
            self._sim_host = [
                torch.zeros((1,), dtype=torch.int64, pin_memory=True) for _ in range(4)
            ]
            self._sim_done = [None] * len(self._sim_host)
        turn = self._sim_turn
        self._sim_turn = (turn + 1) % len(self._sim_host)
        done = self._sim_done[turn]
        if done is not None and not done.query():
            done.synchronize()
        host = self._sim_host[turn]
        host[0] = sample_simulated_acc_len(
            SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, max_len, generator=self._sim_rng
        )
        return host, turn

    def _record_sim_done(self, turn: int) -> None:
        if self._sim_done[turn] is None:
            self._sim_done[turn] = torch.cuda.Event()
        self._sim_done[turn].record()

    def _capture(self, slot: SimpleNamespace) -> None:
        from sglang.srt.distributed.parallel_state import graph_capture

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._side_stream = torch.cuda.Stream()
            self._main_done, self._commit_done = torch.cuda.Event(), torch.cuda.Event()
        graph = torch.cuda.CUDAGraph()
        with graph_capture(stream=torch.cuda.current_stream()) as ctx:
            with torch.cuda.graph(
                graph,
                pool=self._pool,
                stream=ctx.stream,
                capture_error_mode="thread_local",
            ):
                outputs = self._body(slot)
        # Captured after the main graph in the same pool and replayed after it.
        commit_graph = torch.cuda.CUDAGraph()
        self._side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(
            commit_graph,
            pool=self._pool,
            stream=self._side_stream,
            capture_error_mode="thread_local",
        ):
            self._commit(slot, outputs)
        torch.cuda.current_stream().wait_stream(self._side_stream)
        slot.graph, slot.commit_graph, slot.outputs = graph, commit_graph, outputs
        logger.info(
            "Captured the EAGLE post-verify graphs for batch size %d",
            slot.num_requests,
        )


def _post_verify_inputs(
    batch: ScheduleBatch, verify_input: Any
) -> dict[str, torch.Tensor]:
    """The round's inputs the post-verify graphs read, staged into fixed buffers."""
    info = batch.sampling_info
    inputs = {
        "draft_tokens": verify_input.draft_token,
        "retrieve_index": verify_input.retrieve_index,
        "retrieve_next_token": verify_input.retrieve_next_token,
        "retrieve_next_sibling": verify_input.retrieve_next_sibling,
        "seq_lens": batch.seq_lens,
        "req_pool_indices": batch.req_pool_indices,
    }
    if not info.is_all_greedy:
        inputs.update(
            temperatures=info.temperatures, top_ks=info.top_ks, top_ps=info.top_ps
        )
    if batch.mamba_track_indices is not None:
        inputs["track_indices"] = batch.mamba_track_indices
    return inputs
