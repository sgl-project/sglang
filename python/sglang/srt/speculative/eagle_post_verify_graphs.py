# SPDX-License-Identifier: Apache-2.0
"""SGLang's post-verify CUDA graphs for the EAGLE verify
(SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH): sampling and acceptance, the commit
outputs and, on a side stream, the mamba state commit, for every batch a graph can
reproduce; any other batch runs SGLang's eager path."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.kernels.ops.memory.small_field_copy import SmallFieldCopyPlan
from sglang.srt.environ import envs
from sglang.srt.speculative import perfll_step2
from sglang.srt.utils import host_trace

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

# Eager rounds per post-verify key before its graphs are captured.
_NUM_WARMUP = 2
_FUSED_BS1_ACCEPT = os.environ.get("Q38_MTP_POSTVERIFY_FUSED_ACCEPT", "0") == "1"
# XORed into the server seed so these draws differ from other samplers'.
_SIM_ACC_LEN_DOMAIN = 0x0F3D6B8C2A915E47
# Arbitrary bound on post-verify graph sets (key x verify logits buffer); a batch
# past it runs SGLang's eager path.
_MAX_POST_VERIFY_STATES = 64
# SGLANG_Q35MF_MKH_N3C: a round's inputs and its forced length staged in one launch.
_ONE_LAUNCH_LOAD = envs.SGLANG_Q35MF_MKH_N3C.get()
_SHARDED_VERIFY = envs.SGLANG_Q35MF_SHARDED_VERIFY.get()
_SIM_PREDRAW = envs.SGLANG_PERFLL_HOST_SIM_PREDRAW.get()
# SGLANG_PERFLL_STEP5A_BONUS_I64: the bonus tokens in the dtype the relay and the next draft read
_BONUS_DTYPE = torch.int64 if envs.SGLANG_PERFLL_STEP5A_BONUS_I64.get() else torch.int32
# SGLANG_PERFLL_STEP5A_SEQ_LENS: a batch's lengths may be a slot's new_seq_lens output
_STAGED_SEQ_LENS = envs.SGLANG_PERFLL_STEP5A_SEQ_LENS.get()


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
        # SGLANG_PST_SPEC_EXTEND_LAYOUT: the draft extend's layout in the commit-outputs launch
        self._extend_layout = envs.SGLANG_PST_SPEC_EXTEND_LAYOUT.get()
        self._extend_mrope = (
            self._extend_layout and target_worker.model_runner.model_config.model_is_mrope
        )
        self._graphs = _PostVerifyGraphs(
            body=self._body,
            commit=self._commit,
            sim_rng=simulated_acc_len_rng,
        )
        # SGLANG_PERFLL_STEP2 >= 2: the post-verify graph also stages the next round's draft
        # inputs (enable_step2); next_draft is the staging of the last round run here
        self._step2: Optional[SimpleNamespace] = None
        self.next_draft = None
        # SGLANG_PERFLL_RS_D1 (enable_rs_d1): the graph also writes the draft extend runner's
        # inputs, which then ride at the end of the draft-extend outputs
        self._rs_d1: Optional[SimpleNamespace] = None
        if _BONUS_DTYPE is torch.int64:
            logger.info(
                "PERF-LL step5a: the post-verify graph writes the bonus tokens as int64"
            )

    def enable_step2(
        self,
        *,
        num_steps: int,
        window: int,
        seq_len_fill_value: int,
        mrope: bool,
        req_to_token: torch.Tensor,
        runner_buffers: Any = None,
    ) -> None:
        """``runner_buffers``: the draft graph runner's static inputs, which the staging
        then writes in place (mode 3)."""
        self._step2 = SimpleNamespace(
            num_steps=num_steps,
            window=window,
            seq_len_fill_value=seq_len_fill_value,
            mrope=mrope,
            req_to_token=req_to_token,
            runner_buffers=runner_buffers,
        )

    def enable_rs_d1(
        self,
        *,
        runner: Any,
        req_to_token: torch.Tensor,
        published_seq_lens: torch.Tensor,
    ) -> None:
        """SGLANG_PERFLL_RS_D1: for a batch size the draft extend graph ``runner`` captured as
        it is (up to SGLANG_PERFLL_RS_D1_MAX_BS), the graph also writes that runner's static
        inputs (the hidden rows on a branch beside the accept chain) and publishes the new lengths
        to ``published_seq_lens``."""
        max_bs = envs.SGLANG_PERFLL_RS_D1_MAX_BS.get()
        self._rs_d1 = SimpleNamespace(
            buffers=runner.buffers,
            batch_sizes=frozenset(
                bs for bs in runner.capture_bs if max_bs <= 0 or bs <= max_bs
            ),
            width=runner.captured_req_width,
            req_to_token=req_to_token,
            published_seq_lens=published_seq_lens,
            hidden_stream=None,
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
        self.next_draft = None
        key = self._key(
            batch, verify_input, logits_output, can_run_cuda_graph, max_top_k
        )
        if key is None:
            return None
        outputs = self._graphs.run(
            key=key,
            inputs=_post_verify_inputs(batch, verify_input),
            make_slot=lambda staged: self._make_slot(
                staged, batch, verify_input, logits_output, max_top_k
            ),
            sim_max_len=verify_input.max_tree_depth,
        )
        if outputs is not None:
            self.next_draft = self._graphs.last_slot.next_draft
        return outputs

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
            # compute_spec_logprobs reads the gathered logits off logits_output
            or (
                _SHARDED_VERIFY
                and batch.return_logprob
                and logits_output.next_token_logits_vocab_shard is not None
            )
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
            # A slot that stages the immutable write window cannot be reused by a
            # path without that input (or vice versa).
            verify_input.prepared_out_cache_loc is not None,
            # the graphs read the verify logits in place; a runner may own several
            logits_output.next_token_logits.data_ptr(),
            # and with SGLANG_PERFLL_RS_D1 its hidden rows
            (
                logits_output.hidden_states.data_ptr()
                if self._rs_d1 is not None and logits_output.hidden_states is not None
                else 0
            ),
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
        step2 = self._step2
        return SimpleNamespace(
            next_draft=(
                None
                if step2 is None
                else perfll_step2.make_next_draft(
                    num_requests=len(batch.seq_lens),
                    num_steps=step2.num_steps,
                    window=step2.window,
                    mrope=step2.mrope,
                    device=batch.device,
                    runner_buffers=step2.runner_buffers,
                )
            ),
            num_requests=len(batch.seq_lens),
            tokens_per_request=verify_input.draft_token_num,
            max_tree_depth=verify_input.max_tree_depth,
            is_all_greedy=bool(info.is_all_greedy),
            need_top_k=bool(info.need_top_k_sampling),
            need_top_p=bool(info.need_top_p_sampling),
            max_top_k=max_top_k,
            logits=logits_output.next_token_logits,
            vocab_shard=(
                logits_output.next_token_logits_vocab_shard if _SHARDED_VERIFY else None
            ),
            staged=staged,
            device=batch.device,
            forward_mode=batch.forward_mode,
            tree_cache=batch.tree_cache,
            track_interval=(
                mamba_track_grid(batch.tree_cache.page_size)
                if batch.mamba_track_indices is not None
                else 0
            ),
            extend_inputs=(
                None
                if self._rs_d1 is None or "write_window" not in staged
                else self._extend_inputs_for(
                    num_requests=len(batch.seq_lens),
                    width=verify_input.draft_token_num,
                    hidden=logits_output.hidden_states,
                    tracked=batch.mamba_track_indices is not None,
                )
            ),
            hidden=None if self._rs_d1 is None else logits_output.hidden_states,
        )

    def _extend_inputs_for(
        self, *, num_requests: int, width: int, hidden: Any, tracked: bool
    ) -> Optional[perfll_step2.DraftExtendInputs]:
        """The draft extend runner's input views this key's graph writes (SGLANG_PERFLL_RS_D1),
        or None: only for a size the runner replays unpadded, with every input it copies."""
        rs = self._rs_d1
        if (
            num_requests not in rs.batch_sizes
            or width != rs.width
            or hidden is None
            or rs.buffers.hidden_states is None
            or hidden.shape != (num_requests * width, rs.buffers.hidden_states.shape[1])
            or hidden.dtype != rs.buffers.hidden_states.dtype
            or tracked != (rs.buffers.mamba_track_indices is not None)
        ):
            logger.info(
                "PERF-LL rs-d1: the post-verify graph for %d requests leaves the draft "
                "extend's inputs to its runner",
                num_requests,
            )
            return None
        return perfll_step2.make_draft_extend_inputs(
            runner_buffers=rs.buffers,
            num_requests=num_requests,
            width=width,
            mrope=self._extend_mrope,
            published_seq_lens=rs.published_seq_lens,
        )

    def _body(self, slot: SimpleNamespace) -> tuple:
        """Sampling and acceptance, then the commit outputs; recorded in SGLang's
        post-verify graph."""
        from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs

        extend = slot.extend_inputs
        if extend is not None:
            # the hidden rows do not wait for the acceptance: a branch beside the accept chain
            if self._rs_d1.hidden_stream is None:
                self._rs_d1.hidden_stream = torch.cuda.Stream()
            main, hidden_stream = torch.cuda.current_stream(), self._rs_d1.hidden_stream
            hidden_stream.wait_stream(main)
            with torch.cuda.stream(hidden_stream):
                extend.hidden_states.copy_(slot.hidden)
        predict, accept_lens, accept_index = _sample(slot)
        new_seq_lens, bonus_tokens, commit_steps, draft_extend_inputs = (
            prepare_verify_commit_outputs(
                predict,
                accept_index,
                accept_lens,
                slot.staged["seq_lens"],
                num_draft_tokens=slot.tokens_per_request,
                mamba_track_interval=slot.track_interval,
                extend_layout=self._extend_layout,
                extend_mrope=self._extend_mrope,
                bonus_dtype=_BONUS_DTYPE,
                **(
                    {}
                    if extend is None
                    else dict(
                        runner_inputs=extend,
                        req_pool_indices=slot.staged["req_pool_indices"],
                        track_indices=slot.staged.get("track_indices"),
                        runner_write_window=slot.staged["write_window"],
                    )
                ),
            )
        )
        if slot.next_draft is not None:
            step2 = self._step2
            perfll_step2.stage_next_draft(
                slot.next_draft,
                req_pool_indices=slot.staged["req_pool_indices"],
                new_seq_lens=new_seq_lens,
                req_to_token=step2.req_to_token,
                num_steps=step2.num_steps,
                window=step2.window,
                seq_len_fill_value=step2.seq_len_fill_value,
            )
        if extend is not None:
            main.wait_stream(hidden_stream)
            draft_extend_inputs = (*draft_extend_inputs, extend)
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
        # Scoped opt-in to the existing native chain_accept implementation.
        # Other sampling modes/batch sizes keep the original native path.
        _q38_postverify_fused_accept=(
            _FUSED_BS1_ACCEPT and slot.num_requests == 1 and slot.is_all_greedy
        ),
        draft_probs=None,
    )
    return eagle_sample(
        verify_view,
        batch_view,
        SimpleNamespace(
            next_token_logits=slot.logits,
            next_token_logits_vocab_shard=slot.vocab_shard,
        ),
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
        self._sim_np = None
        self._sim_done: list[Optional[torch.cuda.Event]] = []
        self._sim_turn = 0
        # the slot of the last run that returned outputs
        self.last_slot: Optional[SimpleNamespace] = None

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
        if host_trace.ENABLED:
            host_trace.mark(host_trace.PVA)
        sim_len, sim_turn = self._draw_sim_len(sim_max_len)
        if host_trace.ENABLED:
            host_trace.mark(host_trace.PVB)
        slot = state.slots[turn]
        if _STAGED_SEQ_LENS and slot is not None and slot.graph is not None:
            _check_lengths_outlive(slot, inputs["seq_lens"])
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
        if host_trace.ENABLED:
            host_trace.mark(host_trace.PVC)
        self.last_slot = slot
        if slot.graph is None:
            if state.warmups < _NUM_WARMUP * 2:
                state.warmups += 1
                outputs = self._body(slot)
                self._commit(slot, outputs)
                return outputs[:6]
            self._capture(slot)
        if not getattr(slot, "replay_attested", False):
            logger.info(
                "Q38_MTP_GRAPH_REPLAY bs=%d width=%d track=%s one_launch_load=%s",
                slot.num_requests, slot.tokens_per_request,
                "track_indices" in slot.staged, slot.load_plan is not None,
            )
            slot.replay_attested = True
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
            # numpy views of the pinned slots: a store without a torch dispatch
            self._sim_np = [h.numpy() for h in self._sim_host] if _SIM_PREDRAW else None
        turn = self._sim_turn
        self._sim_turn = (turn + 1) % len(self._sim_host)
        done = self._sim_done[turn]
        if done is not None and not done.query():
            done.synchronize()
        host = self._sim_host[turn]
        value = sample_simulated_acc_len(
            SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, max_len, generator=self._sim_rng
        )
        if self._sim_np is not None:
            self._sim_np[turn][0] = value
        else:
            host[0] = value
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


def _check_lengths_outlive(slot: SimpleNamespace, seq_lens: torch.Tensor) -> None:
    """A batch's lengths taken from a slot's new_seq_lens (SGLANG_PERFLL_STEP5A_SEQ_LENS) are
    read through the whole next round, so that slot's replay must not be the one about to run:
    the slots alternate, and the lengths are retaken from the other slot every round."""
    if seq_lens.data_ptr() == slot.outputs[3].data_ptr():
        raise RuntimeError(
            "PERF-LL step5a: the post-verify slot about to replay holds this batch's "
            "lengths; its readers in this round would see the overwrite"
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
    if verify_input.prepared_out_cache_loc is not None:
        inputs["write_window"] = verify_input.prepared_out_cache_loc
    return inputs
