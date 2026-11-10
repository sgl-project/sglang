# SPDX-License-Identifier: Apache-2.0
"""SGLang's post-verify CUDA graphs for the EAGLE verify
(SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH): sampling and acceptance, the commit
outputs and, on a side stream, the mamba state commit, for every batch a graph can
reproduce; any other batch runs SGLang's eager path.

The side-stream commit may run after the scheduler reused a finished request's
row, so it writes the verify forward's slots, staged with the round's inputs."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.kernels.ops.memory.small_field_copy import SmallFieldCopyPlan
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardStateSlots
from sglang.srt.speculative.eagle_worker_common import (
    STOCK_SPECULATIVE_RUNTIME_HOOKS,
    make_post_verify_io,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

# Eager rounds per post-verify key before its graphs are captured.
_NUM_WARMUP = 2
# Arbitrary bound on post-verify graph sets (key x verify logits buffer); a batch
# past it runs SGLang's eager path.
_MAX_POST_VERIFY_STATES = 64
# SGLANG_OPT_SPEC_POST_VERIFY_ONE_LAUNCH_LOAD: a round's inputs and its forced length staged in one launch.
_ONE_LAUNCH_LOAD = envs.SGLANG_OPT_SPEC_POST_VERIFY_ONE_LAUNCH_LOAD.get()


class EaglePostVerifyGraphs:
    """The verify's post-section in SGLang's own CUDA graphs, one set per key: batch
    size, sampling mode, forced acceptance, tracking and the verify logits buffer."""

    def __init__(
        self,
        *,
        target_worker: Any,
        simulated_acc_len_rng: torch.Generator,
        speculative_runtime_hooks: Any = STOCK_SPECULATIVE_RUNTIME_HOOKS,
    ):
        self._target_worker = target_worker
        self._speculative_runtime_hooks = speculative_runtime_hooks
        # Whether the commit writes recurrent state slots; resolved at first use.
        self._stages_state_slots: Optional[bool] = None
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
            inputs=_post_verify_inputs(
                batch, verify_input, self._verify_state_slots(batch)
            ),
            make_slot=lambda staged: self._make_slot(
                staged, batch, verify_input, logits_output, max_top_k
            ),
            sim_max_len=verify_input.max_tree_depth,
        )

    def _verify_state_slots(self, batch: ScheduleBatch) -> Optional[ForwardStateSlots]:
        """The verify forward's state slots, which the mamba commit writes; None
        for a model without recurrent state."""
        model_runner = self._target_worker.model_runner
        if self._stages_state_slots is None:
            self._stages_state_slots = (
                mambaish_config(model_runner.model_config) is not None
            )
        if not self._stages_state_slots:
            return None
        return model_runner.attn_backend.forward_state_slots(
            len(batch.seq_lens), batch.mamba_track_indices
        )

    def eager_simulated_accept_length(
        self, *, max_tree_depth: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """A forced acceptance length (SGLANG_SIMULATE_ACC_LEN) for a round these graphs
        do not replay, drawn from their stream and staged on the device as a replay
        would, so the eager verify takes the same fused chain; None when off."""
        return self._graphs.draw_sim_len_on(max_tree_depth, device)

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
                batch.return_logprob
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
            # the graphs read the verify logits in place; a runner may own several
            logits_output.next_token_logits.data_ptr(),
            # shards borrow the full logits storage, so the pointer alone is ambiguous
            logits_output.next_token_logits_vocab_shard is not None,
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
            unused_sampling_tensor=torch.empty(0, device=batch.device),
            return_logprob=bool(batch.return_logprob),
            logits=logits_output.next_token_logits,
            vocab_shard=logits_output.next_token_logits_vocab_shard,
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

        predict, accept_lens, accept_index = _sample(
            slot, self._speculative_runtime_hooks
        )
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
        staged = slot.staged
        batch_view = SimpleNamespace(
            forward_mode=slot.forward_mode,
            seq_lens=staged["seq_lens"],
            req_pool_indices=staged["req_pool_indices"],
            mamba_track_indices=staged.get("track_indices"),
            tree_cache=slot.tree_cache,
        )
        state_slots = None
        if "state_slots" in staged:
            state_slots = ForwardStateSlots(
                staged["state_slots"], staged.get("track_state_slots")
            )
        commit_mamba_states_after_verify(
            self._target_worker,
            batch_view,
            outputs[1],
            outputs[2],
            slot.tokens_per_request,
            prepared_step_indices=outputs[6],
            state_slots=state_slots,
        )


def _sample(slot: SimpleNamespace, speculative_runtime_hooks: Any) -> tuple:
    """Run post-verify through the same bound strategy as the eager path."""

    staged = slot.staged
    unused = slot.unused_sampling_tensor
    # Only the staged tensors change between replays. Keys exclude min-p,
    # watermark and seeded rounds, and greedy keys read no sampling tensor, so the
    # remaining fields are empty or None: a reader fails rather than seeing the
    # capture round's values.
    assert slot.is_all_greedy or "temperatures" in staged
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
        temperatures=staged.get("temperatures", unused),
        top_ks=staged.get("top_ks", unused),
        top_ps=staged.get("top_ps", unused),
        min_ps=unused,
        max_top_k=None,
        watermark_keys=None,
        watermark_context_windows=None,
        watermark_enabled=None,
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
        custom_mask=None,
        positions=None,
    )
    output = speculative_runtime_hooks.post_verify(
        make_post_verify_io(
            verify_input=verify_view,
            sampling_info=sampling_info,
            sequence_lengths=staged["seq_lens"],
            request_pool_indices=staged["req_pool_indices"],
            device=slot.device,
            logits_output=SimpleNamespace(
                next_token_logits=slot.logits,
                next_token_logits_vocab_shard=slot.vocab_shard,
            ),
            grammar_mask=None,
            # Post-verify graph eligibility excludes every watermark round.
            watermark=None,
            watermark_full_mask=True,
            return_logprob=slot.return_logprob,
            is_idle=False,
            simulated_accept_length=staged.get("sim_accept_len"),
            sparse_target_max_top_k=slot.max_top_k,
            sparse_bonus_argmax=slot.max_top_k is not None,
        )
    )
    return output.token_ids, output.accepted_lengths, output.accepted_indices


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

    def draw_sim_len_on(
        self, max_len: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        sim_len, turn = self._draw_sim_len(max_len)
        if sim_len is None:
            return None
        staged = sim_len.to(device, non_blocking=True)
        self._record_sim_done(turn)
        return staged

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
    batch: ScheduleBatch,
    verify_input: Any,
    state_slots: Optional[ForwardStateSlots] = None,
) -> dict[str, torch.Tensor]:
    """The round's inputs the post-verify graphs read, staged into fixed buffers.
    ``state_slots``: the verify forward's own, which the mamba commit writes."""
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
    if state_slots is not None:
        inputs["state_slots"] = state_slots.working
        if state_slots.track is not None:
            inputs["track_state_slots"] = state_slots.track
    return inputs
