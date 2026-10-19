"""One CUDA graph for the chain that follows the target-verify forward.

After the verify graph, a decode round runs sampling/acceptance (eagle_sample),
the verify commit outputs and the mamba state commit as ~30 eager launches.
For a batch whose Python-visible branches are fixed (same sampling flags,
no grammar / penalties) this replays the same calls, on static copies of
their per-round inputs, as one captured graph; logprobs are computed after
the replay from its outputs, as on the eager path.

The injected acceptance length (benchmark mode) is drawn on the host exactly
as before and enters the graph as a device scalar, so real and injected
acceptance replay the same graph. Two graphs per key alternate so that one
round's outputs stay valid until the host has consumed them.

Enabled by SGLANG_Q35MF_EDGE_POST_VERIFY_GRAPH=1 (default off), for batches of up to
SGLANG_Q35MF_EDGE_POST_VERIFY_MAX_BS requests (default 2).
"""

from __future__ import annotations

import contextlib
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.kernels.ops.memory.small_field_copy import SmallFieldCopyPlan
from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_spec, mamba_track_grid

logger = logging.getLogger(__name__)

_NUM_WARMUP = 2
# Largest batch replayed from the graph; larger batches keep the eager chain.
_MAX_BS = int(os.environ.get("SGLANG_Q35MF_EDGE_POST_VERIFY_MAX_BS", "2"))
# Arbitrary bound on captured graph sets (key x verify logits buffer).
_MAX_STATES = max(16, 8 * _MAX_BS)
# With the side-stream commit: largest batch whose commit runs on the side stream; larger
# batches commit inside the main graph (the side commit slows the overlapping draft extend).
_SIDE_MAX_BS = int(
    os.environ.get("SGLANG_Q35MF_EDGE_POST_VERIFY_SIDE_MAX_BS", str(_MAX_BS))
)


def post_verify_graph_enabled() -> bool:
    return os.environ.get("SGLANG_Q35MF_EDGE_POST_VERIFY_GRAPH", "0") == "1"


_SPARSE_VERIFY = os.environ.get("SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING", "0") == "1"
_SIDE_STREAM_COMMIT = os.environ.get("SGLANG_Q35MF_EDGE_SIDE_STREAM_COMMIT", "0") == "1"
_ONE_LAUNCH_LOAD = envs.SGLANG_Q35MF_MKH_N3C.get()
_SHARDED_IN_GRAPH = envs.SGLANG_Q35MF_D2R_PVG_SHARDED.get()


def _check_sharded_sampling_config() -> None:
    # Without the in-graph sharded sampler, graphed rounds above batch 2 would gather the
    # shard-local verify logits; that combination is slower and was never validated.
    if (
        _MAX_BS > 2
        and envs.SGLANG_Q35MF_PDDEC_SHARDED_VERIFY_SAMPLING.get()
        and not _SHARDED_IN_GRAPH
    ):
        raise ValueError(
            "SGLANG_Q35MF_EDGE_POST_VERIFY_MAX_BS > 2 with "
            "SGLANG_Q35MF_PDDEC_SHARDED_VERIFY_SAMPLING=1 requires "
            "SGLANG_Q35MF_D2R_PVG_SHARDED=1"
        )


class PostVerifySamplingView(SimpleNamespace):
    """Batch sampling info inside the post-verify graph that the sharded verify sampler accepts."""


def sparse_verify_max_top_k(batch) -> Optional[int]:
    """Support size for eagle_sample's compact top-k sampling branch, or None for the dense
    branch. Every request must sample with a finite top-k within the compact limit."""
    if not _SPARSE_VERIFY or batch.forward_mode.is_idle():
        return None
    from sglang.srt.speculative.uno_utils import _SPARSE_TOP_K_LIMIT

    top_ks = [req.sampling_params.top_k for req in batch.reqs]
    if not top_ks or any(k <= 1 or k > _SPARSE_TOP_K_LIMIT for k in top_ks):
        return None
    return max(top_ks)


@contextlib.contextmanager
def _count_acceptance_calls():
    # eagle_sample resolves both kernels through these module attributes at call time.
    from sglang.kernels.ops.speculative import sampling as sampling_ops
    from sglang.srt.speculative import eagle_utils
    from sglang.srt.speculative.sharded_verify_sampling import ShardedVerifySampler

    # [acceptance calls, of which sharded sampler runs]
    calls = [0, 0]
    greedy_fn = eagle_utils.verify_tree_greedy_func
    tree_fn = sampling_ops.tree_speculative_sampling_target_only
    sharded_fn = ShardedVerifySampler.run

    def counted(fn, sharded=False):
        def call(*args, **kwargs):
            calls[0] += 1
            calls[1] += int(sharded)
            return fn(*args, **kwargs)

        return call

    eagle_utils.verify_tree_greedy_func = counted(greedy_fn)
    sampling_ops.tree_speculative_sampling_target_only = counted(tree_fn)
    if eagle_utils._LOCAL_VERIFY_SAMPLING:
        # The one-rank chunked sampler is the acceptance kernel on that path.
        from sglang.srt.speculative import sharded_verify_sampling as local_ops

        local_fn = local_ops.sample_chain_full_vocab
        local_ops.sample_chain_full_vocab = counted(local_fn)
    # The sharded sampler decides acceptance inside its own kernels.
    ShardedVerifySampler.run = counted(sharded_fn, sharded=True)
    try:
        yield calls
    finally:
        eagle_utils.verify_tree_greedy_func = greedy_fn
        sampling_ops.tree_speculative_sampling_target_only = tree_fn
        if eagle_utils._LOCAL_VERIFY_SAMPLING:
            local_ops.sample_chain_full_vocab = local_fn
        ShardedVerifySampler.run = sharded_fn


class _Slot:
    def __init__(self):
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.commit_graph: Optional[torch.cuda.CUDAGraph] = None
        self.inputs: Dict[str, torch.Tensor] = {}
        self.outputs: Optional[Tuple] = None
        self.load_plan = (
            SmallFieldCopyPlan(allow_pinned_src=True) if _ONE_LAUNCH_LOAD else None
        )


class _KeyState:
    def __init__(self):
        self.warmups = 0
        self.slots = [_Slot(), _Slot()]
        self.turn = 0


class PostVerifyGraph:
    def __init__(self, *, device: torch.device, target_worker: Any):
        _check_sharded_sampling_config()
        self.device = device
        self.target_worker = target_worker
        self.disabled = False
        self._states: Dict[tuple, _KeyState] = {}
        self._pool = None
        # The host runs up to ~2 rounds ahead of the device; a pinned slot is
        # rewritten only after the event behind its last async copy has completed.
        self._sim_len_host = [
            torch.zeros((1,), dtype=torch.int64, pin_memory=True) for _ in range(4)
        ]
        self._sim_len_done: list = [None] * len(self._sim_len_host)
        self._sim_len_turn = 0
        self._sim_len_stalls = 0
        self._sim_len_slot = 0
        self._check = os.environ.get("SGLANG_Q35MF_EDGE_POST_VERIFY_CHECK", "0") == "1"
        # The mamba state commit replays on its own stream so it overlaps the draft
        # extend; join() must run before any later work that reads that state.
        self._side_stream = torch.cuda.Stream() if _SIDE_STREAM_COMMIT else None
        self._main_done = torch.cuda.Event() if _SIDE_STREAM_COMMIT else None
        self._commit_done = torch.cuda.Event() if _SIDE_STREAM_COMMIT else None
        self._commit_pending = False
        self._max_top_k: Optional[int] = None
        # Engagement counters: replayed rounds per key and rounds left to the stock path.
        self._graph_rounds: Dict[tuple, int] = {}
        self._stock_rounds = 0
        self._checked = 0
        self._mismatch = 0
        self._load_mismatch = 0

    def try_run(
        self,
        *,
        batch,
        verify_input,
        logits_output,
        num_draft_tokens: int,
        topk: int,
        token_to_kv_pool_allocator,
        uno_target_max_top_k: Optional[int] = None,
    ):
        """Returns (predict, accept_lens, accept_index, new_seq_lens, bonus_tokens,
        prepared_draft_inputs) or None when the stock path must run."""
        if self._commit_pending:
            # The worker did not join after the last round; order the commit before this verify.
            logger.error(
                "post-verify graph: side-stream commit was not joined; joining late"
            )
            self.join()
        if self.disabled:
            return None
        if self._host_hooked():
            # A per-round host hook on the acceptance kernel cannot be replayed from a graph.
            logger.warning(
                "post-verify graph off: verify_tree_greedy_func is wrapped by a host hook"
            )
            self.disabled = True
            return None
        key = self._key(
            batch=batch,
            verify_input=verify_input,
            logits_output=logits_output,
            num_draft_tokens=num_draft_tokens,
            topk=topk,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if key is None:
            self._count_round(None)
            return None
        key = key + (uno_target_max_top_k,)
        # The graph reads the verify logits in place, and the verify runner may hold
        # several logits buffers per batch size, so graphs are kept per buffer.
        state_key = (key, logits_output.next_token_logits.data_ptr())
        if state_key not in self._states and len(self._states) >= _MAX_STATES:
            self._count_round(None)
            return None
        self._max_top_k = uno_target_max_top_k
        self._count_round(key)
        try:
            return self._run(
                state_key, batch, verify_input, logits_output, num_draft_tokens
            )
        except Exception:
            logger.exception(
                "post-verify graph disabled after an error; stock path from now on"
            )
            self.disabled = True
            raise

    def _count_round(self, key) -> None:
        if key is None:
            self._stock_rounds += 1
        else:
            self._graph_rounds[key] = self._graph_rounds.get(key, 0) + 1
        total = self._stock_rounds + sum(self._graph_rounds.values())
        if total % 2000 == 0:
            logger.info(
                "post-verify graph rounds: stock %d, graph %s",
                self._stock_rounds,
                self._graph_rounds,
            )

    def join(self) -> None:
        """Makes the current stream wait for the last side-stream mamba state commit."""
        if self._commit_pending:
            torch.cuda.current_stream().wait_event(self._commit_done)
            self._commit_pending = False

    @staticmethod
    def _host_hooked() -> bool:
        from sglang.srt.speculative import eagle_utils

        return hasattr(eagle_utils.verify_tree_greedy_func, "__wrapped__")

    def _key(
        self,
        *,
        batch,
        verify_input,
        logits_output,
        num_draft_tokens,
        topk,
        token_to_kv_pool_allocator,
    ):
        from sglang.srt.layers.logits_processor import verify_logits_on_shards
        from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN

        info = batch.sampling_info
        model_runner = self.target_worker.model_runner
        bs = len(batch.seq_lens)
        if (
            batch.forward_mode.is_idle()
            or topk != 1
            or bs > _MAX_BS
            or batch.has_grammar
            or batch.hisparse_coordinator is not None
            or get_parallel().pp_size > 1
            or get_spec().speculative_use_rejection_sampling
            or verify_input.draft_probs is not None
            or info.acc_additive_penalties is not None
            or info.acc_scaling_penalties is not None
            or info.logit_bias is not None
            or info.sampling_seed is not None
            or info.need_min_p_sampling
            or (
                model_runner.watermark_state is not None
                and info.has_watermark_candidates
            )
            or getattr(
                token_to_kv_pool_allocator.get_kvcache(),
                "clear_unaccepted_c128_draft_states",
                None,
            )
            is not None
            or not logits_output.next_token_logits.is_cuda
            or (
                (batch.return_logprob or info.has_custom_logit_processor)
                and verify_logits_on_shards(logits_output.next_token_logits)
            )
        ):
            return None
        return (
            bs,
            bool(info.is_all_greedy),
            bool(info.need_top_k_sampling),
            bool(info.need_top_p_sampling),
            SIMULATE_ACC_LEN > 0,
            batch.mamba_track_indices is not None,
            num_draft_tokens,
            verify_input.max_tree_depth,
        )

    def _gather_inputs(self, batch, verify_input) -> Dict[str, torch.Tensor]:
        info = batch.sampling_info
        x = {
            "draft_token": verify_input.draft_token,
            "retrieve_index": verify_input.retrieve_index,
            "retrieve_next_token": verify_input.retrieve_next_token,
            "retrieve_next_sibling": verify_input.retrieve_next_sibling,
            "seq_lens": batch.seq_lens,
            "req_pool_indices": batch.req_pool_indices,
        }
        if not info.is_all_greedy:
            x["temperatures"] = info.temperatures
            x["top_ks"] = info.top_ks
            x["top_ps"] = info.top_ps
        elif _SHARDED_IN_GRAPH:
            # The sharded sampler reads a temperature for every row, greedy batches included.
            x["temperatures"] = info.temperatures
        if batch.mamba_track_indices is not None:
            x["mamba_track_indices"] = batch.mamba_track_indices
        return x

    def _sim_len(self, max_len: int) -> Optional[torch.Tensor]:
        from sglang.srt.speculative.spec_utils import (
            SIMULATE_ACC_LEN,
            SIMULATE_ACC_METHOD,
            sample_simulated_acc_len,
        )

        if SIMULATE_ACC_LEN <= 0:
            return None
        # Same host draw, at the same point of the round, as the eager path.
        turn = self._sim_len_turn
        self._sim_len_turn = (turn + 1) % len(self._sim_len_host)
        done = self._sim_len_done[turn]
        if done is not None and not done.query():
            # Host is more slots ahead than the ring covers; never overwrite a pending source.
            self._sim_len_stalls += 1
            if self._sim_len_stalls == 1:
                logger.warning(
                    "post-verify graph: injected-length ring wrapped before its copy ran"
                )
            done.synchronize()
        host = self._sim_len_host[turn]
        host[0] = sample_simulated_acc_len(
            SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, max_len
        )
        self._sim_len_slot = turn
        return host

    def _run(self, state_key, batch, verify_input, logits_output, num_draft_tokens):
        st = self._states.get(state_key)
        if st is None:
            st = self._states[state_key] = _KeyState()
        slot = st.slots[st.turn]
        st.turn ^= 1
        inputs = self._gather_inputs(batch, verify_input)
        sim_len = self._sim_len(verify_input.max_tree_depth)
        if slot.graph is None:
            if not slot.inputs:
                slot.inputs = {k: v.clone() for k, v in inputs.items()}
                slot.inputs["sim_len"] = torch.zeros(
                    (1,), dtype=torch.int64, device=self.device
                )
            self._load(slot, inputs, sim_len)
            if st.warmups < _NUM_WARMUP * 2:
                st.warmups += 1
                return self._body(
                    slot.inputs, batch, verify_input, logits_output, num_draft_tokens
                )[:6]
            self._capture(
                slot,
                batch,
                verify_input,
                logits_output,
                num_draft_tokens,
                key=state_key[0],
            )
        else:
            self._load(slot, inputs, sim_len)
        if self._check:
            self._check_load(slot, inputs, sim_len)
        slot.graph.replay()
        if slot.commit_graph is not None:
            self._main_done.record()
            self._side_stream.wait_event(self._main_done)
            with torch.cuda.stream(self._side_stream):
                slot.commit_graph.replay()
            self._commit_done.record(self._side_stream)
            self._commit_pending = True
        if self._check:
            self._compare(slot, batch, verify_input, logits_output, num_draft_tokens)
        return slot.outputs[:6]

    def _check_load(self, slot: _Slot, inputs: Dict[str, torch.Tensor], sim_len):
        # Debug only (host syncs): the slot must hold exactly this round's sources.
        same = all(torch.equal(slot.inputs[k], v) for k, v in inputs.items())
        if sim_len is not None:
            same = same and torch.equal(slot.inputs["sim_len"].cpu(), sim_len)
        if not same:
            self._load_mismatch += 1
            logger.error(
                "post-verify graph: slot inputs differ from their sources (%d so far)",
                self._load_mismatch,
            )

    def _compare(
        self, slot: _Slot, batch, verify_input, logits_output, num_draft_tokens
    ):
        # Debug only (host syncs): rerun the eager sampling and commit-output math on the same
        # inputs; the fields compared do not depend on the sampling coins.
        from sglang.srt.speculative.eagle_utils import eagle_sample

        predict, accept_lens, accept_index, new_seq_lens, bonus, _ = slot.outputs[:6]
        views = self._views(slot.inputs, batch, verify_input)
        r_predict, r_lens, r_index = eagle_sample(
            views[1],
            views[0],
            logits_output,
            None,
            uno_target_max_top_k=self._max_top_k,
            simulated_acc_len=slot.inputs["sim_len"],
            sparse_bonus_argmax=self._max_top_k is not None,
        )
        r_seq, r_bonus = prepare_verify_commit_outputs(
            r_predict, r_index, r_lens, slot.inputs["seq_lens"]
        )
        valid = r_index.clamp(min=0).long()
        same = (
            torch.equal(accept_lens, r_lens)
            and torch.equal(accept_index, r_index)
            and torch.equal(new_seq_lens, r_seq)
            and torch.equal(bonus, r_bonus)
            and torch.equal(predict[valid], r_predict[valid])
        )
        self._checked += 1
        self._mismatch += 0 if same else 1
        if not same or self._checked % 1000 == 0:
            logger.info(
                "post-verify graph check: %d rounds, %d mismatches, %d load mismatches",
                self._checked,
                self._mismatch,
                self._load_mismatch,
            )

    def _load(self, slot: _Slot, inputs: Dict[str, torch.Tensor], sim_len):
        if slot.load_plan is not None:
            dsts = [slot.inputs[name] for name in inputs]
            srcs = list(inputs.values())
            if sim_len is not None:
                dsts.append(slot.inputs["sim_len"])
                srcs.append(sim_len)
            if slot.load_plan(dsts, srcs):
                if sim_len is not None:
                    self._record_sim_len_done()
                return
        # One multi-tensor copy per dtype; one memcpy per input costs ~4 us of GPU time each.
        groups: Dict[torch.dtype, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        for name, src in inputs.items():
            dst = slot.inputs[name]
            if dst.data_ptr() != src.data_ptr():
                dsts, srcs = groups.setdefault(src.dtype, ([], []))
                dsts.append(dst)
                srcs.append(src)
        for dsts, srcs in groups.values():
            torch._foreach_copy_(dsts, srcs, non_blocking=True)
        if sim_len is not None:
            slot.inputs["sim_len"].copy_(sim_len, non_blocking=True)
            self._record_sim_len_done()

    def _record_sim_len_done(self) -> None:
        done = self._sim_len_done[self._sim_len_slot]
        if done is None:
            done = self._sim_len_done[self._sim_len_slot] = torch.cuda.Event()
        done.record()

    def _capture(
        self,
        slot: _Slot,
        batch,
        verify_input,
        logits_output,
        num_draft_tokens,
        key=None,
    ):
        from sglang.srt.distributed.parallel_state import graph_capture

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.current_stream()
        graph = torch.cuda.CUDAGraph()
        side = self._side_stream is not None and len(batch.seq_lens) <= _SIDE_MAX_BS
        with _count_acceptance_calls() as acceptance_calls:
            with graph_capture(stream=stream) as ctx:
                with torch.cuda.graph(
                    graph,
                    pool=self._pool,
                    stream=ctx.stream,
                    capture_error_mode="thread_local",
                ):
                    outputs = self._body(
                        slot.inputs,
                        batch,
                        verify_input,
                        logits_output,
                        num_draft_tokens,
                        commit=not side,
                    )
        # Injected acceptance only overrides the result; the real acceptance kernel must be in the graph.
        if acceptance_calls[0] == 0:
            raise RuntimeError(
                "post-verify graph captured without the acceptance kernel"
            )
        if side:
            # Captured after the main graph in the same pool and replayed after it every round.
            commit_graph = torch.cuda.CUDAGraph()
            self._side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(
                commit_graph,
                pool=self._pool,
                stream=self._side_stream,
                capture_error_mode="thread_local",
            ):
                self._commit(
                    slot.inputs, batch, verify_input, outputs, num_draft_tokens
                )
            torch.cuda.current_stream().wait_stream(self._side_stream)
            slot.commit_graph = commit_graph
        slot.graph = graph
        slot.outputs = outputs
        logger.info(
            "Captured post-verify graph for bs=%d key=%s side_commit=%s sharded_sampler=%s",
            len(batch.seq_lens),
            key,
            slot.commit_graph is not None,
            acceptance_calls[1] > 0,
        )

    def _body(
        self, x, batch, verify_input, logits_output, num_draft_tokens, commit=True
    ):
        from sglang.srt.speculative.eagle_utils import eagle_sample

        batch_view, verify_view = self._views(x, batch, verify_input)
        predict, accept_lens, accept_index = eagle_sample(
            verify_view,
            batch_view,
            logits_output,
            None,
            uno_target_max_top_k=self._max_top_k,
            simulated_acc_len=x["sim_len"],
            sparse_bonus_argmax=self._max_top_k is not None,
        )
        track_interval = (
            mamba_track_grid(batch.tree_cache.page_size)
            if batch.mamba_track_indices is not None
            else 0
        )
        new_seq_lens, bonus_tokens, commit_steps, draft_inputs = (
            prepare_verify_commit_outputs(
                predict,
                accept_index,
                accept_lens,
                x["seq_lens"],
                num_draft_tokens=num_draft_tokens,
                mamba_track_interval=track_interval,
            )
        )
        outputs = (
            predict,
            accept_lens,
            accept_index,
            new_seq_lens,
            bonus_tokens,
            draft_inputs,
            commit_steps,
        )
        if commit:
            self._commit(x, batch, verify_input, outputs, num_draft_tokens)
        return outputs

    def _commit(self, x, batch, verify_input, outputs, num_draft_tokens):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        batch_view, _ = self._views(x, batch, verify_input)
        commit_mamba_states_after_verify(
            self.target_worker,
            batch_view,
            outputs[1],
            outputs[2],
            num_draft_tokens,
            prepared_step_indices=outputs[6],
        )

    @staticmethod
    def _views(x, batch, verify_input):
        info = batch.sampling_info
        view_fields = dict(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            has_watermark_candidates=False,
            is_all_greedy=info.is_all_greedy,
            temperatures=x.get("temperatures"),
            top_ks=x.get("top_ks"),
            top_ps=x.get("top_ps"),
            need_top_k_sampling=info.need_top_k_sampling,
            need_top_p_sampling=info.need_top_p_sampling,
            need_min_p_sampling=False,
            sampling_seed=None,
        )
        if _SHARDED_IN_GRAPH:
            info_view = PostVerifySamplingView(
                vocab_size=info.vocab_size,
                has_custom_logit_processor=info.has_custom_logit_processor,
                **view_fields,
            )
        else:
            info_view = SimpleNamespace(**view_fields)
        batch_view = SimpleNamespace(
            device=batch.device,
            forward_mode=batch.forward_mode,
            seq_lens=x["seq_lens"],
            req_pool_indices=x["req_pool_indices"],
            mamba_track_indices=x.get("mamba_track_indices"),
            sampling_info=info_view,
            return_logprob=False,
        )
        verify_view = SimpleNamespace(
            draft_token=x["draft_token"],
            draft_token_num=verify_input.draft_token_num,
            max_tree_depth=verify_input.max_tree_depth,
            retrieve_index=x["retrieve_index"],
            retrieve_next_token=x["retrieve_next_token"],
            retrieve_next_sibling=x["retrieve_next_sibling"],
            tree_topk=verify_input.tree_topk,
            draft_probs=None,
        )
        return batch_view, verify_view
