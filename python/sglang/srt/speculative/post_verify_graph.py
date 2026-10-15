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

Enabled by SGLANG_Q35MF_EDGE_POST_VERIFY_GRAPH=1 (default off).
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs
from sglang.srt.runtime_context import get_parallel, get_spec, mamba_track_grid

logger = logging.getLogger(__name__)

_NUM_WARMUP = 2
_MAX_BS = 2


def post_verify_graph_enabled() -> bool:
    return os.environ.get("SGLANG_Q35MF_EDGE_POST_VERIFY_GRAPH", "0") == "1"


_SPARSE_VERIFY = os.environ.get("SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING", "0") == "1"


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


class _Slot:
    def __init__(self):
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.inputs: Dict[str, torch.Tensor] = {}
        self.outputs: Optional[Tuple] = None
        self.logits_ptr = 0


class _KeyState:
    def __init__(self):
        self.warmups = 0
        self.slots = [_Slot(), _Slot()]
        self.turn = 0


class PostVerifyGraph:
    def __init__(self, *, device: torch.device, target_worker: Any):
        self.device = device
        self.target_worker = target_worker
        self.disabled = False
        self._states: Dict[tuple, _KeyState] = {}
        self._pool = None
        # The host runs up to ~2 rounds ahead of the device; a pinned slot is not
        # rewritten before its async copy has executed.
        self._sim_len_host = [
            torch.zeros((1,), dtype=torch.int64, pin_memory=True) for _ in range(4)
        ]
        self._sim_len_turn = 0
        self._check = os.environ.get("SGLANG_Q35MF_EDGE_POST_VERIFY_CHECK", "0") == "1"
        self._max_top_k: Optional[int] = None
        self._checked = 0
        self._mismatch = 0

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
        if self.disabled:
            return None
        if self._host_hooked():
            # A per-round host hook on the acceptance kernel cannot be replayed from a graph.
            logger.warning("post-verify graph off: verify_tree_greedy_func is wrapped by a host hook")
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
            return None
        key = key + (uno_target_max_top_k,)
        self._max_top_k = uno_target_max_top_k
        try:
            return self._run(key, batch, verify_input, logits_output, num_draft_tokens)
        except Exception:
            logger.exception("post-verify graph disabled after an error; stock path from now on")
            self.disabled = True
            raise

    @staticmethod
    def _host_hooked() -> bool:
        from sglang.srt.speculative import eagle_utils

        return hasattr(eagle_utils.verify_tree_greedy_func, "__wrapped__")

    def _key(self, *, batch, verify_input, logits_output, num_draft_tokens, topk, token_to_kv_pool_allocator):
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
            or (model_runner.watermark_state is not None and info.has_watermark_candidates)
            or getattr(token_to_kv_pool_allocator.get_kvcache(), "clear_unaccepted_c128_draft_states", None)
            is not None
            or not logits_output.next_token_logits.is_cuda
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
        host = self._sim_len_host[self._sim_len_turn]
        self._sim_len_turn = (self._sim_len_turn + 1) % len(self._sim_len_host)
        host[0] = sample_simulated_acc_len(SIMULATE_ACC_LEN, SIMULATE_ACC_METHOD, max_len)
        return host

    def _run(self, key, batch, verify_input, logits_output, num_draft_tokens):
        st = self._states.get(key)
        if st is None:
            st = self._states[key] = _KeyState()
        slot = st.slots[st.turn]
        st.turn ^= 1
        logits_ptr = logits_output.next_token_logits.data_ptr()
        if slot.graph is not None and slot.logits_ptr != logits_ptr:
            raise RuntimeError("verify logits moved; the captured post-verify graph reads a stale buffer")
        slot.logits_ptr = logits_ptr
        inputs = self._gather_inputs(batch, verify_input)
        sim_len = self._sim_len(verify_input.max_tree_depth)
        if slot.graph is None:
            if not slot.inputs:
                slot.inputs = {k: v.clone() for k, v in inputs.items()}
                slot.inputs["sim_len"] = torch.zeros((1,), dtype=torch.int64, device=self.device)
            self._load(slot, inputs, sim_len)
            if st.warmups < _NUM_WARMUP * 2:
                st.warmups += 1
                return self._body(slot.inputs, batch, verify_input, logits_output, num_draft_tokens)
            self._capture(slot, batch, verify_input, logits_output, num_draft_tokens)
        else:
            self._load(slot, inputs, sim_len)
        slot.graph.replay()
        if self._check:
            self._compare(slot, batch, verify_input, logits_output, num_draft_tokens)
        return slot.outputs

    def _compare(self, slot: _Slot, batch, verify_input, logits_output, num_draft_tokens):
        # Debug only (host syncs): rerun the eager sampling and commit-output math on the same
        # inputs; the fields compared do not depend on the sampling coins.
        from sglang.srt.speculative.eagle_utils import eagle_sample

        predict, accept_lens, accept_index, new_seq_lens, bonus, _ = slot.outputs
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
                "post-verify graph check: %d rounds, %d mismatches", self._checked, self._mismatch
            )

    def _load(self, slot: _Slot, inputs: Dict[str, torch.Tensor], sim_len):
        for name, src in inputs.items():
            dst = slot.inputs[name]
            if dst.data_ptr() != src.data_ptr():
                dst.copy_(src, non_blocking=True)
        if sim_len is not None:
            slot.inputs["sim_len"].copy_(sim_len, non_blocking=True)

    def _capture(self, slot: _Slot, batch, verify_input, logits_output, num_draft_tokens):
        from sglang.srt.distributed.parallel_state import graph_capture

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.current_stream()
        graph = torch.cuda.CUDAGraph()
        with graph_capture(stream=stream) as ctx:
            with torch.cuda.graph(
                graph, pool=self._pool, stream=ctx.stream, capture_error_mode="thread_local"
            ):
                outputs = self._body(slot.inputs, batch, verify_input, logits_output, num_draft_tokens)
        slot.graph = graph
        slot.outputs = outputs
        logger.info("Captured post-verify graph for bs=%d", len(batch.seq_lens))

    def _body(self, x, batch, verify_input, logits_output, num_draft_tokens):
        from sglang.srt.speculative.eagle_utils import eagle_sample
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

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
        new_seq_lens, bonus_tokens, commit_steps, draft_inputs = prepare_verify_commit_outputs(
            predict,
            accept_index,
            accept_lens,
            x["seq_lens"],
            num_draft_tokens=num_draft_tokens,
            mamba_track_interval=track_interval,
        )
        commit_mamba_states_after_verify(
            self.target_worker,
            batch_view,
            accept_lens,
            accept_index,
            num_draft_tokens,
            prepared_step_indices=commit_steps,
        )
        return predict, accept_lens, accept_index, new_seq_lens, bonus_tokens, draft_inputs

    @staticmethod
    def _views(x, batch, verify_input):
        info = batch.sampling_info
        info_view = SimpleNamespace(
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
