# SPDX-License-Identifier: Apache-2.0
"""SGLang's side of a megakernel's EAGLE entries (spec.ENTRIES) at the EAGLE workers'
and their graph runners' call sites; kernel_plugins/nvidia/eagle_binding.py documents
the entries and their structs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.kernel_plugins.nvidia.binding import (
    megakernel_agree,
    megakernel_enabled,
    megakernel_entry,
)
from sglang.srt.kernel_plugins.nvidia.eagle_binding import (
    EagleDraftExtendIO,
    EagleDraftLoopIO,
    EagleDraftRoundIO,
    attn_step_metadata,
    kv_pool_buffers,
    split_backend,
)
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel, get_spec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# the draft model's entries
DRAFT_ENTRIES = ("draft_loop", "draft_extend", "begin_draft_round")


def _offers_eagle_entries(*, draft_model: Any) -> bool:
    """Whether the draft's megakernel offers an EAGLE entry."""
    return any(
        megakernel_entry(draft_model, name) is not None for name in DRAFT_ENTRIES
    )


def _unservable_reasons(draft: Any) -> list[str]:
    """Why the EAGLE entries cannot run on this draft worker."""
    reasons = []
    if draft.topk != 1:
        reasons.append("the entries run a chain draft (topk 1)")
    # None for speculative_num_steps <= 1 (DraftBackendFactory.create_decode_backend)
    multi_step = draft.draft_attn_backend
    if multi_step is None:
        reasons.append("the entries run a multi-step draft (speculative_num_steps > 1)")
    if get_spec().speculative_use_rejection_sampling:
        reasons.append("the entries do not run rejection sampling")
    # the draft's top-1 would be a hot-vocab index, and the entries get no map
    if draft.hot_token_id is not None:
        reasons.append("the entries do not run a speculative token map")
    if get_parallel().attn_dp_enabled:
        # the graph runners would pad to other DP ranks' batches, which _bucket does
        # not see
        reasons.append("the entries do not run under DP attention")
    return reasons + _backend_reasons(
        target=draft.target_worker.model_runner.attn_backend,
        draft=(
            *(() if multi_step is None else multi_step.attn_backends),
            draft.draft_extend_attn_backend or draft.draft_runner.attn_backend,
        ),
    )


def init_eagle_megakernel(*, draft_worker: Any) -> Optional[EagleMegakernel]:
    """With --enable-nvidia-megakernel, SGLang's side of the entries the draft's
    megakernel offers; called once the pools and both attention backends exist, before
    any graph is captured."""
    if not megakernel_enabled() or not _offers_eagle_entries(
        draft_model=draft_worker.draft_runner.model
    ):
        return None
    reasons = _unservable_reasons(draft_worker)
    if reasons:
        raise ValueError("the EAGLE entries cannot run: " + "; ".join(reasons))
    return EagleMegakernel(draft_worker=draft_worker)


class EagleMegakernel:
    """SGLang's side of the entries, never handed to a plugin: the batch sizes whose
    captured graphs run each entry, their SGLang-owned outputs, and where each draft
    round's input came from."""

    def __init__(self, *, draft_worker: Any):
        self._draft_worker = draft_worker
        draft_runner = draft_worker.draft_runner
        self._draft_runner = draft_runner
        self._draft_config = draft_runner.model_config
        self._draft_loop = megakernel_entry(draft_runner.model, "draft_loop")
        self._draft_extend = megakernel_entry(draft_runner.model, "draft_extend")
        self._begin_draft_round = megakernel_entry(
            draft_runner.model, "begin_draft_round"
        )
        # runner id -> what bind_*_graphs set up: bs -> step, and the outputs
        self._draft_graphs: dict[int, SimpleNamespace] = {}
        self._extend_graphs: dict[int, SimpleNamespace] = {}
        self._draft_kv = None
        # draft inputs produced so far, and (that count, rids) of the last one a
        # served draft-extend replay produced
        self._input_seq = 0
        self._served_input: Optional[tuple] = None

    # ----- the EAGLE draft and draft-extend graphs -----

    def bind_draft_graphs(self, runner: Any) -> frozenset[int]:
        """Before the runner captures: calls the draft's draft_loop factory once with
        the runner's outputs, and returns the batch sizes whose graphs run it."""
        if self._draft_loop is None:
            return frozenset()
        draft_tokens = torch.zeros(
            (runner.max_bs, runner.speculative_num_steps),
            dtype=torch.int64,
            device=runner.device,
        )
        steps = self._draft_loop(
            num_steps=runner.speculative_num_steps,
            max_bs=runner.max_bs,
            draft_tokens=draft_tokens,
        )
        served = _served(steps, runner.capture_bs, what="draft_loop batch sizes")
        if served:
            self._draft_graphs[id(runner)] = SimpleNamespace(
                steps=served, draft_tokens=draft_tokens
            )
        return frozenset(served)

    def run_draft_loop(self, *, runner: Any, forward_batch: ForwardBatch) -> tuple:
        """Inside the draft graph's capture: draft_loop's step in place of
        draft_forward, with the same outputs."""
        bs = int(forward_batch.batch_size)
        graphs = self._draft_graphs[id(runner)]
        draft_tokens = graphs.draft_tokens[:bs]
        spec_info = forward_batch.spec_info
        graphs.steps[bs](
            EagleDraftLoopIO(
                num_requests=bs,
                num_steps=runner.speculative_num_steps,
                topk_index=spec_info.topk_index,
                hidden_states=spec_info.hidden_states,
                positions=forward_batch.positions,
                mrope_positions=(
                    forward_batch.mrope_positions
                    if self._draft_config.model_is_mrope
                    else None
                ),
                out_cache_loc=forward_batch.out_cache_loc,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                steps=tuple(
                    map(attn_step_metadata, runner.draft_attn_backend.attn_backends)
                ),
                kv=self._kv(),
                draft_tokens_out=draft_tokens,
            )
        )
        worker = self._draft_worker
        # a topk-1 chain's tree is constant: the worker's preallocated parents
        return (
            worker._topk1_parents_prealloc[:bs],
            worker._topk1_score_indices_prealloc[:bs],
            draft_tokens,
            None,
        )

    def bind_extend_graphs(self, runner: Any) -> frozenset[int]:
        """Before the runner captures: calls the draft's draft_extend factory once
        with the runner's outputs, and returns the batch sizes whose graphs run it."""
        if self._draft_extend is None:
            return frozenset()
        out = SimpleNamespace(
            topk_index=torch.zeros(
                (runner.max_bs, 1), dtype=torch.int64, device=runner.device
            ),
            topk_p=torch.ones(
                (runner.max_bs, 1), dtype=torch.float32, device=runner.device
            ),
            hidden=torch.zeros(
                (runner.max_bs, self._draft_config.hidden_size),
                dtype=self._draft_config.dtype,
                device=runner.device,
            ),
        )
        steps = self._draft_extend(
            tokens_per_request=runner.captured_req_width,
            max_bs=runner.max_bs,
            topk_index=out.topk_index,
            topk_p=out.topk_p,
            hidden=out.hidden,
        )
        served = _served(steps, runner.capture_bs, what="draft_extend batch sizes")
        if served:
            self._extend_graphs[id(runner)] = SimpleNamespace(steps=served, out=out)
        return frozenset(served)

    def run_draft_extend(self, *, runner: Any, forward_batch: ForwardBatch) -> Any:
        """Inside the draft-extend graph's capture: draft_extend's step in place of
        the model's forward; the logits stay inside the megakernel."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        bs = int(forward_batch.batch_size)
        graphs = self._extend_graphs[id(runner)]
        out = graphs.out
        spec_info = forward_batch.spec_info
        graphs.steps[bs](
            EagleDraftExtendIO(
                num_requests=bs,
                tokens_per_request=runner.captured_req_width,
                input_ids=forward_batch.input_ids,
                positions=forward_batch.positions,
                mrope_positions=(
                    forward_batch.mrope_positions
                    if self._draft_config.model_is_mrope
                    else None
                ),
                hidden_states=spec_info.hidden_states,
                out_cache_loc=forward_batch.out_cache_loc,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                select_index=spec_info.select_index,
                num_accept_tokens=spec_info.num_accept_tokens,
                attn=attn_step_metadata(get_attn_backend()),
                kv=self._kv(),
                topk_index_out=out.topk_index[:bs],
                topk_p_out=out.topk_p[:bs],
                hidden_out=out.hidden[:bs],
            )
        )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=out.hidden[:bs]
        )

    # ----- the draft worker's host calls -----

    def note_draft_extend(self) -> None:
        """A draft extend (after a prefill or a verify) produces the next draft input."""
        self._input_seq += 1

    def extend_tokens(
        self, *, runner: Any, batch: ScheduleBatch
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """After a draft-extend replay: (topk_p, topk_index) of a graph draft_extend
        ran, else None for SGLang's top-1."""
        graphs = self._extend_graphs.get(id(runner))
        if graphs is None or runner.bs not in graphs.steps:
            return None
        self._served_input = (self._input_seq, _rids(batch))
        raw_bs = runner.raw_bs
        # the next replay rewrites the buffers before the next round reads them
        return (
            graphs.out.topk_p[:raw_bs].clone(),
            graphs.out.topk_index[:raw_bs].clone(),
        )

    def begin_draft_round(
        self, *, runner: Any, batch: ScheduleBatch, draft_input: Any
    ) -> None:
        """Before a draft graph replays; tells a served batch size's draft_loop where
        its input came from."""
        from_served_extend = self._served_input == (self._input_seq, _rids(batch))
        self._served_input = None
        graphs = self._draft_graphs.get(id(runner))
        bs = len(batch.seq_lens)
        if (
            self._begin_draft_round is None
            or graphs is None
            or _bucket(bs, runner.capture_bs) not in graphs.steps
        ):
            return
        self._begin_draft_round(
            EagleDraftRoundIO(
                num_requests=bs,
                from_served_extend=from_served_extend,
                topk_index=draft_input.topk_index,
                hidden_states=draft_input.hidden_states,
            )
        )

    def _kv(self):
        # the draft's pool, allocated before any draft graph is captured
        if self._draft_kv is None:
            runner = self._draft_runner
            self._draft_kv = kv_pool_buffers(
                pool=runner.token_to_kv_pool,
                backend=split_backend(runner.attn_backend)[0],
                model=runner.model,
            )
        return self._draft_kv


def _served(
    steps: Callable[[int], Optional[Callable]], sizes: list[int], *, what: str
) -> dict[int, Callable]:
    """bs -> step of the sizes a factory's result serves, alike on every rank."""
    served = {bs: steps(bs) for bs in sorted(sizes)}
    served = {bs: step for bs, step in served.items() if step is not None}
    # a rank on SGLang's code at a size would wait in collectives the step skips
    megakernel_agree(tuple(served), what=what)
    return served


def _backend_reasons(*, target: Any, draft: tuple) -> list[str]:
    """Why these backends' step metadata cannot be handed over."""
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    reasons = []
    if (
        not isinstance(target, HybridLinearAttnBackend)
        or not isinstance(target.full_attn_backend, TRTLLMHAAttnBackend)
        or not isinstance(target.linear_attn_backend, GDNAttnBackend)
    ):
        reasons.append(
            "the target must run GDN and trtllm_mha under a hybrid linear-attention "
            f"backend, not {type(target).__name__}"
        )
    elif target.full_attn_backend.use_sliding_window_kv_pool:
        # every full-attention layer writes its KV at out_cache_loc
        reasons.append("the target's full attention must not use a sliding-window pool")
    for backend in draft:
        backend = split_backend(backend)[0]
        if not isinstance(backend, TRTLLMHAAttnBackend):
            reasons.append(
                f"the draft must run trtllm_mha, not {type(backend).__name__}"
            )
    return reasons


def _rids(batch: ScheduleBatch) -> tuple[str, ...]:
    return tuple(req.rid for req in batch.reqs)


def _bucket(bs: int, capture_bs: list[int]) -> int:
    """The captured batch size a batch of bs requests replays, without DP attention."""
    return min((size for size in capture_bs if size >= bs), default=-1)
