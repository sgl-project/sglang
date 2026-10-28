from __future__ import annotations

import logging
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    SamplingMaskStatus,
)
from sglang.srt.managers.auxiliary_output import CommittedTokens
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_MATCHED_TOKEN,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    get_required_capture_hidden_mode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_observability,
    get_schedule,
    mamba_track_grid,
)
from sglang.srt.sampling.sampling_params import (
    get_request_reasoning_end_token_ids,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.auxiliary_output import HostAuxiliaryOutput
    from sglang.srt.managers.scheduler_components.bundles import (
        KvResources,
        SchedulerServices,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.logprob_result_processor import (
        SchedulerLogprobResultProcessor,
    )
    from sglang.srt.managers.scheduler_components.pending_steps import (
        PendingStep,
        PendingSteps,
    )
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )
    from sglang.srt.managers.utils import (
        EmbeddingBatchResult,
        GenerationBatchResult,
    )
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

logger = logging.getLogger(__name__)


def _host_sum(lengths: Optional[torch.Tensor]) -> int:
    return int(lengths.sum()) if lengths is not None else 0


def _get_speculative_output_stride(result: GenerationBatchResult) -> int:
    """Return the padded per-request width in flattened speculative output."""
    stride = result.speculative_output_stride
    if stride is None:
        stride = result.speculative_num_draft_tokens
    if stride is None or stride < 1:
        raise RuntimeError("speculative result is missing a positive output row stride")
    return stride


def _accept_grammar_tokens(req: Req, tokens: Union[int, List[int]]) -> List[int]:
    """Advance the grammar over the accepted token(s), stopping at the token
    that terminates it.

    ``tokens`` is a single sampled token (normal decode) or the whole
    verified run (spec decode). Returns the retained prefix; for spec the
    suffix past grammar completion is dropped so it is never committed to KV
    nor emitted. Advances the grammar FSM only -- ``grammar.finished`` is
    synced by the caller once the finish state is updated.
    """
    if isinstance(tokens, int):
        tokens = [tokens]
    retained = []
    try:
        for token_id in tokens:
            req.grammar.accept_token(token_id)
            retained.append(token_id)
            if req.grammar.is_terminated():
                break
    except ValueError as e:
        # accept_token raises ValueError if the token is not in the grammar
        # (misconfigured grammar or invalid token); abort the request.
        logger.error(
            f"Grammar accept_token failed for req {req.rid} with token {tokens}: {e}"
        )
        req.to_finish = FINISH_ABORT()
    return retained


def advance_grammar_fsm(result: GenerationBatchResult, batch: ScheduleBatch) -> None:
    """The grammar commit of one step: advance each req's grammar FSM over
    the tokens the step produced, and (for a spec decode step) keep the
    grammar-truncated runs on ``result``.

    It runs once (``result.grammar_advanced``): inside the next step's verify
    (commit_pending_grammar, so the advance overlaps that forward),
    or in the step's own host commit. It handles both decode (the accepted
    spec run) and extend (the single prefilled token) steps, so the next
    step's verify can commit whatever step precedes it, e.g. at the
    extend->decode boundary.
    """
    if result.grammar_advanced or not batch.has_grammar:
        return
    is_decode = batch.forward_mode.is_decode()
    if not (is_decode or batch.forward_mode.is_extend()):
        return
    if result.copy_done is not None:
        result.copy_done.synchronize()
    next_token_ids = result.next_token_ids.tolist()

    if not is_decode:
        # Extend: advance over the single token each completed-prefill req emitted
        # (mirrors process_batch_result_prefill's per-req token indexing).
        for i, req in enumerate(batch.reqs):
            if (
                req.grammar is None
                or req.is_retracted
                or req.finished()
                or req is batch.chunked_req
            ):
                continue
            _accept_grammar_tokens(req, next_token_ids[i])
        result.grammar_advanced = True
        return

    # Decode: only the spec-v2 path reaches here (the next step's verify and
    # _resolve_spec_v2_tokens). Non-spec grammar decode
    # advances its FSM in process_batch_result_decode and has no accept_lens, so
    # bail out defensively.
    if result.accept_lens is None:
        return
    accept_lens = result.accept_lens.tolist()
    stride = _get_speculative_output_stride(result)
    retained = [None] * len(batch.reqs)
    for i, req in enumerate(batch.reqs):
        if req.grammar is None or req.is_retracted or req.finished():
            continue
        accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]
        # Stop accepting once the grammar terminates so the over-drafted suffix
        # is never committed to KV nor emitted; this advances the FSM.
        retained[i] = _accept_grammar_tokens(req, accept_tokens)
    result.grammar_retained_tokens = retained
    result.grammar_advanced = True


def commit_pending_grammar(pending: Iterable[PendingStep]) -> None:
    """The grammar commit of each pending step: its tokens advance its
    requests' grammar FSMs, once (the step's host commit then skips it). A
    speculative verify runs it right before building the next step's vocab
    mask, so it overlaps the verify forward."""
    for record in pending:
        advance_grammar_fsm(record.result, record.step)


def maybe_update_reasoning_tokens(
    model_config: ModelConfig,
    req: Req,
    next_token_id: Union[int, List[int]],
):
    if not req.require_reasoning:
        return
    think_end_ids = model_config.think_end_ids
    if req._think_end_matcher is None:
        request_think_end_ids = get_request_reasoning_end_token_ids(
            req.sampling_params.custom_params,
            allowed_sequences=getattr(
                model_config,
                "request_selectable_think_end_id_sequences",
                None,
            ),
        )
        if request_think_end_ids is not None:
            think_end_ids = request_think_end_ids
        if not think_end_ids:
            return
    req.update_reasoning_tokens(next_token_id, think_end_ids)


@dataclass(kw_only=True, slots=True)
class ExtendCommit:
    """One extend step's host commit as its rows read it: the step, its
    result in host values, and the cursor into the step's flattened input
    logprobs, which each row advances past its own."""

    batch: ScheduleBatch
    result: GenerationBatchResult
    logits_output: Optional[LogitsProcessorOutput]
    next_token_ids: List[int]
    logprob_pt: int = 0


class ExtendOutcome(Protocol):
    """What an extend commit's rows mean in a mode."""

    def conclude(
        self, commit: ExtendCommit, i: int, req: Req, next_token_id: int
    ) -> None:
        """A row whose last launched prefill chunk this step ran."""

    def continue_chunk(self, commit: ExtendCommit, i: int, req: Req) -> None:
        """The row of the request still chunking."""

    def finish(self, commit: ExtendCommit) -> Optional[List[Req]]:
        """After the rows: stream, and return the requests that go back to
        the front of the waiting list."""


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerBatchResultProcessor:
    is_generation: bool
    model_config: ModelConfig
    kv: KvResources
    services: SchedulerServices
    decode_offload_manager: Optional[DecodeKVCacheOffloadManager]
    metrics_collector: SchedulerMetricsCollector
    workers: Workers
    logprob_result_processor: SchedulerLogprobResultProcessor
    beam_coordinator: BeamCoordinator
    pending: PendingSteps
    request_retirer: RequestRetirer
    enable_overlap: bool = field(
        init=False, default_factory=lambda: not get_schedule().disable_overlap_schedule
    )

    def _maybe_collect_routed_experts(self, req: Req):
        """Collect routed experts for a finished request.

        Returns immediately if `return_routed_experts` was not set on the
        request, so non-opted-in reqs don't pay the host-gather cost.

        Honors the caller's absolute start so the response covers
        `[start_len, seqlen - 1)`. The default start_len is 0, which returns
        the full sequence.

        Logs a soft warning if the resulting tensor's row count differs from
        the expected `seqlen - 1 - start_len`, to catch silent regressions.
        """
        if not req.return_routed_experts:
            return
        capturer = get_global_experts_capturer()
        if capturer is None:
            return
        start_len = req.routed_experts_start_len
        seqlen = len(req.origin_input_ids) + len(req.output_ids_through_stop)
        req.routed_experts = capturer.get_topk(
            req_pool_idx=req.kv.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.kv.req_to_token_pool,
            start_len=start_len,
        )

        expected_rows = max(0, seqlen - 1 - start_len)
        if (
            req.routed_experts is not None
            and req.routed_experts.shape[0] != expected_rows
        ):
            logger.warning(
                "routed_experts row-count mismatch for req %s: got %d, expected %d "
                "(seqlen=%d, raw_seqlen=%d, cached_tokens=%d, start_len=%s). "
                "This indicates a silent bug.",
                req.rid,
                req.routed_experts.shape[0],
                expected_rows,
                seqlen,
                req.seqlen,
                req.cached_tokens,
                req.routed_experts_start_len,
            )

    def _maybe_collect_indexer_topk(self, req: Req):
        capturer = get_global_indexer_capturer()
        if capturer is None:
            return
        seqlen = len(req.origin_input_ids) + len(req.output_ids_through_stop)
        req.indexer_topk = capturer.get_topk(
            req_pool_idx=req.kv.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.kv.req_to_token_pool,
        )

    def _maybe_collect_customized_info(
        self,
        i: int,
        req: Req,
        logits_output: LogitsProcessorOutput,
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                # Copy the element so it doesn't retain the entire batch
                # tensor/array via a view reference.
                elem = v[i]
                if isinstance(elem, torch.Tensor):
                    elem = elem.clone()
                elif hasattr(elem, "copy") and callable(elem.copy):
                    elem = elem.copy()
                req.customized_info[k].append(elem)

    @staticmethod
    def _visible_output_len(req: Req) -> int:
        return req.finished_len if req.finished_len is not None else len(req.output_ids)

    @classmethod
    def snapshot_auxiliary_output_starts(
        cls,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> Optional[List[int]]:
        if result.auxiliary_host_output is None:
            return None
        return [cls._visible_output_len(req) for req in batch.reqs]

    @classmethod
    def _build_auxiliary_commits(
        cls,
        batch: ScheduleBatch,
        output_starts: List[int],
    ) -> List[Optional[CommittedTokens]]:
        commits: List[Optional[CommittedTokens]] = []
        for req, output_start in zip(batch.reqs, output_starts, strict=True):
            output_end = cls._visible_output_len(req)
            if output_end < output_start:
                raise RuntimeError("committed output length moved backwards")
            if output_end == output_start:
                commits.append(None)
                continue
            commits.append(
                CommittedTokens(
                    output_index=output_start,
                    token_ids=tuple(req.output_ids[output_start:output_end]),
                )
            )
        return commits

    @classmethod
    def consume_auxiliary_output(
        cls,
        batch: ScheduleBatch,
        output: HostAuxiliaryOutput,
        output_starts: List[int],
    ) -> None:
        output.consume(batch, cls._build_auxiliary_commits(batch, output_starts))

    def host_prefill_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> Tuple[Optional[LogitsProcessorOutput], List[int], Optional[List[int]]]:
        """Wait for a prefill step's host copies and turn its result into host
        values: the logits output's host copy, the next token ids, and the
        auxiliary output starts."""
        if result.copy_done is not None:
            result.copy_done.synchronize()
        auxiliary_output_starts = self.snapshot_auxiliary_output_starts(batch, result)
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
        logits_output = (
            result.logits_output.host_copy()
            if result.logits_output is not None
            else None
        )
        next_token_ids = result.next_token_ids.tolist()
        self.move_logprobs_to_cpu(batch=batch, logits_output=logits_output)
        self.materialize_sampling_mask_output(batch.reqs, logits_output)
        return logits_output, next_token_ids, auxiliary_output_starts

    def commit_extend(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        outcome: ExtendOutcome,
    ) -> Optional[List[Req]]:
        """The host commit of a generation extend step, row by row: the row
        of the request still chunking continues its chunk, every other row
        concludes. The outcome says what each means in the mode, streams at
        the end and returns the requests that go back to the front of the
        waiting list."""
        self.kv.token_to_kv_pool_allocator.free_group_begin()
        logits_output, next_token_ids, auxiliary_output_starts = (
            self.host_prefill_result(batch, result)
        )
        self._validate_pp_skip_output_comm(batch, result)
        commit = ExtendCommit(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req is batch.chunked_req:
                outcome.continue_chunk(commit, i, req)
            else:
                outcome.conclude(commit, i, req, next_token_id)
        if result.auxiliary_host_output is not None:
            self.consume_auxiliary_output(
                batch,
                result.auxiliary_host_output,
                auxiliary_output_starts,
            )
        self.kv.token_to_kv_pool_allocator.free_group_end()
        return outcome.finish(commit)

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        on_checkpoint: Optional[Callable[[Req], None]] = None,
    ):
        """on_checkpoint takes each concluded request whose KV the commit
        checkpointed (HiSparse stages it until its host copy lands)."""
        if self.is_generation:
            return self.commit_extend(
                batch, result, _PrefillOutcome(self, batch, on_checkpoint)
            )

        # An embedding or reward model.
        self.kv.token_to_kv_pool_allocator.free_group_begin()
        if result.copy_done is not None:
            result.copy_done.synchronize()

        embeddings = self._convert_embeddings(result=result)
        phs = result.pooled_hidden_states

        if phs is not None:
            if isinstance(phs, list):
                phs = [t.cpu().detach() for t in phs]
            else:
                phs = phs.cpu().detach()

        # Check finish conditions
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue

            if req.discard_output_reason is not None and req is not batch.chunked_req:
                req.to_finish = req.discard_output_reason
                req.update_finish_state(0)
                self.request_retirer.retire_finished(req, checkpoint=False)
                req.time_stats.set_completion_time()
                continue

            req.embedding = embeddings[i]
            if req.return_pooled_hidden_states and phs is not None:
                req.pooled_hidden_state = phs[i]
            if req is not batch.chunked_req:
                req.time_stats.set_prefill_finished_time()
                # Dummy output token for embedding models
                req.output_ids.append(0)
                req.update_finish_state()

                if req.finished():
                    self.request_retirer.retire_finished(req, checkpoint=True)
                    req.time_stats.set_completion_time()
                else:
                    checkpoint_kv_cache(req, self.kv.tree_cache)
            else:
                # being chunked reqs' prefill is not finished
                req.time_stats.set_last_chunked_prefill_finish_time()

        self.kv.token_to_kv_pool_allocator.free_group_end()
        self.services.output_streamer.stream_output(
            batch.reqs, batch.return_logprob, None
        )

    def _convert_embeddings(self, *, result: EmbeddingBatchResult) -> list:
        is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

        embeddings = result.embeddings

        if is_sparse:
            batch_ids, token_ids = embeddings.indices()
            values = embeddings.values()

            embeddings = [{} for _ in range(embeddings.size(0))]
            for i in range(batch_ids.shape[0]):
                embeddings[batch_ids[i].item()][token_ids[i].item()] = values[i].item()
        else:
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.tolist()
            else:
                embeddings = [tensor.tolist() for tensor in embeddings]
        return embeddings

    def move_logprobs_to_cpu(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        """Turn the logprobs of a commit's host copy of the logits output
        (`LogitsProcessorOutput.host_copy`) into host values in place."""
        if batch.return_logprob:
            logits_output.finalize_input_logprobs()
            if logits_output.next_token_logprobs is not None:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.tolist()
                )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
            if logits_output.next_token_top_logprobs_val:
                logits_output.next_token_top_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_top_logprobs_val
                ]
                logits_output.next_token_top_logprobs_idx = [
                    x.tolist() for x in logits_output.next_token_top_logprobs_idx
                ]
            if logits_output.next_token_token_ids_logprobs_val:
                logits_output.next_token_token_ids_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_token_ids_logprobs_val
                ]

    def apply_prefill_logprobs(
        self, commit: ExtendCommit, i: int, req: Req, *, store: bool = True
    ) -> None:
        """Advance the logprob cursor past a concluded row, optionally storing
        this request's values."""
        extend_logprob_start_len_per_req = commit.batch.extend_logprob_start_lens
        extend_input_len_per_req = commit.batch.extend_lens
        assert extend_logprob_start_len_per_req is not None
        assert extend_input_len_per_req is not None
        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        extend_input_len = extend_input_len_per_req[i]

        num_input_logprobs = self.logprob_result_processor.calculate_num_input_logprobs(
            req,
            extend_input_len,
            extend_logprob_start_len,
        )

        if store and req.return_logprob:
            self.logprob_result_processor.add_logprob_return_values(
                i,
                req,
                commit.logprob_pt,
                commit.next_token_ids,
                num_input_logprobs,
                commit.logits_output,
            )
        commit.logprob_pt += num_input_logprobs

    @staticmethod
    def _validate_pp_skip_output_comm(
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        """Validate PP skip output comm correctness.

        - When skip=True: all reqs must be middle chunks (batch.chunked_req)
          so placeholder zeros are never consumed via req.output_ids.append().
        - When skip=False: at least one req should consume next_token_ids
          (not a middle chunk), otherwise warn.
        """
        if not envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get():
            return

        if not getattr(result, "skipped_output_comm", False):
            if batch.forward_mode.is_extend():
                has_consumed_output = any(
                    req is not batch.chunked_req
                    for req in batch.reqs
                    if not req.finished() and not req.is_retracted
                )
                if not has_consumed_output and len(batch.reqs) > 0:
                    logger.warning(
                        f"PP non-skip output comm: no req consumed next_token_ids. "
                        f"contains_last_prefill_chunk={batch.contains_last_prefill_chunk}, "
                        f"num_reqs={len(batch.reqs)}, all of them middle chunks"
                    )
            return

        for req in batch.reqs:
            if not req.finished() and not req.is_retracted:
                assert req is batch.chunked_req, (
                    f"PP skip output comm invariant violated: req {req.rid} "
                    f"is not a middle chunk "
                    f"but output was skipped (contains_last_prefill_chunk="
                    f"{batch.contains_last_prefill_chunk}). "
                    f"Placeholder zeros would be appended to output_ids."
                )

    def _append_prefill_hidden_states(
        self,
        *,
        req: Req,
        logits_output: LogitsProcessorOutput,
        hidden_state_offset: int,
        capture_hidden_mode: CaptureHiddenMode,
        extend_input_len: int,
        store: bool = True,
    ) -> int:
        if capture_hidden_mode.is_full():
            start = hidden_state_offset
            hidden_state_offset += extend_input_len
            if not store or not req.return_hidden_states:
                return hidden_state_offset

            req_hidden_states = logits_output.hidden_states[start:hidden_state_offset]
            if req.return_hidden_states is True:
                req.hidden_states.append(req_hidden_states.cpu().clone().tolist())
            elif req.return_hidden_states == "last":
                req.hidden_states.append(req_hidden_states[-1].cpu().tolist())
        elif capture_hidden_mode.is_last():
            index = hidden_state_offset
            hidden_state_offset += 1
            if store and req.return_hidden_states:
                req.hidden_states.append(
                    logits_output.hidden_states[index].cpu().tolist()
                )
        else:
            raise ValueError(
                f"Unexpected hidden states capture mode: {capture_hidden_mode}"
            )
        return hidden_state_offset

    @staticmethod
    def _append_decode_hidden_states(
        *,
        req: Req,
        hidden_states: torch.Tensor,
        start: int,
        accept_len: int,
    ) -> None:
        if accept_len <= 0:
            return

        if req.return_hidden_states == "last":
            valid_accept_len = accept_len
            if req.finished_len is not None:
                step_start = len(req.output_ids) - accept_len
                valid_accept_len = max(
                    0,
                    min(accept_len, req.finished_len - step_start),
                )
            if valid_accept_len > 0:
                req.hidden_states[:] = [
                    hidden_states[start + valid_accept_len - 1].cpu().tolist()
                ]
        else:
            req.hidden_states.extend(
                hidden_states[start : start + accept_len].cpu().tolist()
            )

    @staticmethod
    def _get_prefill_hidden_capture_mode(batch: ScheduleBatch) -> CaptureHiddenMode:
        return get_required_capture_hidden_mode(
            max(
                batch.return_hidden_states_mode,
                get_server_return_hidden_states_mode(),
            ),
            batch.spec_info,
        )

    def apply_prefill_grammar(
        self,
        commit: ExtendCommit,
        req: Req,
        next_token_id: int,
        abort: Callable[[Req, str], None],
    ) -> None:
        """Advance a concluded row's grammar over its token, then sync
        grammar.finished. A token the grammar rejects aborts the request with
        the error (status 500), the way the mode aborts it."""
        # The step's grammar commit may have advanced the FSM over this
        # prefilled token already (inside the next spec step's verify); only
        # advance if not, but always sync grammar.finished.
        if not commit.result.grammar_advanced:
            # FIXME: this try-except block is for handling unexpected xgrammar issue.
            try:
                req.grammar.accept_token(next_token_id)
            except ValueError as e:
                # Grammar accept_token can raise ValueError if the token is not in the
                # grammar. This can happen if the grammar is not set correctly or the
                # token is invalid.
                error = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                logger.error(error)
                abort(req, error)
        req.grammar.finished = req.finished()

    def apply_chunked_prefill_logprobs(
        self, commit: ExtendCommit, i: int, req: Req, *, store: bool = True
    ) -> None:
        """Advance the logprob cursor past a continuing chunk's row,
        optionally storing its input logprobs."""
        extend_logprob_start_len = commit.batch.extend_logprob_start_lens[i]
        extend_input_len = commit.batch.extend_lens[i]
        if extend_logprob_start_len < extend_input_len:
            # Update input logprobs.
            num_input_logprobs = (
                self.logprob_result_processor.calculate_num_input_logprobs(
                    req,
                    extend_input_len,
                    extend_logprob_start_len,
                )
            )
            if store and req.return_logprob:
                self.logprob_result_processor.add_input_logprob_return_values(
                    i,
                    req,
                    commit.logits_output,
                    commit.logprob_pt,
                    num_input_logprobs,
                    last_prefill_chunk=False,
                )
            commit.logprob_pt += num_input_logprobs

    def _resolve_spec_v2_tokens(
        self,
        result: GenerationBatchResult,
        batch: ScheduleBatch,
    ) -> List[List[int]]:
        """Resolve the padded next token ids for spec-v2 (overlap and non-overlap)."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        self.services.metrics_sink.snapshot_spec_decode_metrics(batch, result)

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        stride = _get_speculative_output_stride(result)
        num_correct_drafts_per_req = result.num_correct_drafts_per_req()

        block_accept_lens = (
            result.block_accept_lens.tolist()
            if result.block_accept_lens is not None
            else None
        )
        cap_lens = result.cap_lens.tolist() if result.cap_lens is not None else None

        # Feed the adaptive controller now that accept_lens is on CPU,
        # instead of doing a synchronous GPU→CPU copy in the worker hot path.
        # BaseSpecWorker provides a no-op default for non-adaptive workers.
        self.workers.model_worker.on_verify_complete_cpu(
            num_correct_drafts_per_req, batch_size=len(batch.reqs)
        )

        # The step's grammar commit (idempotent): under overlap the next step's
        # verify may have run it already; otherwise it runs now. The retained
        # runs it left in result.grammar_retained_tokens are used below.
        advance_grammar_fsm(result, batch)

        predict_tokens = []
        for i, req in enumerate(batch.reqs):
            accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]

            if req.is_retracted or req.finished():
                # Nothing to settle: no worker pre-claims the bonus, so
                # kv_committed_len already holds the committed prefix.
                pass
            else:
                if req.grammar is not None:
                    # FSM already advanced + truncated by advance_grammar_fsm; reuse
                    # the retained (grammar-legal) run instead of advancing again.
                    accept_tokens = result.grammar_retained_tokens[i]

                # Commit the full accepted run (drafts + bonus).
                num_accept_tokens = len(accept_tokens)
                req.kv.commit_decode(num_accept_tokens)
                req.spec_stats.verify_ct += 1

                num_correct_drafts = num_correct_drafts_per_req[i]
                req.spec_stats.num_correct_drafts += num_correct_drafts
                req.spec_stats.update_correct_drafts_histogram(num_correct_drafts)

                if block_accept_lens is not None:
                    req.spec_stats.num_block_accept_tokens += block_accept_lens[i]
                if cap_lens is not None:
                    req.spec_stats.num_cap_tokens += cap_lens[i]
                    req.spec_stats.update_cap_lens_histogram(cap_lens[i])

            predict_tokens.append(accept_tokens)

        return predict_tokens

    def process_batch_result_idle(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.services.output_streamer._stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()
        auxiliary_output_starts = self.snapshot_auxiliary_output_starts(batch, result)
        auxiliary_output = result.auxiliary_host_output
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()

        logits_output, next_token_ids, can_run_cuda_graph = (
            (
                result.logits_output.host_copy()
                if result.logits_output is not None
                else None
            ),
            result.next_token_ids,
            result.can_run_cuda_graph,
        )
        self.materialize_sampling_mask_output(batch.reqs, logits_output)

        next_token_ids, next_token_logprobs = self._normalize_decode_outputs(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )

        batch_size = batch.batch_size()
        num_generated_tokens = result.get_num_generated_tokens(batch_size)
        self.services.metrics_sink.num_generated_tokens += num_generated_tokens
        if not batch.spec_algorithm.is_none():
            self.services.metrics_sink.update_spec_metrics(
                batch_size,
                num_generated_tokens - batch_size * result.num_non_draft_tokens_per_req,
                num_accept_tokens=num_generated_tokens,
                num_block_accept_tokens=_host_sum(result.block_accept_lens),
                num_cap_tokens=_host_sum(result.cap_lens),
            )
        if get_observability().enable_metrics:
            self.metrics_collector.increment_decode_cuda_graph_pass(
                value=can_run_cuda_graph
            )

        self.kv.token_to_kv_pool_allocator.free_group_begin()

        # Folds the relay point's selection into the DAG and sets the finish
        # states the loop below observes. Beam + spec is rejected at admission.
        newly_finished_beam_groups = set()
        if batch.spec_algorithm.is_none() and logits_output is not None:
            newly_finished_beam_groups = self.beam_coordinator.commit_decode(batch)

        for i, req in enumerate(batch.reqs):
            req: Req

            if req.beam_group is not None:
                # Under overlap a finished row reappears for one overshoot tick;
                # gate on the committing tick so this runs exactly once.
                if req.finished() and (
                    id(req.beam_group) not in newly_finished_beam_groups
                ):
                    continue
                req.time_stats.set_last_decode_finish_time()
                self._handle_finish_state_updated_req(
                    req, batch, result, i, logits_output
                )
                continue

            if self.enable_overlap and (req.finished() or req.is_retracted):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                continue

            # next_token_id is a per-req list: 1 token for non-spec, the verified
            # run for spec (already grammar-truncated in _resolve_spec_v2_tokens).
            next_token_id = next_token_ids[i]
            is_spec = not batch.spec_algorithm.is_none()

            req.time_stats.set_last_decode_finish_time()
            sampling_mask_finish_reason = None
            if req.return_sampling_mask:
                statuses = logits_output.next_token_sampling_mask_status
                status = None if statuses is None else statuses[i]
                sampling_mask_finish_reason = self.get_sampling_mask_finish_reason(
                    status=status
                )
            if sampling_mask_finish_reason is not None:
                req.to_finish = sampling_mask_finish_reason
                new_accept_len = 0
            else:
                req.output_ids.extend(next_token_id)
                new_accept_len = len(next_token_id)
                maybe_update_reasoning_tokens(self.model_config, req, next_token_id)
            req.update_finish_state(new_accept_len)

            if sampling_mask_finish_reason is not None:
                self._handle_sampling_mask_abort(req)
                continue

            self._handle_finish_state_updated_req(req, batch, result, i, logits_output)

            if req.return_logprob:
                self._apply_decode_logprobs(
                    req=req,
                    i=i,
                    batch=batch,
                    next_token_id=next_token_id,
                    next_token_logprobs=next_token_logprobs,
                    logits_output=logits_output,
                )

            if req.return_sampling_mask:
                self.add_sampling_mask_return_values(
                    i, req, logits_output, accept_len=new_accept_len
                )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                # hidden_states is [bs * stride, hidden_dim], one row per emitted
                # token; speculative workers record their padded row width.
                stride = _get_speculative_output_stride(result) if is_spec else 1
                accept_len = len(next_token_id)
                start = i * stride
                self._append_decode_hidden_states(
                    req=req,
                    hidden_states=logits_output.hidden_states,
                    start=start,
                    accept_len=accept_len,
                )

            if req.grammar is not None:
                if not is_spec:
                    # Normal decode advances the grammar for its single token
                    # here; spec already advanced it in _resolve_spec_v2_tokens.
                    _accept_grammar_tokens(req, next_token_id)
                req.grammar.finished = req.finished()

        if auxiliary_output is not None:
            self.consume_auxiliary_output(
                batch,
                auxiliary_output,
                auxiliary_output_starts,
            )

        self.services.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        self.kv.token_to_kv_pool_allocator.free_group_end()

        self.services.metrics_sink.forward_ct_decode = (
            self.services.metrics_sink.forward_ct_decode + 1
        ) % (1 << 30)
        return num_generated_tokens

    def _normalize_decode_outputs(
        self,
        *,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        logits_output: LogitsProcessorOutput,
        next_token_ids: Union[torch.Tensor, List[int]],
    ) -> Tuple[Union[List[int], List[List[int]]], Optional[List[float]]]:
        next_token_logprobs = None
        # Normalize to a uniform per-req list of accepted tokens (List[List[int]]):
        # spec unpacks the padded verify output; non-spec wraps its single token.
        if not batch.spec_algorithm.is_none():
            next_token_ids = self._resolve_spec_v2_tokens(result, batch)
        else:
            ids = (
                next_token_ids.tolist()
                if torch.is_tensor(next_token_ids)
                else next_token_ids
            )
            next_token_ids = [[t] for t in ids]

        if batch.return_logprob:
            next_token_logprobs = logits_output.next_token_logprobs.tolist()
            if logits_output.next_token_top_logprobs_val:
                logits_output.next_token_top_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_top_logprobs_val
                ]
                logits_output.next_token_top_logprobs_idx = [
                    x.tolist() for x in logits_output.next_token_top_logprobs_idx
                ]

            if logits_output.next_token_token_ids_logprobs_val:
                logits_output.next_token_token_ids_logprobs_val = [
                    v.tolist() for v in logits_output.next_token_token_ids_logprobs_val
                ]
        return next_token_ids, next_token_logprobs

    def _apply_decode_logprobs(
        self,
        *,
        req: Req,
        i: int,
        batch: ScheduleBatch,
        next_token_id: Union[int, List[int]],
        next_token_logprobs: list,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        # accepted_ids is already a per-req list; non-spec logprobs are flat, so
        # the scalar logprob still needs wrapping.
        if not batch.spec_algorithm.is_none():
            accepted_logprobs = next_token_logprobs[i]
            accepted_ids = next_token_id
            max_accept = len(accepted_logprobs)
        else:
            accepted_logprobs = [next_token_logprobs[i]]
            accepted_ids = next_token_id
            max_accept = 1

        for j, tok_id in enumerate(accepted_ids):
            req.logprob.output_token_logprobs_val.append(accepted_logprobs[j])
            req.logprob.output_token_logprobs_idx.append(tok_id)
            if req.logprob.top_logprobs_num > 0:
                flat_idx = i * max_accept + j
                req.logprob.output_top_logprobs_val.append(
                    logits_output.next_token_top_logprobs_val[flat_idx]
                )
                req.logprob.output_top_logprobs_idx.append(
                    logits_output.next_token_top_logprobs_idx[flat_idx]
                )
            if req.logprob.token_ids_logprob is not None:
                flat_idx = i * max_accept + j
                req.logprob.output_token_ids_logprobs_val.append(
                    logits_output.next_token_token_ids_logprobs_val[flat_idx]
                )
                req.logprob.output_token_ids_logprobs_idx.append(
                    logits_output.next_token_token_ids_logprobs_idx[flat_idx]
                )

    def add_sampling_mask_return_values(
        self,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        *,
        accept_len: int = 1,
    ) -> None:
        """Attach sparse sampling support metadata to the return values."""
        masks = output.next_token_sampling_mask_idx[i]
        logprobs = output.next_token_sampling_logprobs[i]
        if not isinstance(masks, list):
            req.sampling_mask_rows.append(masks, logprobs)
            return
        # Speculative rows carry one support per accepted token; drop the ones
        # past finished_len, as output_ids is trimmed there.
        step_start = len(req.output_ids) - accept_len
        num_visible = max(
            0, min(accept_len, self._visible_output_len(req) - step_start)
        )
        for token in range(num_visible):
            req.sampling_mask_rows.append(masks[token], logprobs[token])

    @staticmethod
    def _fill_speculative_sampling_mask_rows(
        *,
        reqs: List[Req],
        batch_indices: List[int],
        num_accept_tokens: List[int],
        lengths: List[List[int]],
        statuses: List[List[int]],
        token_ids: np.ndarray,
        selected_logprobs: np.ndarray,
        support_logprobs: Optional[np.ndarray],
        masks: list,
        logprobs: list,
        status_by_batch: list,
    ) -> None:
        """Speculative rows carry one support per draft position; keep the
        accepted prefix, as a list with one entry per accepted token."""
        assert len(batch_indices) == len(num_accept_tokens)
        support_row = 0
        for row, batch_index in enumerate(batch_indices):
            returns_support_logprobs = (
                reqs[batch_index].sampling_logprobs_mode == "support"
            )
            num_tokens = num_accept_tokens[row]
            status = max(statuses[row][:num_tokens])
            status_by_batch[batch_index] = status
            if status == SamplingMaskStatus.OK:
                row_lengths = lengths[row][:num_tokens]
                masks[batch_index] = [
                    token_ids[row, token, :length]
                    for token, length in enumerate(row_lengths)
                ]
                if returns_support_logprobs:
                    logprobs[batch_index] = [
                        support_logprobs[support_row, token, :length]
                        for token, length in enumerate(row_lengths)
                    ]
                else:
                    logprobs[batch_index] = [
                        selected_logprobs[row, token : token + 1]
                        for token in range(num_tokens)
                    ]
            if returns_support_logprobs:
                support_row += 1

    @staticmethod
    def materialize_sampling_mask_output(
        reqs: List[Req],
        output: Optional[LogitsProcessorOutput],
    ) -> None:
        """Convert opted-in tensor rows to batch-aligned host rows."""
        if output is None or output.sampling_mask_output is None:
            return

        sampling_output = output.sampling_mask_output
        batch_indices = [i for i, req in enumerate(reqs) if req.return_sampling_mask]
        lengths = sampling_output.lengths.tolist()
        statuses = sampling_output.statuses.tolist()
        assert len(batch_indices) == len(lengths)

        batch_size = len(reqs)
        masks = [None] * batch_size
        logprobs = [None] * batch_size
        status_by_batch = [None] * batch_size
        token_ids = sampling_output.token_ids.cpu().numpy()
        selected_logprobs = sampling_output.selected_logprobs.cpu().numpy()
        support_logprobs = (
            None
            if sampling_output.support_logprobs is None
            else sampling_output.support_logprobs.cpu().numpy()
        )
        if sampling_output.num_accept_tokens is not None:
            SchedulerBatchResultProcessor._fill_speculative_sampling_mask_rows(
                reqs=reqs,
                batch_indices=batch_indices,
                num_accept_tokens=sampling_output.num_accept_tokens.tolist(),
                lengths=lengths,
                statuses=statuses,
                token_ids=token_ids,
                selected_logprobs=selected_logprobs,
                support_logprobs=support_logprobs,
                masks=masks,
                logprobs=logprobs,
                status_by_batch=status_by_batch,
            )
            output.next_token_sampling_mask_idx = masks
            output.next_token_sampling_logprobs = logprobs
            output.next_token_sampling_mask_status = status_by_batch
            output.sampling_mask_output = None
            return

        support_row = 0
        for row, batch_index in enumerate(batch_indices):
            returns_support_logprobs = (
                reqs[batch_index].sampling_logprobs_mode == "support"
            )
            status = int(statuses[row])
            length = int(lengths[row])
            status_by_batch[batch_index] = status
            if status == SamplingMaskStatus.OK:
                masks[batch_index] = token_ids[row, :length]
                if returns_support_logprobs:
                    logprobs[batch_index] = support_logprobs[support_row, :length]
                else:
                    logprobs[batch_index] = selected_logprobs[row : row + 1]
            if returns_support_logprobs:
                support_row += 1

        output.next_token_sampling_mask_idx = masks
        output.next_token_sampling_logprobs = logprobs
        output.next_token_sampling_mask_status = status_by_batch
        output.sampling_mask_output = None

    def sampling_mask_finish_reason(
        self, commit: ExtendCommit, i: int
    ) -> Optional[FINISH_ABORT]:
        """The finish reason an unusable sampling mask gives row i, or None."""
        assert commit.logits_output is not None
        statuses = commit.logits_output.next_token_sampling_mask_status
        status = None if statuses is None else statuses[i]
        return self.get_sampling_mask_finish_reason(status=status)

    def get_sampling_mask_finish_reason(
        self,
        *,
        status: Optional[int],
    ) -> Optional[FINISH_ABORT]:
        if status == SamplingMaskStatus.OK:
            return None
        if status == SamplingMaskStatus.OVERFLOW:
            return FINISH_ABORT(
                "Sampling support exceeds --sampling-mask-max-tokens="
                f"{get_exec().features.sampling_mask_max_tokens}, commonly because "
                "of cutoff ties. Lower top_k to leave more headroom or increase "
                "the server limit.",
                HTTPStatus.BAD_REQUEST,
                "BadRequestError",
            )
        if status is None:
            return FINISH_ABORT(
                "The sampling backend did not return captured sampling support.",
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "InternalServerError",
            )
        assert status == SamplingMaskStatus.INVALID
        return FINISH_ABORT(
            "The sampling backend selected a token outside its captured sampling "
            "support.",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "InternalServerError",
        )

    def _handle_sampling_mask_abort(self, req: Req) -> None:
        """Release a request whose sampled token must not be committed."""
        self.request_retirer.release_finished_features(req)
        if get_disagg().disaggregation_decode_enable_offload_kvcache:
            self.request_retirer.retire_offloaded(req)
        else:
            self.request_retirer.retire_finished(req, checkpoint=False)
        req.time_stats.set_completion_time()

    def _handle_finish_state_updated_req(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
        logits_output: LogitsProcessorOutput,
    ):
        # Called here (after update_finish_state) so req.finished() is valid
        # for _mamba_lazy_spec_update inside.
        self._mamba_prefix_cache_update(req, batch, result, i)

        if (
            get_disagg().disaggregation_decode_enable_offload_kvcache
            and not req.finished()
        ):
            self.decode_offload_manager.offload_kv_cache(req)

        if req.finished():
            if not self._mamba_can_cache_finished_req(req, result, i):
                req.skip_radix_cache_insert = True

            # isinstance narrowing: create_worker may also return plain
            # TpModelWorker-based drafts, which carry no spec-worker hooks.
            if isinstance(self.workers.draft_worker, BaseSpecWorker):
                self.workers.draft_worker.note_request_finished(
                    rid=req.rid,
                    natural_stop=isinstance(req.finished_reason, FINISH_MATCHED_TOKEN),
                )

            # delete feature to save memory
            self.request_retirer.release_finished_features(req)
            self._maybe_collect_routed_experts(req)
            self._maybe_collect_indexer_topk(req)

            if get_disagg().disaggregation_decode_enable_offload_kvcache:
                # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                if not self.decode_offload_manager.offload_kv_cache(req):
                    self.request_retirer.retire_offloaded(req)
            else:
                self.request_retirer.retire_finished(
                    req,
                    checkpoint=not self._mamba_lazy_keep_written(req, batch, result, i),
                )

            req.time_stats.set_completion_time()

        self._maybe_collect_customized_info(i, req, logits_output)

    def _mamba_can_cache_finished_req(
        self, req: Req, result: GenerationBatchResult, i: int
    ) -> bool:
        num_correct_drafts = result.num_correct_drafts_per_req()
        if (
            num_correct_drafts is None
            or req.kv.mamba_ping_pong_track_buffer is not None
            or not self.kv.tree_cache.supports_mamba()
        ):
            return True

        # no_buffer donates the live state, which cannot be shortened with the key.
        if req.owned_kv_len() != req.kv.kv_committed_len:
            return False

        retained = result.grammar_retained_tokens
        if retained is None or retained[i] is None:
            return True

        num_verified = num_correct_drafts[i] + result.num_non_draft_tokens_per_req
        return len(retained[i]) == num_verified

    def _mamba_prefix_cache_update(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
    ) -> None:
        """Commit the state this step tracked at a ping-pong boundary.

        Non-lazy: the tracked slot becomes the latest checkpoint and the next
        boundary tracks into the other slot.
        Lazy: the tracked slot becomes the only one; the old one is freed.
        """
        if req.kv.mamba_ping_pong_track_buffer is None:
            return

        lazy = get_exec().mamba.enable_mamba_extra_buffer_lazy
        at_boundary, track_seqlen = self._mamba_check_track_boundary(
            req, batch, result, i
        )

        if lazy and not batch.spec_algorithm.is_none():
            # For spec, at_boundary means this step actually crossed an interval.
            self._mamba_lazy_spec_update(req, batch, i, at_boundary, track_seqlen)
            return

        if not at_boundary:
            return

        track_idx = batch.mamba_track_buffer_indices[i]
        previous_track_seqlen = req.kv.mamba_last_track_seqlen
        req.kv.mamba_last_track_seqlen = track_seqlen
        if lazy:
            # Lazy frees the slot it stops tracking, so nothing names the
            # previous checkpoint there; mamba_prev_track_seqlen stays None.
            self.mamba_lazy_post_decode_at_boundary(req, batch, track_idx)
        else:
            req.kv.mamba_last_track_idx = track_idx
            # The slot that stops being the latest still holds its
            # checkpoint; name it so a short key can fall back to it.
            req.kv.mamba_prev_track_seqlen = previous_track_seqlen
            req.kv.mamba_next_track_idx = (
                batch.req_to_token_pool.get_mamba_ping_pong_other_idx(track_idx)
            )

    def _mamba_lazy_spec_update(
        self, req: Req, batch: ScheduleBatch, i: int, crossed: bool, track_seqlen: int
    ) -> None:
        """Lazy + spec post-processing.

        Running req with a confirmed crossing: the commit scattered the
        crossing state into the planned position; promote it to keep unless
        the plan already pointed at the keep slot (an in-place crossing). A
        finishing req keeps its slots for the donation
        (_mamba_lazy_keep_written).
        """
        planned_pos = batch.mamba_lazy_spec_track_positions_cpu[i]
        if req.finished() or not crossed:
            return
        if planned_pos != req.kv.mamba_next_track_idx:
            # Promote pending -> keep: free the old checkpoint, repoint.
            pool = batch.req_to_token_pool
            keep_idx = req.kv.mamba_next_track_idx
            keep_val = req.kv.mamba_ping_pong_track_buffer[keep_idx]
            pool.mamba_allocator.free(keep_val.unsqueeze(0))
            pool.set_mamba_ping_pong_slot(req, keep_idx, -1)
            req.kv.mamba_next_track_idx = planned_pos
        req.kv.mamba_last_track_idx = planned_pos
        req.kv.mamba_last_track_seqlen = track_seqlen

    def _mamba_lazy_keep_written(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
    ) -> bool:
        """Whether a finishing lazy request's checkpoint slot is written in
        place, by this step's verify (a finishing request's crossing is not
        committed) or by a pending step; the request is then released without
        a checkpoint."""
        if (
            not get_exec().mamba.enable_mamba_extra_buffer_lazy
            or req.kv.mamba_ping_pong_track_buffer is None
        ):
            return False
        keep_idx = req.kv.mamba_last_track_idx
        if (
            not batch.spec_algorithm.is_none()
            and batch.mamba_track_write_position(i) == keep_idx
            and self._mamba_check_track_boundary(req, batch, result, i)[0]
        ):
            return True
        return self._mamba_lazy_keep_pending_write(req)

    def _mamba_lazy_keep_pending_write(self, req: Req) -> bool:
        """Whether a pending step's forward may write a lazy request's
        checkpoint slot in place. Until that step commits, the slot holds no
        state its tracked length names, so the checkpoint is not published."""
        return (
            get_exec().mamba.enable_mamba_extra_buffer_lazy
            and req.kv.mamba_ping_pong_track_buffer is not None
            and self.pending.plans_mamba_track_write(req, req.kv.mamba_last_track_idx)
        )

    def _mamba_check_track_boundary(self, req, batch, result, i):
        """Check if this decode step crosses a mamba track interval boundary.

        Returns (at_boundary, track_seqlen). A non-spec step tracked the rows
        its mask marks, at the step's own length. For spec decode, the boundary
        is detected by comparing the accepted seq_len range against interval
        boundaries.
        """
        interval = mamba_track_grid(self.kv.tree_cache.page_size)

        if batch.spec_algorithm.is_none():
            if batch.mamba_track_mask_cpu is not None and batch.mamba_track_mask_cpu[i]:
                # The counters have advanced: committed ends at this step.
                return True, req.kv.kv_committed_len
        elif result.accept_lens is not None:
            cur = req.seqlen - 1
            num_non_draft = result.num_non_draft_tokens_per_req
            retained = result.grammar_retained_tokens
            if retained is not None and retained[i] is not None:
                # NOTE(kpham-sgl): grammar-truncated run, minus non-draft
                # tokens ahead of the tracked verify window (UNO's root).
                num_committed = max(len(retained[i]) - (num_non_draft - 1), 0)
            else:
                num_committed = int(result.accept_lens[i]) - num_non_draft + 1
            prev = cur - num_committed
            if cur // interval != prev // interval:
                return True, cur // interval * interval

        return False, 0

    def mamba_lazy_post_decode_at_boundary(
        self, req: Req, batch: ScheduleBatch, track_idx: int
    ):
        """Commit a completed lazy-mode boundary and free its old slot."""
        req.kv.mamba_last_track_idx = track_idx
        req.kv.mamba_next_track_idx = track_idx
        other_idx = 1 - track_idx
        pool = batch.req_to_token_pool
        if not req.kv.mamba_ping_pong_empty[other_idx]:
            pool.mamba_allocator.free(
                req.kv.mamba_ping_pong_track_buffer[other_idx].unsqueeze(0)
            )
            pool.set_mamba_ping_pong_slot(req, other_idx, -1)


def _abort_at_next_commit(req: Req, error: str) -> None:
    """Abort a running request: it finishes at its next commit, as a client
    abort does, since the next step may already run it."""
    req.to_finish = FINISH_ABORT(error, HTTPStatus.INTERNAL_SERVER_ERROR)


class _PrefillOutcome:
    """Normal prefill: a concluded row takes its token, then finishes or
    checkpoints; every row but the chunked request's streams."""

    def __init__(
        self,
        processor: SchedulerBatchResultProcessor,
        batch: ScheduleBatch,
        on_checkpoint: Optional[Callable[[Req], None]],
    ):
        self.processor = processor
        self.on_checkpoint = on_checkpoint
        self.first_decode_row = len(batch.reqs) - batch.num_decode_rows
        self.capture_hidden_mode = processor._get_prefill_hidden_capture_mode(batch)
        self.hidden_state_offset = 0
        self.skip_stream_req = None

    def _capture_hidden_states(
        self, commit: ExtendCommit, i: int, req: Req, *, store: bool
    ) -> None:
        if (
            commit.batch.return_hidden_states
            and commit.logits_output.hidden_states is not None
        ):
            assert commit.batch.extend_lens is not None
            self.hidden_state_offset = self.processor._append_prefill_hidden_states(
                req=req,
                logits_output=commit.logits_output,
                hidden_state_offset=self.hidden_state_offset,
                capture_hidden_mode=self.capture_hidden_mode,
                extend_input_len=commit.batch.extend_lens[i],
                store=store,
            )

    def conclude(
        self, commit: ExtendCommit, i: int, req: Req, next_token_id: int
    ) -> None:
        processor = self.processor
        batch = commit.batch
        logits_output = commit.logits_output
        is_decode_row = i >= self.first_decode_row
        should_commit_output = not req.finished() and not req.is_retracted
        # A reason here finishes the request without committing this
        # forward's output: a failed KV load or an unusable sampling mask.
        discard_finish_reason = (
            req.discard_output_reason if should_commit_output else None
        )
        if (
            should_commit_output
            and discard_finish_reason is None
            and req.return_sampling_mask
        ):
            discard_finish_reason = processor.sampling_mask_finish_reason(commit, i)
        self._capture_hidden_states(
            commit,
            i,
            req,
            store=should_commit_output and discard_finish_reason is None,
        )

        if req.finished() or req.is_retracted:
            # Decode req in a mixed batch, or a retracted req.
            return

        req.time_stats.set_prefill_finished_time()

        if discard_finish_reason is not None:
            req.to_finish = discard_finish_reason
            req.update_finish_state(0)
        elif req.beam_group is not None:
            # The relay point already replaced the sampled-token
            # append; the group owns all finish semantics.
            processor.beam_coordinator.commit_prefill(
                req, up_to_tick=batch.forward_iter
            )
        else:
            # req output_ids are set here
            req.output_ids.append(next_token_id)

            maybe_update_reasoning_tokens(processor.model_config, req, next_token_id)

            req.update_finish_state()
        # A mixed spec tail committed its pending bonus token; advance
        # so the next spec prepare_for_decode reserves from the right base.
        if not req.finished() and is_decode_row and not batch.spec_algorithm.is_none():
            req.kv.commit_decode(1)
        if req.finished():
            if discard_finish_reason is None:
                processor._maybe_collect_routed_experts(req)
                processor._maybe_collect_indexer_topk(req)
            processor.request_retirer.retire_finished(
                req,
                checkpoint=discard_finish_reason is None
                and not processor._mamba_lazy_keep_pending_write(req),
            )
            req.time_stats.set_completion_time()
        elif not is_decode_row:
            checkpoint_kv_cache(
                req,
                processor.kv.tree_cache,
                publish=not processor._mamba_lazy_keep_pending_write(req),
            )
            if self.on_checkpoint is not None:
                self.on_checkpoint(req)

        if discard_finish_reason is None:
            processor._maybe_collect_customized_info(i, req, logits_output)

        if batch.return_logprob:
            processor.apply_prefill_logprobs(
                commit, i, req, store=discard_finish_reason is None
            )

        if discard_finish_reason is not None:
            return

        if req.return_sampling_mask:
            processor.add_sampling_mask_return_values(i, req, logits_output)

        if req.grammar is not None:
            processor.apply_prefill_grammar(
                commit, req, next_token_id, abort=_abort_at_next_commit
            )

    def continue_chunk(self, commit: ExtendCommit, i: int, req: Req) -> None:
        self._capture_hidden_states(commit, i, req, store=False)
        if req.is_retracted:
            return
        # Its prefill is not finished, so it does not stream. An aborted
        # middle chunk still comes here, to drain its logprob accounting.
        self.skip_stream_req = req

        # Incrementally update input logprobs.
        if commit.batch.return_logprob:
            self.processor.apply_chunked_prefill_logprobs(commit, i, req)

        req.time_stats.set_last_chunked_prefill_finish_time()

    def finish(self, commit: ExtendCommit) -> None:
        self.processor.services.output_streamer.stream_output(
            commit.batch.reqs, commit.batch.return_logprob, self.skip_stream_req
        )
