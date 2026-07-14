from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.scheduler_components.logprob_result_processor import (
        SchedulerLogprobResultProcessor,
    )
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        SchedulerMetricsReporter,
    )
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.managers.utils import (
        EmbeddingBatchResult,
        GenerationBatchResult,
    )
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerBatchResultProcessor:
    is_generation: bool
    disaggregation_mode: DisaggregationMode
    enable_overlap: bool
    enable_overlap_mlx: bool
    server_args: ServerArgs
    model_config: ModelConfig
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    hisparse_coordinator: Optional[HiSparseCoordinator]
    req_to_token_pool: ReqToTokenPool
    decode_offload_manager: Optional[DecodeKVCacheOffloadManager]
    metrics_collector: SchedulerMetricsCollector
    metrics_reporter: SchedulerMetricsReporter
    draft_worker: BaseTpWorker
    model_worker: BaseTpWorker
    logprob_result_processor: SchedulerLogprobResultProcessor
    output_streamer: SchedulerOutputStreamer
    abort_request: Callable

    def process_batch_result_prebuilt(self, batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        use_free_group = self.server_args.disaggregation_decode_enable_radix_cache
        if use_free_group:
            self.token_to_kv_pool_allocator.free_group_begin()
        for req in batch.reqs:
            req.time_stats.set_decode_prebuilt_finish_time()
            req.update_finish_state()
            if req.finished():
                req.time_stats.set_quick_finish_time()
                if self.server_args.enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                release_kv_cache(req, self.tree_cache)

        # Note: Logprobs should be handled on the prefill engine.
        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        if use_free_group:
            self.token_to_kv_pool_allocator.free_group_end()

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
            req_pool_idx=req.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.req_to_token_pool,
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
            req_pool_idx=req.req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=self.req_to_token_pool,
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

    def process_batch_result_prefill(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()
            if result.routed_experts_output is not None:
                result.routed_experts_output.finalize()
                result.routed_experts_output = None
            if result.indexer_topk_output is not None:
                result.indexer_topk_output.finalize()
                result.indexer_topk_output = None

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            next_token_ids = next_token_ids.tolist()
            self.move_logprobs_to_cpu(batch=batch, logits_output=logits_output)

            self._validate_pp_skip_output_comm(batch, result)

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if (
                    req.finished() and req.inflight_middle_chunks <= 0
                ) or req.is_retracted:
                    # Decode req in a mixed batch, or a retracted req. Keep an
                    # aborted middle chunk in the chunked branch long enough to
                    # drain its accounting without streaming it.
                    continue

                if req.inflight_middle_chunks <= 0:
                    req.time_stats.set_prefill_finished_time()

                    # req output_ids are set here
                    req.output_ids.append(next_token_id)

                    self._maybe_update_reasoning_tokens(req, next_token_id)

                    req.update_finish_state()
                    if req.finished():
                        self._maybe_collect_routed_experts(req)
                        self._maybe_collect_indexer_topk(req)
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        maybe_cache_unfinished_req(req, self.tree_cache)
                        if self.server_args.enable_hisparse:
                            self.hisparse_coordinator.admit_request_into_staging(req)

                    self._maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        logprob_pt = self._apply_prefill_logprobs(
                            req=req,
                            i=i,
                            logits_output=logits_output,
                            extend_input_len_per_req=extend_input_len_per_req,
                            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                            next_token_ids=next_token_ids,
                            logprob_pt=logprob_pt,
                        )

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        hidden_state_offset = self._append_prefill_hidden_states(
                            req=req,
                            logits_output=logits_output,
                            hidden_state_offset=hidden_state_offset,
                        )

                    if req.grammar is not None:
                        self._apply_prefill_grammar(
                            req=req, next_token_id=next_token_id
                        )

                else:
                    # being chunked reqs' prefill is not finished
                    req.inflight_middle_chunks -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if batch.return_logprob:
                        logprob_pt = self._apply_chunked_prefill_logprobs(
                            req=req,
                            i=i,
                            logits_output=logits_output,
                            extend_input_len_per_req=extend_input_len_per_req,
                            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
                            logprob_pt=logprob_pt,
                        )

                    req.time_stats.set_last_chunked_prefill_finish_time()

        else:  # embedding or reward model
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

                req.embedding = embeddings[i]
                if req.return_pooled_hidden_states and phs is not None:
                    req.pooled_hidden_state = phs[i]
                if req.inflight_middle_chunks <= 0:
                    req.time_stats.set_prefill_finished_time()
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.update_finish_state()

                    if req.finished():
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    else:
                        maybe_cache_unfinished_req(req, self.tree_cache)
                else:
                    # being chunked reqs' prefill is not finished
                    req.inflight_middle_chunks -= 1
                    req.time_stats.set_last_chunked_prefill_finish_time()

        self.output_streamer.stream_output(
            batch.reqs, batch.return_logprob, skip_stream_req
        )

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
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
        if batch.return_logprob:
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

    def _apply_prefill_logprobs(
        self,
        *,
        req: Req,
        i: int,
        logits_output: LogitsProcessorOutput,
        extend_input_len_per_req: Optional[List[int]],
        extend_logprob_start_len_per_req: Optional[List[int]],
        next_token_ids: List[int],
        logprob_pt: int,
    ) -> int:
        assert extend_logprob_start_len_per_req is not None
        assert extend_input_len_per_req is not None
        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        extend_input_len = extend_input_len_per_req[i]

        num_input_logprobs = self.logprob_result_processor.calculate_num_input_logprobs(
            req,
            extend_input_len,
            extend_logprob_start_len,
        )

        if req.return_logprob:
            self.logprob_result_processor.add_logprob_return_values(
                i,
                req,
                logprob_pt,
                next_token_ids,
                num_input_logprobs,
                logits_output,
            )
        logprob_pt += num_input_logprobs
        return logprob_pt

    @staticmethod
    def _validate_pp_skip_output_comm(
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        """Validate PP skip output comm correctness.

        - When skip=True: all reqs must be middle chunks (inflight_middle_chunks > 0)
          so placeholder zeros are never consumed via req.output_ids.append().
        - When skip=False: at least one req should consume next_token_ids
          (inflight_middle_chunks <= 0), otherwise warn.
        """
        if not envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get():
            return

        if not getattr(result, "skipped_output_comm", False):
            if batch.forward_mode.is_extend() and not batch.forward_mode.is_prebuilt():
                has_consumed_output = any(
                    req.inflight_middle_chunks <= 0
                    for req in batch.reqs
                    if not req.finished() and not req.is_retracted
                )
                if not has_consumed_output and len(batch.reqs) > 0:
                    chunks = list([r.inflight_middle_chunks for r in batch.reqs])
                    logger.warning(
                        f"PP non-skip output comm: no req consumed next_token_ids. "
                        f"contains_last_prefill_chunk={batch.contains_last_prefill_chunk}, "
                        f"num_reqs={len(batch.reqs)}, all inflight_middle_chunks={chunks}"
                    )
            return

        for req in batch.reqs:
            if not req.finished() and not req.is_retracted:
                assert req.inflight_middle_chunks > 0, (
                    f"PP skip output comm invariant violated: req {req.rid} "
                    f"has inflight_middle_chunks={req.inflight_middle_chunks} "
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
    ) -> int:
        req.hidden_states.append(
            logits_output.hidden_states[
                hidden_state_offset : (
                    hidden_state_offset := hidden_state_offset
                    + len(req.origin_input_ids)
                )
            ]
            .cpu()
            .clone()
            .tolist()
        )
        return hidden_state_offset

    def _apply_prefill_grammar(self, *, req: Req, next_token_id: int) -> None:
        # FIXME: this try-except block is for handling unexpected xgrammar issue.
        try:
            req.grammar.accept_token(next_token_id)
        except ValueError as e:
            # Grammar accept_token can raise ValueError if the token is not in the grammar.
            # This can happen if the grammar is not set correctly or the token is invalid.
            logger.error(
                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
            )
            self.abort_request(AbortReq(rid=req.rid))
        req.grammar.finished = req.finished()

    def _apply_chunked_prefill_logprobs(
        self,
        *,
        req: Req,
        i: int,
        logits_output: LogitsProcessorOutput,
        extend_input_len_per_req: Optional[List[int]],
        extend_logprob_start_len_per_req: Optional[List[int]],
        logprob_pt: int,
    ) -> int:
        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
        extend_input_len = extend_input_len_per_req[i]
        if extend_logprob_start_len < extend_input_len:
            # Update input logprobs.
            num_input_logprobs = (
                self.logprob_result_processor.calculate_num_input_logprobs(
                    req,
                    extend_input_len,
                    extend_logprob_start_len,
                )
            )
            if req.return_logprob:
                self.logprob_result_processor.add_input_logprob_return_values(
                    i,
                    req,
                    logits_output,
                    logprob_pt,
                    num_input_logprobs,
                    last_prefill_chunk=False,
                )
            logprob_pt += num_input_logprobs
        return logprob_pt

    def _resolve_spec_v2_tokens(
        self,
        result: GenerationBatchResult,
        batch: ScheduleBatch,
    ) -> List[List[int]]:
        """Resolve the padded next token ids for spec-v2 (overlap and non-overlap)."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        result.num_correct_drafts = sum(accept_lens) - len(batch.reqs)
        result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]

        # Feed the adaptive controller now that accept_lens is on CPU,
        # instead of doing a synchronous GPU→CPU copy in the worker hot path.
        # BaseSpecWorker provides a no-op default for non-adaptive workers.
        self.model_worker.on_verify_complete_cpu(
            result.num_correct_drafts_per_req_cpu, batch_size=len(batch.reqs)
        )

        predict_tokens = []
        # In adaptive spec-v2, the worker state may already have switched when this
        # delayed result is processed. Use the draft token count recorded on result.
        stride = result.speculative_num_draft_tokens
        assert stride is not None, "spec-v2 result missing speculative_num_draft_tokens"

        for i, req in enumerate(batch.reqs):
            accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]

            if req.is_retracted or req.finished():
                # Nothing to settle: no worker pre-claims the bonus, so
                # kv_committed_len already holds the committed prefix.
                pass
            else:
                if req.grammar is not None:
                    # Stop accepting once the grammar terminates, so the
                    # over-drafted suffix is never committed to KV nor emitted.
                    # This advances the grammar FSM; the result loop only syncs
                    # grammar.finished.
                    accept_tokens = self._accept_grammar_tokens(req, accept_tokens)

                # Commit the full accepted run (drafts + bonus).
                num_accept_tokens = len(accept_tokens)
                req.kv_committed_len += num_accept_tokens
                req.spec_verify_ct += 1

                num_correct_drafts = result.num_correct_drafts_per_req_cpu[i]
                req.spec_num_correct_drafts += num_correct_drafts
                req.update_spec_correct_drafts_histogram(num_correct_drafts)

            predict_tokens.append(accept_tokens)

        return predict_tokens

    def _accept_grammar_tokens(
        self, req: Req, tokens: Union[int, List[int]]
    ) -> List[int]:
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
                f"Grammar accept_token failed for req {req.rid} with token "
                f"{tokens}: {e}"
            )
            self.abort_request(AbortReq(rid=req.rid))
        return retained

    def process_batch_result_idle(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.output_streamer._stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_decode(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        next_token_ids, next_token_logprobs = self._normalize_decode_outputs(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=next_token_ids,
        )

        self.metrics_reporter.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none():
            self.metrics_reporter.update_spec_metrics(
                batch.batch_size(), result.num_correct_drafts
            )
        if self.server_args.enable_metrics:
            self.metrics_collector.increment_decode_cuda_graph_pass(
                value=can_run_cuda_graph
            )

        self.token_to_kv_pool_allocator.free_group_begin()

        for i, req in enumerate(batch.reqs):
            req: Req

            if (self.enable_overlap or self.enable_overlap_mlx) and (
                req.finished() or req.is_retracted
            ):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                continue

            # next_token_id is a per-req list: 1 token for non-spec, the verified
            # run for spec (already grammar-truncated in _resolve_spec_v2_tokens).
            next_token_id = next_token_ids[i]
            is_spec = not batch.spec_algorithm.is_none()

            req.output_ids.extend(next_token_id)
            new_accept_len = len(next_token_id)

            self._maybe_update_reasoning_tokens(req, next_token_id)
            req.time_stats.set_last_decode_finish_time()
            req.update_finish_state(new_accept_len)

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

            if req.return_hidden_states and logits_output.hidden_states is not None:
                # hidden_states is [bs * stride, hidden_dim], one row per emitted
                # token; stride = speculative_num_draft_tokens for spec, 1 for non-spec.
                stride = result.speculative_num_draft_tokens or 1
                accept_len = len(next_token_id)
                start = i * stride
                req.hidden_states.extend(
                    logits_output.hidden_states[start : start + accept_len]
                    .cpu()
                    .tolist()
                )

            if req.grammar is not None:
                if not is_spec:
                    # Normal decode advances the grammar for its single token
                    # here; spec already advanced it in _resolve_spec_v2_tokens.
                    self._accept_grammar_tokens(req, next_token_id)
                req.grammar.finished = req.finished()

        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.forward_ct_decode = (
            self.metrics_reporter.forward_ct_decode + 1
        ) % (1 << 30)
        self.metrics_reporter.report_decode_stats(
            can_run_cuda_graph,
            running_batch=batch,
            num_correct_drafts=result.num_correct_drafts,
        )

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
            # CUDA workers return a device tensor, MLX a host list[int]; both -> list.
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

    def _handle_finish_state_updated_req(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
        logits_output: LogitsProcessorOutput,
    ):
        # Called here (after update_finish_state) so req.finished() is valid
        # for mamba_lazy_post_decode_at_boundary inside.
        self._mamba_prefix_cache_update(req, batch, result, i)

        if (
            self.server_args.disaggregation_decode_enable_offload_kvcache
            and not req.finished()
        ):
            self.decode_offload_manager.offload_kv_cache(req)

        if req.finished():
            # delete feature to save memory
            if req.multimodal_inputs is not None and req.session is None:
                req.multimodal_inputs.release_features()
            self._maybe_collect_routed_experts(req)
            self._maybe_collect_indexer_topk(req)

            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                if not self.decode_offload_manager.offload_kv_cache(req):
                    self.decode_offload_manager.finalize_release_on_finish(req)
            else:
                if self.server_args.enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                prepare_release = getattr(
                    self.model_worker, "prepare_for_kv_cache_release", None
                )
                if callable(prepare_release):
                    prepare_release(req)
                is_insert = (
                    req.mamba_lazy_is_insert
                    if get_global_server_args().enable_mamba_extra_buffer_lazy()
                    else True
                )
                release_kv_cache(req, self.tree_cache, is_insert=is_insert)

            req.time_stats.set_completion_time()

        self._maybe_collect_customized_info(i, req, logits_output)

    def _maybe_update_reasoning_tokens(
        self,
        req: Req,
        next_token_id: Union[int, List[int]],
    ):
        think_end_id = self.model_config.think_end_id
        if req.require_reasoning and think_end_id is not None:
            req.update_reasoning_tokens(next_token_id, think_end_id)

    def _mamba_prefix_cache_update(
        self,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
    ) -> None:
        """Update mamba track state at ping-pong boundaries.

        Non-lazy: swap the ping-pong index so the next forward writes to
        the alternate slot.
        Lazy: keep the same index (prealloc handles the swap) and run
        post-decode cleanup to free the temporary second slot.
        """
        if req.mamba_ping_pong_track_buffer is None:
            return

        lazy = get_global_server_args().enable_mamba_extra_buffer_lazy()
        at_boundary, track_seqlen = self._mamba_check_track_boundary(
            req, batch, result, i
        )

        if not at_boundary:
            return

        req.mamba_last_track_seqlen = track_seqlen
        if lazy:
            self.mamba_lazy_post_decode_at_boundary(req, batch)
        else:
            req.mamba_next_track_idx = (
                batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )

    def _mamba_check_track_boundary(self, req, batch, result, i):
        """Check if this decode step crosses a mamba track interval boundary.

        Returns (at_boundary, track_seqlen).  The boundary condition
        matches what the forward's tracking mask used:
        ``prepare_for_decode`` increments both ``seq_lens_cpu`` and
        ``kv_committed_len`` by 1, then checks
        ``seq_lens_cpu % interval == 0``.  Using ``kv_committed_len``
        here reproduces that check exactly, and the value is always a
        multiple of ``interval`` (hence page-aligned).

        For spec decode, the boundary is detected by comparing the
        accepted seq_len range against interval boundaries.
        """
        interval = get_global_server_args().mamba_track_interval

        if batch.spec_algorithm.is_none():
            if req.kv_committed_len % interval == 0:
                return True, req.kv_committed_len
        elif result.num_correct_drafts_per_req_cpu is not None:
            cur = req.seqlen - 1
            prev = cur - result.num_correct_drafts_per_req_cpu[i] - 1
            if cur // interval != prev // interval:
                return True, cur // interval * interval

        return False, 0

    def mamba_lazy_post_decode_at_boundary(self, req: Req, batch: ScheduleBatch):
        """Post-decode cleanup at a lazy-mode track boundary.

        Finished reqs: if prealloc failed (other slot is -1), the forward
        overwrote the only slot with corrupted state, so mark
        is_insert=False to skip the cache insert.  If the other slot is
        occupied (stale prealloc from an overlap extra forward), free it
        so the prealloc assert in the next prepare_for_decode holds.

        Running reqs: free the old ping-pong slot so we go back to
        holding only 1 slot until the next boundary.
        """
        other_idx = 1 - req.mamba_next_track_idx
        other_val = req.mamba_ping_pong_track_buffer[other_idx].item()
        if other_val != -1:
            pool = batch.req_to_token_pool
            pool.mamba_allocator.free(
                req.mamba_ping_pong_track_buffer[other_idx].unsqueeze(0)
            )
            pool.set_mamba_ping_pong_slot(req, other_idx, -1)
        elif req.finished():
            req.mamba_lazy_is_insert = False
