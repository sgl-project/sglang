from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Optional,
)

import torch
import zmq

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
    CachedTokensDetails,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


DEFAULT_FORCE_STREAM_INTERVAL = envs.SGLANG_FORCE_STREAM_INTERVAL.get()


@dataclass(kw_only=True, slots=True)
class SchedulerOutputStreamer:
    send_to_detokenizer: zmq.Socket
    tree_cache: BasePrefixCache
    ps: ParallelState
    server_args: ServerArgs
    is_generation: bool
    spec_algorithm: SpeculativeAlgorithm
    disaggregation_mode: DisaggregationMode
    enable_hicache_storage: Callable[[], bool]
    _test_stream_output_count: int = 0

    def _get_storage_backend_type(self) -> str:
        """Get storage backend type from tree_cache."""
        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def get_cached_tokens_details(self, req: Req) -> Optional[CachedTokensDetails]:
        """Get detailed cache breakdown for a request, if available.

        Returns:
            - None if no cached tokens at all
            - {"device": X, "host": Y} without storage breakdown
            - {"device": X, "host": Y, "storage": Z} with storage breakdown
        """
        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }
            # In PD mode the L3 hit is produced on prefill and reported on
            # decode via metadata, while decode may not have a local storage backend.
            if req.cached_tokens_storage > 0 or self.enable_hicache_storage():
                details["storage"] = req.cached_tokens_storage
            if self.enable_hicache_storage():
                details["storage_backend"] = self._get_storage_backend_type()
            return details

        if req.cached_tokens > 0:
            return {
                "device": req.cached_tokens,
                "host": 0,
            }

        return None

    def stream_output(
        self,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self._stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self._stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def _stream_output_generation(
        self,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        return_hidden_states = any(
            req.return_hidden_states for req in reqs if req is not skip_req
        )
        return_routed_experts = any(
            req.return_routed_experts for req in reqs if req is not skip_req
        )
        return_indexer_topk = any(
            req.return_indexer_topk for req in reqs if req is not skip_req
        )

        acc = _GenerationStreamAccumulator(
            return_logprob=return_logprob,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            return_indexer_topk=return_indexer_topk,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            default_stream_interval=self.server_args.stream_interval,
            default_force_stream_interval=DEFAULT_FORCE_STREAM_INTERVAL,
            get_cached_tokens_details=self.get_cached_tokens_details,
        )
        for req in reqs:
            if req is skip_req:
                continue
            if req.finished() and req.finished_output:
                # With the overlap schedule, a request will try to output twice and hit this line twice
                # because of the one additional delayed token. This "continue" prevented the dummy output.
                continue

            acc.accept(req=req)
            self._maybe_log_time_stats(req=req)

        # Send to detokenizer
        payload = acc.to_payload(
            dp_rank=self.ps.dp_rank,
            is_idle_batch=is_idle_batch,
        )
        if payload is not None:
            self.send_to_detokenizer.send_output(payload)

    def _maybe_log_time_stats(self, *, req: Req) -> None:
        if (
            req.finished()
            and self.ps.attn_tp_rank == 0
            and self.server_args.enable_request_time_stats_logging
        ):
            req.log_time_stats()

    def _stream_output_embedding(self, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        time_stats = []
        retraction_counts = []
        phs_list = []
        has_phs = False
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self.get_cached_tokens_details(req))
                time_stats.append(req.time_stats)
                retraction_counts.append(req.retraction_count)

                phs = req.pooled_hidden_state
                phs_list.append(phs)
                if phs is not None:
                    has_phs = True

        # Optimize pooled hidden states (PHS) for IPC serialization.
        # Two formats, disambiguated on the receiver side by length:
        #   Stacked:     [stacked_tensor(N, ...)] — len 1, N > 1 requests
        #   Non-stacked: [tensor_0, tensor_1, ...] — len == N
        # Stacking reduces N pickle/__reduce_ex__ calls to 1.
        # Only possible when all entries are non-None and same shape.
        # See paired receiver logic in tokenizer_manager.py.
        stacked_phs = None
        if has_phs:
            all_have_phs = all(t is not None for t in phs_list)
            if all_have_phs:
                if len(phs_list) > 1 and all(
                    t.shape == phs_list[0].shape for t in phs_list
                ):
                    # Stacked: single tensor, wrapped in a list.
                    stacked_phs = [torch.stack(phs_list)]
                else:
                    # Non-stacked: 1 request, mixed shapes, or mixed None.
                    stacked_phs = phs_list
            else:
                # Non-stacked: some requests don't have PHS (None entries).
                stacked_phs = phs_list

        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                time_stats=wrap_as_pickle(time_stats),
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
                pooled_hidden_states=stacked_phs,
            )
        )


@dataclass(slots=True, kw_only=True)
class _GenerationStreamAccumulator:
    return_logprob: bool
    return_hidden_states: bool
    return_routed_experts: bool
    return_indexer_topk: bool
    spec_algorithm: Any
    disaggregation_mode: DisaggregationMode
    default_stream_interval: int
    default_force_stream_interval: int
    get_cached_tokens_details: Callable[[Req], Optional[CachedTokensDetails]]

    rids: list = field(default_factory=list)
    http_worker_ipcs: list = field(default_factory=list)
    finished_reasons: list = field(default_factory=list)
    decoded_texts: list = field(default_factory=list)
    decode_ids_list: list = field(default_factory=list)
    read_offsets: list = field(default_factory=list)
    output_ids: list = field(default_factory=list)
    skip_special_tokens: list = field(default_factory=list)
    spaces_between_special_tokens: list = field(default_factory=list)
    no_stop_trim: list = field(default_factory=list)
    prompt_tokens: list = field(default_factory=list)
    reasoning_tokens: list = field(default_factory=list)
    completion_tokens: list = field(default_factory=list)
    cached_tokens: list = field(default_factory=list)
    cached_tokens_details: list = field(
        default_factory=list
    )  # Detailed breakdown by cache source
    image_tokens: list = field(default_factory=list)
    audio_tokens: list = field(default_factory=list)
    video_tokens: list = field(default_factory=list)
    spec_verify_ct: list = field(default_factory=list)
    spec_num_correct_drafts: list = field(default_factory=list)
    spec_correct_drafts_histogram: list = field(default_factory=list)
    retraction_counts: list = field(default_factory=list)
    output_hidden_states: Optional[list] = None
    routed_experts: Optional[list] = None
    indexer_topk: Optional[list] = None
    customized_info: dict = field(default_factory=dict)
    time_stats: list = field(default_factory=list)
    input_token_logprobs_val: Optional[list] = None
    input_token_logprobs_idx: Optional[list] = None
    output_token_logprobs_val: Optional[list] = None
    output_token_logprobs_idx: Optional[list] = None
    input_top_logprobs_val: Optional[list] = None
    input_top_logprobs_idx: Optional[list] = None
    output_top_logprobs_val: Optional[list] = None
    output_top_logprobs_idx: Optional[list] = None
    input_token_ids_logprobs_val: Optional[list] = None
    input_token_ids_logprobs_idx: Optional[list] = None
    output_token_ids_logprobs_val: Optional[list] = None
    output_token_ids_logprobs_idx: Optional[list] = None

    def __post_init__(self) -> None:
        if self.return_hidden_states:
            self.output_hidden_states = []
        if self.return_routed_experts:
            self.routed_experts = []
        if self.return_indexer_topk:
            self.indexer_topk = []

        if self.return_logprob:
            self.input_token_logprobs_val = []
            self.input_token_logprobs_idx = []
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            self.input_top_logprobs_val = []
            self.input_top_logprobs_idx = []
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            self.input_token_ids_logprobs_val = []
            self.input_token_ids_logprobs_idx = []
            self.output_token_ids_logprobs_val = []
            self.output_token_ids_logprobs_idx = []

    def accept(self, *, req: Req) -> None:
        if req.finished():
            assert not req.finished_output
            req.finished_output = True
            if req.finished_len is None:
                req.finished_len = len(req.output_ids)
            should_output = True
        else:
            if req.stream:
                stream_interval = (
                    req.sampling_params.stream_interval or self.default_stream_interval
                )

                # origin stream_interval logic
                should_output = (
                    len(req.output_ids) % stream_interval == 1
                    if stream_interval > 1
                    else len(req.output_ids) % stream_interval == 0
                )

                if should_output:
                    # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                    should_output &= not req.check_match_stop_str_prefix()
            else:
                should_output = (
                    len(req.output_ids) % self.default_force_stream_interval == 0
                )

        if not should_output:
            return

        send_token_offset = req.send_token_offset
        send_output_token_logprobs_offset = req.send_output_token_logprobs_offset
        self.rids.append(req.rid)
        self.http_worker_ipcs.append(req.http_worker_ipc)
        self.finished_reasons.append(
            req.finished_reason.to_json() if req.finished_reason else None
        )
        self.decoded_texts.append(req.decoded_text)
        decode_ids, read_offset = req.init_incremental_detokenize()

        self.decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

        # Exclude the tokens after stop condition
        output_ids_ = req.output_ids_through_stop

        req.send_decode_id_offset = len(decode_ids)
        self.read_offsets.append(read_offset)
        self.output_ids.append(output_ids_[send_token_offset:])
        req.send_token_offset = len(output_ids_)
        self.skip_special_tokens.append(req.sampling_params.skip_special_tokens)
        self.spaces_between_special_tokens.append(
            req.sampling_params.spaces_between_special_tokens
        )
        self.no_stop_trim.append(req.sampling_params.no_stop_trim)
        self.prompt_tokens.append(len(req.origin_input_ids))
        self.reasoning_tokens.append(req.reasoning_tokens)
        self.completion_tokens.append(len(output_ids_))
        self.cached_tokens.append(req.cached_tokens)

        # Collect detailed cache breakdown if available
        self.cached_tokens_details.append(self.get_cached_tokens_details(req))

        # Multimodal prompt token counts. In disagg decode mode the prefill node
        # already computed these and transferred them via the metadata buffer
        # (req.mm_*), so prefer the pre-stored values; otherwise compute them
        # from the request's multimodal items.
        if req.mm_image_tokens or req.mm_audio_tokens or req.mm_video_tokens:
            image_t = req.mm_image_tokens
            audio_t = req.mm_audio_tokens
            video_t = req.mm_video_tokens
        elif req.multimodal_inputs:
            image_t, audio_t, video_t = req.multimodal_inputs.compute_mm_token_counts()
        else:
            image_t = audio_t = video_t = 0
        self.image_tokens.append(image_t)
        self.audio_tokens.append(audio_t)
        self.video_tokens.append(video_t)

        self.retraction_counts.append(req.retraction_count)

        self.time_stats.append(req.time_stats)

        if not self.spec_algorithm.is_none():
            self.spec_verify_ct.append(req.spec_verify_ct)
            self.spec_num_correct_drafts.append(req.spec_num_correct_drafts)
            self.spec_correct_drafts_histogram.append(req.spec_correct_drafts_histogram)

        if self.return_logprob:
            if (
                req.return_logprob
                and not req.input_logprob_sent
                # Decode server does not send input logprobs
                and self.disaggregation_mode != DisaggregationMode.DECODE
                # Only send when input logprobs have been computed (after prefill)
                and req.logprob.input_token_logprobs_val is not None
            ):
                self.input_token_logprobs_val.append(
                    req.logprob.input_token_logprobs_val
                )
                self.input_token_logprobs_idx.append(
                    req.logprob.input_token_logprobs_idx
                )
                self.input_top_logprobs_val.append(req.logprob.input_top_logprobs_val)
                self.input_top_logprobs_idx.append(req.logprob.input_top_logprobs_idx)
                self.input_token_ids_logprobs_val.append(
                    req.logprob.input_token_ids_logprobs_val
                )
                self.input_token_ids_logprobs_idx.append(
                    req.logprob.input_token_ids_logprobs_idx
                )
                req.input_logprob_sent = True
            else:
                self.input_token_logprobs_val.append([])
                self.input_token_logprobs_idx.append([])
                self.input_top_logprobs_val.append([])
                self.input_top_logprobs_idx.append([])
                self.input_token_ids_logprobs_val.append([])
                self.input_token_ids_logprobs_idx.append([])

            if req.return_logprob:
                logprob_end = max(len(output_ids_), 1)
                self.output_token_logprobs_val.append(
                    req.logprob.output_token_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_logprobs_idx.append(
                    req.logprob.output_token_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_top_logprobs_val.append(
                    req.logprob.output_top_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_top_logprobs_idx.append(
                    req.logprob.output_top_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_ids_logprobs_val.append(
                    req.logprob.output_token_ids_logprobs_val[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                self.output_token_ids_logprobs_idx.append(
                    req.logprob.output_token_ids_logprobs_idx[
                        send_output_token_logprobs_offset:logprob_end
                    ]
                )
                req.send_output_token_logprobs_offset = logprob_end
            else:
                self.output_token_logprobs_val.append([])
                self.output_token_logprobs_idx.append([])
                self.output_top_logprobs_val.append([])
                self.output_top_logprobs_idx.append([])
                self.output_token_ids_logprobs_val.append([])
                self.output_token_ids_logprobs_idx.append([])

        if self.return_hidden_states:
            if req.return_hidden_states:
                # Mirror output_ids_through_stop: spec verify steps can overshoot finished_len.
                hs = req.hidden_states
                if req.finished_len is not None:
                    hs = hs[: req.finished_len]
                self.output_hidden_states.append(hs)
            else:
                self.output_hidden_states.append(None)
        if self.return_routed_experts:
            self.routed_experts.append(
                req.routed_experts if req.return_routed_experts else None
            )
        if self.return_indexer_topk:
            self.indexer_topk.append(
                req.indexer_topk if req.return_indexer_topk else None
            )

        current_output_len = len(self.output_ids[-1])
        if req.customized_info is not None:
            for key, req_values in req.customized_info.items():
                if key not in self.customized_info:
                    self.customized_info[key] = [
                        [None] * len(prev_output_ids)
                        for prev_output_ids in self.output_ids[:-1]
                    ]
                self.customized_info[key].append(
                    [None] * current_output_len
                    if req_values is None
                    else req_values[send_token_offset : len(output_ids_)]
                )

        for per_request_values in self.customized_info.values():
            if len(per_request_values) < len(self.output_ids):
                per_request_values.append([None] * current_output_len)

    def to_payload(
        self, *, dp_rank: int, is_idle_batch: bool
    ) -> Optional[BatchTokenIDOutput]:
        if not (self.rids or is_idle_batch):
            return None
        dp_ranks = [dp_rank] * len(self.rids) if self.rids else None
        return BatchTokenIDOutput(
            rids=self.rids,
            http_worker_ipcs=self.http_worker_ipcs,
            spec_verify_ct=self.spec_verify_ct,
            spec_num_correct_drafts=self.spec_num_correct_drafts,
            spec_correct_drafts_histogram=self.spec_correct_drafts_histogram,
            time_stats=wrap_as_pickle(self.time_stats),
            finished_reasons=self.finished_reasons,
            decoded_texts=self.decoded_texts,
            decode_ids=self.decode_ids_list,
            read_offsets=self.read_offsets,
            output_ids=self.output_ids,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            no_stop_trim=self.no_stop_trim,
            prompt_tokens=self.prompt_tokens,
            reasoning_tokens=self.reasoning_tokens,
            completion_tokens=self.completion_tokens,
            cached_tokens=self.cached_tokens,
            cached_tokens_details=self.cached_tokens_details,
            image_tokens=self.image_tokens,
            audio_tokens=self.audio_tokens,
            video_tokens=self.video_tokens,
            input_token_logprobs_val=self.input_token_logprobs_val,
            input_token_logprobs_idx=self.input_token_logprobs_idx,
            output_token_logprobs_val=self.output_token_logprobs_val,
            output_token_logprobs_idx=self.output_token_logprobs_idx,
            input_top_logprobs_val=self.input_top_logprobs_val,
            input_top_logprobs_idx=self.input_top_logprobs_idx,
            output_top_logprobs_val=self.output_top_logprobs_val,
            output_top_logprobs_idx=self.output_top_logprobs_idx,
            input_token_ids_logprobs_val=self.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=self.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=self.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=self.output_token_ids_logprobs_idx,
            output_token_entropy_val=None,
            output_hidden_states=self.output_hidden_states,
            routed_experts=self.routed_experts,
            indexer_topk=self.indexer_topk,
            customized_info=(
                wrap_as_pickle(self.customized_info) if self.customized_info else None
            ),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=self.retraction_counts,
            dp_ranks=dp_ranks,
        )
