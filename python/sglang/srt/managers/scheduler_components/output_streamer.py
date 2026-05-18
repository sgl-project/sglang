from __future__ import annotations

import logging
from dataclasses import dataclass
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
    GetLoadsReqInput,
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
    load_inquirer_get_loads: Callable[..., Any]
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

    def get_cached_tokens_details(self, req: Req) -> Optional[dict]:
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
            # Only include storage fields if L3 storage is enabled
            if self.enable_hicache_storage():
                details["storage"] = req.cached_tokens_storage
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
        if not hasattr(self, "_test_stream_output_count"):
            self._test_stream_output_count = 0
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
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        reasoning_tokens = []
        completion_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        spec_verify_ct = []
        spec_num_correct_drafts = []
        spec_correct_drafts_histogram = []
        retraction_counts = []
        output_hidden_states = None
        load = self.load_inquirer_get_loads(GetLoadsReqInput(include=["core"]))
        routed_experts = None
        indexer_topk = None
        customized_info = {}

        time_stats = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                if req.finished_len is None:
                    req.finished_len = len(req.output_ids)
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval
                        or self.server_args.stream_interval
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
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                # Exclude the tokens after stop condition
                output_ids_ = req.output_ids_through_stop

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(output_ids_[send_token_offset:])
                req.send_token_offset = len(output_ids_)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                reasoning_tokens.append(req.reasoning_tokens)
                completion_tokens.append(len(output_ids_))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self.get_cached_tokens_details(req))

                retraction_counts.append(req.retraction_count)

                time_stats.append(req.time_stats)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_num_correct_drafts.append(req.spec_num_correct_drafts)
                    spec_correct_drafts_histogram.append(
                        req.spec_correct_drafts_histogram
                    )

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                        # Only send when input logprobs have been computed (after prefill)
                        and req.input_token_logprobs_val is not None
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        logprob_end = max(len(output_ids_), 1)
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        req.send_output_token_logprobs_offset = logprob_end
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                if req.return_routed_experts:
                    if routed_experts is None:
                        routed_experts = []
                    routed_experts.append(req.routed_experts)
                if req.return_indexer_topk:
                    if indexer_topk is None:
                        indexer_topk = []
                    indexer_topk.append(req.indexer_topk)

                if req.customized_info is not None:
                    for k, v in req.customized_info.items():
                        if k not in customized_info:
                            customized_info[k] = []
                        customized_info[k].append(
                            v[send_token_offset : len(output_ids_)]
                        )

            if (
                req.finished()
                and self.ps.attn_tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        dp_ranks = [self.ps.dp_rank] * len(rids) if rids else None

        # Send to detokenizer
        if reqs or is_idle_batch:
            self.send_to_detokenizer.send_output(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_num_correct_drafts=spec_num_correct_drafts,
                    spec_correct_drafts_histogram=spec_correct_drafts_histogram,
                    time_stats=time_stats,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cached_tokens_details=cached_tokens_details,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=None,
                    output_hidden_states=output_hidden_states,
                    routed_experts=routed_experts,
                    indexer_topk=indexer_topk,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    load=load,
                    dp_ranks=dp_ranks,
                )
            )

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

        # Optimize PHS for pickle: torch.stack reduces N __reduce_ex__
        # calls to 1 across the ZMQ IPC boundary.  We can only stack when
        # *every* entry is non-None (homogeneous batch); mixed batches
        # (some requests want PHS, others don't) keep the raw list so
        # positional indexing on the receiver side stays correct.
        stacked_phs = None
        if has_phs:
            all_have_phs = all(t is not None for t in phs_list)
            if all_have_phs:
                if all(t.shape == phs_list[0].shape for t in phs_list):
                    stacked_phs = torch.stack(phs_list)
                else:
                    stacked_phs = phs_list
            else:
                stacked_phs = phs_list

        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                time_stats=time_stats,
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
