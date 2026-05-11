from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Union

import pybase64
import torch

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers import logprob_ops, request_tracing, spec_decoding_meta
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    WatchLoadUpdateReq,
)

logger = logging.getLogger(__name__)
from typing import Any, Dict, Optional

from sglang.srt.managers.tokenizer_manager_components.lora_controller import (
    LoraController,
)
from sglang.srt.managers.tokenizer_manager_components.request_log_manager import (
    RequestLogManager,
)
from sglang.srt.managers.tokenizer_manager_components.request_metrics_recorder import (
    RequestMetricsRecorder,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputProcessorConfig:
    weight_version: Optional[str]
    batch_notify_size: int
    incremental_streaming_output: bool
    enable_metrics: bool
    skip_tokenizer_init: bool
    speculative_algorithm: str
    speculative_num_draft_tokens: int
    dp_size: int
    enable_lora: bool
    served_model_name: str


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputProcessor:
    """Consumes BatchStrOutput / BatchTokenIDOutput / BatchEmbeddingOutput from scheduler."""

    rid_to_state: Dict[str, ReqState]
    tokenizer: Optional[Any]
    request_metrics_recorder: RequestMetricsRecorder
    request_log_manager: RequestLogManager
    lora_controller: LoraController
    send_to_scheduler: Any
    config: OutputProcessorConfig

    async def handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
    ):
        pending_notify: dict[str, ReqState] = {}
        batch_notify_size = self.config.batch_notify_size
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                # Known race: /health_generate pops its rid as soon as ANY message bumps last_receive_tstamp.
                if rid.startswith(HEALTH_CHECK_RID_PREFIX):
                    continue
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.config.weight_version,
                "num_retractions": recv_obj.retraction_counts[i],
            }

            if self.config.enable_metrics:
                if recv_obj.time_stats is not None:
                    scheduler_time_stats = recv_obj.time_stats[i]
                    meta_info.update(scheduler_time_stats.convert_to_output_meta_info())

            if getattr(state.obj, "return_logprob", False):
                logprob_ops.absorb_recv(
                    meta_info,
                    state,
                    top_logprobs_num=state.obj.top_logprobs_num,
                    token_ids_logprob=state.obj.token_ids_logprob,
                    return_text_in_logprobs=state.obj.return_text_in_logprobs
                    and not self.config.skip_tokenizer_init,
                    recv_obj=recv_obj,
                    recv_obj_index=i,
                    tokenizer=self.tokenizer,
                )

            if not isinstance(recv_obj, BatchEmbeddingOutput):
                meta_info.update(
                    {
                        "reasoning_tokens": recv_obj.reasoning_tokens[i],
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )
                # Add detailed cache breakdown if available
                if (
                    hasattr(recv_obj, "cached_tokens_details")
                    and recv_obj.cached_tokens_details
                ):
                    meta_info["cached_tokens_details"] = recv_obj.cached_tokens_details[
                        i
                    ]

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]
            if getattr(recv_obj, "routed_experts", None):
                val = recv_obj.routed_experts[i]
                if val is not None:
                    # BatchStrOutput is pre-encoded by the detokenizer;
                    # BatchTokenIDOutput (skip_tokenizer_init) bypasses it.
                    if isinstance(val, torch.Tensor):
                        val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                    meta_info["routed_experts"] = val
            if getattr(recv_obj, "indexer_topk", None):
                val = recv_obj.indexer_topk[i]
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        val = pybase64.b64encode(val.numpy().tobytes()).decode("utf-8")
                    meta_info["indexer_topk"] = val
            if getattr(recv_obj, "customized_info", None):
                for k, v in recv_obj.customized_info.items():
                    meta_info[k] = v[i]
            if getattr(recv_obj, "dp_ranks", None):
                meta_info["dp_rank"] = recv_obj.dp_ranks[i]

            state.finished = recv_obj.finished_reasons[i] is not None
            if isinstance(recv_obj, BatchStrOutput):
                # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
                is_stream = getattr(state.obj, "stream", False)
                incremental = self.config.incremental_streaming_output and is_stream
                delta_text = recv_obj.output_strs[i]
                delta_output_ids = recv_obj.output_ids[i]
                output_offset = state.last_output_offset
                state.append_text(delta_text)
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        logprob_ops.slice_streaming_output_meta_info(
                            meta_info, output_offset
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "text": delta_text,
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "text": state.get_text(),
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        # Non-incremental intermediate: pass reference (no
                        # copy) and defer text to _wait_one_response to avoid
                        # O(n) per-step cost that compounds to O(n^2).
                        out_dict = {
                            "text": None,
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "text": state.get_text(),
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
            elif isinstance(recv_obj, BatchTokenIDOutput):
                is_stream = getattr(state.obj, "stream", False)
                incremental = self.config.incremental_streaming_output and is_stream
                delta_output_ids = recv_obj.output_ids[i]
                output_offset = state.last_output_offset
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        logprob_ops.slice_streaming_output_meta_info(
                            meta_info, output_offset
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        out_dict = {
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
            else:
                assert isinstance(recv_obj, BatchEmbeddingOutput)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }
                if (
                    recv_obj.pooled_hidden_states is not None
                    and recv_obj.pooled_hidden_states[i] is not None
                ):
                    out_dict["pooled_hidden_state"] = recv_obj.pooled_hidden_states[i]

            # Set first_token_time on the first output batch.
            # This is the single write point for first_token_time.
            if state.time_stats.first_token_time == 0.0:
                state.time_stats.set_first_token_time()

            if state.finished:
                if state.time_stats.trace_ctx.tracing_enable:
                    state.time_stats.trace_ctx.trace_set_root_attrs(
                        request_tracing.make_span_attrs(
                            state=state,
                            recv_obj=recv_obj,
                            i=i,
                            served_model_name=self.config.served_model_name,
                        )
                    )
                state.time_stats.set_finished_time()
                meta_info["e2e_latency"] = state.time_stats.get_e2e_latency()

                if self.config.speculative_algorithm:
                    spec_decoding_meta.fill_spec_decoding_meta(
                        meta_info,
                        recv_obj=recv_obj,
                        i=i,
                        speculative_num_draft_tokens=self.config.speculative_num_draft_tokens,
                    )
                if self.config.enable_metrics:
                    scheduler_time_stats = (
                        recv_obj.time_stats[i]
                        if recv_obj.time_stats is not None
                        else None
                    )
                    completion_tokens = (
                        recv_obj.completion_tokens[i]
                        if not isinstance(recv_obj, BatchEmbeddingOutput)
                        else 0
                    )
                    meta_info.update(
                        state.time_stats.convert_to_output_meta_info(
                            scheduler_time_stats, completion_tokens
                        )
                    )

                del self.rid_to_state[rid]

                # Mark ongoing LoRA request as finished.
                if self.config.enable_lora and state.obj.lora_path:
                    asyncio.create_task(
                        self.lora_controller.lora_registry.release(state.obj.lora_id)
                    )

            if out_dict is not None:
                state.out_list.append(out_dict)
                pending_notify[rid] = state

                if len(pending_notify) >= batch_notify_size:
                    for s in pending_notify.values():
                        s.event.set()
                    pending_notify = {}
                    await asyncio.sleep(0)

            if self.config.enable_metrics and state.obj.log_metrics:
                self.request_metrics_recorder.collect_metrics(state, recv_obj, i)
            if (
                self.request_log_manager.dump_requests_folder
                and state.finished
                and state.obj.log_metrics
            ):
                self.request_log_manager.dump_requests(state, out_dict)
            if (
                self.request_log_manager.crash_dump_folder
                and state.finished
                and state.obj.log_metrics
            ):
                self.request_log_manager.record_request_for_crash_dump(state, out_dict)

        # handle_loop awaits next recv immediately
        for s in pending_notify.values():
            s.event.set()

        # When skip_tokenizer_init is enabled, tokensizer_manager receives
        # BatchTokenIDOutput.
        if (
            self.config.dp_size > 1
            and isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput))
            and recv_obj.load is not None
        ):
            load_update_req = WatchLoadUpdateReq(loads=[recv_obj.load])
            self.send_to_scheduler.send_pyobj(load_update_req)
