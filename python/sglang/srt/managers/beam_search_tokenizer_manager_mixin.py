# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mixin class for handling beam search in TokenizerManager."""

import asyncio
import logging
import time
from http import HTTPStatus
from typing import Any, Dict, Union

import fastapi

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    BatchMultimodalOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)
from sglang.srt.tracing.trace import trace_req_finish

logger = logging.getLogger(__name__)


class BeamSearchTokenizerManagerMixin:
    async def wait_beam_search_response(
        self,
        out: Dict[str, Any],
        state: Any,
        obj: Any,
    ):
        """Handle beam search response and return all beam results as a single array."""

        if not state.response_sent_to_client_ts:
            state.response_sent_to_client_ts = time.time()

        self.request_logger.log_finished_request(
            obj, out, is_multimodal_gen=self.model_config.is_multimodal_gen
        )

        if self.request_metrics_exporter_manager.exporter_enabled():
            # Asynchronously write metrics for this request using the exporter manager.
            asyncio.create_task(
                self.request_metrics_exporter_manager.write_record(obj, out)
            )

        beam_results = out.get("beam_results", [])
        if beam_results:
            first_beam = beam_results[0]
            first_beam["meta_info"][
                "response_sent_to_client_ts"
            ] = state.response_sent_to_client_ts
            finish_reason = first_beam["meta_info"]["finish_reason"]
            if isinstance(finish_reason, dict):
                if (
                    finish_reason.get("type") == "abort"
                    and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                ):
                    if not obj.stream:
                        raise ValueError(finish_reason["message"])

                if finish_reason.get("type") == "abort" and finish_reason.get(
                    "status_code"
                ) in (
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                ):
                    if state.obj.rid in self.rid_to_state:
                        del self.rid_to_state[state.obj.rid]
                    if self.server_args.enable_lora and state.obj.lora_path:
                        await self.lora_registry.release(state.obj.lora_id)

                    if not obj.stream:
                        raise fastapi.HTTPException(
                            status_code=finish_reason["status_code"],
                            detail=finish_reason["message"],
                        )

        return beam_results

    def handle_beam_search_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchMultimodalOutput,
            BatchTokenIDOutput,
        ],
        i: int,
        rid: str,
        state: Any,
        meta_info: Dict[str, Any],
    ) -> bool:
        # Only support BatchTokenIDOutput or BatchStrOutput for beam search
        if not isinstance(recv_obj, (BatchTokenIDOutput, BatchStrOutput)):
            return False

        beam_search_output = (
            recv_obj.beam_search_output[i]
            if recv_obj.beam_search_output and i < len(recv_obj.beam_search_output)
            else None
        )
        has_beam_search = (
            beam_search_output is not None
            and hasattr(beam_search_output, "sequences")
            and beam_search_output.sequences
        )
        if not has_beam_search or recv_obj.finished_reasons[i] is None:
            return False

        meta_info.update(
            {
                "completion_tokens": recv_obj.completion_tokens[i],
                "cached_tokens": recv_obj.cached_tokens[i],
            }
        )

        state.finished = True
        state.finished_time = time.time()
        state.finished_time_perf = time.perf_counter()
        meta_info["e2e_latency"] = state.finished_time - state.created_time

        if self.enable_metrics:
            self._calculate_timing_metrics(meta_info, state, recv_obj, i)

        trace_req_finish(rid, ts=int(state.finished_time * 1e9))

        del self.rid_to_state[rid]

        # Mark ongoing LoRA request as finished.
        if self.server_args.enable_lora and state.obj.lora_path:
            asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

        self._process_beam_search_outputs(
            beam_search_output, meta_info, recv_obj, state
        )

        state.event.set()

        # Log metrics and dump
        if self.enable_metrics and state.obj.log_metrics:
            self.collect_metrics(state, recv_obj, i)
        if self.dump_requests_folder and state.finished and state.obj.log_metrics:
            self.dump_requests(state, state.out_list[-1])
        if self.crash_dump_folder and state.finished and state.obj.log_metrics:
            self.record_request_for_crash_dump(state, state.out_list[-1])

        return True

    def _process_beam_search_outputs(
        self,
        beam_search_output: Any,
        meta_info: Dict[str, Any],
        recv_obj: Union[BatchStrOutput, BatchTokenIDOutput],
        state: Any,
    ) -> None:
        """Process beam search outputs; the first element contains the request's meta_info, followed by all beam results."""
        beam_results = []
        for idx, beam_seq in enumerate(beam_search_output.sequences):
            if isinstance(recv_obj, BatchStrOutput):
                beam_out_dict = {
                    "text": beam_seq.text if beam_seq.text else "",
                    "output_ids": beam_seq.tokens.copy(),
                }
            elif isinstance(recv_obj, BatchTokenIDOutput):
                beam_out_dict = {
                    "output_ids": beam_seq.tokens.copy(),
                }
            else:
                continue
            if idx == 0:
                beam_meta_info = meta_info.copy()
            else:
                beam_meta_info = {}
            beam_meta_info["finish_reason"] = beam_seq.finish_reason
            beam_meta_info["sequence_score"] = beam_seq.beam_score
            beam_out_dict["meta_info"] = beam_meta_info

            beam_results.append(beam_out_dict)

        state.out_list.append({"beam_results": beam_results})
