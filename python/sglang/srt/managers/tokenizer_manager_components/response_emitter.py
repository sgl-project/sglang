from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, Union

import fastapi
from fastapi import BackgroundTasks

from sglang.srt.environ import envs
from sglang.srt.managers import logprob_ops
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)

_REQUEST_STATE_WAIT_TIMEOUT = envs.SGLANG_REQUEST_STATE_WAIT_TIMEOUT.get()

_INCREMENTAL_STREAMING_META_INFO_KEYS = (
    "output_token_logprobs",
    "output_top_logprobs",
    "output_token_ids_logprobs",
)
from typing import Callable, Dict

from sglang.srt.managers.tokenizer_manager_components.lora_controller import (
    LoraController,
)
from sglang.srt.managers.tokenizer_manager_components.request_log_manager import (
    RequestLogManager,
)
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseEmitter:
    """Drains rid_to_state[rid].out_list and yields per-request dicts to HTTP clients."""

    rid_to_state: Dict[str, ReqState]
    lora_controller: LoraController
    request_log_manager: RequestLogManager
    abort_request: Callable[..., None]
    server_args: ServerArgs

    def _coalesce_streaming_chunks(
        self,
        out_list: list,
        rid: str,
    ) -> dict:
        """Coalesce multiple incremental streaming chunks into one.

        Both text and output_ids are incremental deltas, so we concatenate them;
        all other fields (meta_info, etc.) are taken from the last chunk.
        """
        if len(out_list) >= 20:
            logger.warning(
                "Streaming backlog: rid=%s, coalescing %d queued chunks into one. "
                "This may inflate P99 ITL for affected requests.",
                rid,
                len(out_list),
            )
        out = dict(out_list[-1])
        if "output_ids" in out:
            out["output_ids"] = [id for chunk in out_list for id in chunk["output_ids"]]
        if "text" in out:
            out["text"] = "".join(chunk["text"] for chunk in out_list)
        if "meta_info" in out:
            meta_info_list = [chunk["meta_info"] for chunk in out_list]
            meta_info = dict(meta_info_list[-1])
            for key in logprob_ops.INCREMENTAL_STREAMING_META_INFO_KEYS:
                if any(key in m for m in meta_info_list):
                    meta_info[key] = [
                        item for m in meta_info_list for item in m.get(key, [])
                    ]
            out["meta_info"] = meta_info
        return out

    async def _handle_abort_finish_reason(
        self,
        out: dict,
        state: ReqState,
        is_stream: bool,
    ) -> Optional[dict]:
        """Handle abort/error finish reasons from the scheduler.

        Returns the output dict if it should be yielded (stream abort), or None
        for normal flow. Raises ValueError or HTTPException for non-stream aborts.
        """
        finish_reason = out["meta_info"]["finish_reason"]

        if (
            finish_reason.get("type") == "abort"
            and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
        ):
            if not is_stream:
                raise ValueError(finish_reason["message"])
            return out

        if finish_reason.get("type") == "abort" and finish_reason.get(
            "status_code"
        ) in (
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.INTERNAL_SERVER_ERROR,
        ):
            # Delete the key to prevent resending abort request to the scheduler and
            # to ensure aborted request state is cleaned up.
            if state.obj.rid in self.rid_to_state:
                del self.rid_to_state[state.obj.rid]

            # Mark ongoing LoRA request as finished.
            if self.server_args.enable_lora and state.obj.lora_path:
                await self.lora_controller.lora_registry.release(state.obj.lora_id)
            if not is_stream:
                raise fastapi.HTTPException(
                    status_code=finish_reason["status_code"],
                    detail=finish_reason["message"],
                )
            return out

        return None

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]
        # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                    )
                continue

            # Drain all pending outputs atomically.
            out_list = state.out_list
            state.out_list = []
            finished = state.finished
            state.event.clear()

            # With incremental streaming, each chunk is a delta — coalesce
            # multiple queued chunks to avoid dropping token ids.
            incremental_stream = (
                is_stream and self.server_args.incremental_streaming_output
            )
            if incremental_stream and len(out_list) > 1:
                out = self._coalesce_streaming_chunks(out_list, obj.rid)
            else:
                out = out_list[-1]

            # Resolve deferred text for non-incremental streaming.
            # _handle_batch_output sets "text": None on intermediate chunks
            # to avoid O(n) string rebuild per step (O(n^2) total).
            if (
                is_stream
                and not incremental_stream
                and "text" in out
                and out["text"] is None
            ):
                out["text"] = state.get_text()

            if finished:
                # Record response sent time right before we log finished results and metrics.
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()
                self.request_log_manager.request_logger.log_finished_request(
                    obj,
                    out,
                    request=request,
                )

                if (
                    self.request_log_manager.request_metrics_exporter_manager.exporter_enabled()
                ):
                    asyncio.create_task(
                        self.request_log_manager.request_metrics_exporter_manager.write_record(
                            obj, out
                        )
                    )

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    abort_out = await self._handle_abort_finish_reason(
                        out, state, is_stream
                    )
                    if abort_out is not None:
                        yield abort_out
                        break

                yield out
                break

            if is_stream:
                # Record response sent time right before we send response.
                if not state.time_stats.response_sent_to_client_time:
                    state.time_stats.set_response_sent_to_client_time()
                    out["meta_info"][
                        "response_sent_to_client_ts"
                    ] = state.time_stats.get_response_sent_to_client_realtime()
                yield out
            else:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, running)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}"
                    )

    def create_abort_task(self, obj: GenerateReqInput):
        async def abort_request():
            await asyncio.sleep(2)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    async def _handle_batch_request(
        self,
        obj,
        *,
        rids,
        generators,
        request=None,
    ):
        """Wait for all per-request generators and yield outputs (single or stream)."""
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass
