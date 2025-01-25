import asyncio
import dataclasses
import time
from typing import Optional, List, Any

from sglang.srt.metrics.collector import TokenizerMetricsCollector
from sglang.srt.server_args import ServerArgs


class GenerationManager:
    pass


class GenerationConverter:
    pass


class _MetricManager:
    def __init__(self, server_args: ServerArgs):
        self.metrics_collector = TokenizerMetricsCollector(
            labels={
                "model_name": server_args.served_model_name,
                # TODO: Add lora name/path in the future,
            },
        )

    def handle_batch_output_metrics(
        self,
        recv_obj,
        i: int,
        state: "_MetricReqState",
        finished: bool,
        stream: Optional[bool],
    ):
        completion_tokens = (
            recv_obj.completion_tokens[i]
            if getattr(recv_obj, "completion_tokens", None)
            else 0
        )

        if state.first_token_time is None:
            state.first_token_time = time.time()
            self.metrics_collector.observe_time_to_first_token(
                state.first_token_time - state.created_time
            )
        else:
            if completion_tokens >= 2:
                # Compute time_per_output_token for the streaming case
                self.metrics_collector.observe_time_per_output_token(
                    (time.time() - state.first_token_time) / (completion_tokens - 1)
                )

        if finished:
            self.metrics_collector.observe_one_finished_request(
                recv_obj.prompt_tokens[i], completion_tokens
            )
            self.metrics_collector.observe_e2e_request_latency(
                time.time() - state.created_time
            )
            # Compute time_per_output_token for the non-streaming case
            if stream is not None and not stream and completion_tokens >= 1:
                self.metrics_collector.observe_time_per_output_token(
                    (time.time() - state.created_time) / completion_tokens
                )


@dataclasses.dataclass
class _ReqState:
    """Store the state a request."""

    out_list: List
    finished: bool
    event: asyncio.Event
    obj: Any

    # For streaming output
    last_output_offset: int = 0


@dataclasses.dataclass
class _MetricReqState:
    created_time: float
    first_token_time: Optional[float] = None
