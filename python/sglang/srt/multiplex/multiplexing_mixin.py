"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import set_pdmux_status
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
    set_current_stream_idx,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:
    def init_pdmux(self: Scheduler):
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(self.server_args.pdmux_config_path)
        initialize_stream_groups(self.ps.gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self: Scheduler,
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if not self.running_batch.is_empty() and self.split_prefill_batch:
            decode_bs = self.running_batch.batch_size()
            manual_divisions = self.pdmux_config.manual_divisions
            if manual_divisions:
                for i in range(len(manual_divisions)):
                    _, _, threshold = manual_divisions[i]
                    if decode_bs >= threshold:
                        stream_idx = i + 1
            else:
                stream_idx = max(
                    1,
                    min(
                        self.real_sm_group_num - 2,
                        decode_bs
                        * (self.real_sm_group_num - 2)
                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)
        elif not self.running_batch.is_empty():
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(self: Scheduler, sm_count: int) -> bool:
        if self.split_prefill_batch:
            return False

        # add new request
        prefill_plan = self.get_new_batch_prefill(self.running_batch)
        batch = prefill_plan.batch_to_run
        self.running_batch = prefill_plan.running_batch
        if batch and not batch.is_empty():
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            self.split_prefill_batch = batch
            return True
        return False

    def _get_split_forward_count(self: Scheduler) -> int:
        remaining_layers = (
            self.model_config.num_hidden_layers - self.split_prefill_batch.split_index
        )

        # Splitting only benefits decode work that can run between prefill
        # intervals. Without decode work, finish prefill in one model call to
        # avoid repeating the full scheduler/model-runner setup per layer.
        if self.running_batch is None or self.running_batch.is_empty():
            return remaining_layers

        if self.split_prefill_batch.extend_num_tokens <= 0:
            return remaining_layers

        forward_count = max(
            1,
            self.pdmux_config.split_forward_token_budget
            // self.split_prefill_batch.extend_num_tokens,
        )
        return min(forward_count, remaining_layers)

    def _get_pdmux_prefill_token_limit(
        self: Scheduler, max_prefill_tokens: int
    ) -> Optional[int]:
        hard_limit = self.pdmux_max_prefill_plan_tokens
        if not (self.enable_pdmux and hard_limit is not None):
            return None

        # PrefillAdder accounts input tokens in page-aligned units. Align the
        # backend's raw-token limit down so every accepted request can consume
        # the admission budget instead of remaining in the waiting queue.
        limit = min(max_prefill_tokens, hard_limit)
        return limit - limit % self.page_size

    def _get_prefill_admission_config(
        self: Scheduler, max_prefill_tokens: int
    ) -> tuple[int, bool]:
        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
            self, max_prefill_tokens
        )
        if effective_limit is None:
            return max_prefill_tokens, False
        return effective_limit, True

    def _get_max_req_input_len(self: Scheduler, max_req_input_len: int) -> int:
        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
            self, self.max_prefill_tokens
        )
        if effective_limit is None:
            return max_req_input_len
        # Request validation rejects lengths >= max_req_input_len.
        return min(max_req_input_len, effective_limit + 1)

    def _merge_finished_prefill_batch(
        self: Scheduler,
        prefill_result,
        prefill_stream,
        decode_stream,
    ) -> None:
        self.process_batch_result(self.split_prefill_batch, prefill_result)
        if self.running_batch and not self.running_batch.is_empty():
            self.running_batch.merge_batch(self.split_prefill_batch)
        else:
            self.running_batch = self.split_prefill_batch

        self.split_prefill_batch = None

        # merge_batch enqueues tensor concatenations on the prefill stream.
        # The next loop prepares decode before the stream-group synchronization,
        # so publish an explicit dependency before decode indexes those tensors.
        merge_done = prefill_stream.record_event()
        decode_stream.wait_event(merge_done)

    @torch.inference_mode()
    def event_loop_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")

        while True:
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    self.on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if self.running_batch and not self.running_batch.is_empty():
                    decode_result = self.run_batch(self.running_batch)
                    decode_done = True
                else:
                    decode_done = False
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = self._get_split_forward_count()
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(self.running_batch, decode_result)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.ps.tp_size:
                        self._merge_finished_prefill_batch(
                            prefill_result,
                            prefill_stream,
                            decode_stream,
                        )
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
