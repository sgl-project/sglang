"""
Mixin class providing multiplexing scheduling logic
"""

import logging
import time
from typing import Optional

import torch
import torch.distributed as dist
from sgl_kernel import spatial
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import set_pdmux_status
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(self) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        stream_idx = 0
        if not self.running_batch.is_empty() and self.split_prefill_batch:
            stream_idx = 1
        elif not self.running_batch.is_empty():
            stream_idx = 2
        elif self.split_prefill_batch:
            stream_idx = 0

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(self, sm_count: int) -> bool:
        if self.split_prefill_batch:
            return False

        # add new request
        batch = self.get_new_batch_prefill()
        if batch and not batch.is_empty():
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            self.split_prefill_batch = batch
            return True
        return False

    @torch.inference_mode()
    def event_loop_pdmux(self):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = 0
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")
        logger.debug(
            f"sm_counts: {len(self.sm_counts)}\n mem:{torch.cuda.memory_allocated()}"
        )

        while True:
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                if not wait_prefill_kernel_done:
                    sm_count = self.sm_counts[stream_idx][0]
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0
                    and (self.running_batch is None or self.running_batch.is_empty())
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    # print(f"Running batch is None or empty, stream_idx: {stream_idx}, sm_counts: {self.sm_counts}")
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm:{self.sm_counts[stream_idx][0]}, decode sm:{self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if self.running_batch and not self.running_batch.is_empty():
                    # logger.debug("Running decode batch...")
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
                    forward_count = (
                        max(
                            1,
                            self.max_prefill_tokens
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    # logger.debug(
                    #     f"Running split prefill batch {self.split_prefill_batch.split_index} with forward count: {forward_count}..."
                    # )
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
                    # logger.debug(f"Processing decode batch result...")
                    self.process_batch_result(self.running_batch, decode_result)
                    # logger.debug(f"Processing decode batch result... Done")

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
                else:
                    flags = torch.zeros(1, device="cpu", dtype=torch.int32)

                self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                if flags.item() == self.tp_size:
                    # logger.debug(f"Processing prefill batch result...")
                    self.process_batch_result(self.split_prefill_batch, prefill_result)
                    if self.running_batch and not self.running_batch.is_empty():
                        self.running_batch.merge_batch(self.split_prefill_batch)
                    else:
                        self.running_batch = self.split_prefill_batch

                    self.split_prefill_batch = None
                    wait_prefill_kernel_done = False
                    adjust_stream_group = True
