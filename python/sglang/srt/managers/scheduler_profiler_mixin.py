import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_npu

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    torch_npu._apply_patches(patches)

logger = logging.getLogger(__name__)


class SchedulerProfilerMixin:

    def init_profier(self):
        self.torch_profiler = None
        self.torch_profiler_output_dir: Optional[str] = None
        self.profiler_activities: Optional[List[str]] = None
        self.profile_id: Optional[str] = None
        self.profiler_start_forward_ct: Optional[int] = None
        self.profiler_target_forward_ct: Optional[int] = None
        self.profiler_target_prefill_ct: Optional[int] = None
        self.profiler_target_decode_ct: Optional[int] = None
        self.profiler_prefill_ct: Optional[int] = None
        self.profiler_decode_ct: Optional[int] = None
        self.profile_by_stage: bool = False
        self.profile_steps: Optional[int] = None
        self.profile_in_progress: bool = False
        self.rpd_profiler = None

    def init_profile(
        self,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
    ) -> ProfileReqOutput:
        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        self.profile_by_stage = profile_by_stage

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.torch_profiler_output_dir = output_dir
        self.torch_profiler_with_stack = with_stack
        self.torch_profiler_record_shapes = record_shapes
        self.profiler_activities = activities
        self.profile_id = profile_id

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.forward_ct + 1)

        if num_steps:
            self.profile_steps = num_steps
            if self.profile_by_stage:
                self.profiler_target_prefill_ct = num_steps
                self.profiler_target_decode_ct = num_steps
                self.profiler_prefill_ct = 0
                self.profiler_decode_ct = 0
            elif start_step:
                self.profiler_target_forward_ct = (
                    self.profiler_start_forward_ct + num_steps
                )
            else:
                self.profiler_target_forward_ct = self.forward_ct + num_steps
            # The caller will be notified when reaching profiler_target_forward_ct
        else:
            self.profiler_target_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded")

    def start_profile(
        self, stage: Optional[ForwardMode] = None
    ) -> ProfileReqOutput | None:
        stage_str = f" for {stage.__str__()}" if stage else ""
        logger.info(
            f"Profiling starts{stage_str}. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
        )

        activities = self.profiler_activities
        with_stack = self.torch_profiler_with_stack
        record_shapes = self.torch_profiler_record_shapes

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if "RPD" in activities:
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.skipCreate()

            self.rpd_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                "rpd-" + str(time.time()) + f"-TP-{self.tp_rank}" + ".trace.json.gz",
            )

            if self.tp_rank == 0:
                import sqlite3

                from rocpd.schema import RocpdSchema

                if os.path.exists("trace.rpd"):
                    os.unlink("trace.rpd")
                schema = RocpdSchema()
                connection = sqlite3.connect("trace.rpd")
                schema.writeSchema(connection)
                connection.commit()
                del connection
            torch.distributed.barrier(self.tp_cpu_group)

            self.rpd_profiler = rpdTracerControl()
            self.rpd_profiler.setPythonTrace(True)
            self.rpd_profiler.start()
            self.rpd_profiler.rangePush("", "rpd profile range", "")
            self.profile_in_progress = True
        elif torchprof_activities:
            self.torch_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=with_stack if with_stack is not None else True,
                record_shapes=record_shapes if record_shapes is not None else False,
                on_trace_ready=(
                    None
                    if not _is_npu
                    else torch_npu.profiler.tensorboard_trace_handler(
                        self.torch_profiler_output_dir
                    )
                ),
            )
            self.torch_profiler.start()
            self.profile_in_progress = True

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self.profile_in_progress = True

        if "CUDA_PROFILER" in activities:
            torch.cuda.cudart().cudaProfilerStart()
            self.profile_in_progress = True

        return ProfileReqOutput(success=True, message="Succeeded")

    def stop_profile(
        self, stage: Optional[ForwardMode] = None
    ) -> ProfileReqOutput | None:
        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        if not Path(self.torch_profiler_output_dir).exists():
            Path(self.torch_profiler_output_dir).mkdir(parents=True, exist_ok=True)

        stage_suffix = f"-{stage.__str__()}" if stage else ""
        logger.info("Stop profiling" + stage_suffix + "...")
        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            if not _is_npu:
                self.torch_profiler.export_chrome_trace(
                    os.path.join(
                        self.torch_profiler_output_dir,
                        self.profile_id
                        + f"-TP-{self.tp_rank}"
                        + stage_suffix
                        + ".trace.json.gz",
                    )
                )
            torch.distributed.barrier(self.tp_cpu_group)

        if self.rpd_profiler is not None:
            self.rpd_profiler.rangePop()
            self.rpd_profiler.stop()
            self.rpd_profiler.flush()

            torch.distributed.barrier(self.tp_cpu_group)
            if self.tp_rank == 0:
                from sglang.srt.utils import rpd_to_chrome_trace

                rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
            self.rpd_profiler = None
            self.rpd_profiler_path = None

        if self.profiler_activities is not None and "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                str(time.time())
                + f"-TP-{self.tp_rank}-memory"
                + stage_suffix
                + ".pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        if "CUDA_PROFILER" in self.profiler_activities:
            torch.cuda.cudart().cudaProfilerStop()

        logger.info(
            "Profiling done. Traces are saved to: %s",
            self.torch_profiler_output_dir,
        )
        self.torch_profiler = None
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded.")

    def _profile_batch_predicate(self, batch):
        if self.profile_by_stage:
            if batch.forward_mode.is_prefill():
                if self.profiler_prefill_ct == 0:
                    self.start_profile(batch.forward_mode)
                self.profiler_prefill_ct += 1
                if self.profiler_prefill_ct > self.profiler_target_prefill_ct:
                    if self.profile_in_progress:
                        self.stop_profile(stage=ForwardMode.EXTEND)
            elif batch.forward_mode.is_decode():
                if self.profiler_decode_ct == 0:
                    if self.profile_in_progress:
                        # force trace flush
                        self.stop_profile(ForwardMode.EXTEND)
                    self.start_profile(batch.forward_mode)
                self.profiler_decode_ct += 1
                if self.profiler_decode_ct > self.profiler_target_decode_ct:
                    if self.profile_in_progress:
                        self.stop_profile(stage=ForwardMode.DECODE)
            elif batch.forward_mode.is_idle():
                pass
            else:
                raise RuntimeError(f"unsupported profile stage: {batch.forward_mode}")
        else:
            # Check profiler
            if (
                self.profiler_target_forward_ct
                and self.profiler_target_forward_ct <= self.forward_ct
            ):
                self.stop_profile()
            if (
                self.profiler_start_forward_ct
                and self.profiler_start_forward_ct == self.forward_ct
            ):
                self.start_profile()

    def profile(self, recv_req: ProfileReq):
        if recv_req.type == ProfileReqType.START_PROFILE:
            if recv_req.profile_by_stage or recv_req.start_step:
                return self.init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                )
            else:
                self.init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                )
                return self.start_profile(True)
        else:
            return self.stop_profile()
