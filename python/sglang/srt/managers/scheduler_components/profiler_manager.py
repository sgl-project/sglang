from __future__ import annotations

import gc
import gzip
import logging
import os
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
)

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput, ProfileReqType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.step_span_utils import set_detailed_annotations_enabled
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_device
from sglang.srt.utils import is_mps, is_npu
from sglang.srt.utils.profile_merger import ProfileMerger
from sglang.srt.utils.profile_utils import ProfileManager
from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

_is_npu = is_npu()
_is_mps = is_mps()
if _is_npu:
    import torch_npu

    patches = [
        ["profiler.profile", torch_npu.profiler.profile],
        ["profiler.ProfilerActivity.CUDA", torch_npu.profiler.ProfilerActivity.NPU],
        ["profiler.ProfilerActivity.CPU", torch_npu.profiler.ProfilerActivity.CPU],
    ]
    apply_torch_npu_patches(torch_npu, patches)
elif _is_mps:
    from sglang.srt.hardware_backend.mlx.profiler import apply_metal_profiler_patches

    apply_metal_profiler_patches()

logger = logging.getLogger(__name__)


# Compress the trace a few MB at a time: see _export_trace.
_TRACE_GZIP_CHUNK = 4 << 20

# Traces are hundreds of MB each while they wait to be written, so a profile that
# is restarted faster than they can be written out has to wait for one.
_MAX_PENDING_FLUSHES = 2


def _export_trace(profiler: Any, path: str) -> None:
    """Write out a chrome trace, compressing it in large chunks.

    torch's own gzip export copies the json into the archive line by line, so it
    needs the GIL once per line. A thread competing with the scheduler loop for
    the GIL gets it rarely enough to stretch a 20 s write into 130 s; handing
    zlib megabytes per call keeps this thread inside long C calls instead.
    Everything else is torch's own staging block: the json goes to the system temp
    directory and the archive is written in place.
    """
    if not path.endswith(".gz") or _is_mps:
        # On MPS a Metal trace is saved alongside this one under a name derived
        # from the path below, so it has to be the one the profile advertises.
        profiler.export_chrome_trace(path)
        return

    # Not in the directory the traces are collected in: the json is an order of
    # magnitude larger than the archive, so that volume does not have to hold both.
    # The name is torch's rather than the trace's, because a profile id is free to
    # be long enough, or multi-byte enough, to push a derived name past the limit a
    # filesystem puts on one.
    with tempfile.NamedTemporaryFile("w+b", suffix=".json") as handle:
        profiler.export_chrome_trace(handle.name)
        with open(handle.name, "rb") as fin, gzip.open(path, "wb") as fout:
            shutil.copyfileobj(fin, fout, _TRACE_GZIP_CHUNK)


@dataclass
class _PendingTraceFlush:
    """A trace being written out by the flush thread, drained by check_pending_flush.

    output_dir is captured instead of read back from the manager, because a new
    profiling session may already be configured by the time the write finishes.
    """

    export: Future
    output_dir: Path
    # Set for an explicit /stop_profile, which is answered once the trace is on disk.
    reply_to: Optional[ProfileReq]


@dataclass(kw_only=True)
class SchedulerProfilerManager:
    ps: Any
    dp_tp_cpu_group: Any
    get_forward_ct: Callable[[], int]
    # called as send_response(output=..., recv_req=...) to answer a deferred request
    send_response: Callable[[Any, Any], None]

    def __post_init__(self) -> None:
        self._flush_executor: Optional[ThreadPoolExecutor] = None
        self._pending_flushes: deque[_PendingTraceFlush] = deque()

        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager = ProfileManager(
                ps=self.ps,
                cpu_group=self.dp_tp_cpu_group,
            )
            return

        self.torch_profiler = None
        self.torch_profiler_output_dir: Optional[Path] = None
        self.profiler_activities: Optional[List[str]] = None
        self.profile_id: Optional[str] = None

        self.profiler_start_forward_ct: Optional[int] = None
        self.profiler_target_forward_ct: Optional[int] = None

        self.profiler_prefill_ct: Optional[int] = None
        self.profiler_decode_ct: Optional[int] = None
        self.profiler_target_prefill_ct: Optional[int] = None
        self.profiler_target_decode_ct: Optional[int] = None

        self.profile_by_stage: bool = False
        self.profile_in_progress: bool = False
        self.merge_profiles = False
        self.detailed_annotations: bool = False

        # For ROCM
        self.rpd_profiler = None

    def _init_profile(
        self,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
        activities: Optional[List[str]],
        with_stack: Optional[bool],
        record_shapes: Optional[bool],
        profile_by_stage: bool,
        profile_id: str,
        merge_profiles: bool = False,
        profile_prefix: str = "",
        detailed_annotations: bool = False,
        profile_stages: Optional[List[str]] = None,
    ) -> ProfileReqOutput:
        if envs.SGLANG_PROFILE_V2.get():
            self.detailed_annotations = detailed_annotations
            return self._profile_manager.configure(
                output_dir=output_dir,
                start_step=start_step,
                num_steps=num_steps,
                activities=activities,
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=profile_by_stage,
                profile_id=profile_id,
                merge_profiles=merge_profiles,
                profile_prefix=profile_prefix,
                profile_stages=profile_stages,
                detailed_annotations=detailed_annotations,
            )

        if self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is already in progress. Call /stop_profile first.",
            )

        self.profile_by_stage = profile_by_stage
        self.merge_profiles = merge_profiles

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU", "GPU"]

        self.torch_profiler_output_dir = Path(output_dir).expanduser()
        self.torch_profiler_with_stack = with_stack
        self.torch_profiler_record_shapes = record_shapes
        self.profiler_activities = activities
        self.profile_id = profile_id
        self.profile_prefix = profile_prefix
        self.detailed_annotations = detailed_annotations

        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.get_forward_ct() + 1)

        if num_steps:
            if self.profile_by_stage:
                self.profiler_prefill_ct = 0
                self.profiler_decode_ct = 0
                self.profiler_target_prefill_ct = num_steps
                self.profiler_target_decode_ct = num_steps
            elif start_step:
                self.profiler_target_forward_ct = (
                    self.profiler_start_forward_ct + num_steps
                )
            else:
                self.profiler_target_forward_ct = self.get_forward_ct() + num_steps + 1
            # The caller will be notified when reaching profiler_target_forward_ct
        else:
            self.profiler_target_forward_ct = None

        return ProfileReqOutput(success=True, message="Succeeded")

    def _apply_detailed_annotations(self, enabled: bool) -> None:
        # Toggle the process-wide flag read by build_step_span_name; folds the
        # per-phase sq/sqsq/sqsk/sk aggregates (context c_ / generation g_)
        # into the step span while a detailed-annotation profile is active.
        set_detailed_annotations_enabled(enabled)

    def _start_profile(
        self, stage: Optional[ForwardMode] = None
    ) -> ProfileReqOutput | None:
        if envs.SGLANG_PROFILE_V2.get():
            self._apply_detailed_annotations(self.detailed_annotations)
            return self._profile_manager.manual_start()

        stage_str = f" for {stage.name}" if stage else ""
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

        if current_platform.is_out_of_tree():
            if hasattr(
                torch.profiler.ProfilerActivity,
                current_platform.get_torch_profiler_activity_str(),
            ):
                activity_map[current_platform.get_torch_profiler_activity_str()] = (
                    current_platform.get_torch_profiler_activity()
                )
        if hasattr(torch.profiler.ProfilerActivity, "XPU"):
            activity_map["XPU"] = torch.profiler.ProfilerActivity.XPU
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if "RPD" in activities:  # for ROCM
            from rpdTracerControl import rpdTracerControl

            rpdTracerControl.skipCreate()

            self.rpd_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                "rpd-" + str(time.time()) + f"-TP-{self.ps.tp_rank}" + ".trace.json.gz",
            )

            if self.ps.tp_rank == 0:
                import sqlite3

                from rocpd.schema import RocpdSchema

                if os.path.exists("trace.rpd"):
                    os.unlink("trace.rpd")
                schema = RocpdSchema()
                connection = sqlite3.connect("trace.rpd")
                schema.writeSchema(connection)
                connection.commit()
                del connection
            torch.distributed.barrier(self.dp_tp_cpu_group)

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
                        str(self.torch_profiler_output_dir)
                    )
                ),
                experimental_config=(
                    None
                    if not _is_npu
                    else torch_npu.profiler._ExperimentalConfig(
                        export_type=torch_npu.profiler.ExportType.Text,
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        msprof_tx=False,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,
                        op_attr=False,
                        data_simplification=False,
                        record_op_args=False,
                        gc_detect_threshold=None,
                    )
                ),
            )
            try:
                self.torch_profiler.start()
            except RuntimeError as e:
                self.torch_profiler = None
                return ProfileReqOutput(success=False, message=str(e))
            self.profile_in_progress = True

        if "MEM" in activities:
            torch.cuda.memory._record_memory_history(
                max_entries=envs.SGLANG_MEM_PROFILE_MAX_ENTRIES.get()
            )
            self.profile_in_progress = True

        if "CUDA_PROFILER" in activities:
            if self.ps.gpu_id == get_device().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStart()
            self.profile_in_progress = True

        self._apply_detailed_annotations(self.detailed_annotations)
        return ProfileReqOutput(success=True, message="Succeeded")

    def _merge_profile_traces(self) -> str:
        if not self.merge_profiles:
            return ""

        if self.ps.tp_rank != 0:
            return ""
        if self.ps.dp_size > 1 and self.ps.dp_rank != 0:
            return ""
        if self.ps.pp_size > 1 and self.ps.pp_rank != 0:
            return ""
        if self.ps.moe_ep_size > 1 and self.ps.moe_ep_rank != 0:
            return ""

        try:
            logger.info("Starting profile merge...")
            merger = ProfileMerger(self.torch_profiler_output_dir, self.profile_id)
            merged_path = merger.merge_chrome_traces()

            summary = merger.get_merge_summary()
            merge_message = (
                f" Merged trace: {merged_path} "
                f"(Events: {summary.get('total_events', '?')}, "
                f"Files: {summary.get('total_files', '?')})"
            )

            logger.info(f"Profile merge completed: {merged_path}")
        except Exception as e:
            logger.error(f"Failed to merge profiles: {e}", exc_info=True)
            return f" Merge failed: {e!s}"
        else:
            return merge_message

    def _stop_profile(
        self, stage: Optional[ForwardMode] = None, reply_to: Optional[ProfileReq] = None
    ) -> ProfileReqOutput | None:
        if envs.SGLANG_PROFILE_V2.get():
            self._apply_detailed_annotations(False)
            return self._profile_manager.manual_stop()

        if not self.profile_in_progress:
            return ProfileReqOutput(
                success=False,
                message="Profiling is not in progress. Call /start_profile first.",
            )

        self.torch_profiler_output_dir.mkdir(parents=True, exist_ok=True)

        if self.profile_prefix:
            stage_prefix = self.profile_prefix + "-"
        else:
            stage_prefix = ""

        stage_suffix = f"-{stage.name}" if stage else ""
        logger.info("Stop profiling" + stage_suffix + "...")
        pending = None
        if self.torch_profiler is not None:
            self.torch_profiler.stop()
            if not _is_npu:
                # Build filename with only non-zero ranks to maintain backward compatibility
                filename_parts = [self.profile_id, f"TP-{self.ps.tp_rank}"]

                # Only add other ranks if parallelism is enabled (size > 1)
                if self.ps.dp_size > 1:
                    filename_parts.append(f"DP-{self.ps.dp_rank}")
                if self.ps.pp_size > 1:
                    filename_parts.append(f"PP-{self.ps.pp_rank}")
                if self.ps.moe_ep_size > 1:
                    filename_parts.append(f"EP-{self.ps.moe_ep_rank}")

                filename = (
                    stage_prefix
                    + "-".join(filename_parts)
                    + stage_suffix
                    + ".trace.json.gz"
                )
                path = os.path.join(self.torch_profiler_output_dir, filename)

                if self.merge_profiles:
                    # The merge below reads every trace whose name starts with this
                    # profile id, so a trace this rank recorded earlier under the
                    # same id has to be written first -- it publishes to the very
                    # name the export below does. Then stay synchronous, because the
                    # merge also reads the traces of the other ranks.
                    self._drain_flushes(block=True)
                    _export_trace(self.torch_profiler, path)
                else:
                    pending = self._flush_trace_async(path, reply_to)
            torch.distributed.barrier(self.dp_tp_cpu_group)

        if self.rpd_profiler is not None:
            self.rpd_profiler.rangePop()
            self.rpd_profiler.stop()
            self.rpd_profiler.flush()

            torch.distributed.barrier(self.dp_tp_cpu_group)
            if self.ps.tp_rank == 0:
                from sglang.srt.utils.rpd_utils import rpd_to_chrome_trace

                rpd_to_chrome_trace("trace.rpd", self.rpd_profile_path)
            self.rpd_profiler = None
            self.rpd_profile_path = None

        if self.profiler_activities is not None and "MEM" in self.profiler_activities:
            memory_profile_path = os.path.join(
                self.torch_profiler_output_dir,
                stage_prefix
                + str(time.time())
                + f"-TP-{self.ps.tp_rank}-memory"
                + stage_suffix
                + ".pickle",
            )
            torch.cuda.memory._dump_snapshot(memory_profile_path)
            torch.cuda.memory._record_memory_history(enabled=None)

        if "CUDA_PROFILER" in self.profiler_activities:
            if self.ps.gpu_id == get_device().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStop()

        # The session itself ended when stop() returned, so profiling can be started
        # again even while the trace of this one is still being written out.
        self.profile_in_progress = False
        self.profiler_start_forward_ct = None
        self._apply_detailed_annotations(False)

        if pending is not None:
            return None  # answered by check_pending_flush() once the trace is written

        merge_message = self._merge_profile_traces()

        logger.info(
            "Profiling done. Traces are saved to: %s%s",
            self.torch_profiler_output_dir,
            merge_message,
        )

        if self.torch_profiler is not None:
            self.torch_profiler = None
            gc.collect()

        return ProfileReqOutput(success=True, message=f"Succeeded.{merge_message}")

    def _flush_trace_async(
        self, path: str, reply_to: Optional[ProfileReq]
    ) -> Optional[_PendingTraceFlush]:
        """Hand the recorded trace to the flush thread and stop tracking it here.

        Writing it out takes seconds per profiled second, which the scheduler thread
        cannot afford to spend: requests in flight stall for the whole write, and a
        health check that lands in the window fails, so the router drops the rank.

        Returns None if the trace had to be written here after all, in which case
        the caller answers its request itself.
        """
        if self._flush_executor is None:
            # A single worker, so traces are written in the order they were recorded.
            # Its thread is non-daemon, so interpreter exit joins it rather than
            # truncating a trace that is still being written.
            self._flush_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sglang-trace-flush"
            )

        while len(self._pending_flushes) >= _MAX_PENDING_FLUSHES:
            # Wait for the oldest rather than let recorded traces pile up: a stop
            # that has to wait is what this code used to do for every stop.
            self._drain_flushes(block=True, limit=1)

        # The profiler object goes with it: this session is over, and the next one
        # may well start before the write finishes.
        holder = [self.torch_profiler]

        def _export() -> None:
            try:
                _export_trace(holder[0], path)
            finally:
                # Free the trace here as well - it is hundreds of MB for a
                # multi-second window, and reclaiming it is not cheap either. Also
                # on the way out of a failed export, which would otherwise leave it
                # for whenever a collection happens to come around.
                holder[0] = None
                gc.collect()

        try:
            export = self._flush_executor.submit(_export)
        except RuntimeError:
            # The flush thread is gone, which happens once shutdown has drained it.
            # Write the trace here instead of dropping it, and leave torch_profiler
            # in place for the caller to clean up as it did before.
            _export()
            return None

        self.torch_profiler = None
        pending = _PendingTraceFlush(
            export=export,
            output_dir=self.torch_profiler_output_dir,
            reply_to=reply_to,
        )
        self._pending_flushes.append(pending)
        return pending

    def check_pending_flush(self) -> None:
        """Drain trace writes that have finished. Called from the scheduler loop."""
        self._drain_flushes()

    def drain_pending_flushes(self) -> None:
        """Finish traces that are still being written. Called on graceful shutdown.

        The flush thread is not a daemon, so the process waits for the write during
        interpreter teardown either way; waiting for it here means the /stop_profile
        that asked for the trace still gets its answer.
        """
        self._drain_flushes(block=True)
        if self._flush_executor is not None:
            # Kept in place rather than cleared, so a stop that somehow arrives
            # after this writes its trace inline instead of starting a thread that
            # nothing would drain.
            self._flush_executor.shutdown(wait=True)

    def _drain_flushes(self, block: bool = False, limit: Optional[int] = None) -> None:
        drained = 0
        while self._pending_flushes and (limit is None or drained < limit):
            if not block and not self._pending_flushes[0].export.done():
                return
            pending = self._pending_flushes.popleft()
            error = pending.export.exception()
            drained += 1
            if error is None:
                logger.info(
                    "Profiling done. Traces are saved to: %s", pending.output_dir
                )
            else:
                logger.error("Failed to export trace: %s", error, exc_info=error)
            if pending.reply_to is not None:
                self.send_response(
                    output=ProfileReqOutput(
                        success=error is None,
                        message=(
                            "Succeeded."
                            if error is None
                            else f"Failed to export trace: {error}"
                        ),
                    ),
                    recv_req=pending.reply_to,
                )

    def _profile_batch_predicate(self, batch: ScheduleBatch):
        if envs.SGLANG_PROFILE_V2.get():
            self._profile_manager.step(forward_mode=batch.forward_mode)
            return

        if self.profile_by_stage:
            if batch.forward_mode.is_prefill():
                if self.profiler_prefill_ct == 0:
                    self._start_profile(batch.forward_mode)
                self.profiler_prefill_ct += 1
                if self.profiler_prefill_ct > self.profiler_target_prefill_ct:
                    if self.profile_in_progress:
                        self._stop_profile(stage=ForwardMode.EXTEND)
            elif batch.forward_mode.is_decode():
                if self.profiler_decode_ct == 0:
                    if self.profile_in_progress:
                        # force trace flush
                        self._stop_profile(stage=ForwardMode.EXTEND)
                    self._start_profile(batch.forward_mode)
                self.profiler_decode_ct += 1
                if self.profiler_decode_ct > self.profiler_target_decode_ct:
                    if self.profile_in_progress:
                        self._stop_profile(stage=ForwardMode.DECODE)
            elif batch.forward_mode.is_idle():
                pass
            else:
                raise RuntimeError(f"unsupported profile stage: {batch.forward_mode}")
        else:
            # Check profiler
            if (
                self.profiler_target_forward_ct
                and self.profiler_target_forward_ct <= self.get_forward_ct()
            ):
                self._stop_profile()
            if (
                self.profiler_start_forward_ct
                and self.profiler_start_forward_ct == self.get_forward_ct()
            ):
                self._start_profile()

    def _profile(self, recv_req: ProfileReq):
        if recv_req.req_type == ProfileReqType.START_PROFILE:
            if recv_req.profile_by_stage or recv_req.start_step:
                return self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                    recv_req.detailed_annotations,
                    recv_req.profile_stages,
                )
            else:
                self._init_profile(
                    recv_req.output_dir,
                    recv_req.start_step,
                    recv_req.num_steps,
                    recv_req.activities,
                    recv_req.with_stack,
                    recv_req.record_shapes,
                    recv_req.profile_by_stage,
                    recv_req.profile_id,
                    recv_req.merge_profiles,
                    recv_req.profile_prefix,
                    recv_req.detailed_annotations,
                )
                return self._start_profile()
        else:
            return self._stop_profile(reply_to=recv_req)
