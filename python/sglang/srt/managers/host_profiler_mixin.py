"""
HostProfilerMixin provides profiling capabilities for the host process (TokenizerManager).

This mixin profiles operations in the host process, including:
- Text tokenization (CPU)
- Multimodal data preprocessing (CPU + GPU for image/audio processing)
- Request handling overhead

This is complementary to SchedulerProfilerMixin which profiles the scheduler/GPU operations.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class HostProfilerMixin:
    """Mixin to add profiling capabilities to TokenizerManager (host process)."""

    def init_host_profiler(self):
        """Initialize profiler state variables."""
        self.host_profiler = None
        self.host_profiler_output_dir: Optional[Path] = None
        self.host_profiler_activities: Optional[List[str]] = None
        self.host_profile_id: Optional[str] = None
        self.host_profile_in_progress: bool = False
        self.host_profiler_with_stack: bool = True
        self.host_profiler_record_shapes: bool = False
        self.host_profile_request_count: int = 0
        self.host_profile_target_request_count: Optional[int] = None

    def start_host_profile(
        self,
        output_dir: Optional[str] = None,
        activities: Optional[List[str]] = None,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
        num_requests: Optional[int] = None,
        profile_id: Optional[str] = None,
    ) -> dict:
        """
        Start profiling the host process.

        Args:
            output_dir: Directory to save profile traces. Defaults to SGLANG_TORCH_PROFILER_DIR or /tmp
            activities: List of activities to profile. Options: ["CPU", "GPU"].
                       Default is ["CPU", "GPU"] to capture both CPU operations (tokenization)
                       and GPU operations (multimodal preprocessing like image encoding).
            with_stack: Whether to record stack traces
            record_shapes: Whether to record tensor shapes
            num_requests: Number of requests to profile before auto-stopping. None means manual stop.
            profile_id: Optional profile ID to use. If None, generates one based on timestamp.
                       Use the same profile_id as scheduler to enable trace merging.

        Returns:
            dict with success status and message
        """
        if self.host_profile_in_progress:
            return {
                "success": False,
                "message": "Host profiling is already in progress. Call /stop_host_profile first.",
            }

        # Set defaults
        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            # Default to both CPU and GPU since multimodal preprocessing uses GPU
            activities = ["CPU", "GPU"]

        self.host_profiler_output_dir = Path(output_dir).expanduser()
        self.host_profiler_output_dir.mkdir(parents=True, exist_ok=True)
        self.host_profiler_activities = activities
        # Use provided profile_id or generate one
        self.host_profile_id = profile_id if profile_id else str(time.time())
        self.host_profiler_with_stack = with_stack if with_stack is not None else True
        self.host_profiler_record_shapes = (
            record_shapes if record_shapes is not None else False
        )

        # Set up request counting for auto-stop
        if num_requests is not None:
            self.host_profile_request_count = 0
            self.host_profile_target_request_count = num_requests
        else:
            self.host_profile_target_request_count = None

        # Build profiler activities
        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]

        if torchprof_activities:
            self.host_profiler = torch.profiler.profile(
                activities=torchprof_activities,
                with_stack=self.host_profiler_with_stack,
                record_shapes=self.host_profiler_record_shapes,
            )
            self.host_profiler.start()
            self.host_profile_in_progress = True

            logger.info(
                f"Host profiling started (activities: {activities}). "
                f"Traces will be saved to: {self.host_profiler_output_dir} "
                f"(profile id: {self.host_profile_id})"
            )

            return {
                "success": True,
                "message": f"Host profiling started with activities: {activities}",
            }
        else:
            return {
                "success": False,
                "message": f"No valid activities specified. Valid options: {list(activity_map.keys())}",
            }

    def stop_host_profile(self) -> dict:
        """
        Stop profiling and save the trace.

        Returns:
            dict with success status and message
        """
        if not self.host_profile_in_progress:
            return {
                "success": False,
                "message": "Host profiling is not in progress. Call /start_host_profile first.",
            }

        logger.info("Stopping host profiling...")

        if self.host_profiler is not None:
            try:
                self.host_profiler.stop()

                # Build filename
                filename = f"{self.host_profile_id}-host.trace.json.gz"

                # Ensure output directory exists
                if self.host_profiler_output_dir is not None:
                    self.host_profiler_output_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = os.path.join(self.host_profiler_output_dir, filename)
                else:
                    trace_path = os.path.join("/tmp", filename)

                self.host_profiler.export_chrome_trace(trace_path)

                logger.info(f"Host profiling done. Trace saved to: {trace_path}")

                return {
                    "success": True,
                    "message": f"Host profiling stopped. Trace saved to: {trace_path}",
                }
            except Exception as e:
                logger.warning(f"Error stopping host profiler: {e}")
                return {
                    "success": False,
                    "message": f"Error stopping host profiler: {e}",
                }
            finally:
                self.host_profiler = None
                self.host_profile_in_progress = False
                self.host_profile_request_count = 0
                self.host_profile_target_request_count = None

        self.host_profile_in_progress = False
        return {"success": False, "message": "No active profiler to stop."}

    def _check_host_profile_auto_stop(self):
        """
        Check if we should auto-stop profiling based on request count.
        Call this after processing each request.
        """
        if not self.host_profile_in_progress:
            return

        if self.host_profile_target_request_count is not None:
            self.host_profile_request_count += 1
            if (
                self.host_profile_request_count
                >= self.host_profile_target_request_count
            ):
                logger.info(
                    f"Host profile auto-stopping after {self.host_profile_request_count} requests"
                )
                self.stop_host_profile()

    def get_host_profile_status(self) -> dict:
        """
        Get the current host profiling status.

        Returns:
            dict with profiling status information
        """
        return {
            "in_progress": self.host_profile_in_progress,
            "profile_id": self.host_profile_id,
            "output_dir": (
                str(self.host_profiler_output_dir)
                if self.host_profiler_output_dir
                else None
            ),
            "request_count": self.host_profile_request_count,
            "target_request_count": self.host_profile_target_request_count,
        }
