"""Torch-profiler support for the TokenizerManager process.

The scheduler profiler (``SchedulerProfilerManager``) only traces the scheduler /
worker processes. The tokenizer manager runs in its own process and holds all
pre-scheduler overhead (tokenization, multimodal preprocessing, request
bookkeeping, the asyncio event loop), which is therefore invisible in a normal
profile. This mixin adds an opt-in torch profiler for that process; traces are
written with a ``TKN-0`` rank tag so ``ProfileMerger`` can merge them alongside
the scheduler's ``TP-*`` traces.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


class TokenizerProfilerMixin:
    """Mixin that adds torch profiling support to ``TokenizerManager``."""

    def init_tokenizer_profiler(self):
        """Initialize profiler state. Called from ``TokenizerManager.__init__``."""
        self.tokenizer_torch_profiler = None
        self.tokenizer_profiler_output_dir: Optional[Path] = None
        self.tokenizer_profile_id: Optional[str] = None
        self.tokenizer_profile_in_progress: bool = False

    def start_tokenizer_profile(
        self,
        output_dir: Optional[str] = None,
        activities: Optional[List[str]] = None,
        with_stack: Optional[bool] = None,
        record_shapes: Optional[bool] = None,
        profile_id: Optional[str] = None,
    ) -> bool:
        """Start profiling the tokenizer manager process.

        Args:
            output_dir: Directory to save trace files. Defaults to
                ``SGLANG_TORCH_PROFILER_DIR``.
            activities: Activities to profile. Only "CPU" and "GPU" are
                meaningful here; defaults to ``["CPU"]`` because the tokenizer
                manager does not run device work.
            with_stack: Whether to capture stack traces. Defaults to True.
            record_shapes: Whether to record tensor shapes. Defaults to False.
            profile_id: Identifier shared with the scheduler profile so both
                traces land in the same merge group.

        Returns:
            True if profiling started, False otherwise.
        """
        if self.tokenizer_profile_in_progress:
            logger.warning("Tokenizer profiling already in progress")
            return False

        if output_dir is None:
            output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
        if activities is None:
            activities = ["CPU"]
        if profile_id is None:
            profile_id = str(time.time())

        activity_map = {
            "CPU": torch.profiler.ProfilerActivity.CPU,
            "GPU": torch.profiler.ProfilerActivity.CUDA,
        }
        torchprof_activities = [
            activity_map[a] for a in activities if a in activity_map
        ]
        if not torchprof_activities:
            logger.warning(
                "No valid profiler activities for the tokenizer manager: %s", activities
            )
            return False

        self.tokenizer_profiler_output_dir = Path(output_dir).expanduser()
        self.tokenizer_profiler_output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_profile_id = profile_id

        logger.info(
            "Starting tokenizer manager profiling. Traces will be saved to: %s",
            self.tokenizer_profiler_output_dir,
        )

        self.tokenizer_torch_profiler = torch.profiler.profile(
            activities=torchprof_activities,
            with_stack=with_stack if with_stack is not None else True,
            record_shapes=record_shapes if record_shapes is not None else False,
        )
        self.tokenizer_torch_profiler.start()
        self.tokenizer_profile_in_progress = True

        return True

    def stop_tokenizer_profile(self) -> Optional[str]:
        """Stop profiling the tokenizer manager process.

        Returns:
            Path to the exported trace file, or None if profiling was not active.
        """
        if not self.tokenizer_profile_in_progress:
            return None

        trace_path = None
        if self.tokenizer_torch_profiler is not None:
            self.tokenizer_torch_profiler.stop()

            filename = f"{self.tokenizer_profile_id}-TKN-0.trace.json.gz"
            trace_path = os.path.join(self.tokenizer_profiler_output_dir, filename)

            start = time.perf_counter()
            self.tokenizer_torch_profiler.export_chrome_trace(trace_path)
            trace_size_mb = (
                os.path.getsize(trace_path) / (1024 * 1024)
                if os.path.exists(trace_path)
                else 0.0
            )
            logger.info(
                "Tokenizer manager trace exported in %.2fs (%.1f MB) to: %s",
                time.perf_counter() - start,
                trace_size_mb,
                trace_path,
            )

        self.tokenizer_torch_profiler = None
        self.tokenizer_profile_in_progress = False

        return trace_path
