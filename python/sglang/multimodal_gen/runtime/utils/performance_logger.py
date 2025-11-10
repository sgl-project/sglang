# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import json
import logging
import os
import subprocess
import time
from datetime import datetime

from dateutil.tz import UTC

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
LOG_DIR = os.path.join(project_root, "logs")

# Configure a specific logger for performance metrics
perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)
perf_logger.propagate = False  # Prevent perf logs from going to the main logger

# Ensure the logs directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Set up a file handler for the performance logger
handler = logging.FileHandler(os.path.join(LOG_DIR, "performance.log"))
handler.setFormatter(logging.Formatter("%(message)s"))
perf_logger.addHandler(handler)


def get_git_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


class PerformanceLogger:
    """
    A utility class for logging performance metrics.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.monotonic()
        self.step_timings = []
        self.commit_hash = get_git_commit_hash()

    def record_step_start(self):
        """Records the start time of a step."""
        self.step_start_time = time.monotonic()

    def record_step_end(self, step_name: str, step_index: int | None = None):
        """Records the end time of a step and calculates the duration."""
        duration = time.monotonic() - self.step_start_time
        self.step_timings.append(
            {"name": step_name, "index": step_index, "duration_ms": duration * 1000}
        )

    def log_total_duration(self, tag: str):
        """Logs the total duration of the operation and all recorded steps."""
        total_duration = time.monotonic() - self.start_time
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": self.request_id,
            "commit_hash": self.commit_hash,
            "tag": tag,
            "total_duration_ms": total_duration * 1000,
            "steps": self.step_timings,
        }
        perf_logger.info(json.dumps(log_entry))
