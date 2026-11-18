# SPDX-License-Identifier: Apache-2.0
"""NPU trace handling shared by serving and diffusion profilers.

Non-daemon workers keep native analysis. Daemon workers submit a fresh parser
interpreter after capture stops, without waiting for it or importing SGLang in
it. Each parser writes its own log and status and exits when analysis finishes.
"""

import argparse
import contextlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


class NPUTraceHandler:
    """Own NPU profiler patches, callbacks, and capture-stop/analysis policy.

    The public helpers are stateless; callers keep their profiler's lifecycle.
    Imports stay local so the detached parser does not import SGLang or torch
    before reporting its analysis status.
    """

    @staticmethod
    def apply_patches(mode: Literal["diffusion", "srt"]) -> None:
        """Install the existing replacements for one profiler entry point."""
        if mode not in ("diffusion", "srt"):
            raise ValueError(f"Unknown NPU profiler patch mode: {mode!r}")
        from sglang.srt.hardware_backend.npu import get_torch_npu
        from sglang.srt.utils.torch_npu_patch_utils import apply_torch_npu_patches

        torch_npu = get_torch_npu()
        patches = [["profiler.profile", torch_npu.profiler.profile]]
        if mode == "diffusion":
            patches.append(["profiler.schedule", torch_npu.profiler.schedule])
        else:
            patches.extend(
                [
                    [
                        "profiler.ProfilerActivity.CUDA",
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    [
                        "profiler.ProfilerActivity.CPU",
                        torch_npu.profiler.ProfilerActivity.CPU,
                    ],
                ]
            )
        apply_torch_npu_patches(torch_npu, patches)

    @staticmethod
    def get_on_trace_ready(log_dir: str) -> Callable[[Any], None]:
        """Choose native analysis from the process mode, not the workload name."""
        from sglang.srt.hardware_backend.npu import get_torch_npu

        # TODO(https://gitcode.com/Ascend/msprof/issues/265): Remove this
        # daemon-specific background analysis workaround once supported
        # torch_npu versions can automatically parse traces from multiprocessing
        # daemon workers. Native analysis currently creates child processes,
        # which daemon workers cannot spawn.
        return get_torch_npu().profiler.tensorboard_trace_handler(
            log_dir, analyse_flag=not multiprocessing.current_process().daemon
        )

    @staticmethod
    def stop_capture(
        profiler,
        *,
        export_trace: bool = True,
        synchronize: bool = False,
        rank: int | None = None,
        cpu_group=None,
    ) -> None:
        """Stop capture and submit daemon analysis without waiting for parsing.

        A rank opts into coordinated daemon capture-stop errors; an explicit
        CPU group coordinates multiple ranks without using the default group.
        SRT omits both and keeps its existing local stop/barrier behavior.
        ``export_trace=False`` skips coordination and background submission for
        local error cleanup. Native non-daemon callbacks retain their behavior.
        """
        import torch

        daemon = multiprocessing.current_process().daemon
        coordinate_errors = daemon and export_trace and rank is not None
        error = None
        try:
            if synchronize:
                torch.npu.synchronize()
            profiler.stop()
        except Exception as exc:
            if not coordinate_errors:
                raise
            error = f"rank {rank}: {exc}"
            logger.error("NPU profiler stop failed: %s", error)

        if coordinate_errors:
            errors = [error]
            if cpu_group is not None:
                world_size = torch.distributed.get_world_size(group=cpu_group)
                if world_size > 1:
                    # Coordinate capture errors, never background parser status.
                    errors = [None] * world_size
                    torch.distributed.all_gather_object(errors, error, group=cpu_group)
            if any(errors):
                raise RuntimeError(
                    "NPU profiler stop failed: " + "; ".join(e for e in errors if e)
                )

        if daemon and export_trace:
            try:
                status = start_npu_profile_analysis(profiler.prof_if.prof_path)
                logger.info("NPU profile analysis submitted; status: %s", status)
            except Exception:
                # Profiling analysis must not fail an otherwise valid generation.
                logger.exception(
                    "Could not start NPU profile analysis; raw data is retained"
                )


def _trace_path(directory: Path) -> Path:
    trace = directory / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(
            f"NPU analysis produced no nonempty trace_view.json in {directory}"
        )
    return trace


def _write_status(
    directory: Path,
    state: str,
    started_at: float | None,
    *,
    trace: Path | None = None,
    error: str | None = None,
) -> None:
    status = {
        "state": state,
        "started_at": started_at,
        "finished_at": None if state == "running" else time.time(),
        "trace_path": str(trace) if trace is not None else None,
        "error": error,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, prefix=".sglang_analysis-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(status, stream)
            stream.write("\n")
        os.replace(temporary, directory / "sglang_analysis.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def start_npu_profile_analysis(profiler_path: str) -> Path:
    """Submit one rank's capture; the returned status path may not exist yet."""
    if not profiler_path or not Path(profiler_path).is_dir():
        raise RuntimeError(
            f"NPU profiler did not produce a capture directory: {profiler_path!r}"
        )
    directory = Path(profiler_path).resolve()
    log_path = directory / "sglang_analysis.log"
    command = [
        sys.executable,
        "-c",
        # Keep the script directory off sys.path without importing SGLang.
        "import runpy, sys; sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')",
        str(Path(__file__).resolve()),
        str(directory),
    ]
    try:
        with log_path.open("w") as log:
            # No Popen context manager: its __exit__ waits for the parser.
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except Exception as exc:
        with contextlib.suppress(OSError):
            _write_status(
                directory, "failed", None, error=f"Could not start parser: {exc}"
            )
        raise

    # Reap while the worker is alive, but never delay worker shutdown or kill the
    # parser on parent exit. After reparenting, init/subreaper owns its exit status.
    threading.Thread(
        target=process.wait, name="npu-profile-reaper", daemon=True
    ).start()
    return directory / "sglang_analysis.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    started_at = time.time()
    try:
        _write_status(directory, "running", started_at)
        from torch_npu.profiler.profiler import analyse

        analyse(str(directory), max_process_number=1)
        # Native analysis may log errors and return normally. Validate the artifact.
        trace = _trace_path(directory)
        _write_status(directory, "succeeded", started_at, trace=trace)
        print(f"NPU profile analysis succeeded; trace: {trace}", flush=True)
    except BaseException as exc:
        traceback.print_exc()
        with contextlib.suppress(OSError):
            _write_status(
                directory, "failed", started_at, error=f"{type(exc).__name__}: {exc}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
