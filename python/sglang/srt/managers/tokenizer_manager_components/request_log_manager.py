from __future__ import annotations

import asyncio
import logging
import os
import pickle
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from sglang.srt.environ import envs
from sglang.srt.managers.tokenizer_manager_components.request_state import ReqState
from sglang.srt.observability.req_time_stats import (
    convert_time_to_realtime,
    real_time,
)
from sglang.srt.utils.cudacore_pyspy_dump_utils import (
    collect_scheduler_processes,
    pyspy_dump_schedulers,
    trigger_cuda_user_coredump,
)

logger = logging.getLogger(__name__)
from typing import List, Tuple

from sglang.srt.observability.request_metrics_exporter import (
    RequestMetricsExporterManager,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.request_logger import RequestLogger


@dataclass(slots=True, kw_only=True)
class RequestLogManager:
    server_args: ServerArgs
    request_logger: RequestLogger
    request_metrics_exporter_manager: RequestMetricsExporterManager
    dump_requests_folder: str = ""
    dump_requests_threshold: int = 1000
    dump_requests_exclude_meta_keys: List[str] = field(
        default_factory=lambda: ["routed_experts", "hidden_states"]
    )
    crash_dump_folder: str = ""
    dump_request_list: List[Tuple] = field(default_factory=list)
    crash_dump_request_list: deque = field(default_factory=deque)
    crash_dump_performed: bool = False

    @classmethod
    def from_server_args(cls, *, server_args: ServerArgs) -> "RequestLogManager":
        request_logger = RequestLogger(
            log_requests=server_args.log_requests,
            log_requests_level=server_args.log_requests_level,
            log_requests_format=server_args.log_requests_format,
            log_requests_target=server_args.log_requests_target,
        )
        _, obj_skip_names, out_skip_names = request_logger.metadata
        request_metrics_exporter_manager = RequestMetricsExporterManager(
            server_args, obj_skip_names, out_skip_names
        )
        return cls(
            server_args=server_args,
            request_logger=request_logger,
            request_metrics_exporter_manager=request_metrics_exporter_manager,
            crash_dump_folder=server_args.crash_dump_folder,
        )

    def dump_requests(self, state: ReqState, out_dict: dict):
        if self.dump_requests_exclude_meta_keys and isinstance(
            out_dict.get("meta_info"), dict
        ):
            exclude = self.dump_requests_exclude_meta_keys
            if any(k in out_dict["meta_info"] for k in exclude):
                filtered_meta = {
                    k: v for k, v in out_dict["meta_info"].items() if k not in exclude
                }
                out_dict = {**out_dict, "meta_info": filtered_meta}

        self.dump_request_list.append(
            (
                state.obj,
                out_dict,
                convert_time_to_realtime(state.time_stats.created_time),
                convert_time_to_realtime(state.time_stats.finished_time),
            )
        )

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            self._dump_data_to_file(
                data_list=self.dump_request_list,
                filename=filename,
                log_message=f"Dump {len(self.dump_request_list)} requests to {filename}",
            )
            self.dump_request_list = []

    def record_request_for_crash_dump(self, state: ReqState, out_dict: dict):
        current_time = real_time()
        self.crash_dump_request_list.append(
            (
                state.obj,
                out_dict,
                convert_time_to_realtime(state.time_stats.created_time),
                current_time,
            )
        )
        # Remove requests older than 5 minutes based on finish time
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300
        ):
            self.crash_dump_request_list.popleft()

    def _dump_data_to_file(
        self,
        data_list: List[Tuple],
        filename: str,
        log_message: str,
    ):
        logger.info(log_message)
        to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_list.copy(),
        }

        def background_task():
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                try:
                    pickle.dump(to_dump_with_server_args, f)
                except Exception as e:
                    # When the server is launched with --trust-remote-code,
                    # server_args sometimes fails to pickle. Retry without
                    # server_args so the request data still gets persisted.
                    logger.error(
                        f"Failed to pickle dump with server_args: {e!r}; "
                        "retrying without server_args"
                    )
                    f.seek(0)
                    f.truncate()
                    to_dump_with_server_args["server_args"] = None
                    pickle.dump(to_dump_with_server_args, f)

        asyncio.create_task(asyncio.to_thread(background_task))

    def dump_requests_before_crash(
        self,
        *,
        rid_to_state: Dict[str, ReqState],
        hostname: str = os.getenv("HOSTNAME", socket.gethostname()),
    ):
        should_dump_pyspy = envs.SGLANG_PYSPY_DUMP_BEFORE_CRASH.get()
        should_dump_cuda_coredump = envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH.get()
        should_dump_diagnostics = should_dump_pyspy or should_dump_cuda_coredump
        if not self.crash_dump_folder and not should_dump_diagnostics:
            return

        if self.crash_dump_performed:
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return
        else:
            self.crash_dump_performed = True

        # Dump requests info
        if self.crash_dump_folder:
            logger.error(f"Dumping requests before crash. {self.crash_dump_folder=}")

            # Add finished requests from crash_dump_request_list
            data_to_dump = []
            if self.crash_dump_request_list:
                data_to_dump.extend(self.crash_dump_request_list)

            # Add unfinished requests from rid_to_state
            unfinished_requests = []
            for rid, state in rid_to_state.items():
                if not state.finished:
                    state.time_stats.set_finished_time()
                    unfinished_requests.append(
                        (
                            state.obj,
                            (
                                state.out_list[-1]
                                if state.out_list
                                else state.get_crash_dump_output()
                            ),
                            convert_time_to_realtime(state.time_stats.created_time),
                            convert_time_to_realtime(state.time_stats.finished_time),
                        )
                    )
            if unfinished_requests:
                data_to_dump.extend(unfinished_requests)

            if data_to_dump:
                # Create a file
                filename = os.path.join(
                    self.crash_dump_folder,
                    hostname,
                    f'crash_dump_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl',
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                # Write the data to the file
                data_to_dump_with_server_args = {
                    "server_args": self.server_args,
                    "requests": data_to_dump,
                    "launch_command": " ".join(sys.argv),
                }
                with open(filename, "wb") as f:
                    try:
                        pickle.dump(data_to_dump_with_server_args, f)
                    except Exception as e:
                        # When the server is launched with --trust-remote-code,
                        # server_args sometimes fails to pickle. Retry without
                        # server_args so the request data still gets persisted.
                        logger.error(
                            f"Failed to pickle dump with server_args: {e!r}; "
                            "retrying without server_args"
                        )
                        f.seek(0)
                        f.truncate()
                        data_to_dump_with_server_args["server_args"] = None
                        pickle.dump(data_to_dump_with_server_args, f)
                logger.error(
                    f"Dumped {len(self.crash_dump_request_list)} finished and {len(unfinished_requests)} unfinished requests before crash to {filename}"
                )

        # Dump pyspy and cuda coredump
        if should_dump_diagnostics:
            logger.info(
                "Sleeping 5 seconds before crash diagnostics to let GPU activity settle."
            )
            time.sleep(5)

            scheduler_procs = collect_scheduler_processes()
            if scheduler_procs:
                if should_dump_pyspy:
                    pyspy_dump_schedulers(scheduler_only=True)

                if should_dump_cuda_coredump:
                    trigger_cuda_user_coredump(scheduler_only=True)
                    cuda_coredump_wait_secs = (
                        envs.SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS.get()
                    )
                    if cuda_coredump_wait_secs > 0:
                        logger.info(
                            "Waiting %.1f seconds for CUDA coredumps before exiting.",
                            cuda_coredump_wait_secs,
                        )
                        time.sleep(cuda_coredump_wait_secs)
            else:
                logger.error(
                    "No live scheduler processes found; skipping py-spy and CUDA coredump."
                )
