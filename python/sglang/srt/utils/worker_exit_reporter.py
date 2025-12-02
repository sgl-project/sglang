import atexit
import logging
import os
import signal
import threading

import requests

from sglang.srt.utils import kill_process_tree

logger = logging.getLogger(__name__)


class WorkerExitReporter:
    def __init__(self):
        self.router_url = None
        self.worker_url = None
        self.delete_called = False
        self.api_key = None
        self.lock = threading.Lock()

    def init_worker_exit_reporter(self, server_args):
        router_url = getattr(server_args, "router_url", None)
        if router_url:
            worker_url = f"http://{server_args.host}:{server_args.port}"
            atexit.register(self._report_exit)
            self.router_url = router_url
            self.worker_url = worker_url
            self.api_key = server_args.api_key

    def set_child_process_exit_signal_handler(self):
        if self.router_url and threading.current_thread() is threading.main_thread():
            logger.info(
                "Registered SIGCHLD handler. When a child process exits, its status will be patched to the router."
            )
            signal.signal(signal.SIGCHLD, self._signal_handler)

    def _signal_handler(self, signum, frame):
        pid = os.getpid()
        logger.warning(
            f"Parent process {pid} received SIGCHLD ({signum}), indicating a child process terminated. "
            f"Initiating shutdown sequence."
        )
        if self.router_url:
            self._report_exit()
        kill_process_tree(pid)

    def _report_exit(self):
        with self.lock:
            if self.delete_called:
                return
            self.delete_called = True
        try:
            # Report that this worker is no longer healthy.
            logger.info(f"Reporting worker exit for {self.worker_url}")
            encoded_url = requests.utils.quote(self.worker_url, safe="")
            headers = (
                {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            )
            response = requests.patch(
                f"{self.router_url}/workers/{encoded_url}/health",
                json={"healthy": False},
                timeout=5,
                headers=headers,
            )
            if response.status_code == 200:
                logger.info(f"Successfully queued worker exit for {self.worker_url}")
            else:
                logger.error(f"Failed to call PATCH endpoint: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to report worker exit: {e}")


worker_exit_reporter = WorkerExitReporter()
