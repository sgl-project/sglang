import shutil
import subprocess
import time
import unittest

import psutil

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")


class TestTPServerGPUProcesses(CustomTestCase):
    tp_size = 2

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                str(cls.tp_size),
                "--mem-fraction-static",
                "0.70",
                "--disable-cuda-graph",
                "--disable-piecewise-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_tp_server_has_only_worker_gpu_processes(self):
        if shutil.which("nvidia-smi") is None:
            self.skipTest("nvidia-smi is required for GPU process assertions")

        rows = self._wait_for_server_gpu_processes()
        gpu_pids = {row["pid"] for row in rows}

        self.assertNotIn(
            self.process.pid,
            gpu_pids,
            f"server parent process unexpectedly holds a GPU context: "
            f"{self._format_rows(rows)}",
        )
        self.assertEqual(
            len(gpu_pids),
            self.tp_size,
            f"TP={self.tp_size} server should have exactly {self.tp_size} "
            f"GPU worker processes, got {len(gpu_pids)}: {self._format_rows(rows)}",
        )

    def _wait_for_server_gpu_processes(self):
        deadline = time.monotonic() + 60
        stable_since = None
        last_rows = []

        while time.monotonic() < deadline:
            tree_pids = self._server_process_tree_pids()
            rows = [
                row for row in self._query_gpu_processes() if row["pid"] in tree_pids
            ]
            last_rows = rows

            if len({row["pid"] for row in rows}) >= self.tp_size:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 3:
                    return rows
            else:
                stable_since = None

            time.sleep(0.5)

        self.fail(
            f"Timed out waiting for TP={self.tp_size} GPU worker processes. "
            f"Last observed rows: {self._format_rows(last_rows)}"
        )

    def _server_process_tree_pids(self):
        pids = {self.process.pid}
        try:
            parent = psutil.Process(self.process.pid)
            pids.update(child.pid for child in parent.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
        return pids

    def _query_gpu_processes(self):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", maxsplit=2)]
            if len(fields) != 3:
                continue
            gpu_uuid, pid, process_name = fields
            rows.append(
                {
                    "gpu_uuid": gpu_uuid,
                    "pid": int(pid),
                    "process_name": process_name,
                }
            )
        return rows

    def _format_rows(self, rows):
        if not rows:
            return "[]"
        return "[" + ", ".join(str(row) for row in rows) + "]"


if __name__ == "__main__":
    unittest.main()
