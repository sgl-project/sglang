import json
import os
import select
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="2-gpu-large")


_HELPER_SCRIPT = r"""
import json
import os
import signal
import subprocess
import sys
import time
import types


children = []


WORKER_SCRIPT = r'''
import sys
import time

import torch


device = int(sys.argv[1])
torch.cuda.set_device(device)
tensor = torch.empty((1,), device=f"cuda:{device}")

while True:
    _ = tensor
    time.sleep(1)
'''


def install_fake_deep_gemm():
    deep_gemm = types.ModuleType("deep_gemm")
    deep_gemm.__path__ = []

    def set_pdl(enabled):
        if not enabled:
            return
        import torch

        torch.cuda.set_device(0)
        deep_gemm._pdl_tensor = torch.empty((1,), device="cuda")

    deep_gemm.set_pdl = set_pdl

    utils = types.ModuleType("deep_gemm.utils")
    utils.__path__ = []
    layout = types.ModuleType("deep_gemm.utils.layout")
    layout.get_mn_major_tma_aligned_tensor = lambda *args, **kwargs: None

    sys.modules["deep_gemm"] = deep_gemm
    sys.modules["deep_gemm.utils"] = utils
    sys.modules["deep_gemm.utils.layout"] = layout


def cleanup(*_args):
    for proc in children:
        proc.terminate()
    for proc in children:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)

install_fake_deep_gemm()
from sglang.srt.layers.deep_gemm_wrapper import entrypoint  # noqa: F401,E402

for device in range(2):
    children.append(
        subprocess.Popen(
            [sys.executable, "-u", "-c", WORKER_SCRIPT, str(device)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    )

print(
    json.dumps({"parent_pid": os.getpid(), "child_pids": [p.pid for p in children]}),
    flush=True,
)

while True:
    time.sleep(1)
"""


class TestDeepGemmPDLProcessIsolation(CustomTestCase):
    def test_deep_gemm_pdl_does_not_create_parent_gpu_process(self):
        self._require_two_cuda_devices_and_nvidia_smi()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self._first_two_visible_devices(env)
        env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "1"
        env["SGLANG_DEEPGEMM_PDL"] = "1"

        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", textwrap.dedent(_HELPER_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )

        stdout_lines = []
        try:
            process_info = self._read_helper_process_info(proc, stdout_lines)
            expected_worker_pids = set(process_info["child_pids"])
            parent_pid = process_info["parent_pid"]

            observed_pids = self._wait_for_compute_pids(
                parent_pid, expected_worker_pids
            )
            missing_worker_pids = expected_worker_pids - observed_pids
            extra_parent_pids = {parent_pid} & observed_pids

            self.assertFalse(
                extra_parent_pids,
                f"DeepGEMM PDL created an extra parent GPU process: "
                f"{sorted(extra_parent_pids)}; expected only workers "
                f"{sorted(expected_worker_pids)}; observed={sorted(observed_pids)}",
            )
            self.assertFalse(
                missing_worker_pids,
                f"worker GPU processes did not appear in nvidia-smi: "
                f"{sorted(missing_worker_pids)}; observed={sorted(observed_pids)}",
            )
        finally:
            self._terminate_helper(proc)

    def _require_two_cuda_devices_and_nvidia_smi(self):
        if shutil.which("nvidia-smi") is None:
            self.skipTest("nvidia-smi is required for GPU process assertions")

        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("test requires at least two visible CUDA devices")

    def _first_two_visible_devices(self, env):
        visible_devices = env.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            devices = [device.strip() for device in visible_devices.split(",")]
            devices = [device for device in devices if device]
            if len(devices) >= 2:
                return ",".join(devices[:2])
        return "0,1"

    def _read_helper_process_info(self, proc, stdout_lines):
        deadline = time.monotonic() + 30
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                _, stderr = proc.communicate(timeout=5)
                self.fail(
                    f"helper exited before reporting process info "
                    f"(code={proc.returncode}); stdout={stdout_lines}; stderr={stderr}"
                )

            readable, _, _ = select.select([proc.stdout], [], [], 0.5)
            if not readable:
                continue

            line = proc.stdout.readline()
            if not line:
                continue
            stdout_lines.append(line.rstrip())
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

        self.fail(f"timed out waiting for helper process info; stdout={stdout_lines}")

    def _wait_for_compute_pids(self, parent_pid, worker_pids):
        candidate_pids = {parent_pid, *worker_pids}
        deadline = time.monotonic() + 30
        observed_pids = set()
        workers_seen_at = None
        while time.monotonic() < deadline:
            observed_pids |= self._query_compute_pids() & candidate_pids
            if parent_pid in observed_pids:
                return observed_pids
            if worker_pids <= observed_pids:
                if workers_seen_at is None:
                    workers_seen_at = time.monotonic()
                elif time.monotonic() - workers_seen_at >= 1:
                    return observed_pids
            else:
                workers_seen_at = None
            time.sleep(0.25)
        return observed_pids

    def _query_compute_pids(self):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pids = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                pids.add(int(line))
        return pids

    def _terminate_helper(self, proc):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
