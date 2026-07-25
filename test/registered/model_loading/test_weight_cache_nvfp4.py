"""NVFP4 (modelopt_fp4) weight-cache correctness: raw_client_postprocess mode.

NVFP4's process_weights_after_loading repacks/swizzles weights, creates derived
params, and stamps Python-side layout attributes that raw-tensor CUDA IPC cannot
carry. So the weight cache shares NVFP4 in "raw_client_postprocess" mode: the
daemon exports the raw pre-post-process quantized tensors, and each client
re-runs process_weights_after_loading locally after IPC-mapping them
(weight_cache/ipc_loader.py). This test guards that path end to end.

Coverage (dense ModelOptFp4LinearMethod + MoE ModelOptNvFp4FusedMoEMethod):
  1. IPC path actually ran (log marker), in raw_client_postprocess mode.
  2. Numerical parity: greedy output from the daemon+client equals a normal
     disk load. This is the primary gate against silently-wrong weights.
  3. No corruption of the daemon's shared memory: a SECOND client mapped from the
     same still-alive daemon also matches the disk baseline. If the first
     client's local post-processing had written in place into the daemon's
     IPC-shared raw weights, the second client would diverge. (The loader also
     hard-errors on such a write via _assert_ipc_tensors_unmutated; this is the
     black-box end-to-end backstop.)

Requires SM100+ (NVFP4) and the NVFP4 checkpoints on disk / in cache; skipped
otherwise.
"""

import os
import subprocess
import sys
import time
import unittest

import requests

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.srt.weight_cache.protocol import (
    compute_global_rank,
    get_ready_path,
    get_socket_path,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

DENSE_MODEL = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
MOE_MODEL = DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4  # nvidia/Qwen3-30B-A3B-FP4
TP_SIZE = 1
GLOBAL_RANK = compute_global_rank(tp_size=TP_SIZE, pp_rank=0, tp_rank=0)

PROMPTS = [
    "The capital of France is",
    "The first three prime numbers are",
    "Water is made of hydrogen and",
]

register_cuda_ci(est_time=600, stage="base-c", runner_config="4-gpu-b200")


def _greedy_outputs(base_url, model):
    """Deterministic (temperature 0) completions for the fixed prompt set."""
    outputs = []
    for prompt in PROMPTS:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 32,
                "temperature": 0,
            },
        )
        assert resp.status_code == 200, resp.text
        outputs.append(resp.json()["choices"][0]["text"])
    return outputs


class NvFp4WeightCacheParityBase:
    """Launch a disk baseline, then a daemon + client(s), and compare outputs.

    A plain mixin (not a TestCase) so it is not collected on its own; concrete
    classes inherit ``(NvFp4WeightCacheParityBase, CustomTestCase)`` and set
    ``model_path`` / ``check_no_corruption``.
    """

    model_path: str = None
    # A second client (from the same daemon) is expensive, so only the dense
    # class runs the no-corruption check; MoE runs parity only.
    check_no_corruption: bool = False

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(cls.model_path)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.socket_path = get_socket_path(GLOBAL_RANK)
        cls.ready_path = get_ready_path(GLOBAL_RANK)
        cls.daemon_process = None
        cls._client_log = f"/tmp/test_weight_cache_nvfp4_client_{os.getpid()}.log"

        common_args = ["--trust-remote-code", "--quantization", "modelopt_fp4"]

        # 1) Disk baseline (weight cache off) -> reference greedy outputs.
        disk_proc = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[*common_args, "--tp", str(TP_SIZE)],
        )
        try:
            cls.disk_outputs = _greedy_outputs(cls.base_url, cls.model)
        finally:
            kill_process_tree(disk_proc.pid)

        cls._cleanup_daemon_files()

        # 2) Launch the daemon (raw_client_postprocess is auto-selected for
        #    modelopt_fp4) and wait for it to finish loading.
        cls.daemon_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                cls.model,
                "--tp-size",
                str(TP_SIZE),
                "--trust-remote-code",
                "--quantization",
                "modelopt_fp4",
            ]
        )
        cls._wait_daemon_ready()

        # 3) Client #1: loads via IPC + local post-processing.
        cls.client_outputs, cls.client_log_text = cls._run_client(common_args)

        # 4) Client #2 from the same still-alive daemon (no-corruption backstop).
        cls.client2_outputs = None
        if cls.check_no_corruption:
            cls.client2_outputs, _ = cls._run_client(common_args)

    @classmethod
    def _run_client(cls, common_args):
        """Launch a client-mode server, collect greedy outputs + its logs, kill it."""
        with open(cls._client_log, "w") as log:
            proc = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    *common_args,
                    "--tp",
                    str(TP_SIZE),
                    "--weight-cache-mode",
                    "client",
                ],
                return_stdout_stderr=(log, log),
            )
        try:
            outputs = _greedy_outputs(cls.base_url, cls.model)
        finally:
            kill_process_tree(proc.pid)
        log_text = ""
        if os.path.exists(cls._client_log):
            with open(cls._client_log, errors="replace") as f:
                log_text = f.read()
        return outputs, log_text

    @classmethod
    def _wait_daemon_ready(cls):
        start = time.time()
        while not os.path.exists(cls.ready_path):
            if cls.daemon_process.poll() is not None:
                raise RuntimeError(
                    f"Weight cache daemon exited prematurely with code "
                    f"{cls.daemon_process.returncode}"
                )
            if time.time() - start > DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH:
                raise TimeoutError("NVFP4 weight cache daemon not ready in time")
            time.sleep(2)

    @classmethod
    def _cleanup_daemon_files(cls):
        for path in (cls.socket_path, cls.ready_path):
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "daemon_process", None) is not None:
            kill_process_tree(cls.daemon_process.pid)
        cls._cleanup_daemon_files()
        client_log = getattr(cls, "_client_log", None)
        if client_log and os.path.exists(client_log):
            try:
                os.unlink(client_log)
            except OSError:
                pass

    def test_loaded_via_ipc_raw_mode(self):
        """The client loaded over IPC in raw_client_postprocess mode (not disk)."""
        self.assertIn(
            "Loaded model via IPC",
            self.client_log_text,
            "client did not load via IPC (likely fell back to disk)",
        )
        self.assertIn(
            "raw_client_postprocess",
            self.client_log_text,
            "client did not run the raw_client_postprocess path for NVFP4",
        )

    def test_parity_with_disk_load(self):
        """Daemon+client greedy output must equal a normal disk load."""
        self.assertEqual(
            self.client_outputs,
            self.disk_outputs,
            "IPC (raw_client_postprocess) output diverged from disk load",
        )

    def test_no_corruption_second_client(self):
        """A second client from the same daemon must also match disk.

        If client #1's local post-processing wrote in place into the daemon's
        IPC-shared raw weights, client #2 would read corrupted weights and
        diverge from the disk baseline.
        """
        if not self.check_no_corruption:
            self.skipTest("no-corruption check runs only for the dense model")
        self.assertEqual(
            self.client2_outputs,
            self.disk_outputs,
            "second client diverged from disk -- daemon raw weights were "
            "corrupted by the first client's post-processing",
        )


@unittest.skipIf(get_device_sm() < 100, "NVFP4 requires CUDA SM 100 or higher")
class TestNvFp4WeightCacheDense(NvFp4WeightCacheParityBase, CustomTestCase):
    model_path = DENSE_MODEL
    check_no_corruption = True


@unittest.skipIf(get_device_sm() < 100, "NVFP4 requires CUDA SM 100 or higher")
class TestNvFp4WeightCacheMoE(NvFp4WeightCacheParityBase, CustomTestCase):
    model_path = MOE_MODEL
    check_no_corruption = False


if __name__ == "__main__":
    unittest.main(verbosity=3)
