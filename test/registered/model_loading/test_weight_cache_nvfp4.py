import os
import unittest

import requests
from test_weight_cache_daemon import (
    cleanup_daemon_files,
    launch_weight_cache_daemon,
    wait_for_daemon_ready,
)

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

MOE_MODEL = DEFAULT_MODEL_NAME_FOR_TEST_MOE_NVFP4

# The first prompt doubles as the output sanity check: greedy decoding of a fact
# this small must contain the answer, so grossly wrong weights show up as a
# failure rather than as a 200 with fluent nonsense.
SANITY_PROMPT = "The capital of France is"
SANITY_EXPECTED = "Paris"

PROMPTS = [
    SANITY_PROMPT,
    "The first three prime numbers are",
    "Water is made of hydrogen and",
]

# Two classes (TP=1, TP=2), each costing a daemon disk load plus a client IPC
# map (the map itself is ~0.3s). The TP=2 class self-skips below 2 GPUs.
register_cuda_ci(est_time=700, stage="base-c", runner_config="4-gpu-b200")


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


class NvFp4WeightCacheBase:
    """Launch a daemon, then client(s) that map its post-processed weights.

    A plain mixin (not a TestCase) so it is not collected on its own; concrete
    classes inherit ``(NvFp4WeightCacheBase, CustomTestCase)`` and set
    ``model_path`` / ``check_no_corruption``.
    """

    model_path: str = None
    tp_size: int = 1
    # A second client (from the same daemon) costs another server launch, so it
    # is opt-in per class.
    check_no_corruption: bool = False

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(cls.model_path)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.daemon_process = None
        # (path, handle) per launched client; closed in tearDownClass, see
        # _run_client for why they cannot be closed any earlier.
        cls._client_logs = []

        common_args = ["--trust-remote-code", "--quantization", "modelopt_fp4"]

        cleanup_daemon_files(cls.tp_size)

        cls.daemon_process = launch_weight_cache_daemon(
            cls.model, cls.tp_size, extra_args=common_args
        )
        wait_for_daemon_ready(cls.daemon_process, cls.tp_size)

        cls.client_outputs, cls.client_log_text = cls._run_client(common_args)

        cls.client2_outputs = None
        if cls.check_no_corruption:
            cls.client2_outputs, _ = cls._run_client(common_args)

    @classmethod
    def _run_client(cls, common_args):
        """Launch a client-mode server, collect greedy outputs + its logs, kill it.

        The log handle deliberately outlives this call. popen_launch_server tails
        the server's stdout/stderr from background threads that run until the
        child exits, so closing the sink here -- while the server is still up --
        makes every later line raise "I/O operation on closed file" in those
        threads. tearDownClass closes them instead, well after the kill.
        """
        log_path = (
            f"/tmp/test_weight_cache_nvfp4_client{len(cls._client_logs)}"
            f"_{os.getpid()}.log"
        )
        log = open(log_path, "w")
        cls._client_logs.append((log_path, log))

        proc = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_args,
                "--tp",
                str(cls.tp_size),
                "--weight-cache-mode",
                "client",
            ],
            return_stdout_stderr=(log, log),
        )
        try:
            outputs = _greedy_outputs(cls.base_url, cls.model)
        finally:
            kill_process_tree(proc.pid)

        log.flush()
        log_text = ""
        if os.path.exists(log_path):
            with open(log_path, errors="replace") as f:
                log_text = f.read()
        return outputs, log_text

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "daemon_process", None) is not None:
            kill_process_tree(cls.daemon_process.pid)
        cleanup_daemon_files(cls.tp_size)
        for log_path, log in getattr(cls, "_client_logs", []):
            try:
                log.close()
            except OSError:
                pass
            if os.path.exists(log_path):
                try:
                    os.unlink(log_path)
                except OSError:
                    pass

    def test_loaded_via_ipc(self):
        self.assertIn(
            "Loaded model via IPC",
            self.client_log_text,
            "client did not load via IPC (likely fell back to disk)",
        )
        self.assertIn(
            "Applied daemon module attributes",
            self.client_log_text,
            "client did not re-apply the daemon's post-processing layout state",
        )

    def test_output_is_sane(self):
        for prompt, text in zip(PROMPTS, self.client_outputs):
            self.assertGreater(len(text.strip()), 0, f"empty output for: {prompt}")
        self.assertIn(
            SANITY_EXPECTED,
            self.client_outputs[0],
            f"greedy completion of {SANITY_PROMPT!r} did not contain "
            f"{SANITY_EXPECTED!r} -- the IPC-mapped weights are likely wrong",
        )

    def test_no_corruption_second_client(self):
        if not self.check_no_corruption:
            self.skipTest("no-corruption check is opt-in per class")
        self.assertEqual(
            self.client2_outputs,
            self.client_outputs,
            "second client diverged from the first -- the daemon's exported "
            "weights were mutated while serving client #1",
        )


@unittest.skipIf(get_device_sm() < 100, "NVFP4 requires CUDA SM 100 or higher")
class TestNvFp4WeightCacheMoE(NvFp4WeightCacheBase, CustomTestCase):
    model_path = MOE_MODEL

    # Enable tp_size > 1 or second client leads to longer test duration
    tp_size = 1
    check_no_corruption = False


if __name__ == "__main__":
    unittest.main(verbosity=3)
