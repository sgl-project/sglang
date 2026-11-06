import os
import unittest

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuHiCacheHostMemoryFraction(CustomTestCase):
    """Exercise --hicache-host-memory-fraction on Ascend NPU.

    The fraction bounds the HiCache L2 host (CPU RAM) pool budget only when
    neither --hicache-ratio nor --hicache-size is set; an explicit ratio/size
    resolves the fraction to None (auto-sizing off). Sizing itself is
    device-agnostic (psutil + cgroup), so we do not assert an exact byte budget
    here. Instead we prove both ends of the knob work on NPU:

    1. fraction-only (no size/ratio) keeps auto-sizing on: the server boots,
       the host pool is built under the shrunken budget, and L2 reuse still
       works (second identical prompt hits cached_tokens > 0). The
       "HiCache auto-sizing" marker plus "fraction 0.30" prove the value
       reached the sizing path.
    2. an explicit --hicache-size turns auto-sizing off without breaking reuse.

    Numeric budget-scaling (fraction -> per-rank bytes) is covered by the
    device-agnostic CPU unit test test_hicache_auto_size.py.

    [Test Category] HiCache
    [Test Target] --hicache-host-memory-fraction
    """

    model = QWEN3_8B_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    long_text = "What is The capital of France?" * 36

    def _common_args(self):
        return [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--tp-size",
            "1",
            "--enable-hierarchical-cache",
        ]

    def _launch(self, extra_args, tag):
        out_log = open(f"./{tag}_out.log", "w+", encoding="utf-8")
        err_log = open(f"./{tag}_err.log", "w+", encoding="utf-8")
        # ASCEND_USE_FIA=1 makes NPUMHATokenToKVPool expose k_buffer/v_buffer as
        # a per-layer sequence, so the unified HiCache host pool can enumerate
        # device strides. Without it, the single 5D tensor trips `bool(Tensor)`
        # in the host pool's row-stride walk (RuntimeError: Boolean value of
        # Tensor with more than one value is ambiguous).
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=extra_args,
            env={"ASCEND_USE_FIA": "1"},
            return_stdout_stderr=(out_log, err_log),
        )
        return process, out_log, err_log

    def _assert_hicache_reuse(self):
        # Long identical prompt: first request prefill (no hit), second hits L2.
        for i in range(2):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": self.long_text,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 10},
                },
            )
            self.assertEqual(response.status_code, 200)
            cached_tokens = int(response.json()["meta_info"]["cached_tokens"])
            if i == 0:
                self.assertEqual(cached_tokens, 0)
            else:
                self.assertGreater(cached_tokens, 0)

    def _read_logs(self, out_log, err_log):
        out_log.seek(0)
        err_log.seek(0)
        return out_log.read() + err_log.read()

    def _cleanup(self, process, out_log, err_log, tag):
        terminate_and_kill_process_tree(process)
        out_log.close()
        err_log.close()
        os.remove(f"./{tag}_out.log")
        os.remove(f"./{tag}_err.log")

    def test_host_memory_fraction_auto_sizes(self):
        # fraction only (no size/ratio): auto-sizing stays on, L2 still reusable.
        args = self._common_args() + ["--hicache-host-memory-fraction", "0.3"]
        process, out_log, err_log = self._launch(args, "fraction_auto")
        try:
            self._assert_hicache_reuse()
            content = self._read_logs(out_log, err_log)
            self.assertIn("HiCache auto-sizing", content)
            self.assertIn("fraction 0.30", content)
        finally:
            self._cleanup(process, out_log, err_log, "fraction_auto")

    def test_fraction_inactive_with_explicit_size(self):
        # explicit --hicache-size resolves the fraction to None (auto-sizing off).
        args = self._common_args() + [
            "--hicache-host-memory-fraction",
            "0.3",
            "--hicache-size",
            "2",
        ]
        process, out_log, err_log = self._launch(args, "fraction_explicit")
        try:
            self._assert_hicache_reuse()
            content = self._read_logs(out_log, err_log)
            self.assertNotIn("HiCache auto-sizing", content)
        finally:
            self._cleanup(process, out_log, err_log, "fraction_explicit")


if __name__ == "__main__":
    unittest.main()
