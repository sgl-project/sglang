import os
import re
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(
    est_time=500,
    suite="nightly-1-npu-a3",
    nightly=True,
)


class TestAscendCudaGraphGC(unittest.TestCase):
    """
    Testcase: Verify that available memory is larger when --enable-cudagraph-gc is enabled.

    [Test Category] Parameter
    [Test Target] --enable-cudagraph-gc
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.gc_log = "./cudagraph_gc_log.txt"
        cls.server_process = None

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)
        if os.path.exists(cls.gc_log):
            os.remove(cls.gc_log)

    def _launch_and_get_avail_mem(self, enable_cudagraph_gc: bool) -> float:
        other_args = [
            "--trust-remote-code",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.7",
            "--attention-backend",
            "ascend",
        ]

        if enable_cudagraph_gc:
            other_args.append("--enable-cudagraph-gc")

        # Start server and redirect log
        with open(self.gc_log, "w", encoding="utf-8") as f:
            self.server_process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=3600,
                other_args=other_args,
                return_stdout_stderr=(f, f),
            )

        # Parse available memory from log
        with open(self.gc_log, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r"Capture npu graph end\..*avail mem=([\d\.]+) GB", content)
        self.assertIsNotNone(match, "Capture npu graph end log not found")

        avail_mem = float(match.group(1))
        kill_process_tree(self.server_process.pid)
        self.server_process = None

        return avail_mem

    def test_gc_avail_mem_comparison(self):
        # Disable cudagraph GC
        mem_off = self._launch_and_get_avail_mem(enable_cudagraph_gc=False)

        # Enable cudagraph GC
        mem_on = self._launch_and_get_avail_mem(enable_cudagraph_gc=True)

        # Assert available memory increases when GC is enabled
        self.assertGreaterEqual(
            mem_on,
            mem_off,
            f"Available memory with GC should be >= without GC.\n"
            f"Disabled: {mem_off:.2f} GB\nEnabled: {mem_on:.2f} GB",
        )


if __name__ == "__main__":
    unittest.main()
