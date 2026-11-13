import os
import unittest

import requests

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

WARNING_MSG = (
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0 is ignored: "
    "host-bound attention paths present; keeping the mirror."
)


class TestNpuNeedsCpuSeqLensFiaWarning(CustomTestCase):
    """SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0 must be ignored under FIA.

    FIA host IntArrays consume the CPU seq_lens mirror, so the Ascend attention
    backend must keep the mirror (its default of 1) and log WARNING_MSG instead
    of honoring the env var. A regression that silently honors the env var (or
    drops the warning) leaves FIA decode without the host mirror it reads, so
    this case guards the ignore-and-warn contract in ascend_backend.py.
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    OUT_LOG_PATH = "./out_log.txt"
    ERR_LOG_PATH = "./err_log.txt"

    @classmethod
    def setUpClass(cls):
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = (
            "expandable_segments:True,pin_memory_expandable_segments:True"
        )
        cls.out_log_file = open(cls.OUT_LOG_PATH, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.ERR_LOG_PATH, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ],
            env={
                "ASCEND_USE_FIA": "1",
                "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            },
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        for attr, path in (
            ("out_log_file", cls.OUT_LOG_PATH),
            ("err_log_file", cls.ERR_LOG_PATH),
        ):
            log_file = getattr(cls, attr, None)
            if log_file is not None:
                log_file.close()
            if os.path.exists(path):
                os.remove(path)

    def test_needs_cpu_seq_lens_ignored_under_fia(self):
        health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(health_resp.status_code, 200)

        # One FIA decode step exercises the host-bound path the mirror feeds.
        gen_resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        self.assertEqual(gen_resp.status_code, 200)

        self.out_log_file.seek(0)
        self.err_log_file.seek(0)
        logs = self.out_log_file.read() + self.err_log_file.read()
        self.assertIn(WARNING_MSG, logs)


if __name__ == "__main__":
    unittest.main()
