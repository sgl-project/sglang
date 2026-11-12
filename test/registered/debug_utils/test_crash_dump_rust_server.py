"""The embedded Rust server (SGLANG_RUST_SERVER=1) logs requests as the Python
TokenizerManager does (`--log-requests`) and writes the same crash dump when
the scheduler hosting it crashes (`--crash-dump-folder`)."""

import glob
import io
import os
import pickle
import re
import tempfile
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=90, stage="nightly", runner_config="1-gpu-large")


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestRustServerRequestLogAndCrashDump(CustomTestCase):
    MAX_NEW_TOKENS = 4
    NUM_REQUESTS_BEFORE_CRASH = 5

    @classmethod
    def setUpClass(cls):
        cls.crash_dump_folder = tempfile.mkdtemp(prefix="crash_dump_rust_test_")
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        with envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.override(
            cls.NUM_REQUESTS_BEFORE_CRASH * cls.MAX_NEW_TOKENS + 10
        ):
            cls.process = popen_launch_server(
                "Qwen/Qwen3-0.6B",
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                env={"SGLANG_RUST_SERVER": "1"},
                other_args=[
                    "--crash-dump-folder",
                    cls.crash_dump_folder,
                    "--log-requests",
                    "--log-requests-level",
                    "1",
                    "--skip-server-warmup",
                ],
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_request_log_and_crash_dump(self):
        rids = [f"rust-crash-{i}" for i in range(self.NUM_REQUESTS_BEFORE_CRASH * 2)]
        for rid in rids:
            try:
                requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "rid": rid,
                        "text": f"Hello, this is request {rid}.",
                        "sampling_params": {
                            "max_new_tokens": self.MAX_NEW_TOKENS,
                            "temperature": 0,
                        },
                    },
                    headers={"x-smg-routing-key": f"key-{rid}"},
                    timeout=30,
                )
            except requests.exceptions.RequestException:
                pass  # the server is down after the crash

        dump_pattern = os.path.join(self.crash_dump_folder, "*", "crash_dump_*.pkl")
        deadline = time.time() + 60
        while not glob.glob(dump_pattern) and time.time() < deadline:
            time.sleep(1)
        dump_files = glob.glob(dump_pattern)
        self.assertEqual(len(dump_files), 1, f"no single dump in {dump_pattern}")
        with open(dump_files[0], "rb") as f:
            dump = pickle.load(f)

        self.assertIsNotNone(dump["server_args"])
        records = dump["requests"]
        self.assertGreater(len(records), self.NUM_REQUESTS_BEFORE_CRASH)
        finished = [
            r for r in records if r[1].get("meta_info", {}).get("finish_reason")
        ]
        self.assertGreaterEqual(len(finished), self.NUM_REQUESTS_BEFORE_CRASH)
        for obj, out, created_time, finished_time in finished:
            self.assertIsInstance(obj, GenerateReqInput)
            self.assertIn(obj.rid, rids)
            self.assertEqual(out["meta_info"]["id"], obj.rid)
            self.assertEqual(obj.sampling_params["max_new_tokens"], self.MAX_NEW_TOKENS)
            self.assertLessEqual(created_time, finished_time)

        # Every finished request in the dump was logged as Python logs it.
        log = self.stdout.getvalue()
        for obj, *_ in finished:
            receive = re.search(
                rf"^\[[\d\- :]+\] Receive: obj=GenerateReqInput\(rid='{obj.rid}', "
                r".*sampling_params=\{'max_new_tokens': 4, 'temperature': 0\}, .*\)"
                rf", headers=\{{'x-smg-routing-key': 'key-{obj.rid}'\}}$",
                log,
                re.MULTILINE,
            )
            self.assertIsNotNone(receive, f"no Receive line for {obj.rid}")
            finish = re.search(
                rf"^\[[\d\- :]+\] Finish: obj=GenerateReqInput\(rid='{obj.rid}', .*"
                rf"out=\{{'meta_info': \{{'id': '{obj.rid}', .*'finish_reason': "
                r"\{'type': 'length', 'length': 4\}",
                log,
                re.MULTILINE,
            )
            self.assertIsNotNone(finish, f"no Finish line for {obj.rid}")


if __name__ == "__main__":
    unittest.main()
