"""DSV4-Flash 285B MTP performance tests on H200 TP=8.

Manual test (8× H200, 285B FP8 weights). Not registered in CI.
"""

import os
import tempfile
import unittest

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.bench_one_batch_server import run_benchmark as run_one_batch_benchmark
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

DSV4_FLASH_BASE_ENV = {
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_OPT_USE_TOPK_V2": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
}

DSV4_FLASH_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--max-running-requests",
    "8",
]


def _launch_dsv4_flash_server(extra_env=None):
    env = dict(DSV4_FLASH_BASE_ENV)
    if extra_env:
        env.update(extra_env)
    return popen_launch_server(
        DSV4_FLASH_MODEL_PATH,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 4,
        other_args=DSV4_FLASH_SERVER_ARGS,
        env=env,
    )


class TestDSV4FlashMTPSimulatedAcc(CustomTestCase):
    """bs=1 latency at isl=4096 / 900000 with `SGLANG_SIMULATE_ACC_LEN=3`.

    Reference (H200 Flash TP8):
      - isl=4096   → output 258.1 tok/s, accept 2.94
      - isl=900000 → output 222.9 tok/s, accept 2.90
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch_dsv4_flash_server(
            extra_env={"SGLANG_SIMULATE_ACC_LEN": "3"}
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _run_one_batch(self, input_len):
        requests.get(self.base_url + "/flush_cache")
        server_args = ServerArgs(model_path=DSV4_FLASH_MODEL_PATH)
        bench_args = OneBatchBenchArgs(
            run_name=f"dsv4_flash_simacc_isl{input_len}",
            batch_size=(1,),
            input_len=(input_len,),
            output_len=(1024,),
            base_url=self.base_url,
            skip_warmup=True,
            result_filename=os.path.join(
                tempfile.gettempdir(), f"dsv4_flash_simacc_isl{input_len}.jsonl"
            ),
            append_to_github_summary=False,
        )
        results, _ = run_one_batch_benchmark(server_args, bench_args)
        self.assertTrue(results, "bench_one_batch_server returned no results")
        return results[0]

    def test_isl_4096(self):
        r = self._run_one_batch(4096)
        print(
            f"[flash simacc isl=4096] output_throughput={r.output_throughput:.2f} tok/s "
            f"latency={r.latency:.2f}s last_ttft={r.last_ttft:.2f}s "
            f"acc_length={r.acc_length:.2f}"
        )
        # Reference 258.1 tok/s / acc=2.94.
        self.assertGreater(r.output_throughput, 232.0)
        self.assertGreater(r.acc_length, 2.85)

    def test_isl_900k(self):
        r = self._run_one_batch(900_000)
        print(
            f"[flash simacc isl=900k] output_throughput={r.output_throughput:.2f} tok/s "
            f"latency={r.latency:.2f}s last_ttft={r.last_ttft:.2f}s "
            f"acc_length={r.acc_length:.2f}"
        )
        # Reference 222.9 tok/s / acc=2.90.
        self.assertGreater(r.output_throughput, 200.0)
        self.assertGreater(r.acc_length, 2.85)


if __name__ == "__main__":
    unittest.main()
