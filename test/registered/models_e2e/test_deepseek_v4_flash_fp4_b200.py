"""B200 per-commit CI: DeepSeek-V4-Flash FP4 (LowLatency recipe).

Launches TP=4 with the auto-selected flashinfer_mxfp4 MoE runner and EAGLE
speculative decoding.
Runs 12 ServerSanity probes (correctness, streaming, concurrency, determinism)
plus a GSM8K accuracy gate.

Registry: base-c-test-4-gpu-b200 (per-commit, 4x B200)
"""

import os
import re
import shutil
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=465, stage="base-c", runner_config="4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
MTP_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_DEEPEP_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

_PREFILL_GRAPH_REPLAY_RE = re.compile(
    r"Prefill batch.*#new-token: (?P<num_tokens>[0-9]+),.*cuda graph: True"
)


def _wait_for_prefill_graph_replay(log_paths, offsets, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for path, offset in zip(log_paths, offsets, strict=True):
            with open(path, "rb") as log:
                log.seek(offset)
                for line in log.read().decode(errors="replace").splitlines():
                    match = _PREFILL_GRAPH_REPLAY_RE.search(line)
                    if match:
                        return True
        time.sleep(0.2)
    return False


class TestDSV4FlashFP4B200MTP(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """LowLatency recipe: TP=4, FP4 (mxfp4), EAGLE spec decoding."""

    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 2.8
    bs_1_speed_thres = 220

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MTP_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--chunked-prefill-size",
                "4096",
                "--disable-flashinfer-autotune",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4B200DSpark(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """LowLatency recipe: TP=4, FP4 (mxfp4), EAGLE spec decoding."""

    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 4.0
    bs_1_speed_thres = 300

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--speculative-algorithm",
                "DSPARK",
                "--disable-flashinfer-autotune",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4NonSpecB200(
    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
):
    """Non-MTP recipe: TP=4, DP=4, DeepEP, no speculative decoding."""

    gsm8k_accuracy_thres = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
            ],
            env=_DEEPEP_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4BreakableCudaGraphB200(
    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
):
    """BCG recipe: TP=4, DP=4, DeepEP, DP attention, mixed chunk."""

    gsm8k_accuracy_thres = 0.93

    def test_determinism_temp_zero(self):
        # One request leaves three DP ranks idle. Flush before every trial so
        # every request is a fresh prefill, then require the historically bad
        # logical lengths to replay BCG and return identical tokens,
        # selected-token logprobs, and top-5 distributions. DSV4's scheduler
        # log reports its page-size-aligned physical token count, so the log
        # window proves graph replay without comparing that count to the
        # logical input length.
        for num_tokens in (4, 16):
            input_ids = [1] + list(range(1000, 1000 + num_tokens - 1))
            reference = None
            for trial in range(8):
                flush = requests.post(self.base_url + "/flush_cache", timeout=30)
                self.assertEqual(flush.status_code, 200, flush.text)
                offsets = [os.path.getsize(path) for path in self.log_paths]
                response = requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": input_ids,
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": 64,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                        "return_text_in_logprobs": False,
                        "logprob_start_len": -1,
                        "top_logprobs_num": 5,
                    },
                    timeout=180,
                )
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertNotIn("error", result)
                self.assertTrue(
                    _wait_for_prefill_graph_replay(self.log_paths, offsets),
                    f"no prefill BCG replay logged for {num_tokens}-token "
                    f"trial {trial}",
                )
                meta = result["meta_info"]
                observation = {
                    "output_ids": result["output_ids"],
                    "text": result["text"],
                    "output_token_logprobs": meta["output_token_logprobs"],
                    "output_top_logprobs": meta["output_top_logprobs"],
                }
                if reference is None:
                    reference = observation
                else:
                    self.assertEqual(
                        observation,
                        reference,
                        f"temp-0 fresh prefills diverged at {num_tokens} tokens, "
                        f"trial {trial}",
                    )

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.log_dir = tempfile.mkdtemp(prefix="dsv4_bcg_idle_rank_")
        cls.log_paths = (
            os.path.join(cls.log_dir, "server.out"),
            os.path.join(cls.log_dir, "server.err"),
        )
        cls.stdout = open(cls.log_paths[0], "w")  # noqa: SIM115
        cls.stderr = open(cls.log_paths[1], "w")  # noqa: SIM115
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--enable-mixed-chunk",
                "--cuda-graph-backend-prefill",
                "breakable",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--chunked-prefill-size",
                "4096",
                "--piecewise-cuda-graph-max-tokens",
                "1024",
                "--mem-fraction-static",
                "0.80",
                "--cuda-graph-max-bs-decode",
                "16",
                "--max-running-requests",
                "128",
                "--watchdog-timeout",
                "900",
            ],
            env=_DEEPEP_ENV,
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        for name in ("stdout", "stderr"):
            stream = getattr(cls, name, None)
            if stream is not None and not stream.closed:
                stream.close()
        if hasattr(cls, "log_dir"):
            shutil.rmtree(cls.log_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
