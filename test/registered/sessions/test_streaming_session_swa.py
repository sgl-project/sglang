import os
import sys
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

# test/ has no __init__.py; add sibling dir so sibling module is importable
# when this file is run as a script via `python3 <path>`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_streaming_session import (  # noqa: E402
    ABORT_REPRO_CHUNKED_PREFILL_SIZE,
    ABORT_REPRO_CONTEXT_LEN,
    ABORT_REPRO_PAGE_SIZE,
    TestStreamingSession,
    TestStreamingSessionAbortLeakRepro,
)

register_cuda_ci(est_time=559, suite="stage-b-test-1-gpu-large")


SWA_MODEL = "openai/gpt-oss-20b"

# Common gpt-oss-20b launch args. Matches TestSessionLatency/TestSWARadixCacheKL.
SWA_COMMON_ARGS = [
    "--mem-fraction-static",
    "0.70",
    "--disable-piecewise-cuda-graph",
]


class TestStreamingSessionSWA(TestStreamingSession):
    """Baseline streaming session on a hybrid-SWA model."""

    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "512",
                    *SWA_COMMON_ARGS,
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionSWARetractLargePage(TestStreamingSession):
    """SWA under retract decode with page=256."""

    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "4096",
                    "--page-size",
                    "256",
                    *SWA_COMMON_ARGS,
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionSWARetractMixedChunk(TestStreamingSession):
    """SWA under retract decode with --enable-mixed-chunk."""

    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_TEST_RETRACT.override(
            True
        ), envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    "128",
                    "--enable-mixed-chunk",
                    *SWA_COMMON_ARGS,
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestStreamingSessionSWAAbortLeakRepro(TestStreamingSessionAbortLeakRepro):
    """SWA abort-heavy chunked prefill leak repro."""

    @classmethod
    def setUpClass(cls):
        cls.model = SWA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(2):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--enable-streaming-session",
                    "--chunked-prefill-size",
                    str(ABORT_REPRO_CHUNKED_PREFILL_SIZE),
                    "--context-length",
                    str(ABORT_REPRO_CONTEXT_LEN),
                    "--page-size",
                    str(ABORT_REPRO_PAGE_SIZE),
                    "--max-running-requests",
                    "32",
                    "--log-level",
                    "info",
                    *SWA_COMMON_ARGS,
                ],
            )
        cls.tokenizer = get_tokenizer(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
