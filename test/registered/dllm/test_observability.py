from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=500, suite="stage-b-test-1-gpu-large")

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_dllm_observability import (
    DllmObservabilityMixin,
    build_single_prompt,
    build_two_prompts,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

BASE_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.9",
    "--max-running-requests",
    "2",
    "--attention-backend",
    "flashinfer",
    "--cuda-graph-bs",
    "1",
    "2",
    "--disable-radix-cache",
]
INCREMENTAL_OBSERVABILITY_SERVER_ARGS = BASE_SERVER_ARGS + [
    "--incremental-streaming-output",
]
STREAM_INTERVALS = (None, 2, 3)


class ObservabilityServerBase(DllmObservabilityMixin, CustomTestCase):
    dllm_algorithm: str | None = None
    dllm_algorithm_config_path: str | None = None
    incremental_streaming_output = False
    server_args = BASE_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        if cls.dllm_algorithm is None:
            raise unittest.SkipTest("Skip the base observability test class")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls._build_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @classmethod
    def _build_server_args(cls) -> list[str]:
        args = list(cls.server_args)
        args.extend(["--dllm-algorithm", cls.dllm_algorithm])
        if cls.dllm_algorithm_config_path is not None:
            args.extend(["--dllm-algorithm-config", cls.dllm_algorithm_config_path])
        return args

    def _assert_forward_counts(self, prompts: list[str]):
        for stream_interval in STREAM_INTERVALS:
            label = "default" if stream_interval is None else str(stream_interval)
            with self.subTest(stream_interval=label):
                self.assert_generate_stream_cumulative_matches_non_stream(
                    base_url=self.base_url,
                    prompts=prompts,
                    stream_interval=stream_interval,
                    incremental_streaming_output=self.incremental_streaming_output,
                )

    def test_single_request_forward_counts(self):
        self._assert_forward_counts(build_single_prompt())

    def test_two_request_forward_counts(self):
        self._assert_forward_counts(build_two_prompts())


class TestSDARLowConfidenceObservability(ObservabilityServerBase):
    dllm_algorithm = "LowConfidence"
    model = "JetLM/SDAR-8B-Chat"


class TestSDARJointThresholdObservability(ObservabilityServerBase):
    dllm_algorithm = "JointThreshold"
    model = "JetLM/SDAR-8B-Chat"


class TestSDARLowConfidenceIncrementalObservability(ObservabilityServerBase):
    dllm_algorithm = "LowConfidence"
    incremental_streaming_output = True
    model = "JetLM/SDAR-8B-Chat"
    server_args = INCREMENTAL_OBSERVABILITY_SERVER_ARGS


class TestSDARJointThresholdIncrementalObservability(ObservabilityServerBase):
    dllm_algorithm = "JointThreshold"
    incremental_streaming_output = True
    model = "JetLM/SDAR-8B-Chat"
    server_args = INCREMENTAL_OBSERVABILITY_SERVER_ARGS


class TestLLaDALowConfidenceObservability(ObservabilityServerBase):
    dllm_algorithm = "LowConfidence"
    model = "inclusionAI/LLaDA2.0-mini"


class TestLLaDAJointThresholdObservability(ObservabilityServerBase):
    dllm_algorithm = "JointThreshold"
    model = "inclusionAI/LLaDA2.0-mini"


class TestLLaDALowConfidenceIncrementalObservability(ObservabilityServerBase):
    dllm_algorithm = "LowConfidence"
    incremental_streaming_output = True
    model = "inclusionAI/LLaDA2.0-mini"
    server_args = INCREMENTAL_OBSERVABILITY_SERVER_ARGS


class TestLLaDAJointThresholdIncrementalObservability(ObservabilityServerBase):
    dllm_algorithm = "JointThreshold"
    incremental_streaming_output = True
    model = "inclusionAI/LLaDA2.0-mini"
    server_args = INCREMENTAL_OBSERVABILITY_SERVER_ARGS


if __name__ == "__main__":
    unittest.main()
