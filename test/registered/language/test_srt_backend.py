import unittest

import sglang as sgl
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_programs import (
    test_decode_int,
    test_decode_json_regex,
    test_dtype_gen,
    test_expert_answer,
    test_few_shot_qa,
    test_gen_min_new_tokens,
    test_hellaswag_select,
    test_mt_bench,
    test_parallel_decoding,
    test_regex,
    test_select,
    test_stream,
    test_stream_logprobs,
    test_tool_use,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=79, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=120, suite="stage-a-test-1-gpu-small-amd")


class TestSRTBackend(CustomTestCase):
    backend = None
    process = None

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--incremental-streaming-output",
                "--log-level",
                "info",
                "--enable-metrics",
            ],
        )
        cls.backend = sgl.RuntimeEndpoint(DEFAULT_URL_FOR_TEST)
        sgl.set_default_backend(cls.backend)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_few_shot_qa(self):
        test_few_shot_qa()

    def test_mt_bench(self):
        test_mt_bench()

    def test_select(self):
        test_select(check_answer=False)

    def test_decode_int(self):
        test_decode_int()

    @unittest.skip("Skip this flaky test.")
    def test_decode_json_regex(self):
        test_decode_json_regex()

    def test_expert_answer(self):
        test_expert_answer()

    def test_tool_use(self):
        test_tool_use()

    def test_parallel_decoding(self):
        test_parallel_decoding()

    def test_stream(self):
        test_stream()

    def test_stream_logprobs(self):
        test_stream_logprobs()

    def test_regex(self):
        test_regex()

    def test_dtype_gen(self):
        test_dtype_gen()

    def test_hellaswag_select(self):
        # Run twice to capture more bugs
        for _ in range(2):
            accuracy, latency = test_hellaswag_select()
            self.assertGreater(accuracy, 0.60)

    def test_gen_min_new_tokens(self):
        test_gen_min_new_tokens()


if __name__ == "__main__":
    unittest.main()
