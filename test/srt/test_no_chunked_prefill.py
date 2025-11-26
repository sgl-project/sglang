import unittest

from sglang.srt.utils import is_npu
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_bench_serving,
    run_mmlu_test,
)


class TestNoChunkedPrefill(CustomTestCase):

    def test_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, enable_mixed_chunk=False, chunked_prefill_size=-1
        )

    def test_no_chunked_prefill_without_radix_cache(self):
        model = (
            "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
            if is_npu()
            else DEFAULT_MODEL_NAME_FOR_TEST
        )
        other_args = (
            [
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else ["--disable-radix-cache", "--chunked-prefill-size", "-1"]
        )
        res = run_bench_serving(
            model=model,
            num_prompts=10,
            request_rate=float("inf"),
            other_server_args=other_args,
        )

        assert res["completed"] == 10


if __name__ == "__main__":
    unittest.main()
