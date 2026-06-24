"""
Usage:
python3 -m unittest test_triton_attention_backend.TestTritonAttentionBackend.test_mla_triton_attention_backend
"""

import unittest
from functools import wraps

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    CustomTestCase,
    run_bench_serving,
)

register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")


def triton_attention_benchmark(extra_args=None, mem_fraction_static="0.84"):
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(self):
            common_args = [
                "--disable-radix-cache",
                "--trust-remote-code",
                "--tp-size",
                "1",
                "--mem-fraction-static",
                str(mem_fraction_static),
                "--context-length",
                "2050",
                "--attention-backend",
                "triton",
            ]
            full_args = common_args + (extra_args or [])

            model = test_func(self)
            res = run_bench_serving(
                model,
                256,
                float("inf"),
                full_args,
                random_input_len=1024,
                random_output_len=1024,
                need_warmup=False,
            )

        return wrapper

    return decorator


class TestTritonAttentionBackend(CustomTestCase):

    @triton_attention_benchmark(
        [
            "--json-model-override-args",
            '{"num_hidden_layers": 4}',
        ],
    )
    def test_mla_triton_attention_backend(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE


if __name__ == "__main__":
    unittest.main()
