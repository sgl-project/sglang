import unittest

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestHealthCheck(CustomTestCase):
    def test_health_check(self):
        """Test that metrics endpoint returns data when enabled"""
        with self.assertRaises(TimeoutError):
            popen_launch_server(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                DEFAULT_URL_FOR_TEST,
                timeout=60,
                other_args=[
                    "--disable-cuda-graph",
                    "--json-model-override-args",
                    '{"architectures": ["LlamaForCausalLMForHealthTest"]}',
                ],
            )


if __name__ == "__main__":
    unittest.main()
