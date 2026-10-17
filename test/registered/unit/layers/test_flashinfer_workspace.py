import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.attention import flashinfer_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashInferGraphWorkspace(CustomTestCase):
    def test_decode_graph_rows_are_bounded_and_optional(self):
        for configured, expected in (
            ([1, 128, 512], 512),
            ([1, 512, 16_384], 10_000),
            ([], 0),
            (None, 0),
        ):
            with self.subTest(configured=configured):
                graph = SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(bs=configured)
                    )
                )
                with (
                    mock.patch.object(
                        flashinfer_backend,
                        "get_exec",
                        return_value=SimpleNamespace(graph=graph),
                    ),
                    mock.patch.object(
                        flashinfer_backend,
                        "get_cuda_graph_max_batch_size",
                        return_value=10_000,
                    ),
                ):
                    self.assertEqual(
                        flashinfer_backend._flashinfer_decode_graph_max_rows(10_000),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
