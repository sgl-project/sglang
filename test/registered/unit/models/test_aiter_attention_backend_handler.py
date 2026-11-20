import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.deepseek_common import attention_backend_handler as handler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAiterOverCapacityExtend(CustomTestCase):
    def test_over_capacity_extend_stays_chunked_in_graph_context(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )
        with (
            patch.object(handler, "_support_mha_one_shot", return_value=False),
            patch.object(handler, "is_in_breakable_cuda_graph", return_value=True),
        ):
            method = handler.handle_attention_aiter(None, batch)
        self.assertEqual(method, handler.AttnForwardMethod.MHA_CHUNKED_KV)


if __name__ == "__main__":
    unittest.main()
