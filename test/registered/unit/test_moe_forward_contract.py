from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt import runtime_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(0.1, "base-a-test-cpu")


class TestMoeForwardContract(CustomTestCase):
    def setUp(self):
        runtime_context.get_resources().buffers.pop("moe:forward_contract", None)

    def tearDown(self):
        runtime_context.get_resources().buffers.pop("moe:forward_contract", None)

    def test_uses_aggregate_prefill_and_speculative_decode_bounds(self):
        schedule = SimpleNamespace(
            max_prefill_tokens=16384,
            chunked_prefill_size=8192,
            max_running_requests=2,
        )
        graph = SimpleNamespace(decode=SimpleNamespace(max_bs=64))
        spec = SimpleNamespace(
            speculative_algorithm="DSPARK",
            max_speculative_num_draft_tokens=3,
        )
        model_config = SimpleNamespace(is_multimodal=False)

        with (
            patch.object(runtime_context, "get_schedule", return_value=schedule),
            patch.object(
                runtime_context,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(cuda_graph_config=graph)
                ),
            ),
            patch.object(runtime_context, "get_spec", return_value=spec),
            patch.object(
                runtime_context, "process_model_config", return_value=model_config
            ),
        ):
            contract = runtime_context.get_moe_forward_contract()

        self.assertEqual(contract.prefill_base_rows, 16384)
        self.assertEqual(contract.prefill_rows, 16384)
        self.assertEqual(contract.decode_rows, 192)
        self.assertEqual(contract.max_rows, 16384)


if __name__ == "__main__":
    import unittest

    unittest.main()
