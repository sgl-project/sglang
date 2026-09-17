"""CPU coverage for the Kimi K3 FlashMLA DCP model override."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.model_overrides.kimi_k3 import _kimi_k3_overrides
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKimiK3FlashMLADCPOverride(CustomTestCase):
    def test_selects_trtllm_prefill_and_flashmla_decode(self):
        args = SimpleNamespace(
            _resolved_overrides=[],
            dcp_size=4,
            enable_symm_mem=False,
            speculative_algorithm=None,
            attention_backend=None,
            prefill_attention_backend=None,
            decode_attention_backend="flashmla",
            dcp_replicate_q_proj=None,
            dcp_comm_backend="a2a",
        )

        self.assertEqual(
            _kimi_k3_overrides(args, None),
            {
                "prefill_attention_backend": "trtllm_mla",
                "decode_attention_backend": "flashmla",
                "dcp_replicate_q_proj": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
