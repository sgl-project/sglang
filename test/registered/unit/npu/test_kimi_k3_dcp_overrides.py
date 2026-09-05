import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_override_base import _MODEL_OVERRIDE_FNS
from sglang.srt.arg_groups.model_overrides import kimi_k3
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")


class TestKimiK3NpuDcpOverrides(CustomTestCase):
    @staticmethod
    def _args(**changes):
        values = dict(
            dcp_size=2,
            speculative_algorithm=None,
            dcp_replicate_q_proj=False,
            enable_symm_mem=True,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    @patch.object(
        kimi_k3,
        "get_platform",
        return_value=SimpleNamespace(is_npu=True),
    )
    def test_selects_hccl_a2a_without_replicated_q(self, _get_platform):
        self.assertEqual(
            kimi_k3._kimi_k3_overrides(self._args(), hf_config=None),
            {
                "enable_symm_mem": False,
                "dcp_comm_backend": "a2a",
                "dcp_replicate_q_proj": False,
            },
        )

    @patch.object(
        kimi_k3,
        "get_platform",
        return_value=SimpleNamespace(is_npu=True),
    )
    def test_dspark_uses_static_decode_verify(self, _get_platform):
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            self.assertEqual(
                kimi_k3._kimi_k3_overrides(
                    self._args(speculative_algorithm="DSPARK"),
                    hf_config=None,
                ),
                {
                    "enable_symm_mem": False,
                    "speculative_attention_mode": "decode",
                    "dcp_comm_backend": "a2a",
                    "dcp_replicate_q_proj": False,
                },
            )

    @patch.object(
        kimi_k3,
        "get_platform",
        return_value=SimpleNamespace(is_npu=True),
    )
    def test_rejects_unsupported_npu_modes(self, _get_platform):
        with self.assertRaisesRegex(ValueError, "only with DSPARK"):
            kimi_k3._kimi_k3_overrides(
                self._args(speculative_algorithm="EAGLE"),
                hf_config=None,
            )
        with self.assertRaisesRegex(ValueError, "dcp-replicate-q-proj"):
            kimi_k3._kimi_k3_overrides(
                self._args(dcp_replicate_q_proj=True),
                hf_config=None,
            )

    def test_linear_architecture_uses_same_provider(self):
        self.assertIn(
            kimi_k3._kimi_k3_overrides,
            _MODEL_OVERRIDE_FNS["KimiK3LinearForCausalLM"],
        )


if __name__ == "__main__":
    unittest.main()
