"""Unit tests for DCP cascade attention guard logic.

Mirrors the boolean expressions from flashattention_backend.py:
  forward_decode:
    use_cascade_attn = spec_info is not None and topk > 1 and not use_dcp
  forward_extend:
    use_cascade_attn = is_target_verify and topk > 1
                       and not is_swa_layer and dcp_size <= 1
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDCPCascadeGuard(CustomTestCase):
    """Verify cascade attention is disabled when DCP > 1."""

    @staticmethod
    def _decode_cascade(dcp_size, has_spec_info, topk):
        use_dcp = dcp_size > 1
        return has_spec_info and topk > 1 and not use_dcp

    @staticmethod
    def _extend_cascade(dcp_size, is_target_verify, topk, is_swa_layer):
        return is_target_verify and topk > 1 and not is_swa_layer and dcp_size <= 1

    # -- forward_decode truth-table --

    def test_decode_dcp2_spec_topk4_cascade_off(self):
        self.assertFalse(self._decode_cascade(dcp_size=2, has_spec_info=True, topk=4))

    def test_decode_dcp8_spec_topk2_cascade_off(self):
        self.assertFalse(self._decode_cascade(dcp_size=8, has_spec_info=True, topk=2))

    def test_decode_dcp4_spec_topk1_cascade_off(self):
        self.assertFalse(self._decode_cascade(dcp_size=4, has_spec_info=True, topk=1))

    def test_decode_dcp1_spec_topk4_cascade_on(self):
        self.assertTrue(self._decode_cascade(dcp_size=1, has_spec_info=True, topk=4))

    def test_decode_dcp1_no_spec_cascade_off(self):
        self.assertFalse(self._decode_cascade(dcp_size=1, has_spec_info=False, topk=4))

    def test_decode_dcp1_topk0_cascade_off(self):
        self.assertFalse(self._decode_cascade(dcp_size=1, has_spec_info=True, topk=0))

    def test_decode_dcp1_topk1_cascade_off(self):
        self.assertFalse(self._decode_cascade(dcp_size=1, has_spec_info=True, topk=1))

    # -- forward_extend truth-table --

    def test_extend_dcp4_verify_topk2_cascade_off(self):
        self.assertFalse(
            self._extend_cascade(
                dcp_size=4, is_target_verify=True, topk=2, is_swa_layer=False
            )
        )

    def test_extend_dcp2_verify_topk4_cascade_off(self):
        self.assertFalse(
            self._extend_cascade(
                dcp_size=2, is_target_verify=True, topk=4, is_swa_layer=False
            )
        )

    def test_extend_dcp1_verify_topk2_cascade_on(self):
        self.assertTrue(
            self._extend_cascade(
                dcp_size=1, is_target_verify=True, topk=2, is_swa_layer=False
            )
        )

    def test_extend_dcp1_verify_topk2_swa_cascade_off(self):
        self.assertFalse(
            self._extend_cascade(
                dcp_size=1, is_target_verify=True, topk=2, is_swa_layer=True
            )
        )

    def test_extend_dcp1_not_verify_cascade_off(self):
        self.assertFalse(
            self._extend_cascade(
                dcp_size=1, is_target_verify=False, topk=4, is_swa_layer=False
            )
        )

    def test_extend_dcp1_topk1_cascade_off(self):
        self.assertFalse(
            self._extend_cascade(
                dcp_size=1, is_target_verify=True, topk=1, is_swa_layer=False
            )
        )

    # -- exhaustive DCP sizes --

    def test_decode_all_dcp_sizes_block_cascade(self):
        for dcp_size in [2, 3, 4, 8, 16]:
            with self.subTest(dcp_size=dcp_size):
                self.assertFalse(
                    self._decode_cascade(dcp_size=dcp_size, has_spec_info=True, topk=4)
                )

    def test_extend_all_dcp_sizes_block_cascade(self):
        for dcp_size in [2, 3, 4, 8, 16]:
            with self.subTest(dcp_size=dcp_size):
                self.assertFalse(
                    self._extend_cascade(
                        dcp_size=dcp_size,
                        is_target_verify=True,
                        topk=4,
                        is_swa_layer=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
