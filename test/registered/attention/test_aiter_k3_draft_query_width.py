"""K3 draft metadata heads follow the query width the AITER kernel accepts."""

import unittest

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _native_gqa8_backend():
    backend = AiterAttnBackend.__new__(AiterAttnBackend)
    backend._draft_native_gqa8 = True
    backend.num_head = 8
    backend.num_head_padded = 8
    backend.mla_kernel_num_head_padded = 8
    backend.head_pad_mode = "none"
    backend.head_repeat_factor = 1
    return backend


class TestK3DraftQueryWidth(CustomTestCase):
    def test_qlen4_keeps_the_native_8_head_tile(self):
        backend = _native_gqa8_backend()
        backend.configure_k3_draft_query_width(4)
        self.assertTrue(backend._draft_native_gqa8)
        self.assertEqual(backend.mla_kernel_num_head_padded, 8)
        self.assertEqual(backend.head_pad_mode, "none")

    def test_wider_block_repeats_heads_to_16(self):
        backend = _native_gqa8_backend()
        backend.configure_k3_draft_query_width(7)
        self.assertFalse(backend._draft_native_gqa8)
        self.assertEqual(backend.num_head_padded, 16)
        self.assertEqual(backend.mla_kernel_num_head_padded, 16)
        self.assertEqual(backend.head_pad_mode, "repeat")
        self.assertEqual(backend.head_repeat_factor, 2)

    def test_qlen3_also_leaves_the_8_head_kernel(self):
        backend = _native_gqa8_backend()
        backend.configure_k3_draft_query_width(3)
        self.assertFalse(backend._draft_native_gqa8)
        self.assertEqual(backend.mla_kernel_num_head_padded, 16)

    def test_non_native_backend_is_left_alone(self):
        backend = _native_gqa8_backend()
        backend._draft_native_gqa8 = False
        backend.head_pad_mode = "repeat"
        backend.configure_k3_draft_query_width(7)
        self.assertEqual(backend.mla_kernel_num_head_padded, 8)
        self.assertEqual(backend.head_pad_mode, "repeat")


if __name__ == "__main__":
    unittest.main()
