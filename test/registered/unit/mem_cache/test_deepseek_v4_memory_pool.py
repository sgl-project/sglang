import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    get_compress_state_ring_size,
    use_speculative_compress_state_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDeepSeekV4CompressStateLayout(unittest.TestCase):
    def test_pd_uses_speculative_compatible_state_layout(self):
        self.assertFalse(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="null",
                    dsv4_compress_state_layout="auto",
                    dsv4_pd_decode_speculative_algorithm=None,
                )
            )
        )
        self.assertTrue(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm="EAGLE",
                    disaggregation_mode="null",
                    dsv4_compress_state_layout="auto",
                    dsv4_pd_decode_speculative_algorithm=None,
                )
            )
        )
        self.assertTrue(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="prefill",
                    dsv4_compress_state_layout="auto",
                    dsv4_pd_decode_speculative_algorithm=None,
                )
            )
        )
        self.assertTrue(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="decode",
                    dsv4_compress_state_layout="auto",
                    dsv4_pd_decode_speculative_algorithm=None,
                )
            )
        )

    def test_prefill_can_follow_decode_speculative_algorithm(self):
        self.assertFalse(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="prefill",
                    dsv4_compress_state_layout="auto",
                    dsv4_pd_decode_speculative_algorithm="none",
                )
            )
        )
        self.assertTrue(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="prefill",
                    dsv4_compress_state_layout="auto",
                    dsv4_pd_decode_speculative_algorithm="EAGLE",
                )
            )
        )

    def test_explicit_layout_overrides_auto(self):
        self.assertTrue(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm=None,
                    disaggregation_mode="null",
                    dsv4_compress_state_layout="speculative",
                    dsv4_pd_decode_speculative_algorithm=None,
                )
            )
        )
        self.assertFalse(
            use_speculative_compress_state_layout(
                SimpleNamespace(
                    speculative_algorithm="EAGLE",
                    disaggregation_mode="decode",
                    dsv4_compress_state_layout="non-speculative",
                    dsv4_pd_decode_speculative_algorithm="EAGLE",
                )
            )
        )

    def test_speculative_layout_doubles_compress_state_ring_size(self):
        self.assertEqual(get_compress_state_ring_size(4, is_speculative=False), 8)
        self.assertEqual(get_compress_state_ring_size(4, is_speculative=True), 16)
        self.assertEqual(get_compress_state_ring_size(128, is_speculative=False), 128)
        self.assertEqual(get_compress_state_ring_size(128, is_speculative=True), 256)


if __name__ == "__main__":
    unittest.main()
