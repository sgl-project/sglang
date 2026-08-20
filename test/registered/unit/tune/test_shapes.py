import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.tune.shapes import (
    AttnProfile,
    decode_grid,
    parse_decode_key,
    parse_prefill_key,
    prefill_grid,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestShapes(unittest.TestCase):
    def test_grid_sizes_tractable(self):
        # A full sweep must stay in the tens of cells (minutes, not hours).
        self.assertLessEqual(len(decode_grid()), 40)
        self.assertLessEqual(len(prefill_grid()), 20)

    def test_profile_family(self):
        self.assertEqual(AttnProfile(32, 32, 128, "bfloat16").family(), "mha")
        self.assertEqual(AttnProfile(32, 8, 128, "bfloat16").family(), "gqa")
        self.assertEqual(
            AttnProfile(128, 128, 576, "bfloat16", is_mla=True).family(), "mla"
        )

    def test_key_fields_state_parallelism_explicitly(self):
        k = AttnProfile(
            32, 8, 128, "bfloat16", tp_size=4, ep_size=2, dp_size=1
        ).key_fields()
        # topology is explicit, never derived into a shape integer
        self.assertEqual((k["tp"], k["ep"], k["dp"]), (4, 2, 1))

    def test_bucket_key_roundtrip(self):
        for s in decode_grid()[:3]:
            self.assertEqual(parse_decode_key(s.bucket_key()), s)
        for s in prefill_grid()[:3]:
            self.assertEqual(parse_prefill_key(s.bucket_key()), s)


if __name__ == "__main__":
    unittest.main()
