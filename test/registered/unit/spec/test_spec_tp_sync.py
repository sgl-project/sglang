import unittest

from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite, parse_spec_tp_sync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DRAFT = SpecTpSyncSite.DSPARK_DRAFT


class TestParseSpecTpSync(CustomTestCase):
    def test_retired_draft_handles_select_the_merged_site(self):
        for token in ("dspark-graph-greedy", "dspark-draft-sample", "2", "6"):
            with self.subTest(token=token):
                self.assertEqual(parse_spec_tp_sync(token), frozenset({DRAFT}))

    def test_negating_a_retired_handle_drops_the_merged_site(self):
        sites = parse_spec_tp_sync("all,-6")
        self.assertNotIn(DRAFT, sites)
        self.assertEqual(sites, frozenset(SpecTpSyncSite) - {DRAFT})

    def test_draft_noise_is_not_an_rng_site(self):
        # Ranks with the same logits draw the same draft noise.
        self.assertNotIn(DRAFT, parse_spec_tp_sync("rng"))
        self.assertIn(DRAFT, parse_spec_tp_sync("dspark-draft"))


if __name__ == "__main__":
    unittest.main()
