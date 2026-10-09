import unittest

from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite, parse_spec_tp_sync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSpecTpSyncTokens(CustomTestCase):
    def test_retired_dspark_draft_tokens_select_the_merged_site(self):
        merged = frozenset({SpecTpSyncSite.DSPARK_DRAFT})
        for token in ("3", "6", "dspark-draft-greedy", "dspark_graph_sample"):
            self.assertEqual(parse_spec_tp_sync(token), merged)
        self.assertNotIn(
            SpecTpSyncSite.DSPARK_DRAFT, parse_spec_tp_sync("all,-dspark-draft-sample")
        )


if __name__ == "__main__":
    unittest.main()
