import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.speculative.spec_tp_sync import (
    SpecTpSync,
    SpecTpSyncSite,
    parse_spec_tp_sync,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

DRAFT = SpecTpSyncSite.DSPARK_DRAFT
ACCEPT_SAMPLE = SpecTpSyncSite.DSPARK_ACCEPT_SAMPLE


def _tp_sync(env_value):
    """A SpecTpSync on rank 1 of 2; ``env_value`` None leaves SGLANG_SPEC_TP_SYNC unset."""
    group = SimpleNamespace(world_size=2, rank_in_group=1)
    with envs.SGLANG_SPEC_TP_SYNC.override("all"):
        if env_value is None:
            envs.SGLANG_SPEC_TP_SYNC.clear()
        else:
            envs.SGLANG_SPEC_TP_SYNC.set(env_value)
        return SpecTpSync(group)


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


class TestRankIdenticalSkip(CustomTestCase):
    def test_unset_env_skips_only_the_accept_sample_site(self):
        sync = _tp_sync(None)
        for site in SpecTpSyncSite:
            with self.subTest(site=site):
                self.assertTrue(sync.enabled(site))
                self.assertEqual(sync.syncs_rank_identical(site), site != ACCEPT_SAMPLE)

    def test_explicit_env_is_followed_exactly(self):
        # "all" set on purpose (e.g. to debug a divergence) keeps every broadcast.
        self.assertTrue(_tp_sync("all").syncs_rank_identical(ACCEPT_SAMPLE))
        self.assertFalse(_tp_sync("rng,-9").syncs_rank_identical(ACCEPT_SAMPLE))
        self.assertTrue(_tp_sync("rng").syncs_rank_identical(ACCEPT_SAMPLE))


if __name__ == "__main__":
    unittest.main()
