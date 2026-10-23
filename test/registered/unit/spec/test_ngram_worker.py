import unittest
from unittest.mock import Mock

from sglang.srt.sampling.sampling_params import (
    REQUEST_NGRAM_CORPUS_SEEDS_KEY,
    REQUEST_NGRAM_CORPUS_SEEDS_SEPARATOR,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_req(rid, flat=None):
    req = Mock(rid=rid)
    req.sampling_params.custom_params = (
        None if flat is None else {REQUEST_NGRAM_CORPUS_SEEDS_KEY: flat}
    )
    return req


class TestCollectCorpusSeeds(CustomTestCase):
    def test_seeds_are_collected_once_in_batch_order(self):
        """A request stays in the decode batch for many steps; collecting on
        every step would insert the same sequences each step. NgramCorpus.batch_put
        takes one flat list of sequences, so seeds are extended, not nested."""
        maybe_stub_sgl_kernel()
        from sglang.srt.speculative.ngram_worker import _collect_corpus_seeds

        reqs = [
            _make_req("a", [1, 2, REQUEST_NGRAM_CORPUS_SEEDS_SEPARATOR, 3]),
            _make_req("b"),
            _make_req("c", [4, 5]),
        ]
        seeds, seeded_rids = _collect_corpus_seeds(reqs=reqs, seeded_rids=set())
        self.assertEqual(seeds, [[1, 2], [3], [4, 5]])
        self.assertEqual(seeded_rids, {"a", "b", "c"})

        seeds, seeded_rids = _collect_corpus_seeds(reqs=reqs, seeded_rids=seeded_rids)
        self.assertEqual(seeds, [])
        self.assertEqual(seeded_rids, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
