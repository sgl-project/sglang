import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestIndexTopKShareState(unittest.TestCase):
    def test_batch_state_is_noop_when_mtp_reuse_disabled(self):
        batch = SimpleNamespace(reuse_mtp_topk_indices=False, topk_indices="old")
        state = IndexTopKShareState.from_forward_batch(batch)

        self.assertIsNone(state.prev_topk_indices())
        state.store_topk_indices("new")

        self.assertEqual(batch.topk_indices, "old")

    def test_batch_state_carries_topk_when_mtp_reuse_enabled(self):
        batch = SimpleNamespace(reuse_mtp_topk_indices=True, topk_indices="old")
        state = IndexTopKShareState.from_forward_batch(batch)

        self.assertEqual(state.prev_topk_indices(), "old")
        state.store_topk_indices("new")

        self.assertEqual(batch.topk_indices, "new")

    def test_context_manager_clears_batch_state_after_mtp_iteration(self):
        batch = SimpleNamespace(reuse_mtp_topk_indices=False, topk_indices="stale")

        with IndexTopKShareState.enable_mtp_iteration(batch):
            self.assertTrue(batch.reuse_mtp_topk_indices)
            self.assertIsNone(batch.topk_indices)
            batch.topk_indices = "draft-topk"

        self.assertFalse(batch.reuse_mtp_topk_indices)
        self.assertIsNone(batch.topk_indices)

    def test_context_manager_clears_batch_state_on_exception(self):
        batch = SimpleNamespace(reuse_mtp_topk_indices=False, topk_indices=None)

        with self.assertRaises(RuntimeError):
            with IndexTopKShareState.enable_mtp_iteration(batch):
                batch.topk_indices = "draft-topk"
                raise RuntimeError("draft step blew up")

        self.assertFalse(batch.reuse_mtp_topk_indices)
        self.assertIsNone(batch.topk_indices)


if __name__ == "__main__":
    unittest.main()
