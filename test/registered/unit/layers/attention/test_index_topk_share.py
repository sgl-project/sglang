import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _batch(reuse: bool, carried) -> SimpleNamespace:
    return SimpleNamespace(
        reuse_mtp_topk_indices=reuse,
        spec_info=SimpleNamespace(mtp_topk_indices=carried),
    )


class TestIndexTopKShareState(unittest.TestCase):
    def test_batch_state_is_noop_when_mtp_reuse_disabled(self):
        batch = _batch(reuse=False, carried="old")
        state = IndexTopKShareState.from_forward_batch(batch)

        self.assertIsNone(state.prev_topk_indices())
        state.store_topk_indices("new")

        self.assertEqual(batch.spec_info.mtp_topk_indices, "old")

    def test_batch_state_carries_topk_when_mtp_reuse_enabled(self):
        batch = _batch(reuse=True, carried="old")
        state = IndexTopKShareState.from_forward_batch(batch)

        self.assertEqual(state.prev_topk_indices(), "old")
        state.store_topk_indices("new")

        self.assertEqual(batch.spec_info.mtp_topk_indices, "new")

    def test_context_manager_clears_batch_state_after_mtp_iteration(self):
        batch = _batch(reuse=False, carried="stale")

        with IndexTopKShareState.enable_mtp_iteration(batch):
            self.assertTrue(batch.reuse_mtp_topk_indices)
            self.assertIsNone(batch.spec_info.mtp_topk_indices)
            batch.spec_info.mtp_topk_indices = "draft-topk"

        self.assertFalse(batch.reuse_mtp_topk_indices)
        self.assertIsNone(batch.spec_info.mtp_topk_indices)

    def test_context_manager_clears_batch_state_on_exception(self):
        batch = _batch(reuse=False, carried=None)

        with self.assertRaises(RuntimeError):
            with IndexTopKShareState.enable_mtp_iteration(batch):
                batch.spec_info.mtp_topk_indices = "draft-topk"
                raise RuntimeError("draft step blew up")

        self.assertFalse(batch.reuse_mtp_topk_indices)
        self.assertIsNone(batch.spec_info.mtp_topk_indices)


if __name__ == "__main__":
    unittest.main()
