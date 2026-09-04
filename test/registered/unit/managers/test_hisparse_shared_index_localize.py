"""Unit tests for HiSparse's model-global -> pool-local skip-pattern slicing.

`resolve_shared_index_layers` returns a pattern indexed by model-global layer
id, but every per-layer structure in `HiSparseCoordinator` is sized to this
rank's KV pool slice. `_localize_shared_index_layers` bridges the two, and must
bail out (-> synchronous swap-in) rather than mis-index when the mapping does
not hold.
"""

import unittest

from sglang.srt.managers.hisparse_coordinator import _localize_shared_index_layers
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# Anchor / skip / anchor / skip ... as produced by index_topk_freq=2.
ALTERNATING = [False, True] * 4


class TestLocalizeSharedIndexLayers(unittest.TestCase):
    def test_single_stage_is_identity(self):
        """pp_size=1: start_layer=0 and the whole pattern is this rank's slice."""
        self.assertEqual(
            _localize_shared_index_layers(ALTERNATING, start_layer=0, layer_num=8),
            ALTERNATING,
        )

    def test_later_pp_stage_is_sliced(self):
        """A stage starting at layer 4 gets pattern[4:8], not pattern[0:4].

        The slice must be offset by start_layer: returning the head of the
        pattern would make the coordinator treat the wrong layers as anchors.
        """
        pattern = [False, True, True, False, False, True, False, False]
        self.assertEqual(
            _localize_shared_index_layers(pattern, start_layer=4, layer_num=4),
            [False, True, False, False],
        )

    def test_stage_starting_on_skip_layer_disables_prefetch(self):
        """A stage whose first layer is a skip layer has no local anchor.

        The anchor's miss plan lives in the previous rank's device memory, so
        there is nothing to replay here. Must return None (synchronous swap-in)
        instead of a pattern whose layer 0 is a skip layer -- that would make
        `_build_prefetch_groups` assert.
        """
        self.assertIsNone(
            _localize_shared_index_layers(ALTERNATING, start_layer=1, layer_num=4)
        )

    def test_pattern_shorter_than_pool_range_disables_prefetch(self):
        """Attention-layer count != num_hidden_layers (e.g. Longcat doubles it).

        The pattern cannot cover the pool's layer range, so slicing it would
        silently drop layers; fall back to synchronous swap-in.
        """
        self.assertIsNone(
            _localize_shared_index_layers([False, True], start_layer=0, layer_num=8)
        )
        self.assertIsNone(
            _localize_shared_index_layers(ALTERNATING, start_layer=4, layer_num=8)
        )


if __name__ == "__main__":
    unittest.main()
