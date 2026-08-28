# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multimodal weight-loading utilities."""

import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.loader.weight_utils import (
    _disable_runai_streamer_rank_discovery_collective,
    get_lock,
)

_DIST_STREAMER_MOD = "runai_model_streamer.distributed_streamer.distributed_streamer"


class TestDisableRunaiStreamerRankDiscoveryCollective(unittest.TestCase):
    def test_never_touches_torch_distributed_even_when_initialized(self):
        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            _distributedStreamerParams,
        )

        _disable_runai_streamer_rank_discovery_collective()

        with (
            patch(f"{_DIST_STREAMER_MOD}.dist.is_initialized", return_value=True),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_world_size", return_value=2),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_rank", return_value=1),
            patch(f"{_DIST_STREAMER_MOD}.dist.new_group") as mock_new_group,
            patch(f"{_DIST_STREAMER_MOD}.dist.all_gather_object") as mock_all_gather,
            patch(f"{_DIST_STREAMER_MOD}.dist.destroy_process_group") as mock_destroy,
        ):
            result = _distributedStreamerParams().find_local_ranks()

        mock_new_group.assert_not_called()
        mock_all_gather.assert_not_called()
        mock_destroy.assert_not_called()
        # rank is still reported correctly -- only the collective is gone
        self.assertEqual(result, (1, 1, [[1]]))

    def test_reports_rank_zero_when_not_distributed(self):
        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            _distributedStreamerParams,
        )

        _disable_runai_streamer_rank_discovery_collective()

        with patch(f"{_DIST_STREAMER_MOD}.dist.is_initialized", return_value=False):
            result = _distributedStreamerParams().find_local_ranks()

        self.assertEqual(result, (1, 0, [[0]]))

    def test_idempotent_across_repeated_calls(self):
        # Import-time application plus any re-import/re-entry must not stack
        # wrappers or otherwise change behavior.
        _disable_runai_streamer_rank_discovery_collective()
        _disable_runai_streamer_rank_discovery_collective()

        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            _distributedStreamerParams,
        )

        with (
            patch(f"{_DIST_STREAMER_MOD}.dist.is_initialized", return_value=True),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_world_size", return_value=4),
            patch(f"{_DIST_STREAMER_MOD}.dist.get_rank", return_value=3),
            patch(f"{_DIST_STREAMER_MOD}.dist.new_group") as mock_new_group,
        ):
            result = _distributedStreamerParams().find_local_ranks()

        mock_new_group.assert_not_called()
        self.assertEqual(result, (1, 3, [[3]]))

    def test_noop_when_library_missing_attribute(self):
        # Defensive path: if a future runai_model_streamer release renames or
        # removes find_local_ranks, patching must skip (with a warning) rather
        # than crash import.
        import sglang.multimodal_gen.runtime.loader.weight_utils as wu

        class _StubParams:
            pass

        with patch(f"{_DIST_STREAMER_MOD}._distributedStreamerParams", _StubParams):
            wu._disable_runai_streamer_rank_discovery_collective()  # must not raise

        self.assertFalse(hasattr(_StubParams, "find_local_ranks"))


class TestDiffusionWeightLock(unittest.TestCase):
    def test_long_snapshot_path_uses_bounded_lock_filename(self):
        component_path = os.path.join(
            "/scratch",
            "models--" + "very-long-repository-name-" * 8,
            "snapshots",
            "a" * 64,
            "transformer",
            "config.json",
        )

        with tempfile.TemporaryDirectory() as lock_dir:
            lock = get_lock(component_path, lock_dir)
            lock_filename = os.path.basename(lock.lock_file)

            self.assertLessEqual(len(os.fsencode(lock_filename)), 255)
            with lock:
                self.assertTrue(os.path.exists(lock.lock_file))


if __name__ == "__main__":
    unittest.main()
