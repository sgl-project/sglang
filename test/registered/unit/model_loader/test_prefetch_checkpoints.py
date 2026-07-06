"""
Unit tests for coordinated checkpoint prefetch.

Verifies that weights loaded with prefetch enabled are bit-identical
to weights loaded without prefetch.
"""

import os
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import safetensors.torch
import torch

from sglang.srt.model_loader.weight_utils import (
    _prefetch_all_checkpoints,
    buffered_multi_thread_safetensors_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _InlineThread:
    def __init__(self, target, daemon=None):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class _InlineExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def _wait_all(fs, return_when):
    return set(fs), set()


class TestPrefetchCheckpoints(unittest.TestCase):
    """Verify coordinated checkpoint prefetch behavior."""

    def _create_safetensors_files(self, tmpdir, num_shards=3):
        """Create real safetensors files with known tensor content."""
        paths = []
        for i in range(num_shards):
            tensors = {
                f"layer{i}.weight": torch.randn(32, 32),
                f"layer{i}.bias": torch.randn(32),
            }
            path = os.path.join(tmpdir, f"model-{i:05d}.safetensors")
            safetensors.torch.save_file(tensors, path)
            paths.append(path)
        return paths

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_weights_match_with_and_without_prefetch(self, _):
        """Tensors yielded must be bit-identical regardless of prefetch flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_safetensors_files(tmpdir)

            without = dict(safetensors_weights_iterator(paths, prefetch=False))
            with (
                patch("threading.Thread", _InlineThread),
                patch("concurrent.futures.ThreadPoolExecutor", _InlineExecutor),
                patch("concurrent.futures.wait", side_effect=_wait_all),
            ):
                with_pf = dict(safetensors_weights_iterator(paths, prefetch=True))

            self.assertEqual(set(without.keys()), set(with_pf.keys()))
            for name in without:
                torch.testing.assert_close(without[name], with_pf[name])

    def test_prefetch_rejects_invalid_thread_count(self):
        with self.assertRaisesRegex(ValueError, "num_threads"):
            _prefetch_all_checkpoints(["dummy.safetensors"], num_threads=0)

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_prefetch_keeps_bounded_pending_window(self, _):
        paths = [f"model-{i:05d}.safetensors" for i in range(20)]
        pending_sizes = []
        submitted_paths = []

        class RecordingExecutor(_InlineExecutor):
            def submit(self, fn, path):
                submitted_paths.append(path)
                return super().submit(fn, path)

        def record_pending_size(fs, return_when):
            pending_sizes.append(len(fs))
            return _wait_all(fs, return_when)

        with (
            patch("threading.Thread", _InlineThread),
            patch("concurrent.futures.ThreadPoolExecutor", RecordingExecutor),
            patch("concurrent.futures.wait", side_effect=record_pending_size),
            patch("sglang.srt.model_loader.weight_utils._prefetch_checkpoint_file"),
        ):
            _prefetch_all_checkpoints(paths, num_threads=4)

        self.assertEqual(submitted_paths, paths)
        self.assertLessEqual(max(pending_sizes), 4)

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_prefetch_logs_failed_futures(self, _):
        paths = ["bad.safetensors"]

        def fail_prefetch(path):
            raise OSError(f"failed {path}")

        with (
            patch("threading.Thread", _InlineThread),
            patch("concurrent.futures.ThreadPoolExecutor", _InlineExecutor),
            patch("concurrent.futures.wait", side_effect=_wait_all),
            patch(
                "sglang.srt.model_loader.weight_utils._prefetch_checkpoint_file",
                side_effect=fail_prefetch,
            ),
            patch("sglang.srt.model_loader.weight_utils.logger.warning") as warning,
        ):
            _prefetch_all_checkpoints(paths, num_threads=1)

        warning.assert_called_once()
        self.assertEqual(
            warning.call_args.args[0],
            "Failed to prefetch checkpoint file %r.",
        )
        self.assertEqual(warning.call_args.args[1], paths[0])
        self.assertTrue(warning.call_args.kwargs["exc_info"])

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_prefetch_progress_logs_all_crossed_buckets(self, _):
        paths = [f"model-{i:05d}.safetensors" for i in range(3)]

        with (
            patch("threading.Thread", _InlineThread),
            patch("concurrent.futures.ThreadPoolExecutor", _InlineExecutor),
            patch("concurrent.futures.wait", side_effect=_wait_all),
            patch("sglang.srt.model_loader.weight_utils._prefetch_checkpoint_file"),
            patch("sglang.srt.model_loader.weight_utils.logger.info") as log_info,
        ):
            _prefetch_all_checkpoints(paths, num_threads=1)

        progress_pcts = [
            call.args[2]
            for call in log_info.call_args_list
            if call.args
            and call.args[0] == "Rank %d: prefetching checkpoint files: %d%% (%d/%d)"
        ]
        self.assertEqual(progress_pcts, list(range(10, 101, 10)))

    @patch("torch.distributed.is_initialized", return_value=True)
    def test_prefetch_uses_node_local_rank_partitioning(self, _):
        paths = [f"model-{i:05d}.safetensors" for i in range(10)]
        loaded_paths = []

        class FakeWorldGroup:
            local_rank = 1
            local_size = 3
            world_size = 99

        with (
            patch("threading.Thread", _InlineThread),
            patch("concurrent.futures.ThreadPoolExecutor", _InlineExecutor),
            patch("concurrent.futures.wait", side_effect=_wait_all),
            patch(
                "sglang.srt.model_loader.weight_utils.get_world_group",
                return_value=FakeWorldGroup(),
            ),
            patch(
                "sglang.srt.model_loader.weight_utils._prefetch_checkpoint_file",
                side_effect=loaded_paths.append,
            ),
        ):
            _prefetch_all_checkpoints(paths, num_threads=2)

        self.assertEqual(sorted(loaded_paths), sorted(paths[1::3]))

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_buffered_loader_drops_cache_after_each_loaded_shard(self, _):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._create_safetensors_files(tmpdir, num_shards=3)

            with patch(
                "sglang.srt.model_loader.weight_utils._drop_file_cache_after_load"
            ) as drop_cache:
                loaded = list(
                    buffered_multi_thread_safetensors_weights_iterator(
                        paths,
                        max_workers=2,
                        drop_cache_after_load=True,
                    )
                )

            self.assertEqual(len(loaded), 6)
            self.assertEqual(
                [call.args[0] for call in drop_cache.call_args_list],
                paths,
            )


if __name__ == "__main__":
    unittest.main()
