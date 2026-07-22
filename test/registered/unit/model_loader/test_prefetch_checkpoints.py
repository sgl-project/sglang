"""
Unit tests for coordinated checkpoint prefetch.

Verifies that weights loaded with prefetch enabled are bit-identical
to weights loaded without prefetch.
"""

import os
import tempfile
import threading
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import safetensors.torch
import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    _prefetch_all_checkpoints,
    buffered_multi_thread_safetensors_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


class TestPrefetchCheckpoints(CustomTestCase):
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
            def submit(self, fn, path, *args):
                submitted_paths.append(path)
                return super().submit(fn, path, *args)

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

        def fail_prefetch(path, cancel_event):
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
            handle = _prefetch_all_checkpoints(paths, num_threads=1)

        handle.wait()
        self.assertEqual(handle.errors[0][0], paths[0])
        self.assertIsInstance(handle.errors[0][1], OSError)
        warning.assert_called_once()
        self.assertEqual(
            warning.call_args.args[0],
            "Failed to prefetch checkpoint file %r: %s",
        )
        self.assertEqual(warning.call_args.args[1], paths[0])
        self.assertIsInstance(warning.call_args.args[2], OSError)

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
                side_effect=lambda path, cancel_event: loaded_paths.append(path),
            ),
        ):
            _prefetch_all_checkpoints(paths, num_threads=2)

        self.assertEqual(sorted(loaded_paths), sorted(paths[1::3]))

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_prefetch_handle_cancels_before_scheduling_next_shard(self, _):
        paths = [f"model-{i:05d}.safetensors" for i in range(3)]
        started = threading.Event()
        release = threading.Event()
        loaded_paths = []

        def block_first_prefetch(path, cancel_event):
            loaded_paths.append(path)
            started.set()
            self.assertTrue(release.wait(timeout=5))

        with patch(
            "sglang.srt.model_loader.weight_utils._prefetch_checkpoint_file",
            side_effect=block_first_prefetch,
        ):
            handle = _prefetch_all_checkpoints(paths, num_threads=1)
            self.assertTrue(started.wait(timeout=5))
            handle.cancel()
            release.set()
            handle.wait(timeout=5)

        self.assertTrue(handle.cancelled)
        self.assertEqual(loaded_paths, paths[:1])

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


class TestPrefetchDispatch(CustomTestCase):
    """Verify _get_weights_iterator dispatches to the right safetensors
    iterator based on prefetch / multi-thread config.

    Prefetch + default (multi-thread on) must fall back to the
    single-threaded iterator; an explicit enable_multithread_load=true is
    honored as an opt-out; FASTSAFETENSORS and disable_mmap bypass the
    override.
    """

    def _make_loader(self, extra_config, load_format=LoadFormat.SAFETENSORS):
        load_config = LoadConfig(
            load_format=load_format,
            model_loader_extra_config=extra_config,
        )
        return DefaultModelLoader(load_config)

    def _make_source(self):
        # model_config=None skips maybe_add_mtp_safetensors.
        return SimpleNamespace(
            model_or_path="/dummy",
            revision=None,
            fall_back_to_pt=False,
            model_config=None,
            prefix="",
        )

    def _server_args(self, prefetch, disable_mmap=False):
        return SimpleNamespace(
            weight_loader_disable_mmap=disable_mmap,
            weight_loader_prefetch_checkpoints=prefetch,
            weight_loader_prefetch_num_threads=4,
            weight_loader_drop_cache_after_load=False,
        )

    def _run(self, loader, **iterator_kwargs):
        # _get_weights_iterator returns a generator wrapping the chosen
        # iterator; consuming it forces the eager dispatch (the if/elif/else
        # that calls the iterator factory) to execute.
        list(
            loader._get_weights_iterator(
                self._make_source(),
                **iterator_kwargs,
            )
        )

    def _patch_dispatch(self, prefetch, disable_mmap=False):
        return (
            patch.object(
                DefaultModelLoader,
                "_prepare_weights",
                return_value=("/dummy", ["f.safetensors"], True),
            ),
            patch(
                "sglang.srt.model_loader.loader.get_server_args",
                return_value=self._server_args(prefetch, disable_mmap),
            ),
            patch(
                "sglang.srt.model_loader.loader."
                "buffered_multi_thread_safetensors_weights_iterator",
                return_value=iter([]),
            ),
            patch(
                "sglang.srt.model_loader.loader.safetensors_weights_iterator",
                return_value=iter([]),
            ),
            patch("sglang.srt.model_loader.loader.logger.warning"),
        )

    def test_prefetch_uses_single_thread_for_default_config(self):
        """Prefetch on + no explicit multithread config -> single-threaded,
        and the opt-out warning fires once."""
        loader = self._make_loader({})
        p_prep, p_args, p_buffered, p_single, p_warn = self._patch_dispatch(
            prefetch=True
        )
        with (
            p_prep,
            p_args,
            p_buffered as mock_buffered,
            p_single as mock_single,
            p_warn as mock_warning,
        ):
            self._run(loader)
        mock_single.assert_called_once()
        mock_buffered.assert_not_called()
        mock_warning.assert_called_once()

    def test_explicit_enable_multithread_keeps_buffered_with_prefetch(self):
        """Explicit enable_multithread_load=true is the escape hatch; the
        override and its warning must not fire."""
        loader = self._make_loader({"enable_multithread_load": True})
        p_prep, p_args, p_buffered, p_single, p_warn = self._patch_dispatch(
            prefetch=True
        )
        with (
            p_prep,
            p_args,
            p_buffered as mock_buffered,
            p_single as mock_single,
            p_warn as mock_warning,
        ):
            self._run(loader)
        mock_buffered.assert_called_once()
        mock_single.assert_not_called()
        mock_warning.assert_not_called()

    def test_num_threads_only_keeps_buffered_with_prefetch(self):
        """num_threads alone (relying on the enable_multithread_load=True
        default) also signals multi-thread intent, so the override must not
        fire and num_threads stays live."""
        loader = self._make_loader({"num_threads": 64})
        p_prep, p_args, p_buffered, p_single, p_warn = self._patch_dispatch(
            prefetch=True
        )
        with (
            p_prep,
            p_args,
            p_buffered as mock_buffered,
            p_single as mock_single,
            p_warn as mock_warning,
        ):
            self._run(loader)
        mock_buffered.assert_called_once()
        # num_threads is forwarded as max_workers to the buffered iterator.
        self.assertEqual(mock_buffered.call_args.kwargs["max_workers"], 64)
        mock_single.assert_not_called()
        mock_warning.assert_not_called()

    def test_no_prefetch_uses_multithread(self):
        """Prefetch off -> multi-threaded iterator is used (default), no
        override warning."""
        loader = self._make_loader({})
        p_prep, p_args, p_buffered, p_single, p_warn = self._patch_dispatch(
            prefetch=False
        )
        with (
            p_prep,
            p_args,
            p_buffered as mock_buffered,
            p_single as mock_single,
            p_warn as mock_warning,
        ):
            self._run(loader)
        mock_buffered.assert_called_once()
        mock_single.assert_not_called()
        mock_warning.assert_not_called()

    def test_startup_prefetch_reuses_existing_background_handle(self):
        """Startup commit reuses resolved shards and the active prefetch handle."""
        loader = self._make_loader({})
        source = self._make_source()
        resolved_source = DefaultModelLoader.ResolvedSource(
            source=source,
            hf_folder="/dummy",
            weight_files=("f.safetensors",),
            use_safetensors=True,
        )
        p_prep, p_args, p_buffered, p_single, p_warn = self._patch_dispatch(
            prefetch=False
        )
        with (
            p_prep as mock_prepare,
            p_args,
            p_buffered as mock_buffered,
            p_single as mock_single,
            p_warn as mock_warning,
        ):
            list(
                loader._get_weights_iterator(
                    source,
                    resolved_source=resolved_source,
                    startup_prefetch_active=True,
                )
            )

        mock_prepare.assert_not_called()
        mock_single.assert_called_once()
        self.assertFalse(mock_single.call_args.kwargs["prefetch"])
        mock_buffered.assert_not_called()
        mock_warning.assert_called_once()

    def test_prefetch_does_not_override_when_mmap_disabled(self):
        """Prefetch is a no-op without mmap, so the override and its warning
        must not fire."""
        loader = self._make_loader({})
        p_prep, p_args, p_buffered, p_single, p_warn = self._patch_dispatch(
            prefetch=True, disable_mmap=True
        )
        with (
            p_prep,
            p_args,
            p_buffered as mock_buffered,
            p_single as mock_single,
            p_warn as mock_warning,
        ):
            self._run(loader)
        mock_buffered.assert_called_once()
        mock_single.assert_not_called()
        mock_warning.assert_not_called()

    def test_prefetch_does_not_override_for_fastsafetensors(self):
        """FASTSAFETENSORS ignores both flags; override + warning must not
        fire."""
        loader = self._make_loader({}, load_format=LoadFormat.FASTSAFETENSORS)
        p_prep, p_args, p_buffered, p_single, p_warn = self._patch_dispatch(
            prefetch=True
        )
        with (
            patch(
                "sglang.srt.model_loader.loader.fastsafetensors_weights_iterator",
                return_value=iter([]),
            ) as mock_fast,
            p_prep,
            p_args,
            p_buffered as mock_buffered,
            p_single as mock_single,
            p_warn as mock_warning,
        ):
            self._run(loader)
        mock_fast.assert_called_once()
        mock_buffered.assert_not_called()
        mock_single.assert_not_called()
        mock_warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
