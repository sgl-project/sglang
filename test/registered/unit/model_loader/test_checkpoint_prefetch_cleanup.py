import threading
import unittest
from functools import partial
from unittest.mock import MagicMock, patch

from sglang.srt.model_loader import weight_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCheckpointPrefetchCleanup(CustomTestCase):
    def iterators(self):
        return (
            weight_utils.safetensors_weights_iterator,
            partial(
                weight_utils.buffered_multi_thread_safetensors_weights_iterator,
                max_workers=1,
            ),
        )

    def test_prefetch_stops_only_after_loading_finishes(self):
        for iterator in self.iterators():
            with self.subTest(iterator=iterator):
                cancelled = threading.Event()
                worker = threading.Thread(target=cancelled.wait, daemon=True)
                handle = weight_utils.CheckpointFilePrefetchHandle(
                    thread=worker,
                    cancel_event=cancelled,
                    succeeded_event=threading.Event(),
                    errors=[],
                )
                tensor = object()
                progress = MagicMock()
                progress.__iter__.return_value = iter(["model.safetensors"])
                worker.start()
                try:
                    with (
                        patch.object(weight_utils, "tqdm", return_value=progress),
                        patch.object(
                            weight_utils.torch.distributed,
                            "is_initialized",
                            return_value=False,
                        ),
                        patch.object(
                            weight_utils,
                            "_prefetch_all_checkpoints",
                            return_value=handle,
                        ) as prefetch,
                        patch.object(weight_utils.safetensors, "safe_open") as opened,
                    ):
                        checkpoint = opened.return_value.__enter__.return_value
                        checkpoint.keys.return_value = ["weight"]
                        checkpoint.get_tensor.return_value = tensor
                        weights = iterator(["model.safetensors"], prefetch=True)
                        self.assertEqual(next(weights), ("weight", tensor))
                        self.assertFalse(handle.cancelled)
                        self.assertFalse(handle.done)
                        with self.assertRaises(StopIteration):
                            next(weights)
                        prefetch.assert_called_once_with(
                            ["model.safetensors"], num_threads=4
                        )
                        self.assertTrue(handle.cancelled)
                        self.assertTrue(handle.done)
                finally:
                    cancelled.set()
                    worker.join(timeout=5)

    def test_no_prefetch_when_disabled_or_mmap_is_disabled(self):
        for iterator in self.iterators():
            for prefetch, disable_mmap in [(False, False), (True, True)]:
                with (
                    self.subTest(
                        iterator=iterator, prefetch=prefetch, disable_mmap=disable_mmap
                    ),
                    patch.object(weight_utils, "tqdm", return_value=MagicMock()),
                    patch.object(
                        weight_utils.torch.distributed,
                        "is_initialized",
                        return_value=False,
                    ),
                    patch.object(weight_utils, "_prefetch_all_checkpoints") as start,
                ):
                    self.assertEqual(
                        list(
                            iterator([], prefetch=prefetch, disable_mmap=disable_mmap)
                        ),
                        [],
                    )
                    start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
