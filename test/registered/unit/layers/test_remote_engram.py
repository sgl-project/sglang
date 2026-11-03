import importlib.util
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def load_adapter():
    path = Path(__file__).parents[4] / "python/sglang/srt/layers/remote_engram.py"
    spec = importlib.util.spec_from_file_location("test_remote_engram_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemoteEngramValidationTest(CustomTestCase):
    def setUp(self):
        self.module = load_adapter()

    def test_checkpoint_shapes_and_dtypes(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        weight = torch.empty((0, 256), dtype=torch.float8_e4m3fn)
        scale = torch.empty((0, 8), dtype=torch.float8_e8m0fnu)
        table.check_weight(weight, torch.empty((4, 256), dtype=weight.dtype), weight)
        table.check_weight(scale, torch.empty((4, 8), dtype=scale.dtype), weight)
        self.assertEqual(table.loaded, {"weight", "scale"})
        with self.assertRaises(ValueError):
            table.check_weight(
                weight, torch.empty((3, 256), dtype=weight.dtype), weight
            )
        with self.assertRaises(ValueError):
            table.check_weight(scale, torch.empty((4, 8)), weight)

    def test_requires_both_checkpoint_components(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        with self.assertRaisesRegex(RuntimeError, "not loaded"):
            table.finish_load(0, 4)

    def test_rejects_wrong_remote_table_size(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        table.loaded = {"weight", "scale"}
        self.module._runtime = mock.Mock(
            storage_layout="row264", metadata={"lengths": [1]}
        )
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            table.finish_load(0, 4)

    def test_unsupported_layout_and_uninitialized_prefetch_rejected(self):
        with self.assertRaises(ValueError):
            self.module.RemoteEngramTable(4, 128, 1)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            self.module.prepare_forward(True)

    def test_prefetch_lookup_rejects_capture_before_submitting_read(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        table.runtime = mock.Mock(prefetch_enabled=True)
        indices = torch.tensor([0], dtype=torch.int64)
        with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(
                RuntimeError, "unsupported during CUDA graph capture"
            ):
                table.lookup(indices, 0, 4, lambda ids: torch.empty((ids.numel(), 256)))
        table.runtime.fetch.assert_not_called()

    def test_capture_does_not_apply_stale_prefetch_representatives(self):
        table = self.module.RemoteEngramTable(4, 256, 1)
        runtime = mock.Mock(prefetch_enabled=False, storage_layout="row264")
        runtime.fetch.return_value = (torch.tensor([0]), torch.empty(1, 256), None)
        runtime.context.representatives.return_value = torch.tensor([123])
        table.runtime = runtime
        seen = []

        def gather(_weight, _ids, out, _lo, _hi, *, query_order, representatives):
            seen.append((query_order, representatives))
            out.fill_(7)

        with (
            mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
            mock.patch.dict(
                "sys.modules", {"row264_kernels": SimpleNamespace(gather_row264=gather)}
            ),
        ):
            result = table.lookup(
                torch.tensor([0]), 0, 4, lambda _: torch.zeros(1, 256)
            )
        self.assertTrue(torch.all(result == 7))
        self.assertEqual(seen, [(True, None)])

    def test_plan_is_primed_only_after_both_layers_and_deferred_read_after_reduce(self):
        runtime = mock.Mock()
        runtime.storage_layout = "row264"
        runtime.metadata = {"lengths": [4 * 264, 4 * 264]}
        runtime.ranges = {}
        runtime.prefetch_enabled = True
        runtime.defer_layer14 = True
        self.module._runtime = runtime
        one = self.module.RemoteEngramTable(4, 256, 1)
        two = self.module.RemoteEngramTable(4, 256, 14)
        one.loaded = two.loaded = {"weight", "scale"}
        with (
            mock.patch.dict(os.environ, {"ENGRAM_SHARED_READ_PLAN": "1"}),
            mock.patch("torch.cuda.stream", return_value=nullcontext()),
        ):
            one.finish_load(0, 2)
            runtime.context.prime.assert_not_called()
            two.finish_load(2, 2)
        runtime.context.prime.assert_called_once()
        self.assertEqual(runtime.ranges, {1: (0, 2), 14: (2, 4)})
        one.after_reduction()
        runtime.issue_prefetch_layer.assert_called_once_with(14)


if __name__ == "__main__":
    unittest.main()
