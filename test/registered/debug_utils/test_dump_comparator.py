import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="default", nightly=True)


class TestDumpComparator(CustomTestCase):
    def test_calc_rel_diff(self):
        from sglang.srt.debug_utils.dump_comparator import _calc_rel_diff

        x = torch.randn(10, 10)
        self.assertAlmostEqual(_calc_rel_diff(x, x).item(), 0.0, places=5)
        self.assertAlmostEqual(
            _calc_rel_diff(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])).item(),
            1.0,
            places=5,
        )

    def test_argmax_coord(self):
        from sglang.srt.debug_utils.dump_comparator import _argmax_coord

        x = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        self.assertEqual(_argmax_coord(x), (1, 2, 3))

    def test_try_unify_shape(self):
        from sglang.srt.debug_utils.dump_comparator import _try_unify_shape

        target = torch.Size([3, 4])
        self.assertEqual(
            _try_unify_shape(torch.randn(1, 1, 3, 4), target).shape, target
        )
        self.assertEqual(
            _try_unify_shape(torch.randn(2, 3, 4), target).shape, (2, 3, 4)
        )

    def test_compute_smaller_dtype(self):
        from sglang.srt.debug_utils.dump_comparator import _compute_smaller_dtype

        self.assertEqual(
            _compute_smaller_dtype(torch.float32, torch.bfloat16), torch.bfloat16
        )
        self.assertIsNone(_compute_smaller_dtype(torch.float32, torch.float32))

    def test_einops_pattern(self):
        from sglang.srt.debug_utils.dump_comparator import (
            _get_einops_dim_index,
            _split_einops_pattern,
        )

        self.assertEqual(_split_einops_pattern("a (b c) d"), ["a", "(b c)", "d"])
        self.assertEqual(_get_einops_dim_index("a b c", "b"), 1)

    def test_load_object(self):
        from sglang.srt.debug_utils.dump_comparator import _load_object

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tensor.pt"
            torch.save(torch.randn(5, 5), path)
            self.assertEqual(_load_object(path).shape, (5, 5))

            torch.save({"dict": 1}, path)
            self.assertIsNone(_load_object(path))

        self.assertIsNone(_load_object("/nonexistent.pt"))

    def test_compute_and_print_diff(self):
        from sglang.srt.debug_utils.dump_comparator import _compute_and_print_diff

        x = torch.ones(10, 10)
        self.assertAlmostEqual(
            _compute_and_print_diff(x, x, 1e-3)["max_abs_diff"], 0.0, places=5
        )
        self.assertAlmostEqual(
            _compute_and_print_diff(x, x + 0.5, 1e-3)["max_abs_diff"], 0.5, places=4
        )


class TestEndToEnd(CustomTestCase):
    def test_main(self):
        from argparse import Namespace

        from sglang.srt.debug_utils.dump_comparator import main
        from sglang.srt.debug_utils.dumper import _Dumper

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            baseline_tensor = torch.randn(10, 10)
            target_tensor = baseline_tensor + torch.randn(10, 10) * 0.01

            dump_dirs = []
            for d, tensor in [(d1, baseline_tensor), (d2, target_tensor)]:
                with _with_env("SGLANG_DUMPER_DIR", d), _with_env(
                    "SGLANG_DUMPER_SERVER_PORT", "-1"
                ):
                    dumper = _Dumper()
                    dumper.on_forward_pass_start()
                    dumper.dump("tensor_a", tensor)
                    dumper.on_forward_pass_start()
                    dumper.dump("tensor_b", tensor * 2)
                    dump_dirs.append(Path(d) / f"sglang_dump_{dumper._partial_name}")

            args = Namespace(
                baseline_path=str(dump_dirs[0]),
                target_path=str(dump_dirs[1]),
                start_id=1,
                end_id=2,
                baseline_start_id=1,
                diff_threshold=1e-3,
                filter=None,
            )
            main(args)


@contextmanager
def _with_env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


if __name__ == "__main__":
    unittest.main()
