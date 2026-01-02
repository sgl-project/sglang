import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import torch

from sglang.test.test_utils import CustomTestCase

# TODO it needs 0 gpu
register_cuda_ci(est_time=60, suite="nightly-1-gpu", nightly=True)


@contextmanager
def with_env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


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


class TestDumpLoader(CustomTestCase):
    def test_read_meta(self):
        from sglang.srt.debug_utils.dump_loader import read_meta

        with tempfile.TemporaryDirectory() as tmpdir:
            for fn in [
                "forward_pass_id=1___rank=0___dump_index=1___name=a.pt",
                "forward_pass_id=2___rank=0___dump_index=2___name=b.pt",
            ]:
                torch.save(torch.randn(5), Path(tmpdir) / fn)

            df = read_meta(tmpdir)
            self.assertEqual(len(df), 2)
            self.assertTrue(
                all(c in df.columns for c in ["forward_pass_id", "rank", "name"])
            )

    def test_find_row(self):
        from sglang.srt.debug_utils.dump_loader import find_row

        df = pl.DataFrame({"id": [1, 2], "name": ["a", "b"], "file": ["f1", "f2"]})
        self.assertEqual(find_row(df, {"id": 2})["file"], "f2")
        self.assertIsNone(find_row(df, {"id": 999}))

        df_dup = pl.DataFrame({"id": [1, 1], "file": ["f1", "f2"]})
        self.assertIsNone(find_row(df_dup, {"id": 1}))

    def test_cast_to_polars_dtype(self):
        from sglang.srt.debug_utils.dump_loader import _cast_to_polars_dtype

        self.assertEqual(_cast_to_polars_dtype("42", pl.Int64), 42)
        self.assertEqual(_cast_to_polars_dtype("3.14", pl.Float64), 3.14)

    def test_add_duplicate_index(self):
        from sglang.srt.debug_utils.dump_loader import _add_duplicate_index

        df = pl.DataFrame(
            {
                "name": ["a", "a", "b"],
                "dump_index": [1, 2, 3],
                "filename": ["f1", "f2", "f3"],
            }
        )
        result = _add_duplicate_index(df)
        self.assertEqual(
            result.filter(pl.col("name") == "a")
            .sort("dump_index")["duplicate_index"]
            .to_list(),
            [0, 1],
        )


class TestEndToEnd(CustomTestCase):
    def test_dump_load_compare(self):
        from sglang.srt.debug_utils.dump_comparator import _compute_and_print_diff
        from sglang.srt.debug_utils.dump_loader import find_row, read_meta

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            baseline_tensor = torch.randn(10, 10)
            noise = torch.randn(10, 10) * 0.01

            with with_env("SGLANG_DUMPER_DIR", d1):
                from sglang.srt.debug_utils.dumper import _Dumper

                dumper1 = _Dumper()
                dumper1.on_forward_pass_start()
                dumper1.dump("x", baseline_tensor)
                dump_dir1 = Path(d1) / f"sglang_dump_{dumper1._partial_name}"

            with with_env("SGLANG_DUMPER_DIR", d2):
                dumper2 = _Dumper()
                dumper2.on_forward_pass_start()
                dumper2.dump("x", baseline_tensor + noise)
                dump_dir2 = Path(d2) / f"sglang_dump_{dumper2._partial_name}"

            df1, df2 = read_meta(dump_dir1), read_meta(dump_dir2)
            self.assertEqual(len(df1), 1)
            self.assertEqual(len(df2), 1)

            t1 = torch.load(
                dump_dir1 / find_row(df1, {"name": "x"})["filename"], weights_only=False
            )
            t2 = torch.load(
                dump_dir2 / find_row(df2, {"name": "x"})["filename"], weights_only=False
            )

            result = _compute_and_print_diff(t1.float(), t2.float(), 0.1)
            self.assertAlmostEqual(
                result["max_abs_diff"], noise.abs().max().item(), places=3
            )

    def test_dump_dict(self):
        from sglang.srt.debug_utils.dump_loader import read_meta

        with tempfile.TemporaryDirectory() as tmpdir:
            with with_env("SGLANG_DUMPER_DIR", tmpdir):
                from sglang.srt.debug_utils.dumper import _Dumper

                dumper = _Dumper()
                dumper.on_forward_pass_start()
                dumper.dump_dict("layer", {"w": torch.randn(5), "b": torch.randn(3)})
                dump_dir = Path(tmpdir) / f"sglang_dump_{dumper._partial_name}"

            df = read_meta(dump_dir)
            self.assertEqual(set(df["name"].to_list()), {"layer_w", "layer_b"})


if __name__ == "__main__":
    unittest.main()
