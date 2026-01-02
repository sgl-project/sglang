import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch

from sglang.test.test_utils import CustomTestCase


def _create_dumper(base_dir, partial_name):
    from sglang.srt.debug_utils.dumper import _Dumper

    dumper = _Dumper()
    dumper._base_dir = Path(base_dir)
    dumper._enable = True
    dumper._enable_write_file = True
    dumper._partial_name = partial_name
    return dumper


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
        self.assertEqual(_try_unify_shape(torch.randn(1, 1, 3, 4), target).shape, target)
        self.assertEqual(_try_unify_shape(torch.randn(2, 3, 4), target).shape, (2, 3, 4))

    def test_compute_smaller_dtype(self):
        from sglang.srt.debug_utils.dump_comparator import _compute_smaller_dtype

        self.assertEqual(_compute_smaller_dtype(torch.float32, torch.bfloat16), torch.bfloat16)
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
        self.assertAlmostEqual(_compute_and_print_diff(x, x, 1e-3)["max_abs_diff"], 0.0, places=5)
        self.assertAlmostEqual(_compute_and_print_diff(x, x + 0.5, 1e-3)["max_abs_diff"], 0.5, places=4)


class TestDumpLoader(CustomTestCase):
    def test_read_meta(self):
        from sglang.srt.debug_utils.dump_loader import read_meta

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, fn in enumerate(["forward_pass_id=1___rank=0___dump_index=1___name=a.pt",
                                    "forward_pass_id=2___rank=0___dump_index=2___name=b.pt"]):
                torch.save(torch.randn(5), Path(tmpdir) / fn)

            df = read_meta(tmpdir)
            self.assertEqual(len(df), 2)
            self.assertTrue(all(c in df.columns for c in ["forward_pass_id", "rank", "name"]))

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

        df = pl.DataFrame({"name": ["a", "a", "b"], "dump_index": [1, 2, 3], "filename": ["f1", "f2", "f3"]})
        result = _add_duplicate_index(df)
        self.assertEqual(result.filter(pl.col("name") == "a").sort("dump_index")["duplicate_index"].to_list(), [0, 1])


class TestEndToEnd(CustomTestCase):
    def test_dump_load_compare(self):
        from sglang.srt.debug_utils.dump_comparator import _compute_and_print_diff
        from sglang.srt.debug_utils.dump_loader import find_row, read_meta

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            baseline_tensor = torch.randn(10, 10)
            noise = torch.randn(10, 10) * 0.01

            for dumper, tensor, d in [
                (_create_dumper(d1, "b"), baseline_tensor, d1),
                (_create_dumper(d2, "t"), baseline_tensor + noise, d2),
            ]:
                dumper.on_forward_pass_start()
                dumper.dump("x", tensor)

            for d, name in [(d1, "b"), (d2, "t")]:
                df = read_meta(Path(d) / f"sglang_dump_{name}")
                self.assertEqual(len(df), 1)
                row = find_row(df, {"name": "x"})
                self.assertIsNotNone(row)

            t1 = torch.load(Path(d1) / "sglang_dump_b" / find_row(read_meta(Path(d1) / "sglang_dump_b"), {"name": "x"})["filename"], weights_only=False)
            t2 = torch.load(Path(d2) / "sglang_dump_t" / find_row(read_meta(Path(d2) / "sglang_dump_t"), {"name": "x"})["filename"], weights_only=False)

            result = _compute_and_print_diff(t1.float(), t2.float(), 0.1)
            self.assertAlmostEqual(result["max_abs_diff"], noise.abs().max().item(), places=3)

    def test_dump_dict(self):
        from sglang.srt.debug_utils.dump_loader import read_meta

        with tempfile.TemporaryDirectory() as tmpdir:
            dumper = _create_dumper(tmpdir, "test")
            dumper.on_forward_pass_start()
            dumper.dump_dict("layer", {"w": torch.randn(5), "b": torch.randn(3)})

            df = read_meta(Path(tmpdir) / "sglang_dump_test")
            self.assertEqual(set(df["name"].to_list()), {"layer_w", "layer_b"})


if __name__ == "__main__":
    unittest.main()
