import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch

from sglang.test.test_utils import CustomTestCase


class TestDumpComparatorPureFunctions(CustomTestCase):
    def test_calc_rel_diff_identical(self):
        from sglang.srt.debug_utils.dump_comparator import _calc_rel_diff

        x = torch.randn(10, 10)
        rel_diff = _calc_rel_diff(x, x)
        self.assertAlmostEqual(rel_diff.item(), 0.0, places=5)

    def test_calc_rel_diff_different(self):
        from sglang.srt.debug_utils.dump_comparator import _calc_rel_diff

        x = torch.ones(10, 10)
        y = torch.ones(10, 10) * 2
        rel_diff = _calc_rel_diff(x, y)
        self.assertGreater(rel_diff.item(), 0.0)
        self.assertLess(rel_diff.item(), 1.0)

    def test_calc_rel_diff_orthogonal(self):
        from sglang.srt.debug_utils.dump_comparator import _calc_rel_diff

        x = torch.tensor([1.0, 0.0])
        y = torch.tensor([0.0, 1.0])
        rel_diff = _calc_rel_diff(x, y)
        self.assertAlmostEqual(rel_diff.item(), 1.0, places=5)

    def test_argmax_coord_1d(self):
        from sglang.srt.debug_utils.dump_comparator import _argmax_coord

        x = torch.tensor([1.0, 5.0, 3.0])
        coord = _argmax_coord(x)
        self.assertEqual(coord, (1,))

    def test_argmax_coord_2d(self):
        from sglang.srt.debug_utils.dump_comparator import _argmax_coord

        x = torch.zeros(3, 4)
        x[1, 2] = 10.0
        coord = _argmax_coord(x)
        self.assertEqual(coord, (1, 2))

    def test_argmax_coord_3d(self):
        from sglang.srt.debug_utils.dump_comparator import _argmax_coord

        x = torch.zeros(2, 3, 4)
        x[1, 2, 3] = 10.0
        coord = _argmax_coord(x)
        self.assertEqual(coord, (1, 2, 3))

    def test_try_unify_shape_squeeze_leading_dims(self):
        from sglang.srt.debug_utils.dump_comparator import _try_unify_shape

        x = torch.randn(1, 1, 3, 4)
        target_shape = torch.Size([3, 4])
        result = _try_unify_shape(x, target_shape)
        self.assertEqual(result.shape, target_shape)

    def test_try_unify_shape_no_change(self):
        from sglang.srt.debug_utils.dump_comparator import _try_unify_shape

        x = torch.randn(3, 4)
        target_shape = torch.Size([3, 4])
        result = _try_unify_shape(x, target_shape)
        self.assertEqual(result.shape, target_shape)

    def test_try_unify_shape_incompatible(self):
        from sglang.srt.debug_utils.dump_comparator import _try_unify_shape

        x = torch.randn(2, 3, 4)
        target_shape = torch.Size([3, 4])
        result = _try_unify_shape(x, target_shape)
        self.assertEqual(result.shape, x.shape)

    def test_compute_smaller_dtype(self):
        from sglang.srt.debug_utils.dump_comparator import _compute_smaller_dtype

        result = _compute_smaller_dtype(torch.float32, torch.bfloat16)
        self.assertEqual(result, torch.bfloat16)

        result = _compute_smaller_dtype(torch.bfloat16, torch.float32)
        self.assertEqual(result, torch.bfloat16)

        result = _compute_smaller_dtype(torch.float32, torch.float32)
        self.assertIsNone(result)

    def test_split_einops_pattern_simple(self):
        from sglang.srt.debug_utils.dump_comparator import _split_einops_pattern

        result = _split_einops_pattern("batch num_tokens hidden")
        self.assertEqual(result, ["batch", "num_tokens", "hidden"])

    def test_split_einops_pattern_with_parens(self):
        from sglang.srt.debug_utils.dump_comparator import _split_einops_pattern

        result = _split_einops_pattern("batch (num_heads head_dim) hidden")
        self.assertEqual(result, ["batch", "(num_heads head_dim)", "hidden"])

    def test_get_einops_dim_index(self):
        from sglang.srt.debug_utils.dump_comparator import _get_einops_dim_index

        pattern = "batch num_tokens hidden"
        self.assertEqual(_get_einops_dim_index(pattern, "batch"), 0)
        self.assertEqual(_get_einops_dim_index(pattern, "num_tokens"), 1)
        self.assertEqual(_get_einops_dim_index(pattern, "hidden"), 2)

    def test_load_object_success(self):
        from sglang.srt.debug_utils.dump_comparator import _load_object

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tensor = torch.randn(10, 10)
            torch.save(tensor, f.name)
            result = _load_object(f.name)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, tensor.shape)
            Path(f.name).unlink()

    def test_load_object_not_tensor(self):
        from sglang.srt.debug_utils.dump_comparator import _load_object

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"not": "a tensor"}, f.name)
            result = _load_object(f.name)
            self.assertIsNone(result)
            Path(f.name).unlink()

    def test_load_object_file_not_exist(self):
        from sglang.srt.debug_utils.dump_comparator import _load_object

        result = _load_object("/nonexistent/path/to/file.pt")
        self.assertIsNone(result)

    def test_check_tensor_pair_same_shape(self):
        from sglang.srt.debug_utils.dump_comparator import check_tensor_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = torch.randn(10, 10)
            target = baseline.clone()

            baseline_path = Path(tmpdir) / "baseline.pt"
            target_path = Path(tmpdir) / "target.pt"
            torch.save(baseline, baseline_path)
            torch.save(target, target_path)

            check_tensor_pair(
                path_baseline=baseline_path,
                path_target=target_path,
                diff_threshold=1e-3,
                name="test_tensor",
            )

    def test_check_tensor_pair_different_values(self):
        from sglang.srt.debug_utils.dump_comparator import check_tensor_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = torch.ones(10, 10)
            target = torch.ones(10, 10) * 1.5

            baseline_path = Path(tmpdir) / "baseline.pt"
            target_path = Path(tmpdir) / "target.pt"
            torch.save(baseline, baseline_path)
            torch.save(target, target_path)

            check_tensor_pair(
                path_baseline=baseline_path,
                path_target=target_path,
                diff_threshold=1e-3,
                name="test_tensor",
            )

    def test_check_tensor_pair_different_dtype(self):
        from sglang.srt.debug_utils.dump_comparator import check_tensor_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = torch.randn(10, 10, dtype=torch.float32)
            target = baseline.clone().to(torch.bfloat16)

            baseline_path = Path(tmpdir) / "baseline.pt"
            target_path = Path(tmpdir) / "target.pt"
            torch.save(baseline, baseline_path)
            torch.save(target, target_path)

            check_tensor_pair(
                path_baseline=baseline_path,
                path_target=target_path,
                diff_threshold=1e-3,
                name="test_tensor",
            )

    def test_check_tensor_pair_shape_mismatch(self):
        from sglang.srt.debug_utils.dump_comparator import check_tensor_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = torch.randn(10, 10)
            target = torch.randn(10, 20)

            baseline_path = Path(tmpdir) / "baseline.pt"
            target_path = Path(tmpdir) / "target.pt"
            torch.save(baseline, baseline_path)
            torch.save(target, target_path)

            check_tensor_pair(
                path_baseline=baseline_path,
                path_target=target_path,
                diff_threshold=1e-3,
                name="test_tensor",
            )

    def test_check_tensor_pair_none_target(self):
        from sglang.srt.debug_utils.dump_comparator import check_tensor_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = torch.randn(10, 10)

            baseline_path = Path(tmpdir) / "baseline.pt"
            target_path = Path(tmpdir) / "target.pt"
            torch.save(baseline, baseline_path)

            check_tensor_pair(
                path_baseline=baseline_path,
                path_target=target_path,
                diff_threshold=1e-3,
                name="test_tensor",
            )

    def test_compute_and_print_diff(self):
        from sglang.srt.debug_utils.dump_comparator import _compute_and_print_diff

        x_baseline = torch.ones(10, 10)
        x_target = torch.ones(10, 10) + 0.001

        result = _compute_and_print_diff(
            x_baseline=x_baseline,
            x_target=x_target,
            diff_threshold=1e-2,
        )
        self.assertIn("max_abs_diff", result)
        self.assertAlmostEqual(result["max_abs_diff"], 0.001, places=4)


class TestDumpLoader(CustomTestCase):
    def test_read_meta(self):
        from sglang.srt.debug_utils.dump_loader import read_meta

        with tempfile.TemporaryDirectory() as tmpdir:
            filename1 = "forward_pass_id=1___rank=0___dump_index=1___name=tensor_a.pt"
            filename2 = "forward_pass_id=1___rank=1___dump_index=1___name=tensor_a.pt"
            filename3 = "forward_pass_id=2___rank=0___dump_index=2___name=tensor_b.pt"

            for fn in [filename1, filename2, filename3]:
                torch.save(torch.randn(5), Path(tmpdir) / fn)

            df = read_meta(tmpdir)
            self.assertEqual(len(df), 3)
            self.assertIn("forward_pass_id", df.columns)
            self.assertIn("rank", df.columns)
            self.assertIn("dump_index", df.columns)
            self.assertIn("name", df.columns)
            self.assertIn("filename", df.columns)
            self.assertIn("duplicate_index", df.columns)

    def test_find_row_single_match(self):
        from sglang.srt.debug_utils.dump_loader import find_row

        df = pl.DataFrame(
            {
                "forward_pass_id": [1, 2, 3],
                "rank": [0, 0, 0],
                "name": ["a", "b", "c"],
                "filename": ["f1.pt", "f2.pt", "f3.pt"],
            }
        )

        row = find_row(df, conditions={"forward_pass_id": 2, "name": "b"})
        self.assertIsNotNone(row)
        self.assertEqual(row["filename"], "f2.pt")

    def test_find_row_no_match(self):
        from sglang.srt.debug_utils.dump_loader import find_row

        df = pl.DataFrame(
            {
                "forward_pass_id": [1, 2, 3],
                "rank": [0, 0, 0],
                "name": ["a", "b", "c"],
                "filename": ["f1.pt", "f2.pt", "f3.pt"],
            }
        )

        row = find_row(df, conditions={"forward_pass_id": 999})
        self.assertIsNone(row)

    def test_find_row_ambiguous(self):
        from sglang.srt.debug_utils.dump_loader import find_row

        df = pl.DataFrame(
            {
                "forward_pass_id": [1, 1],
                "rank": [0, 0],
                "name": ["a", "a"],
                "filename": ["f1.pt", "f2.pt"],
            }
        )

        row = find_row(df, conditions={"forward_pass_id": 1, "name": "a"})
        self.assertIsNone(row)

    def test_cast_to_polars_dtype(self):
        from sglang.srt.debug_utils.dump_loader import _cast_to_polars_dtype

        self.assertEqual(_cast_to_polars_dtype("42", pl.Int64), 42)
        self.assertEqual(_cast_to_polars_dtype("3.14", pl.Float64), 3.14)
        self.assertEqual(_cast_to_polars_dtype(1, pl.Boolean), True)
        self.assertEqual(_cast_to_polars_dtype(42, pl.String), "42")
        self.assertEqual(_cast_to_polars_dtype("unknown", pl.Utf8), "unknown")

    def test_add_duplicate_index(self):
        from sglang.srt.debug_utils.dump_loader import _add_duplicate_index

        df = pl.DataFrame(
            {
                "forward_pass_id": [1, 1, 1, 2],
                "rank": [0, 0, 0, 0],
                "name": ["a", "a", "b", "a"],
                "dump_index": [1, 2, 3, 4],
                "filename": ["f1.pt", "f2.pt", "f3.pt", "f4.pt"],
            }
        )

        result = _add_duplicate_index(df)
        self.assertIn("duplicate_index", result.columns)

        a_rows = result.filter(
            (pl.col("forward_pass_id") == 1) & (pl.col("name") == "a")
        ).sort("dump_index")
        dup_indices = a_rows["duplicate_index"].to_list()
        self.assertEqual(dup_indices, [0, 1])


class TestEndToEndIntegration(CustomTestCase):
    def test_dump_load_compare_flow(self):
        from sglang.srt.debug_utils.dump_comparator import check_tensor_pair
        from sglang.srt.debug_utils.dump_loader import find_row, read_meta
        from sglang.srt.debug_utils.dumper import _Dumper

        with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as target_dir:
            baseline_dumper = _Dumper()
            baseline_dumper._base_dir = Path(baseline_dir)
            baseline_dumper._enable = True
            baseline_dumper._enable_write_file = True
            baseline_dumper._partial_name = "baseline"

            baseline_tensor = torch.randn(10, 10)
            baseline_dumper.on_forward_pass_start()
            baseline_dumper.dump("test_tensor", baseline_tensor, layer_id=0)

            target_dumper = _Dumper()
            target_dumper._base_dir = Path(target_dir)
            target_dumper._enable = True
            target_dumper._enable_write_file = True
            target_dumper._partial_name = "target"

            target_tensor = baseline_tensor.clone() + torch.randn(10, 10) * 0.001
            target_dumper.on_forward_pass_start()
            target_dumper.dump("test_tensor", target_tensor, layer_id=0)

            baseline_dump_dir = Path(baseline_dir) / "sglang_dump_baseline"
            target_dump_dir = Path(target_dir) / "sglang_dump_target"

            df_baseline = read_meta(baseline_dump_dir)
            df_target = read_meta(target_dump_dir)

            self.assertEqual(len(df_baseline), 1)
            self.assertEqual(len(df_target), 1)

            row_baseline = find_row(
                df_baseline, conditions={"name": "test_tensor", "layer_id": "0"}
            )
            row_target = find_row(
                df_target, conditions={"name": "test_tensor", "layer_id": "0"}
            )

            self.assertIsNotNone(row_baseline)
            self.assertIsNotNone(row_target)

            baseline_path = baseline_dump_dir / row_baseline["filename"]
            target_path = target_dump_dir / row_target["filename"]

            check_tensor_pair(
                path_baseline=baseline_path,
                path_target=target_path,
                diff_threshold=1e-2,
                name="test_tensor",
            )

    def test_dump_dict_load_compare(self):
        from sglang.srt.debug_utils.dump_loader import read_meta
        from sglang.srt.debug_utils.dumper import _Dumper

        with tempfile.TemporaryDirectory() as tmpdir:
            dumper = _Dumper()
            dumper._base_dir = Path(tmpdir)
            dumper._enable = True
            dumper._enable_write_file = True
            dumper._partial_name = "dict_test"

            data = {
                "weight": torch.randn(5, 5),
                "bias": torch.randn(5),
            }

            dumper.on_forward_pass_start()
            dumper.dump_dict("layer", data)

            dump_dir = Path(tmpdir) / "sglang_dump_dict_test"
            df = read_meta(dump_dir)

            self.assertEqual(len(df), 2)
            names = set(df["name"].to_list())
            self.assertIn("layer_weight", names)
            self.assertIn("layer_bias", names)


if __name__ == "__main__":
    unittest.main()

