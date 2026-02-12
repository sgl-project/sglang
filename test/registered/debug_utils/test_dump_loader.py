import tempfile
import unittest
from pathlib import Path

import polars as pl
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="default", nightly=True)


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


if __name__ == "__main__":
    unittest.main()
