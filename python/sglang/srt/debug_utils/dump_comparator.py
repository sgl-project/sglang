import argparse
import functools
import re
from pathlib import Path

import polars as pl
import torch

from sglang.srt.debug_utils.dumper import get_truncated_value


def main(args):
    df_target = read_meta(args.target_path)
    df_target = df_target.sort("rank", "dump_index")
    df_target = df_target.filter(
        (pl.col("forward_pass_id") >= args.start_id)
        & (pl.col("forward_pass_id") <= args.end_id)
    )
    assert all(
        c in df_target.columns
        for c in ["rank", "forward_pass_id", "dump_index", "name"]
    )

    df_baseline = read_meta(args.baseline_path)
    print("df_target", df_target)
    print("df_baseline", df_baseline)

    for row in df_target.iter_rows(named=True):
        rows_baseline = df_baseline.filter(
            (
                pl.col("forward_pass_id")
                == row["forward_pass_id"] - args.start_id + args.baseline_start_id
            )
            & functools.reduce(
                lambda a, b: a & b,
                [
                    pl.col(col) == row[col]
                    for col in row.keys()
                    if col not in ["forward_pass_id", "dump_index", "filename"]
                ],
            )
        )
        assert len(rows_baseline) == 1, f"{rows_baseline=}"
        row_baseline = rows_baseline.to_dicts()[0]

        path_baseline = Path(args.baseline_path) / row_baseline["filename"]
        path_target = Path(args.target_path) / row["filename"]
        print(f"Check: target={str(path_target)} baseline={str(path_baseline)}")
        check_tensor_pair(path_baseline=path_baseline, path_target=path_target)
        print()


def read_meta(directory):
    directory = Path(directory)
    assert directory.is_dir(), f"{directory=} should be a directory"

    rows = []
    for p in directory.glob("*.pt"):
        full_kwargs = {}
        for kv in p.stem.split("___"):
            k, v = kv.split("=")
            full_kwargs[k] = v
        rows.append(
            {
                "filename": str(p.name),
                **full_kwargs,
            }
        )

    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("forward_pass_id").cast(int),
        pl.col("rank").cast(int),
    )
    return df


def check_tensor_pair(path_baseline, path_target):
    x_baseline = torch.load(path_baseline, weights_only=True)
    x_target = torch.load(path_target, weights_only=True)

    print(
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[dtype] {x_baseline.dtype} vs {x_target.dtype}"
    )

    if x_baseline.shape != x_target.shape:
        print(f"❌ Shape mismatch")
        return

    raw_abs_diff = (x_target - x_baseline).abs()

    max_abs_diff = raw_abs_diff.max().item()
    mean_abs_diff = raw_abs_diff.mean().item()
    rel_diff = _calc_rel_diff(x_target, x_baseline)

    needs_print = max_abs_diff > 1e-3

    print(
        "\t".join(
            f"{'❌' if value > 1e-3 else '✅'} {name}={value}"
            for name, value in [
                ("rel_diff", rel_diff),
                ("max_abs_diff", max_abs_diff),
                ("mean_abs_diff", mean_abs_diff),
            ]
        )
    )

    if needs_print:
        print(f"x_baseline(sample)={get_truncated_value(x_baseline)}")
        print(f"x_target(sample)={get_truncated_value(x_target)}")


# Copied from DeepGEMM
def _calc_rel_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=1000000)
    parser.add_argument("--baseline-start-id", type=int, default=0)
    args = parser.parse_args()
    main(args)
