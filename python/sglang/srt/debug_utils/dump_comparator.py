import argparse
import functools
from pathlib import Path

import polars as pl
import torch

from sglang.srt.debug_utils.dump_loader import find_row, read_meta
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
        path_target = Path(args.target_path) / row["filename"]

        row_baseline = find_row(
            df_baseline,
            conditions=dict(
                forward_pass_id=row["forward_pass_id"]
                - args.start_id
                + args.baseline_start_id,
                **{
                    k: v
                    for k, v in row.items()
                    if k not in ["forward_pass_id", "dump_index", "filename"]
                },
            ),
        )

        if row_baseline is None:
            print(f"Skip: target={str(path_target)} since no baseline")
            x_target = _load_object(path_target)
            if x_target is not None:
                print(f"x_target(sample)={get_truncated_value(x_target)}")
            continue

        path_baseline = Path(args.baseline_path) / row_baseline["filename"]
        print(f"Check: target={str(path_target)} baseline={str(path_baseline)}")
        check_tensor_pair(
            path_baseline=path_baseline, path_target=path_target, name=row["name"]
        )
        print()


def check_tensor_pair(path_baseline, path_target, name=""):
    x_baseline = _load_object(path_baseline)
    x_target = _load_object(path_target)

    print(
        f"Raw "
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[dtype] {x_baseline.dtype} vs {x_target.dtype}"
    )

    x_baseline, x_target = _comparison_preprocessor(x_baseline, x_target, name=name)
    x_baseline = _try_unify_shape(x_baseline, target_shape=x_target.shape)

    print(
        f"After preprocessor "
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[dtype] {x_baseline.dtype} vs {x_target.dtype}"
    )

    x_target = x_target.float()
    x_baseline = x_baseline.float()

    for name, fn in (
        ("mean", torch.mean),
        ("std", torch.std),
        ("min", torch.min),
        ("max", torch.max),
        ("p1", functools.partial(torch.quantile, q=0.01)),
        ("p5", functools.partial(torch.quantile, q=0.05)),
        ("p95", functools.partial(torch.quantile, q=0.95)),
        ("p99", functools.partial(torch.quantile, q=0.99)),
    ):
        value_baseline = fn(x_baseline).item()
        value_target = fn(x_target).item()
        print(
            f"[{name}] {value_baseline :.4f} vs {value_target:.4f} (diff: {value_target - value_baseline:.4f})"
        )

    if x_baseline.shape != x_target.shape:
        print(f"⚠️ Shape mismatch")
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


def _try_unify_shape(x: torch.Tensor, target_shape):
    x_shape = x.shape
    num_dim_to_remove = len(x_shape) - len(target_shape)
    if (x_shape[num_dim_to_remove:] == target_shape) and all(
        val == 1 for val in x_shape[:num_dim_to_remove]
    ):
        out = functools.reduce(lambda a, _: a.squeeze(0), range(num_dim_to_remove), x)
        print(f"Unify shape: {x_shape} -> {out.shape} (to match {target_shape})")
        return out

    return x


# Copied from DeepGEMM
def _calc_rel_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _comparison_preprocessor(x_baseline, x_target, name):
    # can insert arbitrary adhoc postprocessing logic here
    return x_baseline, x_target


def _load_object(path):
    x = torch.load(path, weights_only=False)
    if not isinstance(x, torch.Tensor):
        print(f"Skip load {path} since {type(x)=} is not a Tensor")
        return None
    return x.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=1000000)
    parser.add_argument("--baseline-start-id", type=int, default=0)
    args = parser.parse_args()
    main(args)
