import argparse
import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import einops
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

    location_info_of_target_pass_id = _get_location_info_of_target_pass_id()
    tensor_dim_descs = _get_tensor_dim_descs()

    for row in df_target.iter_rows(named=True):
        path_target = Path(args.target_path) / row["filename"]

        if location_info_of_target_pass_id is not None:
            location_info = location_info_of_target_pass_id.get(row["forward_pass_id"])
            if location_info is None:
                continue
            baseline_forward_pass_id = location_info.baseline_forward_pass_id
            baseline_token_slice = location_info.baseline_token_slice
        else:
            baseline_forward_pass_id = (
                row["forward_pass_id"] - args.start_id + args.baseline_start_id
            )
            baseline_token_slice = None

        tensor_dim_desc = None
        if tensor_dim_descs is not None:
            tensor_dim_descs_filtered = [
                desc
                for desc in tensor_dim_descs
                if re.search(desc["pattern"], row["filename"]) is not None
            ]
            if tensor_dim_descs_filtered:
                tensor_dim_desc = tensor_dim_descs_filtered[0]

        row_baseline = find_row(
            df_baseline,
            conditions=dict(
                forward_pass_id=baseline_forward_pass_id,
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
            path_baseline=path_baseline,
            path_target=path_target,
            diff_threshold=args.diff_threshold,
            name=row["name"],
            baseline_token_slice=baseline_token_slice,
            tensor_dim_desc=tensor_dim_desc,
        )
        print()


def _split_einops_pattern(pattern):
    return re.findall(r"\([^()]*\)|\S+", pattern)


def _get_einops_dim_index(pattern: str, dim_name: str):
    pattern_list = _split_einops_pattern(pattern)
    return pattern_list.index(dim_name)


def check_tensor_pair(
    path_baseline,
    path_target,
    diff_threshold: float = 1e-3,
    name="",
    baseline_token_slice=None,
    tensor_dim_desc: Optional["TensorDimDesc"] = None,
):
    x_baseline = _load_object(path_baseline)
    x_target = _load_object(path_target)

    print(
        f"Raw "
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[{'' if x_baseline.dtype == x_target.dtype else '🟠'}dtype] {x_baseline.dtype} vs {x_target.dtype}"
    )

    if tensor_dim_desc is not None:
        if (s := baseline_token_slice) is not None:
            dim = _get_einops_dim_index(tensor_dim_desc.baseline_desc, "num_tokens")
            x_baseline = x_baseline.narrow(
                dim=dim, start=s.start, length=s.stop - s.start
            )
        x_baseline = einops.rearrange(
            x_baseline,
            tensor_dim_desc.baseline_desc + " -> " + tensor_dim_desc.target_desc,
        )
        if (f := tensor_dim_desc.baseline_cropper) is not None:
            print("Apply baseline_cropper")
            x_baseline = f(x_baseline)

    x_baseline, x_target = _comparison_preprocessor(x_baseline, x_target, name=name)
    x_baseline = _try_unify_shape(x_baseline, target_shape=x_target.shape)

    print(
        f"After preprocessor "
        f"[shape] {x_baseline.shape} vs {x_target.shape}\t"
        f"[dtype] {x_baseline.dtype} vs {x_target.dtype}"
    )

    x_baseline_original_dtype = x_baseline.dtype
    x_target_original_dtype = x_target.dtype

    x_target = x_target.float()
    x_baseline = x_baseline.float()

    for name, fn in [
        ("mean", torch.mean),
        ("std", torch.std),
        ("min", torch.min),
        ("max", torch.max),
        *(
            [
                ("p1", functools.partial(torch.quantile, q=0.01)),
                ("p5", functools.partial(torch.quantile, q=0.05)),
                ("p95", functools.partial(torch.quantile, q=0.95)),
                ("p99", functools.partial(torch.quantile, q=0.99)),
            ]
            if x_baseline.numel() < 10_000_000
            else []
        ),
    ]:
        value_baseline = fn(x_baseline).item()
        value_target = fn(x_target).item()
        print(
            f"[{name}] {value_baseline :.4f} vs {value_target:.4f} (diff: {value_target - value_baseline:.4f})"
        )

    if x_baseline.shape != x_target.shape:
        print(f"⚠️ Shape mismatch")
        return

    diff_info = _compute_and_print_diff(
        x_baseline=x_baseline,
        x_target=x_target,
        diff_threshold=diff_threshold,
    )
    needs_print = diff_info["max_abs_diff"] > 1e-3

    if (x_baseline_original_dtype != x_target_original_dtype) and (
        (
            downcast_dtype := _compute_smaller_dtype(
                x_baseline_original_dtype, x_target_original_dtype
            )
        )
        is not None
    ):
        _compute_and_print_diff(
            x_baseline=x_baseline.to(downcast_dtype),
            x_target=x_target.to(downcast_dtype),
            diff_threshold=diff_threshold,
            prefix_text=f"When downcast to {downcast_dtype}: ",
        )

    if needs_print:
        print(f"x_baseline(sample)={get_truncated_value(x_baseline)}")
        print(f"x_target(sample)={get_truncated_value(x_target)}")


def _compute_and_print_diff(
    x_baseline, x_target, diff_threshold: float, prefix_text=""
):
    raw_abs_diff = (x_target - x_baseline).abs()

    max_abs_diff = raw_abs_diff.max().item()
    mean_abs_diff = raw_abs_diff.mean().item()
    rel_diff = _calc_rel_diff(x_target, x_baseline)

    print(
        prefix_text
        + "\t".join(
            f"{'❌' if value > diff_threshold else '✅'} {name}={value}"
            for name, value in [
                ("rel_diff", rel_diff),
                ("max_abs_diff", max_abs_diff),
                ("mean_abs_diff", mean_abs_diff),
            ]
        )
    )

    return dict(max_abs_diff=max_abs_diff)


def _compute_smaller_dtype(dtype_a, dtype_b):
    info_dict = {
        (torch.float32, torch.bfloat16): torch.bfloat16,
        # ... add more ...
    }
    return info_dict.get((dtype_a, dtype_b)) or info_dict.get((dtype_b, dtype_a))


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


def _load_object(path):
    x = torch.load(path, weights_only=False)
    if not isinstance(x, torch.Tensor):
        print(f"Skip load {path} since {type(x)=} is not a Tensor ({x=})")
        return None
    return x.cuda()


# TODO may make customization endpoints configurable via args pointing to code file
def _comparison_preprocessor(x_baseline, x_target, name):
    """Customization endpoint. Can insert arbitrary adhoc postprocessing logic here."""
    return x_baseline, x_target


@dataclass
class LocationInfo:
    baseline_forward_pass_id: int
    baseline_token_slice: slice


def _get_location_info_of_target_pass_id() -> Dict[int, LocationInfo]:
    """Customization endpoint."""
    return {}


@dataclass
class TensorDimDesc:
    baseline_desc: str
    target_desc: str
    baseline_cropper: Optional[Callable[[torch.Tensor], torch.Tensor]]


def _get_tensor_dim_descs() -> List[TensorDimDesc]:
    """Customization endpoint."""
    return []


if __name__ == "__main__":
    # python -m sglang.srt.debug_utils.dump_comparator --baseline-path ... --target-path ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=str)
    parser.add_argument("--target-path", type=str)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=1000000)
    parser.add_argument("--baseline-start-id", type=int, default=0)
    parser.add_argument("--diff-threshold", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
