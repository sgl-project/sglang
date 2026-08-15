import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

_DESCRIPTION = """Compare and find differences to benchmark outputs.

Supported inputs:
* The samples jsonl from `lm_eval --log_samples --output_path FOLDER_NAME`
* The output from `gsm8k/bench_sglang.py --raw-result-file FILE_NAME` (or mmlu)
"""


def main(args):
    if args.data_type == "simple_evals":
        df_input = _compute_df_input_mode_simple_evals(args)
    else:
        df_input = _transform_df_input(_compute_df_raw(args))

    assert all(
        c in df_input.columns
        for c in ["category", "trial_index", "prompt_id", "prompt", "output", "correct"]
    )

    df_meta = _compute_df_meta(df_input)

    df_correctness_per_trial = df_input.group_by(
        "category", "trial_index", maintain_order=True
    ).agg(pl.col("correct").mean())
    df_correctness_delta = (
        df_meta.group_by("correctness_delta").len().sort("correctness_delta")
    )
    df_good_to_bad = df_meta.filter(pl.col("correctness_delta") < 0)
    df_bad_to_good = df_meta.filter(pl.col("correctness_delta") > 0)

    print(f"Dump output to {args.output_path}")
    Path(args.output_path).write_text(
        json.dumps(
            dict(
                df_meta=df_meta.to_dicts(),
                df_good_to_bad=df_good_to_bad.to_dicts(),
                df_bad_to_good=df_bad_to_good.to_dicts(),
            ),
            indent=4,
        ),
    )

    if not args.disable_print_details:
        with pl.Config(
            fmt_str_lengths=10000,
            tbl_cols=-1,
            tbl_rows=-1,
            tbl_width_chars=-1,
            tbl_formatting="UTF8_FULL",
        ):
            print("====== Correctness per trial ======")
            print(df_correctness_per_trial)

            print(
                "====== Correctness Delta (-1.0 means all-right becomes all-wrong) ======"
            )
            print(df_correctness_delta)

            for name, df in [
                ("Good->Bad", df_good_to_bad),
                ("Bad->Good", df_bad_to_good),
            ]:
                print(f"====== Concrete Examples: {name} ======")
                print(df)


def _compute_df_input_mode_simple_evals(args):
    return pl.concat(
        [
            _compute_df_input_one_mode_simple_evals(**info)
            for info in _get_file_infos(args=args)
        ]
    )


def _compute_df_input_one_mode_simple_evals(path, category, trial_index):
    data = json.loads(Path(path).read_text())
    rows = []

    for single_eval_result in data["metadata"]["single_eval_results"]:
        prompt = single_eval_result["example_level_metadata"][
            "actual_queried_prompt_messages"
        ]
        score = single_eval_result["score"]
        assert score in {0.0, 1.0}, f"{score=}"

        row = dict(
            category=category,
            trial_index=trial_index,
            prompt_id=_compute_id_from_object(prompt),
            prompt=json.dumps(prompt),
            output=single_eval_result["example_level_metadata"]["response_text"],
            correct=score == 1.0,
        )
        rows.append(row)

    return pl.DataFrame(rows)


def _compute_id_from_object(obj):
    if isinstance(obj, pl.Series):
        obj = obj.to_list()
    json_str = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def _compute_df_raw(args):
    return pl.concat(
        [
            _read_df_raw(
                path=info["path"],
                category=info["category"],
                trial_index=info["trial_index"],
            )
            for info in _get_file_infos(args=args)
        ]
    )


def _get_file_infos(args):
    return [
        dict(path=path, category=category, trial_index=trial_index)
        for category, paths in [
            ("baseline", args.baseline_path),
            ("target", args.target_path),
        ]
        for trial_index, path in enumerate(paths)
    ]


def _read_df_raw(path: str, category: str, trial_index: int):
    return pl.read_ndjson(path).with_columns(
        category=pl.lit(category), trial_index=trial_index
    )


def _transform_df_input(df: pl.DataFrame):
    if "doc_id" in df.columns:
        print("Transform mode: lm_eval")

        filter_names = df["filter"].unique(maintain_order=True).to_list()
        if len(filter_names) > 1:
            filter_name = filter_names[0]
            print(f"Choose {filter_name=} among {filter_names}")
            df = df.filter(pl.col("filter") == filter_name)

        df = df.select(
            pl.col("category"),
            pl.col("trial_index"),
            prompt_id=pl.col("doc_id"),
            prompt=pl.col("arguments").struct.field("gen_args_0").struct.field("arg_0"),
            output=pl.col("resps").list.get(0).list.get(0),
            correct=pl.col("exact_match").cast(bool),
        )

        return df
    elif "prompt_id" in df.columns:
        print("Transform mode: SGLang bench")
        return df
    else:
        raise Exception(
            f"Unknown data: {df.columns}. You may need to set `--data-type` if using e.g. simple_evals."
        )


def _compute_df_meta(df_input: pl.DataFrame):
    df_input = df_input.sort("prompt_id", "category", "trial_index")
    df_meta = pl.DataFrame(
        [
            _handle_one_prompt(df_one_prompt)
            for df_one_prompt in df_input.partition_by("prompt_id", maintain_order=True)
        ]
    )
    df_meta = df_meta.with_columns(
        correctness_delta=pl.col("correctness_target") - pl.col("correctness_baseline"),
    )
    df_meta = df_meta.sort("correctness_delta", "output_same_prefix_len")
    return df_meta


def _handle_one_prompt(df_one_prompt: pl.DataFrame):
    assert (
        len(set(_compute_id_from_object(obj) for obj in df_one_prompt["prompt"])) == 1
    )

    df_baseline = df_one_prompt.filter(pl.col("category") == "baseline")
    df_target = df_one_prompt.filter(pl.col("category") == "target")

    outputs_baseline = df_baseline["output"].to_list()
    outputs_target = df_target["output"].to_list()

    output_same_prefix_len = max(
        _compute_str_prefix_len(output_baseline, output_target)
        for output_baseline in outputs_baseline
        for output_target in outputs_target
    )

    return dict(
        prompt_id=df_one_prompt[0, "prompt_id"],
        correctness_baseline=df_baseline["correct"].mean(),
        correctness_target=df_target["correct"].mean(),
        output_same_prefix_len=output_same_prefix_len,
        prompt=df_one_prompt[0, "prompt"],
        outputs_baseline=outputs_baseline,
        outputs_target=outputs_target,
    )


def _compute_str_prefix_len(a: str, b: str) -> int:
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] != b[i]:
            return i
    return min_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument("--data-type", type=str, default="auto")
    parser.add_argument("--baseline-path", type=str, nargs="+")
    parser.add_argument("--target-path", type=str, nargs="+")
    parser.add_argument(
        "--output-path", type=str, default="/tmp/text_comparator_output.json"
    )
    parser.add_argument("--disable-print-details", action="store_true")
    args = parser.parse_args()
    main(args)
