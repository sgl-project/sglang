import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

import einops
import polars as pl
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@dataclass
class ExpertDistributionModeDetailPerTokenAndBenchServingPack:
    topk_ids: torch.Tensor
    df_metadata: pl.DataFrame


def read_expert_distribution_mode_detail_per_token_and_bench_serving(dir_data):
    pack_expert_distribution = read_expert_distribution_mode_detail_per_token(dir_data)
    df_bench_serving = read_bench_serving(_single(list(Path(dir_data).glob("*.jsonl"))))

    df = df_bench_serving.join(pack_expert_distribution["df_metadata"], on="rid", how="inner")

    _check_list_is_prefix(df, "history_ids", "input_ids")
    _check_list_is_prefix(df, "input_ids", "all_ids")

    df = df.with_columns(
        input_except_history_ids=pl.col("input_ids").list.slice(pl.col("history_ids").list.len(), None),
        pack_input_except_history_start_index=pl.col("pack_start_index") + pl.col("history_ids").list.len(),
        pack_output_start_index=pl.col("pack_start_index") + pl.col("input_ids").list.len(),
    )

    df = df.sort("dataset_timestamp")

    return ExpertDistributionModeDetailPerTokenAndBenchServingPack(
        topk_ids=pack_expert_distribution["topk_ids"],
        df_metadata=df,
    )


def _check_list_is_prefix(df, col_a, col_b):
    df_violation = df.filter(~_expr_list_is_prefix(col_a, col_b))
    assert len(df_violation) == 0, f"Expect {col_a} to be prefix of {col_b}. Violation: {df_violation=}"


def _expr_list_is_prefix(col_a, col_b):
    return pl.col(col_b).list.slice(0, pl.col(col_a).list.len()) == pl.col(col_a)


def read_expert_distribution_mode_detail_per_token(dir_data):
    """
    Read `expert_distribution_recorder`'s output data when it is in mode `detail_per_token`
    """

    def _handle_record(record):
        rids_raw = torch.tensor([_rid_str_to_int64(rid) for rid in record["rids"]])
        input_ids = record["input_ids"]
        extend_seq_lens = torch.tensor(record["extend_seq_lens"])
        forward_mode = record["forward_mode"]
        topk_ids = einops.rearrange(record["topk_ids_of_layer"],
                                    "num_layer num_token top_k -> num_token num_layer top_k")

        rids_repeat_num = extend_seq_lens if forward_mode == ForwardMode.EXTEND.value else torch.full(
            (len(rids_raw),), 1)
        rids_repeated = torch.repeat_interleave(rids_raw, rids_repeat_num)

        # forward_mode_repeated = torch.full((len(input_ids),), forward_mode)

        return dict(
            rids=rids_repeated,
            # forward_modes=forward_mode_repeated,
            input_ids=input_ids,
            topk_ids=topk_ids,
        )

    def _concat_records(processed_records):
        return {
            k: torch.concat([r[k] for r in processed_records], dim=0)
            for k in processed_records[0].keys()
        }

    def _sort_by_rid(pack):
        sort_index = torch.argsort(pack["rids"], stable=True)
        return {k: v[sort_index, ...] for k, v in pack.items()}

    def _compute_df_metadata(pack):
        rids_raw = pack["rids"]

        pack_start_index = [0] + (1 + torch.argwhere(rids_raw[1:] != rids_raw[:-1])[:, 0]).tolist()
        pack_end_index = pack_start_index[1:] + [len(rids_raw)]
        all_ids = [
            pack["input_ids"][start_index:end_index]
            for start_index, end_index in zip(pack_start_index, pack_end_index, strict=True)
        ]

        df = pl.DataFrame(dict(
            rid=rids_raw[pack_start_index].tolist(),
            all_ids=all_ids,
            pack_start_index=pack_start_index,
            pack_end_index=pack_end_index,
        ))
        return {"topk_ids": pack["topk_ids"], "df_metadata": df}

    processed_records = []
    for path in tqdm(list(Path(dir_data).glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        processed_records += [_handle_record(record) for record in data_pack["records"]]

    pack = _concat_records(processed_records)
    pack = _sort_by_rid(pack)
    pack = _compute_df_metadata(pack)
    return pack


def read_bench_serving(path: Path):
    """
    Read `bench_serving.py`'s outputs
    """
    data_raw = json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(data_raw["tokenizer_id"])

    df = pl.DataFrame(dict(
        rid=[_rid_str_to_int64(x["rid"]) for x in data_raw["output_metadata"]],
        dataset_timestamp=[x["dataset_timestamp"] for x in data_raw["output_metadata"]],
        input_text=data_raw["prompts"],
        output_text=data_raw["generated_texts"],
        history_text=[x["history_text"] for x in data_raw["output_metadata"]],
    ))

    df = df.with_columns(
        input_ids=pl.col("input_text").map_elements(tokenizer.encode, return_dtype=pl.List(pl.Int32)),
        output_ids=pl.col("output_text").map_elements(tokenizer.encode, return_dtype=pl.List(pl.Int32)),
        history_ids=pl.col("history_text").map_elements(tokenizer.encode, return_dtype=pl.List(pl.Int32)),
        dataset_timestamp=pl.col("dataset_timestamp").str.to_datetime(),
    )

    return df


def unnest_all(df: pl.DataFrame, separator=".") -> pl.DataFrame:
    """https://github.com/pola-rs/polars/issues/12353"""
    return df.select(_unnest_all(df.schema, separator))


def _unnest_all(schema, separator):
    for (col, *fields), dtype in _unnest(schema, []):
        expr = pl.col(col)

        for field in fields:
            expr = expr.struct[field]

        if col == "":
            name = separator.join(fields)
        else:
            name = separator.join([col] + fields)

        yield expr.alias(name)


def _unnest(schema, path):
    for name, dtype in schema.items():
        base_type = dtype.base_type()

        if base_type == pl.Struct:
            yield from _unnest(dtype.to_schema(), path + [name])
        else:
            yield path + [name], dtype


def _rid_str_to_int64(rid: str):
    return int(rid, 16) & ((1 << 64) - 1)


def _single(arr: List[Any]):
    assert len(arr) == 1, f"{len(arr)=}"
    return arr[0]
