import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import einops
import polars as pl
import torch
from sglang.srt.eplb_simulator.configs import MY_MODEL_CONFIG_FOR_EXPERT_LOCATION, MY_MODEL_CONFIG_NUM_EXPERTS_PER_TOK
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from tqdm.auto import tqdm
from transformers import AutoTokenizer

_DEVICE = "cuda"


@dataclass
class ExpertDistributionModeDetailPerTokenAndBenchServingPack:
    topk_ids: torch.Tensor  # (num_tokens, num_layers, num_topk)
    df_metadata: pl.DataFrame


def read_expert_distribution_mode_detail_per_token_and_bench_serving(dir_data):
    with torch.device("cuda"):
        pack_expert_distribution = read_expert_distribution_mode_detail_per_token(dir_data)
        df_bench_serving = read_bench_serving(_single(list(Path(dir_data).glob("*.jsonl"))))

        df = df_bench_serving.join(
            pack_expert_distribution["df_metadata"], on="rid", how="inner"
        )

        _check_list_is_prefix(df, "history_ids", "input_ids")
        _check_list_is_prefix(df, "input_ids", "all_ids")

        df = df.with_columns(
            input_except_history_ids=pl.col("input_ids").list.slice(
                pl.col("history_ids").list.len(), None
            ),
            pack_input_except_history_start_index=pl.col("pack_start_index")
                                                  + pl.col("history_ids").list.len(),
            pack_output_start_index=pl.col("pack_start_index")
                                    + pl.col("input_ids").list.len(),
        )

        df = df.sort("dataset_timestamp", maintain_order=True)

        return ExpertDistributionModeDetailPerTokenAndBenchServingPack(
            topk_ids=pack_expert_distribution["topk_ids"],
            df_metadata=df,
        )


def _check_list_is_prefix(df, col_a, col_b):
    df_violation = df.filter(~_expr_list_is_prefix(col_a, col_b))
    if len(df_violation) > 0:
        with pl.Config(
            fmt_str_lengths=10000,
            tbl_cols=-1,
            tbl_rows=50,
            fmt_table_cell_list_len=10000,
            tbl_width_chars=-1,
        ):
            print(f"{df_violation=}")
        raise AssertionError(f"Expect {col_a} to be prefix of {col_b}")


def _expr_list_is_prefix(col_a, col_b):
    return pl.col(col_b).list.slice(0, pl.col(col_a).list.len()) == pl.col(col_a)


def read_expert_distribution_mode_detail_per_token(
    dir_data,
    model_config_for_expert_location=MY_MODEL_CONFIG_FOR_EXPERT_LOCATION,
):
    """
    Read `expert_distribution_recorder`'s output data when it is in mode `detail_per_token`
    """

    def _handle_record(record):
        rids_raw = torch.tensor([_rid_str_to_int64(rid) for rid in record["rids"]], dtype=torch.int64)

        rids_repeat_num = (
            torch.tensor(record["extend_seq_lens"], dtype=torch.int32)
            if record["forward_mode"] == ForwardMode.EXTEND.value
            else torch.full((len(rids_raw),), 1)
        )

        input_ids = torch.tensor(record["input_ids"], dtype=torch.int32)

        return dict(
            rids_raw=rids_raw.cuda(),
            rids_repeat_num=rids_repeat_num.cuda(),
            input_ids=input_ids.cuda(),
        )

    def _concat_records(processed_records):
        return {
            k: torch.concat([r[k] for r in processed_records], dim=0)
            for k in tqdm(list(processed_records[0].keys()))
        }

    def _compute_topk_ids(pack, raw_data_packs):
        total_num_token, = pack["input_ids"].shape

        topk_ids = torch.empty(
            (total_num_token, model_config_for_expert_location.num_layers, MY_MODEL_CONFIG_NUM_EXPERTS_PER_TOK),
            dtype=torch.int16)
        counter = 0

        for raw_data_pack in tqdm(raw_data_packs, desc="compute topk ids"):
            for record in raw_data_pack["records"]:
                topk_ids_of_record = einops.rearrange(
                    record["topk_ids_of_layer"].cuda(),
                    "num_layer num_token top_k -> num_token num_layer top_k",
                )
                counter_next = counter + topk_ids_of_record.shape[0]
                topk_ids[counter:counter_next, :, :] = topk_ids_of_record
                counter = counter_next

        assert counter == total_num_token
        return dict(**pack, topk_ids=topk_ids)

    def _compute_rid(pack):
        pack = {**pack}
        rids_raw = pack.pop("rids_raw")
        rids_repeat_num = pack.pop("rids_repeat_num")
        pack["rids"] = torch.repeat_interleave(rids_raw, rids_repeat_num)
        return pack

    def _sort_by_rid(pack):
        sort_index = torch.argsort(pack["rids"], stable=True)
        return {k: v[sort_index, ...] for k, v in pack.items()}

    def _compute_df_metadata(pack):
        rids_raw = pack["rids"]

        pack_start_index = [0] + (
            1 + torch.argwhere(rids_raw[1:] != rids_raw[:-1])[:, 0]
        ).tolist()
        pack_end_index = pack_start_index[1:] + [len(rids_raw)]
        all_ids = [
            pack["input_ids"][start_index:end_index].tolist()
            for start_index, end_index in zip(
                pack_start_index, pack_end_index, strict=True
            )
        ]

        df = pl.DataFrame(
            dict(
                rid=rids_raw[pack_start_index].tolist(),
                all_ids=all_ids,
                pack_start_index=pack_start_index,
                pack_end_index=pack_end_index,
            )
        )
        return {"topk_ids": pack["topk_ids"], "df_metadata": df}

    raw_data_packs = [
        torch.load(path, weights_only=True, map_location="cpu")
        for path in tqdm(list(Path(dir_data).glob("*.pt")), desc="read raw data packs")
    ]

    processed_records = [_handle_record(record) for raw_data_pack in raw_data_packs for record in
                         raw_data_pack["records"]]

    pack = _concat_records(processed_records)
    pack = _compute_topk_ids(pack, raw_data_packs)
    pack = _compute_rid(pack)
    pack = _sort_by_rid(pack)
    pack = _compute_df_metadata(pack)
    return pack


def read_bench_serving(path: Path):
    """
    Read `bench_serving.py`'s outputs
    """

    print("[read_bench_serving] load json")
    data_raw = json.loads(path.read_text())

    print("[read_bench_serving] load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(data_raw["tokenizer_id"])

    print("[read_bench_serving] create data frame")
    df = pl.DataFrame(
        dict(
            rid=[_rid_str_to_int64(x["rid"]) for x in data_raw["output_metadata"]],
            dataset_timestamp=[
                x.get("dataset_timestamp", "2000-01-01T00:00:00") for x in data_raw["output_metadata"]
            ],
            input_text=data_raw["prompts"],
            output_text=data_raw["generated_texts"],
            history_text=[x.get("history_text", "") for x in data_raw["output_metadata"]],
        )
    )

    print("[read_bench_serving] enhance data frame")
    df = df.with_columns(
        input_ids=pl.col("input_text").map_elements(
            tokenizer.encode, return_dtype=pl.List(pl.Int32)
        ),
        output_ids=pl.col("output_text").map_elements(
            tokenizer.encode, return_dtype=pl.List(pl.Int32)
        ),
        history_ids=pl.col("history_text").map_elements(
            tokenizer.encode, return_dtype=pl.List(pl.Int32)
        ),
        dataset_timestamp=pl.col("dataset_timestamp").str.to_datetime(),
    )

    print("[read_bench_serving] end")
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
    val = int(rid, 16) & ((1 << 64) - 1)
    if val >= (1 << 63):
        val -= (1 << 64)
    return val


def _single(arr: List[Any]):
    assert len(arr) == 1, f"{len(arr)=}"
    return arr[0]
