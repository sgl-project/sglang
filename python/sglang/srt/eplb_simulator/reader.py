from pathlib import Path

import einops
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from tqdm.auto import tqdm


def read_mode_detail_per_token(dir_data):
    def _handle_record(record):
        rids_raw = torch.tensor([int(rid, 16) & ((1 << 64) - 1) for rid in record["rids"]])
        input_ids = record["input_ids"]
        extend_seq_lens = torch.tensor(record["extend_seq_lens"])
        forward_mode = record["forward_mode"]
        topk_ids = einops.rearrange(record["topk_ids_of_layer"],
                                    "num_layer num_token top_k -> num_token num_layer top_k")

        rids_repeat_num = extend_seq_lens if forward_mode == ForwardMode.EXTEND.value else torch.full(
            (len(rids_raw),), 1)
        rids_repeated = torch.repeat_interleave(rids_raw, rids_repeat_num)

        forward_mode_repeated = torch.full((len(input_ids),), forward_mode)

        return dict(
            rids=rids_repeated,
            forward_modes=forward_mode_repeated,
            input_ids=input_ids,
            topk_ids=topk_ids,
        )

    def _concat_tokens(processed_records):
        return {
            k: torch.concat([r[k] for r in processed_records], dim=0)
            for k in processed_records[0].keys()
        }

    def _sort_by_rid(pack):
        sort_index = torch.argsort(pack["rids"], stable=True)
        return {k: v[sort_index, ...] for k, v in pack.items()}

    def _compute_df_metadata(pack):
        df_metadata = TODO
        return {**pack, "df_metadata": df_metadata}

    processed_records = []
    for path in tqdm(list(Path(dir_data).glob("*.pt"))):
        data_pack = torch.load(path, weights_only=True)
        processed_records += [_handle_record(record) for record in data_pack["records"]]

    pack = _concat_tokens(processed_records)
    pack = _sort_by_rid(pack)
    pack = _compute_df_metadata(pack)
    return pack


def read_bench_serving_output(path: Path):
    TODO
