import json
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict, List

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    MOONCAKE_DATASET_URL,
    BaseDataset,
    DatasetRow,
)
from sglang.benchmark.utils import download_and_cache_file


@dataclass
class MooncakeDataset(BaseDataset):
    dataset_path: str
    mooncake_workload: str
    num_requests: int
    num_rounds: int
    block_size: int

    @classmethod
    def from_args(cls, args: Namespace) -> "MooncakeDataset":
        return cls(
            dataset_path=args.dataset_path,
            mooncake_workload=args.mooncake_workload,
            num_requests=args.num_prompts,
            num_rounds=args.mooncake_num_rounds,
            block_size=args.mooncake_block_size,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        if tokenizer is None:
            raise ValueError("MooncakeDataset requires a tokenizer to expand sessions.")

        if not self.dataset_path:
            local_path = os.path.join("/tmp", self.mooncake_workload + "_trace.jsonl")
        else:
            local_path = self.dataset_path

        if not os.path.exists(local_path):
            download_and_cache_file(
                MOONCAKE_DATASET_URL[self.mooncake_workload], local_path
            )

        with open(local_path, "r") as f:
            all_requests_data = [json.loads(line) for line in f if line.strip()]

        return expand_mooncake_requests(
            all_requests_data[: self.num_requests],
            tokenizer,
            num_rounds=self.num_rounds,
            block_size=self.block_size,
        )


def build_mooncake_session_requests(
    record: Dict,
    tokenizer: PreTrainedTokenizerBase,
    num_rounds: int,
    block_size: int = 128,
) -> List[DatasetRow]:
    """Expand one Mooncake trace session into concrete benchmark requests."""
    user_query_base = ""
    hash_ids = record.get("hash_ids", [])
    for hash_id in hash_ids:
        user_query_base += f"{hash_id}" + " ".join(["hi"] * block_size)
    user_query_base += "Tell me a story based on this context."

    output_len_per_round = record.get("output_length", 256)
    chat_history = []
    requests = []

    for i in range(num_rounds):
        chat_history.append(
            {"role": "user", "content": f"Round {i + 1}: {user_query_base}"}
        )

        try:
            full_prompt_text = tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True,
                return_dict=False,
            )
        except Exception:
            full_prompt_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in chat_history]
            )

        prompt_len = len(tokenizer.encode(full_prompt_text))
        requests.append(
            DatasetRow(
                prompt=full_prompt_text,
                prompt_len=prompt_len,
                output_len=output_len_per_round,
                timestamp=record.get("timestamp") if i == 0 else None,
            )
        )

        placeholder_response = " ".join(["story"] * output_len_per_round)
        chat_history.append({"role": "assistant", "content": placeholder_response})

    return requests


def expand_mooncake_requests(
    records: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    num_rounds: int,
    block_size: int = 128,
) -> List[DatasetRow]:
    return [
        request
        for record in records
        for request in build_mooncake_session_requests(
            record,
            tokenizer,
            num_rounds=num_rounds,
            block_size=block_size,
        )
    ]
