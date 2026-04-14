import json
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class TokenTraceDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "TokenTraceDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
        )

    def load(self, tokenizer, model_id=None) -> List[DatasetRow]:
        return sample_token_trace_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            fixed_output_len=self.fixed_output_len,
        )


def sample_token_trace_requests(
    dataset_path: str,
    num_requests: int,
    fixed_output_len: Optional[int] = None,
) -> List[DatasetRow]:
    """
    Load a token-level trace JSONL file without shuffling.

    Each line should contain:
    - input_tokens: list[int]
    - output_tokens: list[int] or num_output_tokens: int
    - optional ts: float timestamp
    """
    dataset: List[DatasetRow] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if num_requests > 0 and len(dataset) >= num_requests:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_tokens = record.get("input_tokens")
            if not input_tokens:
                continue

            prompt_len = int(record.get("num_input_tokens", len(input_tokens)))
            if fixed_output_len is not None:
                output_len = fixed_output_len
            elif "num_output_tokens" in record:
                output_len = int(record["num_output_tokens"])
            else:
                output_tokens = record.get("output_tokens", [])
                output_len = len(output_tokens)

            if prompt_len < 1 or output_len < 1:
                continue

            dataset.append(
                DatasetRow(
                    prompt=input_tokens,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    timestamp=record.get("ts"),
                )
            )

    print(f"Loaded {len(dataset)} token-trace requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    return dataset
