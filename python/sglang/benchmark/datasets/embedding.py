import json
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


EmbeddingInput = Union[str, List[str]]


@dataclass
class EmbeddingDataset(BaseDataset):
    """Load OpenAI-compatible embedding requests from a JSONL file."""

    dataset_path: str
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "EmbeddingDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_embedding_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
        )


def _get_embedding_input(value: Any) -> EmbeddingInput | None:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list) and value and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return value
    return None


def sample_embedding_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[DatasetRow]:
    """Load embedding requests from JSONL without requiring model inference.

    Each line must contain an ``input`` string or non-empty list of strings.
    Other fields are passed through to the OpenAI-compatible embeddings API.
    """
    requests: List[DatasetRow] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if num_requests > 0 and len(requests) >= num_requests:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            input_value = _get_embedding_input(data.get("input"))
            if input_value is None:
                continue

            inputs = [input_value] if isinstance(input_value, str) else input_value
            prompt_len = sum(len(tokenizer.encode(text)) for text in inputs)
            extra_body = {
                key: value for key, value in data.items() if key != "input"
            }
            requests.append(
                DatasetRow(
                    prompt=input_value,
                    prompt_len=prompt_len,
                    output_len=0,
                    extra_request_body=extra_body,
                )
            )

    print(f"Loaded {len(requests)} embedding requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in requests])}")
    return requests

