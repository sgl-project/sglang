import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

LONGBENCH_V2_REPO_ID = "THUDM/LongBench-v2"
LONGBENCH_V2_DEFAULT_OUTPUT_LEN = 10  # answer letter + short explanation


def _format_prompt(example: dict) -> str:
    return (
        f"{example['context']}\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['choice_A']}\n"
        f"B. {example['choice_B']}\n"
        f"C. {example['choice_C']}\n"
        f"D. {example['choice_D']}\n"
        f"Answer:"
    )


@dataclass
class LongBenchV2Dataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    context_len: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "LongBenchV2Dataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_longbench_v2_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            context_len=self.context_len,
        )


def sample_longbench_v2_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
) -> List[DatasetRow]:
    output_len = (
        fixed_output_len
        if fixed_output_len is not None
        else LONGBENCH_V2_DEFAULT_OUTPUT_LEN
    )

    # Load dataset
    if dataset_path:
        # Local file (parquet or JSON lines)
        import pandas as pd

        if dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
            examples = df.to_dict(orient="records")
        else:
            import json

            with open(dataset_path) as f:
                examples = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        ds = load_dataset(LONGBENCH_V2_REPO_ID, split="train")
        examples = list(ds)

    random.shuffle(examples)

    rows: List[DatasetRow] = []
    for example in examples:
        if len(rows) >= num_requests:
            break

        prompt = _format_prompt(example)
        prompt_ids = tokenizer(prompt).input_ids
        prompt_len = len(prompt_ids)

        if context_len is not None and prompt_len + output_len > context_len:
            continue

        rows.append(
            DatasetRow(prompt=prompt, prompt_len=prompt_len, output_len=output_len)
        )

    return rows
