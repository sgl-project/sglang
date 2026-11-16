import json
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    SHAREGPT_FILENAME,
    SHAREGPT_REPO_ID,
    BaseDataset,
    DatasetRow,
    first_two_sharegpt_turns,
    tokenize_and_filter_turn_pairs,
)
from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json


@dataclass
class ShareGPTDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    context_len: Optional[int]
    prompt_suffix: str
    apply_chat_template: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "ShareGPTDataset":
        assert not getattr(args, "tokenize_prompt", False)
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_sharegpt_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
            context_len=self.context_len,
            prompt_suffix=self.prompt_suffix,
            apply_chat_template=self.apply_chat_template,
        )


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[DatasetRow]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not is_file_valid_json(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    dataset = first_two_sharegpt_turns(dataset)

    # Shuffle the dataset.
    random.shuffle(dataset)

    return tokenize_and_filter_turn_pairs(
        dataset,
        num_requests=num_requests,
        tokenizer=tokenizer,
        fixed_output_len=fixed_output_len,
        context_len=context_len,
        prompt_suffix=prompt_suffix,
        apply_chat_template=apply_chat_template,
    )
