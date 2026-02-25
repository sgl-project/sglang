import json
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    SHAREGPT_FILENAME,
    SHAREGPT_REPO_ID,
    BaseDataset,
    DatasetRow,
    compute_random_lens,
)
from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json


@dataclass
class RandomDataset(BaseDataset):
    input_len: int
    output_len: int
    num_requests: int
    range_ratio: float
    dataset_path: str
    return_text: bool
    random_sample: bool

    @classmethod
    def from_args(cls, args: Namespace) -> "RandomDataset":
        return cls(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_requests=args.num_prompts,
            range_ratio=args.random_range_ratio,
            dataset_path=args.dataset_path,
            return_text=not getattr(args, "tokenize_prompt", False),
            random_sample=(args.dataset_name == "random"),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_random_requests(
            input_len=self.input_len,
            output_len=self.output_len,
            num_prompts=self.num_requests,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            random_sample=self.random_sample,
            return_text=self.return_text,
        )


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    random_sample: bool = True,
    return_text: bool = True,
) -> List[DatasetRow]:
    input_lens = compute_random_lens(
        full_len=input_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )
    output_lens = compute_random_lens(
        full_len=output_len,
        range_ratio=range_ratio,
        num=num_prompts,
    )

    if return_text:
        # Need to truncate input_len as server encode will add special token.
        num_special_tokens = int(tokenizer.num_special_tokens_to_add())
        for i in range(num_prompts):
            input_lens[i] = max(0, input_lens[i] - num_special_tokens)

    if random_sample:
        # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

        # Download sharegpt if necessary
        if not is_file_valid_json(dataset_path):
            dataset_path = download_and_cache_hf_file(
                repo_id=SHAREGPT_REPO_ID,
                filename=SHAREGPT_FILENAME,
            )

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [
            data
            for data in dataset
            if len(data.get("conversations", data.get("conversation", []))) >= 2
        ]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data.get("conversations", data.get("conversation", []))[0]["value"],
                data.get("conversations", data.get("conversation", []))[1]["value"],
            )
            for data in dataset
        ]
        # Shuffle the dataset.
        random.shuffle(dataset)

        # Filter out sequences that are too long or too short
        input_requests: List[DatasetRow] = []
        for data in dataset:
            i = len(input_requests)
            if i == num_prompts:
                break

            # Tokenize the prompts and completions.
            prompt = data[0]
            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)

            # Skip empty prompt
            if prompt_len == 0:
                continue

            if prompt_len > input_lens[i]:
                input_ids = prompt_token_ids[: input_lens[i]]
            else:
                ratio = (input_lens[i] + prompt_len - 1) // prompt_len
                input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
            input_content = input_ids
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )
    else:
        # Sample token ids from random integers. This can cause some NaN issues.
        offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
        input_requests = []
        for i in range(num_prompts):
            # Use int() to convert numpy.int64 to native Python int for JSON serialization
            input_content = [
                int((offsets[i] + i + j) % tokenizer.vocab_size)
                for j in range(input_lens[i])
            ]
            if return_text:
                input_content = tokenizer.decode(input_content)
            input_requests.append(
                DatasetRow(
                    prompt=input_content,
                    prompt_len=input_lens[i],
                    output_len=output_lens[i],
                )
            )

    print(f"#Input tokens: {np.sum(input_lens)}")
    print(f"#Output tokens: {np.sum(output_lens)}")
    return input_requests
