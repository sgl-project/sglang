import json
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
)


@dataclass
class MultimodalDataset(BaseDataset):
    input_len: int
    output_len: int
    num_requests: int
    range_ratio: float
    dataset_path: str
    return_text: bool
    random_sample: bool = True

    @classmethod
    def from_args(cls, args: Namespace) -> "MultimodalDataset":
        return cls(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_requests=args.num_prompts,
            range_ratio=args.random_range_ratio,
            dataset_path=args.dataset_path,
            return_text=not getattr(args, "tokenize_prompt", False),
            random_sample=getattr(args, "random_sample", True),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_multimodal_requests(
            input_len=self.input_len,
            output_len=self.output_len,
            num_prompts=self.num_requests,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            dataset_path=self.dataset_path,
            random_sample=self.random_sample,
            return_text=self.return_text,
        )


def sample_multimodal_requests(
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str,
    input_len: int,
    output_len: int,
    range_ratio: float,
    random_sample: bool = True,
    return_text: bool = True,
):
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )

    # Load the dataset.
    try:
        with open(dataset_path) as f:
            dataset = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid dataset path: {dataset_path}") from e

    parsed = []
    for data in dataset:
        if "conversations" in data:
            # Format: {"conversations": [{"from": "human", "value": ...}, ...],
            #          "images": [...], "audios": [...], "videos": [...]}
            question = None
            for turn in data["conversations"]:
                if turn.get("from") == "human":
                    question = turn.get("value", "")
                    break
            if question is None:
                continue
            parsed.append(
                (
                    question,
                    data.get("images", []),
                    data.get("audios", []),
                    data.get("videos", []),
                )
            )
        else:
            # Format: {"question": ..., "image_urls": [...], ...}
            question = data.get("question")
            if question is None:
                continue
            parsed.append(
                (
                    question,
                    data.get("image_urls", []),
                    data.get("audio_urls", []),
                    data.get("video_urls", []),
                )
            )
    dataset = parsed
    if random_sample:
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
        output_len = int(output_lens[i])

        # Prune too short sequences.
        if prompt_len < 2 or output_len < 2:
            continue

        # Prune too long sequences.
        if input_len and prompt_len + output_len > input_len:
            continue

        input_content = prompt if return_text else prompt_token_ids

        # Generate image data if requested

        image_data_list = list(data[1])
        audio_data_list = list(data[2])
        video_data_list = list(data[3])

        input_requests.append(
            DatasetRow(
                prompt=input_content,
                prompt_len=prompt_len,
                output_len=output_len,
                image_data=image_data_list,
                audio_data=audio_data_list,
                video_data=video_data_list,
            )
        )
    return input_requests
