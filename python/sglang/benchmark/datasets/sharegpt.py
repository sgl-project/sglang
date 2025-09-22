import json
import random
from typing import List, Optional

import numpy as np

from sglang.benchmark.datasets.common import BaseDatasetLoader, DatasetRow
from sglang.benchmark.utils import download_and_cache_file, is_file_valid_json

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
ASSISTANT_SUFFIX = "Assistant:"


def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text


class ShareGPTLoader(BaseDatasetLoader):
    def load(self) -> List[DatasetRow]:
        assert not self.args.tokenize_prompt

        dataset_path = self.args.dataset_path
        num_requests = self.args.num_prompts
        fixed_output_len = self.args.sharegpt_output_len
        context_len = self.args.sharegpt_context_len
        prompt_suffix = self.args.prompt_suffix
        apply_chat_template = self.args.apply_chat_template

        if fixed_output_len is not None and fixed_output_len < 4:
            raise ValueError("sharegpt output_len is too small")

        # Download sharegpt if necessary
        if not is_file_valid_json(dataset_path) and dataset_path == "":
            dataset_path = download_and_cache_file(SHAREGPT_URL)

        # Load the dataset.
        with open(dataset_path) as f:
            dataset = json.load(f)

        # Filter out conversations with less than 2 turns.
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
        filtered_dataset: List[DatasetRow] = []
        for i in range(len(dataset)):
            if len(filtered_dataset) == num_requests:
                break

            prompt = dataset[i][0]
            if prompt_suffix:
                prompt = (
                    remove_suffix(prompt, ASSISTANT_SUFFIX)
                    + prompt_suffix
                    + ASSISTANT_SUFFIX
                )

            if apply_chat_template:
                templated_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                if self.tokenizer.bos_token:
                    prompt = templated_prompt.replace(self.tokenizer.bos_token, "")

            prompt_token_ids = self.tokenizer.encode(prompt)
            completion = dataset[i][1]
            completion_token_ids = self.tokenizer.encode(completion)
            prompt_len = len(prompt_token_ids)
            output_len = (
                len(completion_token_ids)
                if fixed_output_len is None
                else fixed_output_len
            )

            if prompt_len < 4 or output_len < 2:
                # Prune too short sequences.
                continue

            if context_len and prompt_len + output_len > context_len:
                # Prune too long sequences.
                continue

            filtered_dataset.append(
                DatasetRow(prompt=prompt, prompt_len=prompt_len, output_len=output_len)
            )

        print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
        print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
        return filtered_dataset
