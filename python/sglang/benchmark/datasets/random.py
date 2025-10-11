import json
import random
from typing import List

import numpy as np

from sglang.benchmark.datasets.common import BaseDatasetLoader, DatasetRow
from sglang.benchmark.datasets.sharegpt import SHAREGPT_URL
from sglang.benchmark.utils import download_and_cache_file, is_file_valid_json


class RandomLoader(BaseDatasetLoader):
    def load(self) -> List[DatasetRow]:
        dataset_path = self.args.dataset_path
        num_prompts = self.args.num_prompts
        input_len = self.args.random_input_len
        output_len = self.args.random_output_len
        range_ratio = self.args.random_range_ratio
        return_text = not self.args.tokenize_prompt
        random_sample = self.args.dataset_name == "random"

        input_lens = np.random.randint(
            max(int(input_len * range_ratio), 1),
            input_len + 1,
            size=num_prompts,
        )
        output_lens = np.random.randint(
            int(output_len * range_ratio),
            output_len + 1,
            size=num_prompts,
        )

        if random_sample:
            # Sample token ids from ShareGPT and repeat/truncate them to satisfy the input_lens

            # Download sharegpt if necessary
            if not is_file_valid_json(dataset_path):
                dataset_path = download_and_cache_file(SHAREGPT_URL)

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
                prompt_token_ids = self.tokenizer.encode(prompt)
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
                    input_content = self.tokenizer.decode(input_content)
                input_requests.append(
                    DatasetRow(
                        prompt=input_content,
                        prompt_len=int(input_lens[i]),
                        output_len=int(output_lens[i]),
                    )
                )
        else:
            # Sample token ids from random integers. This can cause some NaN issues.
            offsets = np.random.randint(0, self.tokenizer.vocab_size, size=num_prompts)
            input_requests = []
            for i in range(num_prompts):
                input_content = [
                    (offsets[i] + i + j) % self.tokenizer.vocab_size
                    for j in range(input_lens[i])
                ]
                if return_text:
                    input_content = self.tokenizer.decode(input_content)
                input_requests.append(
                    DatasetRow(
                        prompt=input_content,
                        prompt_len=int(input_lens[i]),
                        output_len=int(output_lens[i]),
                    )
                )

        print(f"#Input tokens: {np.sum(input_lens)}")
        print(f"#Output tokens: {np.sum(output_lens)}")
        return input_requests
