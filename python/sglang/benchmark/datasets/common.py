import random
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sglang.benchmark.utils import remove_suffix

ASSISTANT_SUFFIX = "Assistant:"
SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
MOONCAKE_DATASET_URL = {
    "mooncake": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
    "conversation": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
    "synthetic": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
    "toolagent": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
}


@dataclass
class DatasetRow:
    prompt: Any
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None
    routing_key: Optional[str] = None
    extra_request_body: Optional[Dict[str, Any]] = None  # Per-request API parameters

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len
        if self.vision_prompt_len is None:
            self.vision_prompt_len = 0
        if self.extra_request_body is None:
            self.extra_request_body = {}


@dataclass
class BaseDataset(ABC):
    @classmethod
    @abstractmethod
    def from_args(cls, args: Namespace) -> "BaseDataset": ...

    @abstractmethod
    def load(
        self,
        tokenizer: Any,
        model_id: Optional[str] = None,
    ) -> List[DatasetRow]: ...


def compute_random_lens(full_len: int, range_ratio: float, num: int) -> List[int]:
    # full_len=0 is valid for embedding benchmarks where no output tokens are generated
    if full_len <= 0:
        return [0] * num
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    ).tolist()


@lru_cache(maxsize=1)
def get_available_tokens(tokenizer):
    """Get valid token ids from the tokenizer vocabulary."""
    # Canonical order: vocab dict iteration order varies across tokenizers
    # versions, which would break --seed reproducibility.
    return sorted(
        token_id
        for token_id in tokenizer.get_vocab().values()
        if isinstance(token_id, int)
    )


def gen_prompt_ids(tokenizer, token_num):
    """Generate random token ids of specified length using tokenizer vocabulary."""
    all_available_tokens = get_available_tokens(tokenizer)
    return random.choices(all_available_tokens, k=token_num)


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    return tokenizer.decode(gen_prompt_ids(tokenizer, token_num))


@lru_cache(maxsize=1)
def get_available_multimodal_text_tokens(tokenizer, image_pad_id):
    """Get valid token ids for synthetic multimodal text prompts."""
    excluded_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if image_pad_id is not None:
        excluded_token_ids.add(image_pad_id)
    return [
        token_id
        for token_id in get_available_tokens(tokenizer)
        if token_id not in excluded_token_ids
    ]


def gen_mm_prompt(tokenizer, image_pad_id, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = get_available_multimodal_text_tokens(tokenizer, image_pad_id)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def first_two_sharegpt_turns(dataset: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    return [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]


def tokenize_and_filter_turn_pairs(
    dataset: List[Tuple[str, str]],
    num_requests: int,
    tokenizer: Any,
    fixed_output_len: Optional[int],
    context_len: Optional[int],
    prompt_suffix: Optional[str],
    apply_chat_template: bool,
) -> List[DatasetRow]:
    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                return_dict=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset
