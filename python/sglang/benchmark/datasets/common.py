import json
import random
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sglang.benchmark.utils import download_and_cache_hf_file, is_file_valid_json

ASSISTANT_SUFFIX = "Assistant:"
SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
MOONCAKE_DATASET_URL = {
    "mooncake": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
    "conversation": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
    "synthetic": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
    "toolagent": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
}


def load_sharegpt_conversations(dataset_path: str) -> List[Tuple[str, str]]:
    """Load ShareGPT dataset and return (prompt, completion) pairs.

    Downloads the dataset if the path is invalid, filters to conversations
    with ≥2 turns, extracts the first two turns, and shuffles the result.
    """
    if not is_file_valid_json(dataset_path):
        dataset_path = download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )
    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]
    if not dataset:
        raise ValueError(
            f"No valid conversations with at least 2 turns found in dataset: {dataset_path}"
        )
    random.shuffle(dataset)
    return dataset


def load_sharegpt_prompts(dataset_path: str) -> List[str]:
    """Load ShareGPT dataset and return first-turn user messages only."""
    return [prompt for prompt, _ in load_sharegpt_conversations(dataset_path)]


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
    return [
        token_id
        for token_id in tokenizer.get_vocab().values()
        if isinstance(token_id, int)
    ]


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


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
