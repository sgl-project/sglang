import random
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np

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
    return np.random.randint(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    ).tolist()


@lru_cache(maxsize=1)
def get_available_tokens(tokenizer):
    """Get all available token ids from the tokenizer vocabulary."""
    return list(tokenizer.get_vocab().values())


def gen_prompt(tokenizer, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def gen_mm_prompt(tokenizer, image_pad_id, token_num):
    """Generate a random prompt of specified token length using tokenizer vocabulary."""
    all_available_tokens = list(tokenizer.get_vocab().values())
    if image_pad_id:
        all_available_tokens.remove(image_pad_id)
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)
