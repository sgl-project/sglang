from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    prompt: str
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


class BaseDatasetLoader(ABC):
    @abstractmethod
    def load(self, args, tokenizer, model_id=None) -> List[Any]:
        raise NotImplementedError
