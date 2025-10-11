from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from transformers import PreTrainedTokenizerBase


@dataclass
class DatasetRow:
    """Represents a single data point for a benchmark request."""

    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None
    raw_data: Optional[str] = None

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len
        if self.vision_prompt_len is None:
            self.vision_prompt_len = 0


@dataclass
class RequestFuncInput:
    """Input parameters for a request function."""

    prompt: Any
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: Optional[str]
    image_data: Optional[List[str]]
    extra_request_body: Dict[str, Any]
    timestamp: Optional[float] = None


@dataclass
class RequestFuncOutput:
    """Output results from a request function."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""

    @staticmethod
    def from_input(request_input: RequestFuncInput):
        """Initializes an output object from an input object."""
        return RequestFuncOutput(prompt_len=request_input.prompt_len)


class BaseDatasetLoader(ABC):

    def __init__(self, args: Namespace, tokenizer: PreTrainedTokenizerBase):
        self.args = args
        self.tokenizer = tokenizer
        self.input_requests: List[DatasetRow] = []

    @abstractmethod
    def load(self) -> List[DatasetRow]:
        pass

    def get_request_generator(
        self,
    ) -> Optional[Callable[[], AsyncGenerator[DatasetRow, None]]]:
        return None


def create_mm_data_row(text_prompt, images, images_base64, output_len, processor):
    try:
        content_items = [
            {"type": "image_url", "image_url": {"url": img_url}}
            for img_url in images_base64
        ]
        content_items.append({"type": "text", "text": text_prompt})
        prompt_str = processor.apply_chat_template(
            [{"role": "user", "content": content_items}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        # Some tokenizers do not support list content; fall back to a placeholder in the text
        prompt_str = f"<image>{text_prompt}"

    # Calculate total tokens (text + vision)
    prompt_len = processor(
        text=[prompt_str],
        images=images,
        padding=False,
        return_tensors="pt",
    )["input_ids"].numel()

    # Calculate text-only tokens
    try:
        # Create text-only version of the prompt
        text_only_prompt = processor.apply_chat_template(
            [{"role": "user", "content": text_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        text_prompt_len = processor(
            text=[text_only_prompt],
            padding=False,
            return_tensors="pt",
        )["input_ids"].numel()
    except Exception:
        # Fallback: just tokenize the text prompt directly
        text_prompt_len = len(processor.tokenizer.encode(text_prompt))

    # Vision tokens = total tokens - text tokens
    vision_prompt_len = prompt_len - text_prompt_len

    return DatasetRow(
        prompt=text_prompt,
        prompt_len=prompt_len,
        output_len=output_len,
        text_prompt_len=text_prompt_len,
        vision_prompt_len=vision_prompt_len,
        image_data=images_base64,
    )
