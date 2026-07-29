import json
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class OpenAIDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenAIDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_openai_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
        )


def sample_openai_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[DatasetRow]:
    """
    Load OpenAI-compatible chat completion requests from a JSONL file.

    Each line should be a JSON object with:
    - "messages": list of {"role": str, "content": str}
    - "max_tokens": int (used as output_len if fixed_output_len not set)
    - "tools": optional list of tool definitions
    - "temperature": optional temperature value
    - "top_p": optional top_p value
    - Other OpenAI API parameters are also extracted and passed through
    """
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            if num_requests > 0 and len(dataset) >= num_requests:
                break
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

    # Fields that should NOT be passed through extra_request_body
    # These are either handled separately or are metadata
    # max_tokens is excluded because it's handled via output_len -> max_completion_tokens
    # max_completion_tokens is also excluded to avoid conflicts
    EXCLUDED_FIELDS = {"messages", "max_tokens", "max_completion_tokens", "model"}

    filtered_dataset: List[DatasetRow] = []
    for data in dataset:
        messages = data.get("messages", [])
        if not messages:
            continue

        # Use max_tokens from the request, or fall back to fixed_output_len
        output_len = fixed_output_len or data.get("max_tokens", 256)

        # Extract extra request body parameters (tools, temperature, top_p, etc.)
        extra_body = {k: v for k, v in data.items() if k not in EXCLUDED_FIELDS}

        # Calculate prompt length by applying chat template
        # This includes the messages but not the tools
        prompt_len = len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )

        # If tools are present, we need to add their token count
        # Tools are sent as part of the request and count toward input tokens
        if "tools" in extra_body:
            # Encode tools as JSON string to estimate token count
            tools_str = json.dumps(extra_body["tools"])
            tools_tokens = len(tokenizer.encode(tools_str))
            prompt_len += tools_tokens

        # Pass messages list directly - bench_serving handles List[Dict] prompts
        filtered_dataset.append(
            DatasetRow(
                prompt=messages,
                prompt_len=prompt_len,
                output_len=output_len,
                extra_request_body=extra_body,  # Store per-request parameters
            )
        )

    print(f"Loaded {len(filtered_dataset)} OpenAI-format requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset
