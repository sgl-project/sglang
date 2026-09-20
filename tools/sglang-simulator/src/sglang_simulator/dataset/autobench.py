"""Simulator-owned loader for timestamped Autobench JSONL traces.

The trace format is a public SGLang Simulator input contract.  Keep its parser
here instead of importing SGLang's benchmark-internal Autobench module, which
may be moved or removed independently of the simulator.
"""

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

_RESERVED_FIELDS = {
    "prompt",
    "messages",
    "prompt_origin",
    "output_len",
    "max_tokens",
    "max_completion_tokens",
    "completion_tokens",
    "prompt_len",
    "text_prompt_len",
    "vision_prompt_len",
    "image_data",
    "timestamp",
    "routing_key",
    "metadata",
    "extra_request_body",
    "param_send",
}


def _load_json_if_needed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value or value[0] not in "[{":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_messages(messages: Any) -> Optional[list[dict[str, Any]]]:
    messages = _load_json_if_needed(messages)
    if not isinstance(messages, list) or not messages:
        return None
    if not all(isinstance(message, dict) for message in messages):
        return None

    normalized = []
    for message in messages:
        if "role" not in message or message.get("content") is None:
            return None
        normalized.append({"role": message["role"], "content": message["content"]})
    return normalized


def _normalize_prompt(row: dict[str, Any]) -> tuple[Any, str]:
    for key in ("messages", "prompt_origin"):
        normalized = _normalize_messages(row.get(key))
        if normalized is not None:
            return normalized, "messages"

    prompt = _load_json_if_needed(row.get("prompt"))
    if isinstance(prompt, list) and prompt:
        if isinstance(prompt[0], dict):
            normalized = _normalize_messages(prompt)
            if normalized is not None:
                return normalized, "messages"
        if all(isinstance(item, int) for item in prompt):
            return prompt, "token_ids"
        if all(isinstance(item, str) for item in prompt):
            return prompt, "multi_turn"
        if all(
            isinstance(turn, list)
            and turn
            and all(
                isinstance(message, dict) and "role" in message and "content" in message
                for message in turn
            )
            for turn in prompt
        ):
            return prompt, "multi_turn"
    if isinstance(prompt, str) and prompt:
        return prompt, "prompt"

    if isinstance(row.get("content"), list):
        turns = [str(item) for item in row["content"]]
        if len(turns) % 2 == 0:
            turns = turns[:-1]
        messages = []
        if row.get("system"):
            messages.append({"role": "system", "content": str(row["system"])})
        messages.extend(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": turn,
            }
            for index, turn in enumerate(turns)
        )
        if messages:
            return messages, "messages"

    raise ValueError("Unsupported Autobench row: missing prompt/messages")


def _prompt_lengths(
    row: dict[str, Any],
    prompt: Any,
    prompt_kind: str,
    tokenizer: Optional[PreTrainedTokenizerBase],
) -> tuple[int, int, int]:
    if row.get("prompt_len") is not None:
        prompt_len = int(row["prompt_len"])
        return (
            prompt_len,
            int(row.get("text_prompt_len", prompt_len)),
            int(row.get("vision_prompt_len", 0)),
        )
    if prompt_kind == "token_ids":
        return len(prompt), len(prompt), 0
    if tokenizer is None:
        raise ValueError("Autobench rows without prompt_len require a tokenizer")
    if prompt_kind == "messages":
        prompt_len = len(
            tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
        )
        return prompt_len, prompt_len, 0
    if prompt_kind == "prompt":
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        return prompt_len, prompt_len, 0
    return 0, 0, 0


def _extra_request_body(row: dict[str, Any]) -> dict[str, Any]:
    extra = {}
    param_send = _load_json_if_needed(row.get("param_send"))
    if isinstance(param_send, dict):
        extra.update(param_send)
    extra.update(
        {key: value for key, value in row.items() if key not in _RESERVED_FIELDS}
    )
    explicit = _load_json_if_needed(row.get("extra_request_body"))
    if isinstance(explicit, dict):
        extra.update(explicit)
    return extra


def sample_autobench_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: Optional[PreTrainedTokenizerBase],
    fixed_output_len: Optional[int] = None,
) -> list[DatasetRow]:
    dataset = []
    with Path(dataset_path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if num_requests > 0 and len(dataset) >= num_requests:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                prompt, prompt_kind = _normalize_prompt(row)
                prompt_len, text_prompt_len, vision_prompt_len = _prompt_lengths(
                    row, prompt, prompt_kind, tokenizer
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid Autobench row {line_number} in {dataset_path}: {error}"
                ) from error

            output_len = fixed_output_len
            for key in (
                "output_len",
                "max_tokens",
                "max_completion_tokens",
                "completion_tokens",
            ):
                output_len = output_len or row.get(key)
            dataset.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=int(output_len or 256),
                    text_prompt_len=text_prompt_len,
                    vision_prompt_len=vision_prompt_len,
                    image_data=row.get("image_data"),
                    timestamp=row.get("timestamp"),
                    routing_key=row.get("routing_key"),
                    extra_request_body=_extra_request_body(row),
                )
            )

    print(f"Loaded {len(dataset)} Autobench requests")
    print(f"#Input tokens: {np.sum([row.prompt_len for row in dataset])}")
    print(f"#Output tokens: {np.sum([row.output_len for row in dataset])}")
    return dataset


@dataclass
class AutoBenchmarkDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "AutoBenchmarkDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=getattr(args, "sharegpt_output_len", None),
        )

    def load(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_id: Optional[str] = None,
    ) -> list[DatasetRow]:
        return sample_autobench_requests(
            self.dataset_path,
            self.num_requests,
            tokenizer,
            self.fixed_output_len,
        )


def register_autobench_dataset() -> None:
    """Register the simulator trace contract with SGLang's serving benchmark."""
    from sglang.benchmark import datasets

    datasets.DATASET_MAPPING["autobench"] = AutoBenchmarkDataset
