import json
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

AUTOBENCH_RESERVED_FIELDS = {
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
    if not value:
        return value
    if value[0] not in "[{":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_messages(messages: Any) -> Optional[List[Dict[str, Any]]]:
    messages = _load_json_if_needed(messages)
    if not isinstance(messages, list) or not messages:
        return None
    if not all(isinstance(message, dict) for message in messages):
        return None

    normalized = []
    for message in messages:
        if "role" not in message:
            return None
        content = message.get("content")
        if content is None:
            return None
        normalized.append({"role": message["role"], "content": content})
    return normalized


def _normalize_legacy_system_content(
    system_prompt: Any, content_list: Any
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(content_list, list) or not content_list:
        return None

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    turns = [str(item) for item in content_list]
    # In the old auto_benchmark helpers, an even number of items usually means the
    # last assistant reply is present and should be removed before benchmarking.
    if len(turns) % 2 == 0:
        turns = turns[:-1]
    if not turns:
        return None

    for index, turn in enumerate(turns):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append({"role": role, "content": turn})
    return messages


def _normalize_prompt(row: Dict[str, Any]) -> Tuple[Any, str]:
    prompt = row.get("prompt")
    messages = row.get("messages")
    prompt_origin = row.get("prompt_origin")

    if messages is not None:
        normalized = _normalize_messages(messages)
        if normalized is not None:
            return normalized, "messages"

    if prompt is not None:
        prompt = _load_json_if_needed(prompt)
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            normalized = _normalize_messages(prompt)
            if normalized is not None:
                return normalized, "messages"
        if (
            isinstance(prompt, list)
            and prompt
            and all(isinstance(item, str) for item in prompt)
        ):
            return prompt, "multi_turn"
        if (
            isinstance(prompt, list)
            and prompt
            and all(isinstance(item, int) for item in prompt)
        ):
            return prompt, "token_ids"
        if isinstance(prompt, str) and prompt:
            return prompt, "prompt"

    if prompt_origin is not None:
        normalized = _normalize_messages(prompt_origin)
        if normalized is not None:
            return normalized, "messages"

    if "system" in row and "content" in row:
        normalized = _normalize_legacy_system_content(
            row.get("system"), row.get("content")
        )
        if normalized is not None:
            return normalized, "messages"

    raise ValueError("Unsupported auto benchmark row: missing prompt/messages")


def _estimate_prompt_lens(
    prompt: Any,
    prompt_kind: str,
    tokenizer: PreTrainedTokenizerBase,
    row: Dict[str, Any],
) -> Tuple[int, int, int]:
    if row.get("prompt_len") is not None:
        prompt_len = int(row["prompt_len"])
        text_prompt_len = int(row.get("text_prompt_len", prompt_len))
        vision_prompt_len = int(row.get("vision_prompt_len", 0))
        return prompt_len, text_prompt_len, vision_prompt_len

    if prompt_kind == "messages":
        text_prompt_len = len(
            tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
        )
        vision_prompt_len = 0
        return text_prompt_len, text_prompt_len, vision_prompt_len

    if prompt_kind == "prompt":
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        return prompt_len, prompt_len, 0

    if prompt_kind == "token_ids":
        prompt_len = len(prompt)
        return prompt_len, prompt_len, 0

    # Multi-turn prompt lists are handled specially by bench_serving and do not
    # contribute reliable static prompt lengths.
    return 0, 0, 0


def _collect_extra_request_body(row: Dict[str, Any]) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}

    param_send = row.get("param_send")
    if param_send is not None:
        parsed = _load_json_if_needed(param_send)
        if isinstance(parsed, dict):
            extra.update(parsed)

    for key, value in row.items():
        if key not in AUTOBENCH_RESERVED_FIELDS:
            extra[key] = value

    explicit_extra = row.get("extra_request_body")
    explicit_extra = _load_json_if_needed(explicit_extra)
    if isinstance(explicit_extra, dict):
        extra.update(explicit_extra)

    return extra


def serialize_dataset_row_to_autobench(
    row: DatasetRow, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "prompt": row.prompt,
        "output_len": row.output_len,
    }
    if row.prompt_len:
        record["prompt_len"] = row.prompt_len
    if row.text_prompt_len not in (None, row.prompt_len):
        record["text_prompt_len"] = row.text_prompt_len
    if row.vision_prompt_len:
        record["vision_prompt_len"] = row.vision_prompt_len
    if row.image_data:
        record["image_data"] = row.image_data
    if row.timestamp is not None:
        record["timestamp"] = row.timestamp
    if row.routing_key is not None:
        record["routing_key"] = row.routing_key
    if row.extra_request_body:
        record["extra_request_body"] = row.extra_request_body
    if metadata:
        record["metadata"] = metadata
    return record


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
            fixed_output_len=args.sharegpt_output_len,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_autobench_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
        )


def sample_autobench_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[DatasetRow]:
    dataset: List[DatasetRow] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if num_requests > 0 and len(dataset) >= num_requests:
                break

            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            prompt, prompt_kind = _normalize_prompt(row)
            prompt_len, text_prompt_len, vision_prompt_len = _estimate_prompt_lens(
                prompt, prompt_kind, tokenizer, row
            )

            output_len = fixed_output_len or row.get("output_len")
            output_len = output_len or row.get("max_tokens")
            output_len = output_len or row.get("max_completion_tokens")
            output_len = output_len or row.get("completion_tokens")
            output_len = int(output_len or 256)

            dataset.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    text_prompt_len=text_prompt_len,
                    vision_prompt_len=vision_prompt_len,
                    image_data=row.get("image_data"),
                    timestamp=row.get("timestamp"),
                    routing_key=row.get("routing_key"),
                    extra_request_body=_collect_extra_request_body(row),
                )
            )

    print(f"Loaded {len(dataset)} auto benchmark requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in dataset])}")
    return dataset
