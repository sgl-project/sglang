from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from .types import (
    CODEFORCES_LANGUAGE_ALIASES,
    CODEFORCES_OPTIONAL_COLUMNS,
    CODEFORCES_REQUIRED_COLUMNS,
    DAPO_MATH_17K_DEFAULT_PROMPT_COLUMN,
    DEFAULT_PROMPT_COLUMN_CANDIDATES,
    PromptSample,
)


def infer_prompt_column(
    available_columns: list[str],
) -> str:
    """Choose a prompt column from common names in a parquet schema."""
    for candidate in DEFAULT_PROMPT_COLUMN_CANDIDATES:
        if candidate in available_columns:
            return candidate

    raise ValueError(
        "Unable to auto-detect the prompt column. "
        f"Available columns: {available_columns}"
    )


def resolve_dapo_math_17k_prompt_column(
    available_columns: list[str],
    prompt_column: str | None = None,
) -> str:
    """Validate and return the DAPO-Math-17k column that stores chat prompts."""
    selected_column = prompt_column or DAPO_MATH_17K_DEFAULT_PROMPT_COLUMN
    if selected_column not in available_columns:
        raise ValueError(
            "dataset-format=dapo_math_17k requires a prompt-style column with "
            "chat messages. "
            f"Requested column: {selected_column!r}. "
            f"Available columns: {available_columns}"
        )
    return selected_column


def _looks_like_chat_message(value: Any) -> bool:
    """Return whether a value resembles a single chat message dict."""
    return isinstance(value, dict) and "role" in value and "content" in value


def _is_chat_message_list(value: Any) -> bool:
    """Return whether a value is a non-empty list of chat message dicts."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_looks_like_chat_message(item) for item in value)
    )


def _flatten_message_content(content: Any) -> str:
    """Flatten string or multimodal-style message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_segments: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_segments.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_segments.append(item["text"])
        return "".join(text_segments)
    return str(content)


def _messages_to_fallback_text(messages: list[dict[str, Any]]) -> str:
    """Render chat messages into a simple role-prefixed text fallback."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _flatten_message_content(message.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _maybe_parse_json_prompt(value: Any) -> Any:
    """Parse prompt strings that contain JSON or Python literal structures."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value



_CHATML_ROLE_PATTERN = re.compile(r"<\|im_start\|>(system|user|assistant)\n")


def _maybe_append_chatml_generation_prompt(
    prompt: str, *, enable_thinking: bool = False
) -> str:
    """Ensure ChatML text prompts end with an assistant generation prefix."""
    stripped = prompt.rstrip()
    if "<|im_start|>" not in stripped or "<|im_end|>" not in stripped:
        return prompt

    role_matches = list(_CHATML_ROLE_PATTERN.finditer(stripped))
    if not role_matches:
        return prompt

    last_role = role_matches[-1].group(1)
    thinking_suffix = "<think>\n" if enable_thinking else ""

    # The prompt already ends with an assistant generation prefix.
    if last_role == "assistant" and not stripped.endswith("<|im_end|>"):
        if enable_thinking and not stripped.endswith("<think>"):
            return stripped + thinking_suffix
        return stripped

    # ChatML user/system turns should terminate with an assistant prefix for generation.
    if last_role in {"system", "user"} and stripped.endswith("<|im_end|>"):
        return stripped + "\n<|im_start|>assistant\n" + thinking_suffix

    return prompt


def _build_chat_template_renderer(model_path: str, *, enable_thinking: bool = False):
    """Create a tokenizer-backed chat-template renderer when available."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return None
    if not getattr(tokenizer, "chat_template", None):
        return None

    def render(messages: list[dict[str, Any]]) -> str:
        """Render chat messages through the loaded tokenizer chat template."""
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = _maybe_append_chatml_generation_prompt(
                prompt, enable_thinking=enable_thinking
            )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("tokenizer.apply_chat_template returned an empty prompt")
        return prompt

    return render


def _normalize_prompt(
    value: Any,
    row_index: int,
    column_name: str,
    chat_template_renderer,
    *,
    enable_thinking: bool = False,
) -> str:
    """Normalize a raw dataset value into the final string prompt."""
    if value is None:
        raise ValueError(
            f"Row {row_index} in column {column_name!r} is null, cannot build a prompt."
        )
    value = _maybe_parse_json_prompt(value)
    if isinstance(value, str):
        return _maybe_append_chatml_generation_prompt(
            value, enable_thinking=enable_thinking
        )
    if _is_chat_message_list(value):
        if chat_template_renderer is not None:
            try:
                return chat_template_renderer(value)
            except Exception:
                pass
        return _messages_to_fallback_text(value)
    return str(value)


def _build_dapo_math_17k_prompt(
    row: dict[str, Any],
    *,
    row_index: int,
    prompt_column: str,
    chat_template_renderer,
    enable_thinking: bool = False,
) -> str:
    """Build one prompt from a DAPO-Math-17k parquet row."""
    if prompt_column not in row:
        raise ValueError(
            f"Row {row_index} is missing DAPO-Math prompt column {prompt_column!r}."
        )
    return _normalize_prompt(
        row.get(prompt_column),
        row_index,
        prompt_column,
        chat_template_renderer,
        enable_thinking=enable_thinking,
    )


def _build_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to count prompt tokens for this benchmark."
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_path)


def _encode_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.encode(prompt))


def _stringify_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_codeforces_examples(examples: Any) -> str:
    if not isinstance(examples, list):
        return ""

    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            continue
        input_text = _stringify_optional_text(example.get("input"))
        output_text = _stringify_optional_text(example.get("output"))
        if not input_text and not output_text:
            continue

        parts = [f"Example {index}"]
        if input_text:
            parts.append("Input:")
            parts.append(input_text)
        if output_text:
            parts.append("Output:")
            parts.append(output_text)
        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


def _build_codeforces_raw_prompt(
    row: dict[str, Any],
    *,
    row_index: int,
    chat_template_renderer,
    code_language: str,
) -> str:
    missing_values = [
        column
        for column in CODEFORCES_REQUIRED_COLUMNS
        if not _stringify_optional_text(row.get(column))
    ]
    if missing_values:
        raise ValueError(
            f"Row {row_index} is missing required Codeforces fields: {missing_values}"
        )

    normalized_language = CODEFORCES_LANGUAGE_ALIASES.get(
        code_language.lower(), code_language
    )
    title = _stringify_optional_text(row.get("title"))
    description = _stringify_optional_text(row.get("description"))
    input_format = _stringify_optional_text(row.get("input_format"))
    output_format = _stringify_optional_text(row.get("output_format"))
    note = _stringify_optional_text(row.get("note"))
    input_mode = _stringify_optional_text(row.get("input_mode"))
    interaction_format = _stringify_optional_text(row.get("interaction_format"))
    examples_text = _format_codeforces_examples(row.get("examples"))

    limit_parts = [
        f"Time limit: {_stringify_optional_text(row.get('time_limit'))} seconds",
        f"Memory limit: {_stringify_optional_text(row.get('memory_limit'))} MB",
    ]
    if input_mode:
        if input_mode == "stdio":
            limit_parts.append("I/O mode: standard input and standard output")
        else:
            limit_parts.append(f"I/O mode: {input_mode}")

    user_sections = [
        (
            f"Write a correct and efficient {normalized_language} solution for the "
            "following competitive programming problem."
        ),
        "\n".join(limit_parts),
    ]
    if title:
        user_sections.append(f"Title: {title}")
    user_sections.extend(
        [
            "Problem description:",
            description,
            "Input format:",
            input_format,
            "Output format:",
            output_format,
        ]
    )
    if interaction_format:
        user_sections.extend(["Interaction format:", interaction_format])
    if note:
        user_sections.extend(["Notes:", note])
    if examples_text:
        user_sections.extend(["Examples:", examples_text])
    user_sections.append(
        (
            f"Return only the final {normalized_language} source code. "
            "Do not include explanations or Markdown fences."
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert competitive programmer. Produce a correct, "
                "efficient solution that respects the stated constraints."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(section for section in user_sections if section),
        },
    ]

    if chat_template_renderer is not None:
        return chat_template_renderer(messages)
    return _messages_to_fallback_text(messages)


def load_prompt_samples(
    args: argparse.Namespace,
) -> tuple[str, list[PromptSample], int]:
    if args.prompt is not None and args.dataset_path is not None:
        raise ValueError("--prompt and --dataset-path are mutually exclusive")
    if args.prompt is None and args.dataset_path is None:
        raise ValueError("Either --prompt or --dataset-path must be provided")
    if args.context_length <= 0:
        raise ValueError("context-length must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_prompt_length is not None and args.max_prompt_length <= 0:
        raise ValueError("max-prompt-length must be positive when set")

    tokenizer_path = args.tokenizer_path or args.target_model_path
    tokenizer = _build_tokenizer(tokenizer_path)

    if args.prompt is not None:
        prompt = args.prompt
        if not args.disable_chat_template:
            chat_template_renderer = _build_chat_template_renderer(
                tokenizer_path, enable_thinking=args.enable_thinking
            )
            if chat_template_renderer is not None:
                prompt = chat_template_renderer(
                    [{"role": "user", "content": args.prompt}]
                )
        prompt_input_ids = _encode_prompt_tokens(tokenizer, prompt)
        prompt_tokens = len(prompt_input_ids)
        if prompt_tokens == 0:
            raise ValueError("--prompt produced zero prompt tokens")
        if (
            args.max_prompt_length is not None
            and prompt_tokens > args.max_prompt_length
        ):
            raise ValueError(
                f"--prompt has {prompt_tokens} tokens, exceeding --max-prompt-length "
                f"{args.max_prompt_length}"
            )
        return (
            "prompt",
            [
                PromptSample(
                    row_index=index,
                    prompt=prompt,
                    prompt_input_ids=list(prompt_input_ids),
                    prompt_tokens=prompt_tokens,
                )
                for index in range(args.batch_size)
            ],
            args.batch_size,
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read parquet datasets.") from exc

    if args.offset < 0:
        raise ValueError("offset must be non-negative")

    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)
    total_rows = parquet_file.metadata.num_rows
    if args.offset >= total_rows:
        raise ValueError(
            f"offset {args.offset} is out of range for {dataset_path}; "
            f"total rows: {total_rows}"
        )

    column_names = parquet_file.schema_arrow.names
    if args.dataset_format == "codeforces_raw":
        missing_columns = [
            column
            for column in CODEFORCES_REQUIRED_COLUMNS
            if column not in column_names
        ]
        if missing_columns:
            raise ValueError(
                "dataset-format=codeforces_raw requires Codeforces problem fields. "
                f"Missing columns: {missing_columns}. Available columns: {column_names}"
            )
        prompt_column = (
            "codeforces_raw["
            f"{CODEFORCES_LANGUAGE_ALIASES.get(args.code_language.lower(), args.code_language)}"
            "]"
        )
        read_columns = [
            column
            for column in [*CODEFORCES_REQUIRED_COLUMNS, *CODEFORCES_OPTIONAL_COLUMNS]
            if column in column_names
        ]
        selected_prompt_column = None
    elif args.dataset_format == "dapo_math_17k":
        selected_prompt_column = resolve_dapo_math_17k_prompt_column(
            column_names,
            prompt_column=args.prompt_column,
        )
        prompt_column = f"dapo_math_17k[{selected_prompt_column}]"
        read_columns = [selected_prompt_column]
    else:
        selected_prompt_column = args.prompt_column or infer_prompt_column(column_names)
        if selected_prompt_column not in column_names:
            raise ValueError(
                f"prompt column {selected_prompt_column!r} not found. "
                f"Available columns: {column_names}"
            )
        prompt_column = selected_prompt_column
        read_columns = [selected_prompt_column]

    chat_template_renderer = None
    if not args.disable_chat_template:
        chat_template_renderer = _build_chat_template_renderer(
            tokenizer_path, enable_thinking=args.enable_thinking
        )

    samples: list[PromptSample] = []
    skipped_empty = 0
    skipped_too_long = 0
    skipped_invalid = 0
    current_row = 0
    remaining_skip = args.offset
    reader_batch_size = max(args.batch_size, 1024)

    for record_batch in parquet_file.iter_batches(
        batch_size=reader_batch_size,
        columns=read_columns,
    ):
        if args.dataset_format in {"codeforces_raw", "dapo_math_17k"}:
            batch_rows = record_batch.to_pylist()
        else:
            batch_rows = record_batch.column(0).to_pylist()

        if remaining_skip >= len(batch_rows):
            remaining_skip -= len(batch_rows)
            current_row += len(batch_rows)
            continue

        start_index = remaining_skip
        for local_index in range(start_index, len(batch_rows)):
            row_index = current_row + local_index
            try:
                if args.dataset_format == "codeforces_raw":
                    prompt = _build_codeforces_raw_prompt(
                        batch_rows[local_index],
                        row_index=row_index,
                        chat_template_renderer=chat_template_renderer,
                        code_language=args.code_language,
                    )
                elif args.dataset_format == "dapo_math_17k":
                    prompt = _build_dapo_math_17k_prompt(
                        batch_rows[local_index],
                        row_index=row_index,
                        prompt_column=selected_prompt_column or "prompt",
                        chat_template_renderer=chat_template_renderer,
                        enable_thinking=args.enable_thinking,
                    )
                else:
                    prompt = _normalize_prompt(
                        batch_rows[local_index],
                        row_index,
                        prompt_column,
                        chat_template_renderer,
                        enable_thinking=args.enable_thinking,
                    )
            except Exception:
                skipped_invalid += 1
                continue

            if not prompt:
                skipped_empty += 1
                continue

            prompt_input_ids = _encode_prompt_tokens(tokenizer, prompt)
            prompt_tokens = len(prompt_input_ids)
            if prompt_tokens == 0:
                skipped_empty += 1
                continue

            if (
                args.max_prompt_length is not None
                and prompt_tokens > args.max_prompt_length
            ):
                skipped_too_long += 1
                continue

            samples.append(
                PromptSample(
                    row_index=row_index,
                    prompt=prompt,
                    prompt_input_ids=prompt_input_ids,
                    prompt_tokens=prompt_tokens,
                )
            )
            if len(samples) >= args.batch_size:
                return prompt_column, samples, total_rows

        current_row += len(batch_rows)
        remaining_skip = 0

    raise ValueError(
        "Not enough valid prompts were found. "
        f"requested={args.batch_size}, collected={len(samples)}, "
        f"skipped_empty={skipped_empty}, skipped_too_long={skipped_too_long}, "
        f"skipped_invalid={skipped_invalid}, offset={args.offset}, "
        f"total_rows={total_rows}, prompt_column={prompt_column!r}"
    )

