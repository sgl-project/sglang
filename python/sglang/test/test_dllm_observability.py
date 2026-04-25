import json
from typing import Any

import requests

OBSERVABILITY_MAX_NEW_TOKENS = 100
OBSERVABILITY_SEED = 42
FORWARD_COUNTS_KEY = "dllm_forward_counts"
PROMPT_1 = (
    "Human: What is the capital of France and how is that city like. "
    "Give me 3 trivial information about that city. "
    "Write in a format of json.\nAssistant:"
)
PROMPT_2 = (
    "Human: What is the capital of Germany and how is that city like. "
    "Give me 3 trivial information about that city. "
    "Write in a format of json.\nAssistant:"
)


def build_single_prompt() -> list[str]:
    return [PROMPT_1]


def build_two_prompts() -> list[str]:
    return [PROMPT_1, PROMPT_2]


def _build_generate_payload(
    prompts: list[str],
    stream: bool,
    stream_interval: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": prompts[0] if len(prompts) == 1 else prompts,
        "sampling_params": {
            "sampling_seed": OBSERVABILITY_SEED,
            "temperature": 0.0,
            "max_new_tokens": OBSERVABILITY_MAX_NEW_TOKENS,
            "ignore_eos": True,
        },
        "stream": stream,
    }
    if stream_interval is not None:
        payload["sampling_params"]["stream_interval"] = stream_interval
    return payload


def _extract_forward_counts(output: dict[str, Any]) -> list[int]:
    meta_info = output.get("meta_info")
    if not isinstance(meta_info, dict):
        raise AssertionError(f"Missing meta_info: {output}")

    counts = meta_info.get(FORWARD_COUNTS_KEY)
    if not isinstance(counts, list):
        raise AssertionError(
            f"Expected {FORWARD_COUNTS_KEY} to be a list, got: {type(counts)}"
        )

    for count in counts:
        if not isinstance(count, int):
            raise AssertionError(
                f"Every {FORWARD_COUNTS_KEY} entry should be an int, got: {count!r}"
            )
        if count < 1:
            raise AssertionError(
                f"Every {FORWARD_COUNTS_KEY} entry should be >= 1, got: {count}"
            )
    return counts


def get_forward_counts_from_generate_non_stream(
    base_url: str,
    prompts: list[str],
) -> dict[int, list[int]]:
    response = requests.post(
        f"{base_url}/generate",
        json=_build_generate_payload(
            prompts,
            stream=False,
            stream_interval=None,
        ),
    )
    outputs = response.json()
    if not isinstance(outputs, list):
        outputs = [outputs]
    return {
        output.get("index", index): _extract_forward_counts(output)
        for index, output in enumerate(outputs)
    }


def get_forward_counts_from_generate_stream(
    base_url: str,
    prompts: list[str],
    stream_interval: int | None = None,
    incremental_streaming_output: bool = False,
) -> dict[int, list[int]]:
    response = requests.post(
        f"{base_url}/generate",
        json=_build_generate_payload(
            prompts,
            stream=True,
            stream_interval=stream_interval,
        ),
        stream=True,
    )

    chunks_by_index: dict[int, list[dict[str, Any]]] = {}
    incremental_counts_by_index: dict[int, list[int]] = {}
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            break

        chunk = json.loads(line[len("data:") :].strip())
        index = chunk.get("index", 0)
        if incremental_streaming_output:
            incremental_counts_by_index.setdefault(index, []).extend(
                _extract_forward_counts(chunk)
            )
        else:
            chunks_by_index.setdefault(index, []).append(chunk)

    if incremental_streaming_output:
        return incremental_counts_by_index

    return {
        index: _extract_forward_counts(chunks[-1])
        for index, chunks in chunks_by_index.items()
        if chunks
    }


class DllmObservabilityMixin:
    def assert_generate_stream_cumulative_matches_non_stream(
        self,
        base_url: str,
        prompts: list[str],
        stream_interval: int | None = None,
        incremental_streaming_output: bool = False,
    ) -> None:
        stream_label = "default" if stream_interval is None else str(stream_interval)
        non_stream_forward_counts_by_index = (
            get_forward_counts_from_generate_non_stream(base_url, prompts)
        )
        stream_forward_counts_by_index = get_forward_counts_from_generate_stream(
            base_url,
            prompts,
            stream_interval,
            incremental_streaming_output=incremental_streaming_output,
        )

        self.assertEqual(
            len(non_stream_forward_counts_by_index.items()),
            len(stream_forward_counts_by_index.items()),
            "Number of non-stream and stream outputs should match.",
        )
        self.assertEqual(
            set(non_stream_forward_counts_by_index),
            set(stream_forward_counts_by_index),
            "Non-stream and stream request indexes should match.",
        )

        for index in sorted(non_stream_forward_counts_by_index):
            self.assertEqual(
                non_stream_forward_counts_by_index[index],
                stream_forward_counts_by_index[index],
                f"Streamed {FORWARD_COUNTS_KEY} should match non-stream output for "
                f"request {index} (stream_interval={stream_label}, "
                f"incremental_streaming_output={incremental_streaming_output}).",
            )
