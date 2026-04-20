import asyncio
import json
import os
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    MOONCAKE_DATASET_URL,
    BaseDataset,
    DatasetRow,
)
from sglang.benchmark.utils import download_and_cache_file


@dataclass
class MooncakeDataset(BaseDataset):
    dataset_path: str
    mooncake_workload: str
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "MooncakeDataset":
        return cls(
            dataset_path=args.dataset_path,
            mooncake_workload=args.mooncake_workload,
            num_requests=args.num_prompts,
        )

    def load(self, tokenizer=None, model_id=None) -> List[Dict]:
        if not self.dataset_path:
            local_path = os.path.join("/tmp", self.mooncake_workload + "_trace.jsonl")
        else:
            local_path = self.dataset_path

        if not os.path.exists(local_path):
            download_and_cache_file(
                MOONCAKE_DATASET_URL[self.mooncake_workload], local_path
            )

        with open(local_path, "r") as f:
            all_requests_data = [json.loads(line) for line in f if line.strip()]

        return all_requests_data[: self.num_requests]


async def get_mooncake_request_over_time(
    input_requests: List[Dict],
    tokenizer: PreTrainedTokenizerBase,
    slowdown_factor: float,
    num_rounds: int,
) -> AsyncGenerator[DatasetRow, None]:
    """
    An async generator that yields requests based on the timestamps in the Mooncake trace file,
    with support for multi-round sessions.
    """
    if not input_requests:
        return

    input_requests.sort(key=lambda r: r["timestamp"])

    start_time = time.perf_counter()
    trace_start_time_ms = input_requests[0]["timestamp"]

    for record in input_requests:
        # Calculate when this entire session should start
        relative_arrival_time_s = (record["timestamp"] - trace_start_time_ms) / 1000.0
        target_arrival_time_s = relative_arrival_time_s * slowdown_factor

        current_elapsed_time_s = time.perf_counter() - start_time
        sleep_duration_s = target_arrival_time_s - current_elapsed_time_s
        if sleep_duration_s > 0:
            await asyncio.sleep(sleep_duration_s)

        # Once the session starts, generate all rounds for it as a burst
        # This simulates a user engaging in a multi-turn conversation

        # Base user query constructed from hash_ids
        user_query_base = ""
        hash_ids = record.get("hash_ids", [])
        for hash_id in hash_ids:
            user_query_base += f"{hash_id}" + " ".join(
                ["hi"] * 128
            )  # Shorter for multi-round
        user_query_base += "Tell me a story based on this context."

        output_len_per_round = record.get("output_length", 256)
        chat_history = []

        for i in range(num_rounds):
            # Add user query for the current round
            chat_history.append(
                {"role": "user", "content": f"Round {i + 1}: {user_query_base}"}
            )

            # Form the full prompt from history
            try:
                full_prompt_text = tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            except Exception:
                full_prompt_text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in chat_history]
                )

            prompt_len = len(tokenizer.encode(full_prompt_text))

            yield DatasetRow(
                prompt=full_prompt_text,
                prompt_len=prompt_len,
                output_len=output_len_per_round,
            )

            # Add a placeholder assistant response for the next round's context
            # We use a placeholder because we don't know the real response
            placeholder_response = " ".join(["story"] * output_len_per_round)
            chat_history.append({"role": "assistant", "content": placeholder_response})
