import asyncio
import json
import os
import random
import time
from typing import AsyncGenerator, Callable, List

import numpy as np

from sglang.benchmark.datasets.common import BaseDatasetLoader, DatasetRow
from sglang.benchmark.utils import download_and_cache_file

MOONCAKE_DATASET_URLS = {
    "mooncake": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
    "conversation": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
    "synthetic": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
    "toolagent": "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
}


class MooncakeLoader(BaseDatasetLoader):
    def load(self) -> List[DatasetRow]:
        workload = self.args.mooncake_workload
        url = MOONCAKE_DATASET_URLS[workload]
        local_path = self.args.dataset_path or os.path.join(
            "/tmp", f"mooncake_{workload}_trace.jsonl"
        )

        if not os.path.exists(local_path):
            download_and_cache_file(url, local_path)

        with open(local_path, "r") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        for record in all_records[: self.args.num_prompts]:
            self.input_requests.append(
                DatasetRow(
                    prompt="",
                    prompt_len=0,
                    output_len=record.get("output_length", 256),
                    timestamp=record.get("timestamp"),
                    raw_data=record,
                )
            )

        if not self.input_requests:
            return []

        # For warmup
        first_record = self.input_requests[0].raw_data
        warmup_prompt_text = (
            " ".join([f"id_{h}" for h in first_record.get("hash_ids", [])])
            + " Tell me a short story based on this context."
        )
        warmup_prompt_len = len(self.tokenizer.encode(warmup_prompt_text))
        warmup_row = DatasetRow(
            prompt=warmup_prompt_text,
            prompt_len=warmup_prompt_len,
            output_len=self.input_requests[0].output_len,
            timestamp=self.input_requests[0].timestamp,
            raw_data=first_record,
        )
        self.input_requests[0] = warmup_row

        print(f"\n# Loaded {len(self.input_requests)} Mooncake request sessions.")
        return self.input_requests

    def get_request_generator(
        self,
    ) -> Callable[[], AsyncGenerator[DatasetRow, None]]:
        return self._mooncake_generator

    async def _mooncake_generator(self) -> AsyncGenerator[DatasetRow, None]:
        self.input_requests.sort(key=lambda r: r.timestamp)
        start_time = time.perf_counter()
        trace_start_time_ms = (
            self.input_requests[0].timestamp if self.input_requests else 0
        )

        for session_row in self.input_requests:
            record = session_row.raw_data

            relative_arrival_time_s = (
                record["timestamp"] - trace_start_time_ms
            ) / 1000.0
            target_arrival_time = start_time + (
                relative_arrival_time_s * self.args.mooncake_slowdown_factor
            )

            sleep_duration = target_arrival_time - time.perf_counter()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)

            # Once the session starts, generate all rounds for it as a burst
            # This simulates a user engaging in a multi-turn conversation

            # Base user query constructed from hash_ids
            chat_history = []
            user_query_base = (
                " ".join([f"id_{h}" for h in record.get("hash_ids", [])])
                + " Tell me a story based on this context."
            )

            for i in range(self.args.mooncake_num_rounds):
                chat_history.append(
                    {"role": "user", "content": f"Round {i+1}: {user_query_base}"}
                )

                try:
                    full_prompt_text = self.tokenizer.apply_chat_template(
                        chat_history, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    full_prompt_text = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
                    )

                prompt_len = len(self.tokenizer.encode(full_prompt_text))
                output_len = record.get("output_length", 256)

                yield DatasetRow(
                    prompt=full_prompt_text,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    timestamp=session_row.timestamp,
                )

                placeholder_response = " ".join(["placeholder"] * output_len)
                chat_history.append(
                    {"role": "assistant", "content": placeholder_response}
                )
