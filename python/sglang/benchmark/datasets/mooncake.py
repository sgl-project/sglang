import asyncio
import json
import os
import random
import time
from argparse import Namespace
from array import array
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    MOONCAKE_DATASET_URL,
    BaseDataset,
    DatasetRow,
)
from sglang.benchmark.utils import download_and_cache_file

# A trace record lists its prompt as ids of 512-token blocks (Mooncake, FAST'25);
# a block id two requests share is a shared prefix.
MOONCAKE_BLOCK_TOKENS = 512
# Stands in for an answer the trace does not contain.
_PLACEHOLDER_TOKEN = 0


class MooncakeBlocks:
    """Token ids for a trace's blocks: a block id always expands to the same
    MOONCAKE_BLOCK_TOKENS ids, drawn from the model's vocabulary."""

    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size
        self._blocks: Dict[int, array] = {}

    def block(self, hash_id: int) -> array:
        ids = self._blocks.get(hash_id)
        if ids is None:
            rng = random.Random(hash_id)
            ids = array(
                "i",
                (rng.randrange(self._vocab_size) for _ in range(MOONCAKE_BLOCK_TOKENS)),
            )
            self._blocks[hash_id] = ids
        return ids

    def prompt(self, record: Dict) -> List[int]:
        ids: List[int] = []
        for hash_id in record.get("hash_ids", []):
            ids.extend(self.block(hash_id))
        # The last block is partial: the record's input_length is the real count.
        return ids[: record.get("input_length", len(ids))]


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
    with support for multi-round sessions. Prompts are token ids: each block id of a
    record expands to the same MOONCAKE_BLOCK_TOKENS ids for every record, so the
    trace's prefix sharing and lengths are kept.
    """
    if not input_requests:
        return

    input_requests.sort(key=lambda r: r["timestamp"])
    blocks = MooncakeBlocks(tokenizer.vocab_size)

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
        query_ids = blocks.prompt(record)
        output_len_per_round = record.get("output_length", 256)
        history: List[int] = []

        for _ in range(num_rounds):
            prompt_ids = history + query_ids
            yield DatasetRow(
                prompt=prompt_ids,
                prompt_len=len(prompt_ids),
                output_len=output_len_per_round,
            )
            # The next round sees this prompt, a placeholder for the answer the
            # trace does not contain, and the query again.
            history = prompt_ids + [_PLACEHOLDER_TOKEN] * output_len_per_round
