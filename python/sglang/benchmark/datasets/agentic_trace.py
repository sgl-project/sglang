import json
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow

# Default per-turn output length when --sharegpt-output-len is not given.
# Matches the OpenHands/agentic recipe used by the evalscope `swe_smith` bench
# (--max-tokens 220 with ignore_eos).
DEFAULT_AGENTIC_OUTPUT_LEN = 220


@dataclass
class AgenticTraceDataset(BaseDataset):
    """Multi-turn agentic trace loader.

    Reads a pre-built trace JSON of the shape produced by the OpenHands /
    SWE-smith dataset builders (``build_openhands_padded_dataset.py``)::

        {
          "metadata": {...},
          "conversations": [
            [   # one conversation == a list of turns
              {"messages": [{"role": "system", ...}, {"role": "user", ...}],
               "prompt_tokens": 73821},
              {"messages": [{"role": "user", ...}], "prompt_tokens": 74894},
              ...
            ],
            ...
          ]
        }

    Each turn's ``messages`` is the *delta* (the new, non-assistant messages for
    that turn). One conversation becomes one :class:`DatasetRow` whose ``prompt``
    is the list of per-turn message deltas. ``bench_serving`` detects this shape
    as multi-turn and replays each conversation round by round, feeding the
    server's real assistant reply back into the next round's history.

    Use with a chat backend (``--backend sglang-oai-chat``).
    """

    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]
    offset: int
    max_turns: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "AgenticTraceDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
            offset=getattr(args, "dataset_offset", 0) or 0,
            max_turns=getattr(args, "agentic_max_turns", None),
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = data.get("conversations", [])
        if not conversations:
            raise ValueError(
                f"No 'conversations' found in {self.dataset_path}. Expected a "
                "trace built by build_openhands_padded_dataset.py / "
                "build_swe_smith_dataset.py."
            )

        # Rotate by offset so successive sweep steps start on fresh
        # conversations (mirrors the evalscope --dataset-offset behavior).
        offset = self.offset % len(conversations)
        if offset:
            conversations = conversations[offset:] + conversations[:offset]

        output_len = self.fixed_output_len or DEFAULT_AGENTIC_OUTPUT_LEN

        filtered_dataset: List[DatasetRow] = []
        for conversation in conversations:
            if self.num_requests > 0 and len(filtered_dataset) >= self.num_requests:
                break

            # prompt == list of per-turn message deltas (each a List[Dict]).
            prompt = [turn.get("messages", []) for turn in conversation]
            prompt = [turn for turn in prompt if turn]  # drop empty turns
            if self.max_turns is not None and self.max_turns > 0:
                prompt = prompt[: self.max_turns]
            if not prompt:
                continue

            # First turn's accumulated prompt tokens (cosmetic for multi-turn:
            # calculate_metrics ignores per-row prompt_len in this mode).
            first_turn = conversation[0] if conversation else {}
            prompt_len = int(first_turn.get("prompt_tokens", 0))

            filtered_dataset.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    output_len=output_len,
                )
            )

        if not filtered_dataset:
            raise ValueError(
                f"No usable conversations loaded from {self.dataset_path}."
            )

        num_turns = [len(row.prompt) for row in filtered_dataset]
        print(
            f"#Conversations: {len(filtered_dataset)} "
            f"(offset={offset}, turns/conv min={min(num_turns)} "
            f"max={max(num_turns)} avg={np.mean(num_turns):.1f})"
        )
        print(f"#Output tokens per turn: {output_len}")
        return filtered_dataset
