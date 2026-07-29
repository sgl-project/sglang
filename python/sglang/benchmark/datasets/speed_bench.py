"""SPEED-Bench (nvidia/SPEED-Bench) dataset for SGLang bench_serving.

Reads the pre-downloaded throughput_1k JSONL produced by prepare_speed_bench.sh
(or equivalent), optionally filtering by category (low_entropy / mixed /
high_entropy) and fixing the output length.

CLI args consumed:
  --dataset-path            Path to the local JSONL file.
  --speed-bench-category    Category filter: low_entropy | mixed | high_entropy
                            (default: all categories).
  --speed-bench-output-len  Fixed number of output tokens per request (default: 512).
  --num-prompts             Number of requests to sample (capped by available rows).
"""

import json
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class SpeedBenchDataset(BaseDataset):
    dataset_path: str
    category: Optional[str]
    output_len: int
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "SpeedBenchDataset":
        if not args.dataset_path:
            raise ValueError(
                "--dataset-path must point to the SPEED-Bench JSONL file "
                "(run prepare_speed_bench.sh to generate it)."
            )
        return cls(
            dataset_path=args.dataset_path,
            category=getattr(args, "speed_bench_category", None) or None,
            output_len=getattr(args, "speed_bench_output_len", 512),
            num_requests=args.num_prompts,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        unique_prompts = []
        with open(self.dataset_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if self.category and row.get("category") != self.category:
                    continue
                # turns is a list of strings; use the first user turn as the prompt
                turns = row.get("turns", [])
                if not turns:
                    continue
                unique_prompts.append(turns[0])

        if not unique_prompts:
            raise ValueError(
                f"No rows found in {self.dataset_path}"
                + (f" for category={self.category}" if self.category else "")
            )

        # Tokenize unique prompts once to avoid redundant work
        unique_dataset_rows: List[DatasetRow] = []
        for prompt_text in unique_prompts:
            # Apply chat template to match vllm bench behaviour
            try:
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                prompt = tokenizer.decode(prompt_ids)
            except Exception:
                prompt_ids = tokenizer.encode(prompt_text)
                prompt = prompt_text

            unique_dataset_rows.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=len(prompt_ids),
                    output_len=self.output_len,
                )
            )

        # Sample (with replacement if needed); shuffle oversampled rows for
        # a realistic request distribution
        if self.num_requests <= len(unique_dataset_rows):
            dataset_rows = random.sample(unique_dataset_rows, self.num_requests)
        else:
            dataset_rows = unique_dataset_rows * (
                self.num_requests // len(unique_dataset_rows) + 1
            )
            dataset_rows = dataset_rows[: self.num_requests]
            random.shuffle(dataset_rows)

        return dataset_rows
