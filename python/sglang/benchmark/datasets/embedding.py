import json
from argparse import Namespace
from dataclasses import dataclass
from typing import List

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class EmbeddingDataset(BaseDataset):
    """Load string or string-array inputs for OpenAI-compatible embeddings."""

    dataset_path: str
    num_requests: int

    @classmethod
    def from_args(cls, args: Namespace) -> "EmbeddingDataset":
        return cls(dataset_path=args.dataset_path, num_requests=args.num_prompts)

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        rows = []
        with open(self.dataset_path, encoding="utf-8") as f:
            for line in f:
                if self.num_requests > 0 and len(rows) >= self.num_requests:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue

                prompt = record.get("input")
                if isinstance(prompt, str):
                    texts = [prompt]
                elif isinstance(prompt, list):
                    texts = prompt
                else:
                    continue
                if not texts or not all(
                    isinstance(text, str) and text.strip() for text in texts
                ):
                    continue

                rows.append(
                    DatasetRow(
                        prompt=prompt,
                        prompt_len=sum(len(tokenizer.encode(text)) for text in texts),
                        output_len=0,
                        extra_request_body={
                            key: value
                            for key, value in record.items()
                            if key not in {"input", "model"}
                        },
                    )
                )

        print(f"Loaded {len(rows)} embedding requests")
        print(f"#Input tokens: {sum(row.prompt_len for row in rows)}")
        print("#Output tokens: 0")
        return rows
