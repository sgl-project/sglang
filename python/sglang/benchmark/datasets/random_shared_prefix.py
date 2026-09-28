"""Dataset adapter for exact shared random-token prefixes."""

from argparse import Namespace
from dataclasses import dataclass, replace
from typing import Any, List

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.datasets.random import RandomDataset


@dataclass
class RandomWithSharedPrefixDataset(RandomDataset):
    """Generate random requests that share one exact token-id prefix.

    The first generated request is the representative. Its first
    ``shared_prefix_len`` token ids replace the same span in every request;
    each request keeps its original suffix, length, and metadata.
    """

    shared_prefix_len: int

    @classmethod
    def from_args(cls, args: Namespace) -> "RandomWithSharedPrefixDataset":
        if args.dataset_name != "random-ids-shared-prefix":
            raise ValueError(
                "RandomWithSharedPrefixDataset requires "
                "dataset_name='random-ids-shared-prefix'"
            )
        if args.random_shared_prefix_len <= 0:
            raise ValueError(
                "random-ids-shared-prefix requires "
                "--random-shared-prefix-len to be positive"
            )
        if not getattr(args, "tokenize_prompt", False):
            raise ValueError("random-ids-shared-prefix requires --tokenize-prompt")
        source = RandomDataset.from_args(args)
        return cls(
            input_len=source.input_len,
            output_len=source.output_len,
            num_requests=source.num_requests,
            range_ratio=source.range_ratio,
            dataset_path=source.dataset_path,
            return_text=source.return_text,
            random_sample=source.random_sample,
            shared_prefix_len=args.random_shared_prefix_len,
        )

    def load(self, tokenizer: Any, model_id: Any = None) -> List[DatasetRow]:
        rows = super().load(tokenizer=tokenizer, model_id=model_id)
        return share_representative_prefix(rows, self.shared_prefix_len)


def share_representative_prefix(
    rows: List[DatasetRow], shared_prefix_len: int
) -> List[DatasetRow]:
    """Return ``rows`` with copied prompts sharing row zero's token-id prefix."""
    if shared_prefix_len < 0:
        raise ValueError(
            f"shared_prefix_len must be non-negative, got {shared_prefix_len}"
        )
    if not rows or shared_prefix_len == 0:
        return list(rows)

    for index, row in enumerate(rows):
        if not isinstance(row.prompt, list) or not all(
            isinstance(token_id, int) for token_id in row.prompt
        ):
            raise ValueError(f"row {index} prompt must be a list of token ids")
        if shared_prefix_len > len(row.prompt):
            raise ValueError(
                f"shared_prefix_len {shared_prefix_len} exceeds row {index} "
                f"prompt length {len(row.prompt)}"
            )

    prefix = rows[0].prompt[:shared_prefix_len]
    return [
        replace(row, prompt=[*prefix, *row.prompt[shared_prefix_len:]]) for row in rows
    ]
