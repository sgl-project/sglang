import sys
from argparse import Namespace
from unittest.mock import Mock

import numpy as np
import pytest

from sglang.benchmark.datasets import get_dataset
from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.datasets.random_shared_prefix import (
    RandomWithSharedPrefixDataset,
    share_representative_prefix,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_share_representative_prefix_preserves_suffixes_and_metadata() -> None:
    rows = [
        DatasetRow(prompt=[1, 2, 3, 4], prompt_len=4, output_len=8),
        DatasetRow(prompt=[5, 6, 7, 8], prompt_len=4, output_len=9),
        DatasetRow(prompt=[9, 10, 11, 12], prompt_len=4, output_len=10),
    ]
    transformed = share_representative_prefix(rows, shared_prefix_len=2)

    assert [row.prompt for row in transformed] == [
        [1, 2, 3, 4],
        [1, 2, 7, 8],
        [1, 2, 11, 12],
    ]
    assert [row.prompt_len for row in transformed] == [4, 4, 4]
    assert [row.output_len for row in transformed] == [8, 9, 10]
    assert [row.prompt for row in rows] == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]


def test_from_args_builds_exact_random_id_dataset() -> None:
    dataset = RandomWithSharedPrefixDataset.from_args(
        Namespace(
            dataset_name="random-ids-shared-prefix",
            tokenize_prompt=True,
            random_shared_prefix_len=3,
            random_input_len=8,
            random_output_len=2,
            num_prompts=4,
            random_range_ratio=1.0,
            dataset_path="",
            backend="sglang",
        )
    )

    assert dataset.shared_prefix_len == 3
    assert dataset.input_len == 8
    assert dataset.output_len == 2
    assert dataset.num_requests == 4
    assert not dataset.random_sample
    assert not dataset.return_text


def test_generated_random_dataset_shares_prefix() -> None:
    np.random.seed(42)
    args = Namespace(
        dataset_name="random-ids-shared-prefix",
        tokenize_prompt=True,
        random_shared_prefix_len=4,
        random_input_len=8,
        random_output_len=2,
        num_prompts=4,
        random_range_ratio=1.0,
        dataset_path="",
        backend="sglang",
    )
    tokenizer = Mock(vocab_size=128)

    rows = get_dataset(args, tokenizer)

    assert all(row.prompt[:4] == rows[0].prompt[:4] for row in rows)


@pytest.mark.parametrize(
    ("rows", "shared_prefix_len", "message"),
    [
        (
            [DatasetRow(prompt=[1, 2], prompt_len=2, output_len=1)],
            3,
            "exceeds row 0 prompt length 2",
        ),
        (
            [DatasetRow(prompt="hello", prompt_len=1, output_len=1)],
            1,
            "must be a list of token ids",
        ),
    ],
)
def test_share_representative_prefix_rejects_invalid_rows(
    rows: list[DatasetRow], shared_prefix_len: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        share_representative_prefix(rows, shared_prefix_len=shared_prefix_len)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
