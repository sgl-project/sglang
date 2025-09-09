from argparse import Namespace
from typing import Type

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDatasetLoader
from sglang.benchmark.datasets.sharegpt import ShareGPTLoader

DATASET_MAPPING = {
    "sharegpt": ShareGPTLoader,
}


def get_dataset_loader(
    args: Namespace, tokenizer: PreTrainedTokenizerBase
) -> BaseDatasetLoader:
    dataset_class: Type[BaseDatasetLoader] = DATASET_MAPPING.get(args.dataset_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return dataset_class(args, tokenizer)
