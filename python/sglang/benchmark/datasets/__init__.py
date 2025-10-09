from argparse import Namespace
from typing import Type

from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDatasetLoader
from sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixLoader,
)
from sglang.benchmark.datasets.image import ImageLoader
from sglang.benchmark.datasets.mmmu import MMMULoader
from sglang.benchmark.datasets.mooncake import MooncakeLoader
from sglang.benchmark.datasets.random import RandomLoader
from sglang.benchmark.datasets.sharegpt import ShareGPTLoader

DATASET_MAPPING = {
    "sharegpt": ShareGPTLoader,
    "random": RandomLoader,
    "random-ids": RandomLoader,
    "image": ImageLoader,
    "generated-shared-prefix": GeneratedSharedPrefixLoader,
    "mmmu": MMMULoader,
    "mooncake": MooncakeLoader,
}


def get_dataset_loader(
    args: Namespace, tokenizer: PreTrainedTokenizerBase
) -> BaseDatasetLoader:
    args.tokenize_prompt = getattr(args, "tokenize_prompt", False)
    dataset_class: Type[BaseDatasetLoader] = DATASET_MAPPING.get(args.dataset_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return dataset_class(args, tokenizer)
