from typing import Dict, Tuple, Type

from sglang.benchmark.datasets.common import (
    BaseDatasetArgs,
    BaseDatasetLoader,
    DatasetRow,
)
from sglang.benchmark.datasets.custom import CustomArgs, CustomDatasetLoader
from sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixArgs,
    GeneratedSharedPrefixDatasetLoader,
)
from sglang.benchmark.datasets.image import ImageArgs, ImageDatasetLoader
from sglang.benchmark.datasets.mmmu import MMMUArgs, MMMUDatasetLoader
from sglang.benchmark.datasets.mooncake import MooncakeArgs, MooncakeDatasetLoader
from sglang.benchmark.datasets.openai_dataset import OpenAIArgs, OpenAIDatasetLoader
from sglang.benchmark.datasets.random import RandomArgs, RandomDatasetLoader
from sglang.benchmark.datasets.sharegpt import ShareGPTArgs, ShareGPTDatasetLoader

DATASET_MAPPING: Dict[str, Tuple[Type[BaseDatasetArgs], BaseDatasetLoader]] = {
    "sharegpt": (ShareGPTArgs, ShareGPTDatasetLoader()),
    "custom": (CustomArgs, CustomDatasetLoader()),
    "openai": (OpenAIArgs, OpenAIDatasetLoader()),
    "random": (RandomArgs, RandomDatasetLoader(random_sample=True)),
    "random-ids": (RandomArgs, RandomDatasetLoader(random_sample=False)),
    "generated-shared-prefix": (
        GeneratedSharedPrefixArgs,
        GeneratedSharedPrefixDatasetLoader(),
    ),
    "mmmu": (MMMUArgs, MMMUDatasetLoader()),
    "image": (ImageArgs, ImageDatasetLoader()),
    "mooncake": (MooncakeArgs, MooncakeDatasetLoader()),
}


def get_dataset(args, tokenizer, model_id=None):
    dataset_name = args.dataset_name
    if dataset_name.startswith("random") and dataset_name not in DATASET_MAPPING:
        dataset_name = "random-ids"

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    args_cls, loader = DATASET_MAPPING[dataset_name]
    config = args_cls.from_args(args)
    return loader.load(config=config, tokenizer=tokenizer, model_id=model_id)


__all__ = [
    "DATASET_MAPPING",
    "DatasetRow",
    "get_dataset",
]
