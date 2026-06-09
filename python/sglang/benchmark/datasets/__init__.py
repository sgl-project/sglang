from typing import Dict, Tuple, Type

from sglang.benchmark.datasets.autobench import AutoBenchmarkDataset
from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sglang.benchmark.datasets.custom import CustomDataset
from sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixDataset,
)
from sglang.benchmark.datasets.image import ImageDataset
from sglang.benchmark.datasets.longbench_v2 import LongBenchV2Dataset
from sglang.benchmark.datasets.mmmu import MMMUDataset
from sglang.benchmark.datasets.mooncake import MooncakeDataset
from sglang.benchmark.datasets.openai_dataset import OpenAIDataset
from sglang.benchmark.datasets.random import RandomDataset
from sglang.benchmark.datasets.sharegpt import ShareGPTDataset
from sglang.benchmark.datasets.speed_bench import SpeedBenchDataset

DATASET_MAPPING: Dict[str, Type[BaseDataset]] = {
    "autobench": AutoBenchmarkDataset,
    "sharegpt": ShareGPTDataset,
    "custom": CustomDataset,
    "openai": OpenAIDataset,
    # TODO: "random" vs "random-ids" should be a flag (e.g. --random-source=sharegpt|integers),
    # not two separate dataset names sharing the same class.
    "random": RandomDataset,
    "random-ids": RandomDataset,
    "generated-shared-prefix": GeneratedSharedPrefixDataset,
    "mmmu": MMMUDataset,
    "image": ImageDataset,
    "mooncake": MooncakeDataset,
    "longbench_v2": LongBenchV2Dataset,
    "speed-bench": SpeedBenchDataset,
}

DATASET_RESULT_ARG_KEYS: Dict[str, Tuple[str, ...]] = {
    "autobench": ("sharegpt_output_len",),
    "sharegpt": ("sharegpt_output_len",),
    "custom": ("sharegpt_output_len",),
    "openai": ("sharegpt_output_len",),
    "longbench_v2": ("sharegpt_output_len",),
    "random": ("random_input_len", "random_output_len", "random_range_ratio"),
    "random-ids": ("random_input_len", "random_output_len", "random_range_ratio"),
    "generated-shared-prefix": (
        "gsp_num_groups",
        "gsp_prompts_per_group",
        "gsp_system_prompt_len",
        "gsp_question_len",
        "gsp_output_len",
        "gsp_range_ratio",
        "gsp_group_distribution",
        "gsp_zipf_alpha",
    ),
    "mmmu": ("random_output_len",),
    "image": ("random_input_len", "random_output_len", "random_range_ratio"),
}


def get_dataset(args, tokenizer, model_id=None):
    dataset_name = args.dataset_name
    if dataset_name.startswith("random") and dataset_name not in DATASET_MAPPING:
        dataset_name = "random-ids"

    if dataset_name not in DATASET_MAPPING:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    dataset_cls = DATASET_MAPPING[dataset_name]
    dataset = dataset_cls.from_args(args)
    return dataset.load(tokenizer=tokenizer, model_id=model_id)


__all__ = [
    "DATASET_MAPPING",
    "DATASET_RESULT_ARG_KEYS",
    "DatasetRow",
    "get_dataset",
]
