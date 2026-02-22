from sglang.benchmark.datasets.common import (
    ASSISTANT_SUFFIX,
    MOONCAKE_DATASET_URL,
    SHAREGPT_FILENAME,
    SHAREGPT_REPO_ID,
    DatasetRow,
    compute_random_lens,
    gen_mm_prompt,
    gen_prompt,
    get_available_tokens,
)
from sglang.benchmark.datasets.custom import (
    CustomArgs,
    CustomDatasetLoader,
    sample_custom_requests,
)
from sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixArgs,
    GeneratedSharedPrefixDatasetLoader,
    get_gen_prefix_cache_path,
    sample_generated_shared_prefix_requests,
)
from sglang.benchmark.datasets.image import (
    ImageArgs,
    ImageDatasetLoader,
    create_mm_data_row,
    parse_image_resolution,
    sample_image_requests,
)
from sglang.benchmark.datasets.mmmu import (
    MMMUArgs,
    MMMUDatasetLoader,
    sample_mmmu_requests,
)
from sglang.benchmark.datasets.mooncake import (
    MooncakeArgs,
    MooncakeDatasetLoader,
    get_mooncake_request_over_time,
)
from sglang.benchmark.datasets.openai_dataset import (
    OpenAIArgs,
    OpenAIDatasetLoader,
    sample_openai_requests,
)
from sglang.benchmark.datasets.random import (
    RandomArgs,
    RandomDatasetLoader,
    sample_random_requests,
)
from sglang.benchmark.datasets.sharegpt import (
    ShareGPTArgs,
    ShareGPTDatasetLoader,
    sample_sharegpt_requests,
)

DATASET_MAPPING = {
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
    "ASSISTANT_SUFFIX",
    "DATASET_MAPPING",
    "MOONCAKE_DATASET_URL",
    "SHAREGPT_FILENAME",
    "SHAREGPT_REPO_ID",
    "DatasetRow",
    "compute_random_lens",
    "create_mm_data_row",
    "gen_mm_prompt",
    "gen_prompt",
    "get_available_tokens",
    "get_dataset",
    "get_gen_prefix_cache_path",
    "get_mooncake_request_over_time",
    "parse_image_resolution",
    "sample_custom_requests",
    "sample_generated_shared_prefix_requests",
    "sample_image_requests",
    "sample_mmmu_requests",
    "sample_openai_requests",
    "sample_random_requests",
    "sample_sharegpt_requests",
]
