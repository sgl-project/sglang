from typing import Dict

from sglang.benchmark.utils import get_processor

from .common import (
    ASSISTANT_SUFFIX,
    MOONCAKE_DATASET_URL,
    SHAREGPT_FILENAME,
    SHAREGPT_REPO_ID,
    BaseDatasetLoader,
    DatasetRow,
)
from .custom import sample_custom_requests
from .generated_shared_prefix import (
    gen_mm_prompt,
    gen_prompt,
    get_available_tokens,
    get_gen_prefix_cache_path,
    sample_generated_shared_prefix_requests,
)
from .image import create_mm_data_row, parse_image_resolution, sample_image_requests
from .mmmu import sample_mmmu_requests
from .mooncake import get_mooncake_request_over_time, load_mooncake_requests
from .openai_dataset import sample_openai_requests
from .random import compute_random_lens, sample_random_requests
from .sharegpt import sample_sharegpt_requests


class ShareGPTDatasetLoader(BaseDatasetLoader):
    def load(self, args, tokenizer, model_id=None):
        tokenize_prompt = getattr(args, "tokenize_prompt", False)
        assert not tokenize_prompt
        return sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )


class RandomDatasetLoader(BaseDatasetLoader):
    def __init__(self, random_sample: bool):
        self.random_sample = random_sample

    def load(self, args, tokenizer, model_id=None):
        tokenize_prompt = getattr(args, "tokenize_prompt", False)
        return sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            random_sample=self.random_sample,
            return_text=not tokenize_prompt,
        )


class ImageDatasetLoader(BaseDatasetLoader):
    def load(self, args, tokenizer, model_id=None):
        processor = get_processor(model_id)
        return sample_image_requests(
            num_requests=args.num_prompts,
            image_count=args.image_count,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            range_ratio=args.random_range_ratio,
            processor=processor,
            image_content=args.image_content,
            image_format=args.image_format,
            image_resolution=args.image_resolution,
            backend=args.backend,
            random_image_count=args.random_image_count,
        )


class GeneratedSharedPrefixDatasetLoader(BaseDatasetLoader):
    def load(self, args, tokenizer, model_id=None):
        tokenize_prompt = getattr(args, "tokenize_prompt", False)
        assert not tokenize_prompt
        return sample_generated_shared_prefix_requests(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=getattr(args, "gsp_range_ratio", 1.0),
            tokenizer=tokenizer,
            args=args,
        )


class MMMUDatasetLoader(BaseDatasetLoader):
    def load(self, args, tokenizer, model_id=None):
        processor = get_processor(model_id)
        return sample_mmmu_requests(
            num_requests=args.num_prompts,
            processor=processor,
            backend=args.backend,
            fixed_output_len=args.random_output_len,
            random_sample=True,
        )


class MooncakeDatasetLoader(BaseDatasetLoader):
    def load(self, args, tokenizer, model_id=None):
        return load_mooncake_requests(
            dataset_path=args.dataset_path,
            mooncake_workload=args.mooncake_workload,
            num_prompts=args.num_prompts,
        )


class CustomDatasetLoader(BaseDatasetLoader):
    def load(self, args, tokenizer, model_id=None):
        tokenize_prompt = getattr(args, "tokenize_prompt", False)
        assert not tokenize_prompt
        return sample_custom_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )


class OpenAIDatasetLoader(BaseDatasetLoader):
    def load(self, args, tokenizer, model_id=None):
        return sample_openai_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )


DATASET_MAPPING: Dict[str, BaseDatasetLoader] = {
    "sharegpt": ShareGPTDatasetLoader(),
    "random": RandomDatasetLoader(random_sample=True),
    "random-ids": RandomDatasetLoader(random_sample=False),
    "image": ImageDatasetLoader(),
    "generated-shared-prefix": GeneratedSharedPrefixDatasetLoader(),
    "mmmu": MMMUDatasetLoader(),
    "mooncake": MooncakeDatasetLoader(),
    "custom": CustomDatasetLoader(),
    "openai": OpenAIDatasetLoader(),
}


def get_dataset_loader(name: str) -> BaseDatasetLoader:
    if name in DATASET_MAPPING:
        return DATASET_MAPPING[name]
    raise ValueError(f"Unknown dataset: {name}")


def get_dataset(args, tokenizer, model_id=None):
    dataset_name = args.dataset_name
    if dataset_name.startswith("random") and dataset_name not in DATASET_MAPPING:
        dataset_name = "random"
    loader = get_dataset_loader(dataset_name)
    return loader.load(args=args, tokenizer=tokenizer, model_id=model_id)


__all__ = [
    "ASSISTANT_SUFFIX",
    "MOONCAKE_DATASET_URL",
    "SHAREGPT_FILENAME",
    "SHAREGPT_REPO_ID",
    "DATASET_MAPPING",
    "BaseDatasetLoader",
    "DatasetRow",
    "compute_random_lens",
    "create_mm_data_row",
    "gen_mm_prompt",
    "gen_prompt",
    "get_available_tokens",
    "get_dataset",
    "get_dataset_loader",
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
