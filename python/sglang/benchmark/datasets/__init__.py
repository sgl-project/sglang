import json
import os

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
from sglang.benchmark.datasets.custom import sample_custom_requests
from sglang.benchmark.datasets.generated_shared_prefix import (
    get_gen_prefix_cache_path,
    sample_generated_shared_prefix_requests,
)
from sglang.benchmark.datasets.image import (
    create_mm_data_row,
    parse_image_resolution,
    sample_image_requests,
)
from sglang.benchmark.datasets.mmmu import sample_mmmu_requests
from sglang.benchmark.datasets.mooncake import get_mooncake_request_over_time
from sglang.benchmark.datasets.openai_dataset import sample_openai_requests
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.datasets.sharegpt import sample_sharegpt_requests
from sglang.benchmark.utils import download_and_cache_file, get_processor


def get_dataset(args, tokenizer, model_id=None):
    tokenize_prompt = getattr(args, "tokenize_prompt", False)
    if args.dataset_name == "sharegpt":
        assert not tokenize_prompt
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )
    elif args.dataset_name.startswith("random"):
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
            dataset_path=args.dataset_path,
            random_sample=args.dataset_name == "random",
            return_text=not tokenize_prompt,
        )
    elif args.dataset_name == "image":
        processor = get_processor(model_id)
        input_requests = sample_image_requests(
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
    elif args.dataset_name == "generated-shared-prefix":
        assert not tokenize_prompt
        input_requests = sample_generated_shared_prefix_requests(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=getattr(args, "gsp_range_ratio", 1.0),
            tokenizer=tokenizer,
            args=args,
        )
    elif args.dataset_name == "mmmu":
        processor = get_processor(model_id)
        input_requests = sample_mmmu_requests(
            num_requests=args.num_prompts,
            processor=processor,
            backend=args.backend,
            fixed_output_len=args.random_output_len,
            random_sample=True,
        )
    elif args.dataset_name == "mooncake":
        # For mooncake, we don't generate the prompts here.
        # We just load the raw trace data. The async generator will handle the rest.
        if not args.dataset_path:
            local_path = os.path.join("/tmp", args.mooncake_workload + "_trace.jsonl")
        else:
            local_path = args.dataset_path

        if not os.path.exists(local_path):
            download_and_cache_file(
                MOONCAKE_DATASET_URL[args.mooncake_workload], local_path
            )

        with open(local_path, "r") as f:
            all_requests_data = [json.loads(line) for line in f if line.strip()]

        # Limit the number of requests based on --num-prompts
        input_requests = all_requests_data[: args.num_prompts]
    elif args.dataset_name == "custom":
        assert not tokenize_prompt
        input_requests = sample_custom_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            context_len=args.sharegpt_context_len,
            prompt_suffix=args.prompt_suffix,
            apply_chat_template=args.apply_chat_template,
        )
    elif args.dataset_name == "openai":
        input_requests = sample_openai_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    return input_requests


__all__ = [
    "ASSISTANT_SUFFIX",
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
