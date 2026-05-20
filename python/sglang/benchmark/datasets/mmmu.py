import io
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import pybase64
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
from sglang.benchmark.datasets.image import create_mm_data_row
from sglang.benchmark.utils import get_processor


@dataclass
class MMMUDataset(BaseDataset):
    num_requests: int
    backend: str
    fixed_output_len: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "MMMUDataset":
        return cls(
            num_requests=args.num_prompts,
            backend=args.backend,
            fixed_output_len=args.random_output_len,
        )

    def load(self, tokenizer=None, model_id=None) -> List[DatasetRow]:
        processor = get_processor(model_id)
        return sample_mmmu_requests(
            num_requests=self.num_requests,
            processor=processor,
            backend=self.backend,
            fixed_output_len=self.fixed_output_len,
        )


def sample_mmmu_requests(
    num_requests: int,
    processor: AutoProcessor | AutoTokenizer,
    backend: str = "sglang",
    fixed_output_len: Optional[int] = None,
    random_sample: bool = True,
) -> List[DatasetRow]:
    """
    Sample requests from the MMMU dataset using HuggingFace datasets.

    Args:
        num_requests: Number of requests to sample.
        fixed_output_len: If provided, use this fixed output length for all requests.
        random_sample: Whether to randomly sample or take the first N.

    Returns:
        List of tuples (prompt, prompt_token_len, output_token_len).
    """
    print("Loading MMMU dataset from HuggingFace...")

    try:
        print("Attempting to load MMMU Math dataset...")
        mmmu_dataset = load_dataset("MMMU/MMMU", "Math", split="test")
        print(
            f"Successfully loaded MMMU Math dataset from HuggingFace with {len(mmmu_dataset)} examples"
        )
    except Exception as e:
        print(f"Failed to load MMMU Math dataset: {e}")
        raise ValueError(f"Failed to load MMMU dataset: {e}")

    # Sample from the dataset
    if len(mmmu_dataset) > num_requests:
        if random_sample:
            # Random sample
            indices = random.sample(range(len(mmmu_dataset)), num_requests)
            sample_dataset = mmmu_dataset.select(indices)
        else:
            # Take first N
            sample_dataset = mmmu_dataset.select(
                range(min(num_requests, len(mmmu_dataset)))
            )
    else:
        print(f"Dataset has less than {num_requests} examples, using all examples")
        sample_dataset = mmmu_dataset

    print(f"Selected {len(sample_dataset)} examples for benchmarking")

    # Create prompts
    filtered_dataset = []

    for i, example in enumerate(sample_dataset):
        try:
            # Extract image_1
            image = example.get("image_1")

            if image is not None:
                if hasattr(image, "save"):
                    # Convert RGBA images to RGB before encoding
                    if image.mode == "RGBA":
                        image = image.convert("RGB")

                    # Encode image to base64 (save as PNG to support palette/alpha modes)
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = pybase64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_data = f"data:image/png;base64,{img_str}"
                else:
                    continue

                # Extract the question
                question = example.get("question")

                # Construct the prompt
                text_prompt = f"Question: {question}\n\nAnswer: "
                output_len = fixed_output_len if fixed_output_len is not None else 256
                data_row = create_mm_data_row(
                    text_prompt, [image], [image_data], output_len, processor, backend
                )
                filtered_dataset.append(data_row)

        except Exception as e:
            print(f"Error processing example {i}: {e}")

    print(f"\nCreated {len(filtered_dataset)} MMMU prompts")
    return filtered_dataset
