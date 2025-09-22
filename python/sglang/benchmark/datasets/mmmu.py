import random
from typing import List

from sglang.benchmark.datasets.common import BaseDatasetLoader, DatasetRow


class MMMULoader(BaseDatasetLoader):
    def load(self) -> List[DatasetRow]:
        """
        Sample requests from the MMMU dataset using HuggingFace datasets.

        Args:
            num_requests: Number of requests to sample.
            tokenizer: Tokenizer to use for token counting.
            fixed_output_len: If provided, use this fixed output length for all requests.
            apply_chat_template: Whether to apply the chat template to the prompt.
            random_sample: Whether to randomly sample or take the first N.

        Returns:
            List of tuples (prompt, prompt_token_len, output_token_len).
        """
        try:
            import io

            import pybase64
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        num_requests = self.args.num_prompts
        fixed_output_len = self.args.random_output_len
        apply_chat_template = self.args.apply_chat_template
        random_sample = True

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
                        img_str = pybase64.b64encode(buffered.getvalue()).decode(
                            "utf-8"
                        )
                        image_data = f"data:image/png;base64,{img_str}"
                    else:
                        continue

                    # Extract the question
                    question = example.get("question")

                    # Construct the prompt
                    prompt = f"Question: {question}\n\nAnswer: "
                    if apply_chat_template:
                        try:
                            prompt = self.tokenizer.apply_chat_template(
                                [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": image_data},
                                            },
                                            {"type": "text", "text": prompt},
                                        ],
                                    }
                                ],
                                add_generation_prompt=True,
                                tokenize=False,
                            )
                        except Exception as e:
                            # Note (Xinyuan): This is a workaround for an issue where some tokenizers do not support content as a list. (e.g. InternVL)
                            print(
                                f"Error applying chat template: {e}, fallback to <image> tag"
                            )
                            prompt = f"<image>{prompt}"

                    # Calculate token lengths for text only (without image data)
                    prompt_token_ids = self.tokenizer.encode(prompt)
                    prompt_len = len(prompt_token_ids)

                    output_len = (
                        fixed_output_len if fixed_output_len is not None else 256
                    )

                    filtered_dataset.append(
                        DatasetRow(
                            prompt=prompt,
                            prompt_len=prompt_len,
                            output_len=output_len,
                            image_data=[image_data],
                        )
                    )

            except Exception as e:
                print(f"Error processing example {i}: {e}")

        print(f"\nCreated {len(filtered_dataset)} MMMU prompts")
        return filtered_dataset
