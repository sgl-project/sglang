import json
import os
import random
import string

import numpy as np
from PIL import Image
from transformers import AutoTokenizer


def save_jsonl(data, file_path):
    """Save a list of dicts to a JSONL file, one JSON object per line."""
    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_custom_dataset(
    tokenizer_path,
    target_tokens,
    num_prompts,
    trust_remote_code=False,
):
    """Generate synthetic fixed-length text for throughput measurements."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    output = []
    for index in range(num_prompts):
        tokens = tokenizer.encode(
            f"Document {index}: This is synthetic input for a serving performance test. ",
            add_special_tokens=False,
        )
        padded = (tokens * (target_tokens // len(tokens) + 1))[:target_tokens]
        output.append({"question": tokenizer.decode(padded), "answer": "none"})
    return output


def generate_random_images(mm_dataset_data, size):
    """Generate random image files for a multimodal dataset.

    Creates random RGB images at the specified resolution for each image path
    listed in the dataset entries.

    Args:
        mm_dataset_data: List of multimodal data entries, each with a "path" field
            containing a list of image file paths.
        size: Image size tuple (width, height), e.g. (1080, 1920).
    """
    total_image_num = len(mm_dataset_data)
    print(f"begin to generate images, total {total_image_num}")

    file_count = 0
    for item in mm_dataset_data:
        image_paths = item.get("path")

        for image_path in image_paths:
            if not image_path:
                print("Error: The image path is none.")
                continue

            dir_name = os.path.dirname(image_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            random_array = np.random.randint(
                0, 256, (size[1], size[0], 3), dtype=np.uint8
            )

            img = Image.fromarray(random_array)
            img.save(image_path, quality=95)
            if os.path.isfile(image_path):
                file_count += 1

    print(f"Finish images generation. Image num: {file_count}")


def generate_mm_dataset(
    tokenizer_path,
    target_tokens=3500,
    num_prompts=1024,
    trust_remote_code=False,
    image_dir="/tmp/datasets/image",
    size=None,
):
    """Generate a multimodal (text + image) dataset.

    First generates fixed-length text data via generate_fixed_len_dataset, then
    attaches random image paths and type labels to each entry, and generates
    the corresponding random image files.

    Args:
        tokenizer_path: Path to the tokenizer.
        target_tokens: Target input token length.
        num_prompts: Number of prompts to generate.
        trust_remote_code: Whether to trust remote code when loading the tokenizer.
        image_dir: Directory to save generated image files.
        size: Image size string in "widthxheight" format, e.g. "1080x1920".

    Returns:
        list[dict]: Each item contains "question", "answer", "type", and "path" fields.
    """
    output_data = []
    text_data = generate_custom_dataset(
        tokenizer_path,
        target_tokens,
        num_prompts,
        trust_remote_code,
    )

    for item in text_data:
        random_string = "".join(
            random.choices(string.ascii_letters + string.digits, k=10)
        )
        item["type"] = "image"
        item["path"] = [f"{image_dir}/{random_string}.jpg"]
        output_data.append(item)

    size = tuple(map(int, size.split("x")))
    generate_random_images(output_data, size)
    return output_data


def generate_random_dataset(
    model_path,
    source_dataset_path,
    batch_size,
    input_len,
    output_file,
    output_len=1024,
    range_ratio=1,
):
    """Generate a random dataset with logic matching bench_serving's --dataset-name random.

    Samples real conversation text from the ShareGPT dataset as prompts, adjusting
    to the target token length via truncation or repetition. Input/output lengths
    are randomly sampled from [target*range_ratio, target]. Output format is a
    JSON array compatible with ais_bench's ShareGPTDataset.

    If source_dataset_path is not a valid JSON file, automatically downloads the
    ShareGPT dataset from HuggingFace (anon8231489123/ShareGPT_Vicuna_unfiltered).

    Args:
        model_path: Model path used to load the tokenizer.
        source_dataset_path: Path to the ShareGPT JSON file; auto-downloaded if invalid.
        batch_size: Number of samples to generate.
        input_len: Target input token length.
        output_file: Output JSON file path.
        output_len: Target output token length, default 1024.
        range_ratio: Random range ratio for input/output lengths. Actual lengths are
            uniformly sampled from [target*range_ratio, target]. Default 1 (fixed length).
    """
    SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

    def _is_file_valid_json(path):
        """Check if the path points to a valid JSON file (exists and parseable)."""
        if not os.path.isfile(path):
            return False
        try:
            with open(path, encoding="utf-8") as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            return False

    def _download_and_cache_hf_file(repo_id, filename, repo_type="dataset"):
        """Download and cache a file from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Randomly sample input/output lengths per request in [target*range_ratio, target]
    input_lens = np.random.randint(
        max(int(input_len * range_ratio), 1),
        input_len + 1,
        size=batch_size,
    ).tolist()
    output_lens = np.random.randint(
        max(int(output_len * range_ratio), 1),
        output_len + 1,
        size=batch_size,
    ).tolist()

    # Subtract special tokens to ensure the actual encoded length does not exceed target
    num_special_tokens = int(tokenizer.num_special_tokens_to_add())
    for i in range(batch_size):
        input_lens[i] = max(1, input_lens[i] - num_special_tokens)

    # Auto-download ShareGPT dataset from HuggingFace if local file is invalid
    if not _is_file_valid_json(source_dataset_path):
        print(
            f"source_dataset_path '{source_dataset_path}' is not a valid file, downloading from HuggingFace..."
        )
        source_dataset_path = _download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )

    # Load ShareGPT dataset, filter for >=2 turns, take the first turn (human) as prompt
    with open(source_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]
    random.shuffle(dataset)

    # Sample prompts, truncating or repeating tokens to reach target input length
    input_requests = []
    for data in dataset:
        i = len(input_requests)
        if i == batch_size:
            break

        prompt = data[0]
        prompt_token_ids = tokenizer.encode(prompt)
        prompt_len = len(prompt_token_ids)

        if prompt_len == 0:
            continue

        if prompt_len > input_lens[i]:
            input_ids = prompt_token_ids[: input_lens[i]]
        else:
            ratio = (input_lens[i] + prompt_len - 1) // prompt_len
            input_ids = (prompt_token_ids * ratio)[: input_lens[i]]
        input_content = tokenizer.decode(input_ids)
        # Output format compatible with ais_bench ShareGPTDataset
        input_requests.append(
            {
                "id": str(i),
                "conversations": [
                    {"from": "human", "value": input_content},
                    {"from": "gpt", "value": "none"},
                ],
            }
        )

    print(f"#Input tokens: {np.sum(input_lens[: len(input_requests)])}")
    print(f"#Output tokens: {np.sum(output_lens[: len(input_requests)])}")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Output as JSON array format, compatible with ais_bench's json.load()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(input_requests, f, ensure_ascii=False, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic text with fixed input token length"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output jsonl path"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to model tokenizer"
    )
    parser.add_argument(
        "--target_tokens", type=int, default=3500, help="Target input token length"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for tokenizer",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1024,
        help="Number of prompts to generate",
    )
    args = parser.parse_args()

    output_data = generate_custom_dataset(
        tokenizer_path=args.tokenizer_path,
        target_tokens=args.target_tokens,
        num_prompts=args.num_prompts,
        trust_remote_code=args.trust_remote_code,
    )
    save_jsonl(output_data, args.output_path)
    print(f"Done! Output {len(output_data)} items to {args.output_path}")


if __name__ == "__main__":
    main()
