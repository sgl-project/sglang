import json
import os
import random
import string

import numpy as np
from PIL import Image
from transformers import AutoTokenizer


def load_jsonl(path):
    """Load data from a JSONL file, one JSON object per line."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save a list of dicts to a JSONL file, one JSON object per line."""
    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_qa(item):
    """Format a GSM8K data entry into QA text for the few-shot pool."""
    question = item["question"]
    answer = item["answer"]
    return f"Question: {question}\nLet's think step by step\nAnswer:\n{answer}\n\n"


def pad_to_target_tokens(
    question,
    few_shot_pool_token_ids,
    tokenizer,
    target_tokens,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
):
    """Pad a question text to the target token length.

    Tokenizes the question using the test_template, calculates the remaining tokens
    needed, and prepends randomly sampled few-shot token ids from the pool to reach
    target_tokens. If the few-shot pool is insufficient, repeats the first sample
    to fill the remaining gap.

    Args:
        question: The test question text.
        few_shot_pool_token_ids: List of token id lists from the few-shot training pool.
        tokenizer: The tokenizer instance.
        target_tokens: Target input token length.
        test_template: Question template string, defaults to GSM8K format.
    """
    test_prompt = test_template.format(question=question)
    test_token_ids = tokenizer.encode(test_prompt, add_special_tokens=False)

    remaining_tokens = target_tokens - len(test_token_ids)
    if remaining_tokens <= 0:
        return tokenizer.decode(
            test_token_ids[:target_tokens], skip_special_tokens=True
        )

    shuffled_ids = list(range(len(few_shot_pool_token_ids)))
    random.shuffle(shuffled_ids)

    prefix_ids = []
    for idx in shuffled_ids:
        fs_ids = few_shot_pool_token_ids[idx]
        if len(prefix_ids) + len(fs_ids) <= remaining_tokens:
            prefix_ids.extend(fs_ids)
        else:
            partial_gap = remaining_tokens - len(prefix_ids)
            if partial_gap > 0:
                prefix_ids.extend(fs_ids[:partial_gap])
            break

    if len(prefix_ids) < remaining_tokens and few_shot_pool_token_ids:
        padding_source_ids = few_shot_pool_token_ids[shuffled_ids[0]]
        repeat_count = (remaining_tokens // len(padding_source_ids)) + 1
        padding_ids = (padding_source_ids * repeat_count)[
            : remaining_tokens - len(prefix_ids)
        ]
        prefix_ids.extend(padding_ids)

    full_ids = prefix_ids + test_token_ids
    return tokenizer.decode(full_ids[:target_tokens], skip_special_tokens=True)


def generate_custom_dataset(
    train_path,
    test_path,
    tokenizer_path,
    target_tokens,
    num_prompts,
    trust_remote_code=False,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
):
    """Generate a custom dataset with a fixed input token length.

    Builds a few-shot pool from the training set and pads test questions to the
    specified token length. If the test set has fewer samples than num_prompts,
    it cycles and repeats to fill the required count.

    Args:
        train_path: Path to the GSM8K training JSONL file.
        test_path: Path to the GSM8K test JSONL file.
        tokenizer_path: Path to the tokenizer.
        target_tokens: Target input token length.
        num_prompts: Number of prompts to generate; 0 means use all test samples.
        trust_remote_code: Whether to trust remote code when loading the tokenizer.
        test_template: Question template string.

    Returns:
        list[dict]: Each item contains fields defined in test_template.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)
    if num_prompts > 0 and num_prompts > len(test_data):
        multiplier = (num_prompts // len(test_data)) + 1
        test_data = (test_data * multiplier)[:num_prompts]
    elif num_prompts > 0:
        test_data = test_data[:num_prompts]

    few_shot_pool = [format_qa(item) for item in train_data]
    few_shot_pool_token_ids = [
        tokenizer.encode(fs, add_special_tokens=False) for fs in few_shot_pool
    ]

    output_data = []
    for i, test_item in enumerate(test_data):
        padded_question = pad_to_target_tokens(
            question=test_item["question"],
            few_shot_pool_token_ids=few_shot_pool_token_ids,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            test_template=test_template,
        )
        output_data.append(
            {
                "question": padded_question,
                "answer": test_item["answer"],
            }
        )
        if (i + 1) % 100 == 0:
            actual_tokens = len(
                tokenizer.encode(padded_question, add_special_tokens=False)
            )
            print(
                f"Processed {i + 1}/{len(test_data)}, last item tokens: {actual_tokens}"
            )

    token_counts = [
        len(tokenizer.encode(item["question"], add_special_tokens=False))
        for item in output_data
    ]
    print(
        f"Token count stats: min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)/len(token_counts):.1f}"
    )

    return output_data


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
    train_path,
    test_path,
    tokenizer_path,
    target_tokens=3500,
    num_prompts=1024,
    trust_remote_code=False,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
    image_dir="/tmp/datasets/image",
    size=None,
):
    """Generate a multimodal (text + image) dataset.

    First generates fixed-length text data via generate_fixed_len_dataset, then
    attaches random image paths and type labels to each entry, and generates
    the corresponding random image files.

    Args:
        train_path: Path to the GSM8K training JSONL file.
        test_path: Path to the GSM8K test JSONL file.
        tokenizer_path: Path to the tokenizer.
        target_tokens: Target input token length.
        num_prompts: Number of prompts to generate.
        trust_remote_code: Whether to trust remote code when loading the tokenizer.
        test_template: Question template string.
        image_dir: Directory to save generated image files.
        size: Image size string in "widthxheight" format, e.g. "1080x1920".

    Returns:
        list[dict]: Each item contains "question", "answer", "type", and "path" fields.
    """
    output_data = []
    text_data = generate_custom_dataset(
        train_path,
        test_path,
        tokenizer_path,
        target_tokens,
        num_prompts,
        trust_remote_code,
        test_template,
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


def generate_gsm8k_dataset(
    model_path, source_dataset_path, batch_size, input_len, output_file
):
    """Generate a dataset with a fixed input token length from GSM8K (JSONL format).

    Reads GSM8K source data, repeats or truncates each question's tokens to input_len,
    then trims or replicates the dataset to batch_size entries, shuffles, and writes
    to the output file.

    Args:
        model_path: Model path used to load the tokenizer.
        source_dataset_path: Path to the GSM8K source JSONL file.
        batch_size: Number of samples to generate.
        input_len: Target input token length.
        output_file: Output JSONL file path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = []
    with open(source_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data["question"])

    dataset_new = []
    for sentence in dataset:
        words = tokenizer.tokenize(sentence)
        len_num = len(words) // input_len
        if len_num == 0:
            multiplier = (input_len // len(words)) + 1
            repeated_len = words * multiplier
            words = repeated_len[:input_len]
            decoded_text = tokenizer.convert_tokens_to_string(words)
            if len(words) != input_len:
                print(
                    f"Generate DataSet Error: the length of new input is {len(words)}, not {input_len}"
                )
            dataset_new.append(decoded_text)

    batch_num = len(dataset_new) // batch_size
    if batch_num == 0:
        multiplier = (batch_size // len(dataset_new)) + 1
        repeated_batch = dataset_new * multiplier
        dataset_new = repeated_batch[:batch_size]
    else:
        dataset_new = dataset_new[:batch_size]

    random.shuffle(dataset_new)

    if len(dataset_new) != batch_size:
        print(
            f"Generate DataSet Error: the size of new dataset is {len(dataset_new)}, not {batch_size}"
        )

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(dataset_new)):
            f.write(
                json.dumps(
                    {"question": f"{dataset_new[i]}", "answer": "none"},
                    ensure_ascii=False,
                )
            )
            f.write("\n")


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

    print(f"#Input tokens: {np.sum(input_lens[:len(input_requests)])}")
    print(f"#Output tokens: {np.sum(output_lens[:len(input_requests)])}")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Output as JSON array format, compatible with ais_bench's json.load()
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(input_requests, f, ensure_ascii=False, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate GSM8K dataset with exact input token length"
    )
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to GSM8K train.jsonl"
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to GSM8K test.jsonl"
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
        default=0,
        help="Number of prompts to generate, 0 means all",
    )
    args = parser.parse_args()

    output_data = generate_custom_dataset(
        train_path=args.train_path,
        test_path=args.test_path,
        tokenizer_path=args.tokenizer_path,
        target_tokens=args.target_tokens,
        num_prompts=args.num_prompts,
        trust_remote_code=args.trust_remote_code,
    )
    save_jsonl(output_data, args.output_path)
    print(f"Done! Output {len(output_data)} items to {args.output_path}")


if __name__ == "__main__":
    main()
