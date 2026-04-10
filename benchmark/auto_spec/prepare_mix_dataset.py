"""
Prepare a mixed benchmark dataset by downloading and integrating
GSM8K, HumanEval, MT-Bench, ShareGPT, and medium-length datasets into a single JSONL file.

The data order is: GSM8K -> HumanEval -> MT-Bench -> ShareGPT -> Medium-length datasets (not shuffled).

Usage:
    python prepare_mix_dataset.py --output mix_spec_dataset.jsonl

    # Control per-dataset sample count
    python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
        --num-gsm8k 200 --num-humaneval 164 --num-mtbench 80 --num-sharegpt 500

    # Use locally downloaded files (when remote URLs are not accessible)
    python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
        --gsm8k-path /path/to/test.jsonl \
        --humaneval-path /path/to/HumanEval.jsonl \
        --mtbench-path /path/to/question.jsonl \
        --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json

    # Include medium-length datasets with local files
    python prepare_mix_dataset.py --output mix_spec_dataset.jsonl \
        --include-medium-length \
        --hotpotqa-path /path/to/hotpot_dev_fullwiki_v1.json \
        --squad-path /path/to/dev-v2.0.json \
        --drop-path /path/to/drop_dataset \
        --mbpp-path /path/to/mbpp.jsonl
"""

import argparse
import gzip
import json
import os

import requests
from tqdm import tqdm

# Dataset download URLs
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)
MTBENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Medium-length dataset URLs (1k-10k tokens) covering different aspects:
# Reasoning/QA - HotpotQA (multi-hop reasoning)
HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
# Reading Comprehension - SQuAD (Stanford QA)
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
# Numerical Reasoning - DROP (Discrete Reasoning Over Paragraphs)
DROP_URL = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip"
# Code - MBPP (Mostly Basic Python Problems)
MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"

# Default cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")


def download_file(url, filename, desc=None):
    """Download a file from a URL with progress bar. Returns the local path or None on failure."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        print(f"  [Cached] {filename}")
        return filename

    print(f"  Downloading from {url}")
    print(f"  Saving to {filename}")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
    except (requests.RequestException, ConnectionError) as e:
        print(f"\n  WARNING: Failed to download from {url}")
        print(f"  Error: {e}")
        print(f"  Skipping this dataset. To include it, manually download and use:")
        print(f"    wget -O {filename} '{url}'")
        print(f"  See README.md for manual download instructions.")
        return None

    total_size = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as f, tqdm(
        desc=desc or os.path.basename(filename),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def load_gsm8k(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load GSM8K dataset and convert to unified format."""
    print("\n[1/4] Loading GSM8K dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            GSM8K_URL,
            os.path.join(cache_dir, "gsm8k_test.jsonl"),
            desc="GSM8K",
        )
    if not path:
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            prompt = "Question: " + data["question"] + "\nAnswer:"
            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": 512,
                    "source": "gsm8k",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} GSM8K samples")
    return records


def load_humaneval(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load HumanEval dataset and convert to unified format."""
    print("\n[2/4] Loading HumanEval dataset...")
    if not path or not os.path.exists(path):
        gz_path = os.path.join(cache_dir, "HumanEval.jsonl.gz")
        jsonl_path = os.path.join(cache_dir, "HumanEval.jsonl")

        # Check if already decompressed
        if os.path.exists(jsonl_path):
            path = jsonl_path
        else:
            result = download_file(HUMANEVAL_URL, gz_path, desc="HumanEval")
            if not result:
                return []
            # Decompress .gz file
            print("  Decompressing HumanEval.jsonl.gz...")
            with gzip.open(gz_path, "rt", encoding="utf-8") as gz_f:
                with open(jsonl_path, "w", encoding="utf-8") as out_f:
                    out_f.write(gz_f.read())
            path = jsonl_path

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            # Use the prompt field (function signature + docstring)
            prompt = (
                "Read the following function signature and docstring, "
                "and fully implement the function described. "
                "Your response should only contain the code for this function.\n\n"
                + data["prompt"]
            )
            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": 512,
                    "source": "human_eval",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} HumanEval samples")
    return records


def load_mtbench(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load MT-Bench dataset and convert to unified format."""
    print("\n[3/4] Loading MT-Bench dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            MTBENCH_URL,
            os.path.join(cache_dir, "mt_bench_question.jsonl"),
            desc="MT-Bench",
        )
    if not path:
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            # Use the first turn question as prompt
            prompt = data["turns"][0]
            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": 256,
                    "source": "mtbench",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} MT-Bench samples")
    return records


def load_sharegpt(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load ShareGPT dataset and convert to unified format."""
    print("\n[4/4] Loading ShareGPT dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            SHAREGPT_URL,
            os.path.join(cache_dir, "ShareGPT_V3_unfiltered_cleaned_split.json"),
            desc="ShareGPT",
        )
    if not path:
        return []

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Filter out conversations with less than 2 turns
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]

    records = []
    for data in dataset:
        convos = data.get("conversations", data.get("conversation", []))
        prompt = convos[0]["value"]
        completion = convos[1]["value"]

        if not prompt or not completion:
            continue

        # Estimate output length based on completion text
        # Use character count / 4 as a rough token estimate
        estimated_output_len = max(len(completion) // 4, 16)

        records.append(
            {
                "prompt": prompt,
                "expected_output_len": estimated_output_len,
                "source": "sharegpt",
            }
        )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} ShareGPT samples")
    return records


def load_hotpotqa(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load HotpotQA dataset and convert to unified format.

    HotpotQA is a multi-hop reasoning QA dataset with contexts typically 1k-5k tokens.
    """
    print("\n[5/8] Loading HotpotQA dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            HOTPOTQA_URL,
            os.path.join(cache_dir, "hotpot_dev_fullwiki_v1.json"),
            desc="HotpotQA",
        )
    if not path:
        return []

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    records = []
    for data in dataset:
        # Combine question with context paragraphs
        question = data.get("question", "")
        context = data.get("context", [])

        # Build context text from paragraphs
        context_texts = []
        for ctx in context:
            if isinstance(ctx, list) and len(ctx) > 1:
                context_texts.append(" ".join(ctx[1]))

        context_str = "\n\n".join(context_texts)
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"

        # Estimate output length (answer is typically short)
        answer = data.get("answer", "")
        estimated_output_len = max(len(answer) // 4, 32)

        records.append(
            {
                "prompt": prompt,
                "expected_output_len": estimated_output_len,
                "source": "hotpotqa",
            }
        )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} HotpotQA samples")
    return records


def load_squad(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load SQuAD dataset and convert to unified format.

    SQuAD is a reading comprehension dataset with contexts typically 1k-3k tokens.
    """
    print("\n[6/8] Loading SQuAD dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            SQUAD_URL,
            os.path.join(cache_dir, "dev-v2.0.json"),
            desc="SQuAD",
        )
    if not path:
        return []

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    records = []
    for article in dataset.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")

                prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

                # Estimate output length from answers
                answers = qa.get("answers", [])
                if answers:
                    answer_text = answers[0].get("text", "")
                    estimated_output_len = max(len(answer_text) // 4, 32)
                else:
                    estimated_output_len = 64

                records.append(
                    {
                        "prompt": prompt,
                        "expected_output_len": estimated_output_len,
                        "source": "squad",
                    }
                )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} SQuAD samples")
    return records


def load_drop(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load DROP dataset and convert to unified format.

    DROP is a discrete reasoning dataset with numerical reasoning over paragraphs.
    Contexts are typically 1k-4k tokens.
    """
    print("\n[7/8] Loading DROP dataset...")
    if not path or not os.path.exists(path):
        zip_path = os.path.join(cache_dir, "drop_dataset.zip")
        json_path = os.path.join(cache_dir, "drop_dataset", "drop_dataset_train.json")

        if os.path.exists(json_path):
            path = json_path
        else:
            result = download_file(DROP_URL, zip_path, desc="DROP")
            if not result:
                return []
            # Extract zip file
            import zipfile

            print("  Extracting DROP dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(cache_dir)
            path = json_path

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    records = []
    for passage_id, passage_data in dataset.items():
        passage_text = passage_data.get("passage", "")
        for qa_pair in passage_data.get("qa_pairs", []):
            question = qa_pair.get("question", "")

            prompt = f"Passage:\n{passage_text}\n\nQuestion: {question}\nAnswer:"

            # Estimate output length from answer
            answer = qa_pair.get("answer", "")
            if isinstance(answer, dict):
                answer_text = answer.get("number", "") or str(answer)
            else:
                answer_text = str(answer)
            estimated_output_len = max(len(answer_text) // 4, 16)

            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": estimated_output_len,
                    "source": "drop",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} DROP samples")
    return records


def load_mbpp(path, num_samples=None, cache_dir=CACHE_DIR):
    """Load MBPP dataset and convert to unified format.

    MBPP (Mostly Basic Python Problems) is a code generation dataset.
    Prompts are typically 500-2k tokens.
    """
    print("\n[8/8] Loading MBPP dataset...")
    if not path or not os.path.exists(path):
        path = download_file(
            MBPP_URL,
            os.path.join(cache_dir, "mbpp.jsonl"),
            desc="MBPP",
        )
    if not path:
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            text = data.get("text", "")
            test_list = data.get("test_list", [])

            # Create a code generation prompt
            prompt = f"""{text}

Your code should pass these tests:
"""
            for test in test_list[:3]:  # Include up to 3 test cases
                prompt += f"{test}\n"

            prompt += "\nProvide your solution:"

            # Estimate output length from reference code
            code = data.get("code", "")
            estimated_output_len = max(len(code) // 4, 128)

            records.append(
                {
                    "prompt": prompt,
                    "expected_output_len": estimated_output_len,
                    "source": "mbpp",
                }
            )

    if num_samples is not None and num_samples < len(records):
        records = records[:num_samples]

    print(f"  Loaded {len(records)} MBPP samples")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mixed benchmark dataset (GSM8K + HumanEval + MT-Bench + ShareGPT + Medium-length datasets)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mix_spec_dataset.jsonl"
        ),
        help="Output JSONL file path. Default: ./mix_spec_dataset.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=CACHE_DIR,
        help="Directory to cache downloaded datasets.",
    )

    # Per-dataset sample count
    parser.add_argument(
        "--num-gsm8k",
        type=int,
        default=None,
        help="Number of GSM8K samples to include. Default: all (1319 test samples).",
    )
    parser.add_argument(
        "--num-humaneval",
        type=int,
        default=None,
        help="Number of HumanEval samples to include. Default: all (164 samples).",
    )
    parser.add_argument(
        "--num-mtbench",
        type=int,
        default=None,
        help="Number of MT-Bench samples to include. Default: all (80 samples).",
    )
    parser.add_argument(
        "--num-sharegpt",
        type=int,
        default=None,
        help="Number of ShareGPT samples to include. Default: all.",
    )

    # Medium-length datasets (1k-10k tokens)
    parser.add_argument(
        "--num-hotpotqa",
        type=int,
        default=None,
        help="Number of HotpotQA samples to include (multi-hop reasoning QA, 1k-5k tokens). Default: 0 (skip).",
    )
    parser.add_argument(
        "--num-squad",
        type=int,
        default=None,
        help="Number of SQuAD samples to include (reading comprehension, 1k-3k tokens). Default: 0 (skip).",
    )
    parser.add_argument(
        "--num-drop",
        type=int,
        default=None,
        help="Number of DROP samples to include (numerical reasoning, 1k-4k tokens). Default: 0 (skip).",
    )
    parser.add_argument(
        "--num-mbpp",
        type=int,
        default=None,
        help="Number of MBPP samples to include (Python code generation, 500-2k tokens). Default: 0 (skip).",
    )

    # Per-dataset local file paths (for manual download)
    parser.add_argument(
        "--gsm8k-path",
        type=str,
        default="",
        help="Path to locally downloaded GSM8K test.jsonl.",
    )
    parser.add_argument(
        "--humaneval-path",
        type=str,
        default="",
        help="Path to locally downloaded HumanEval.jsonl (decompressed).",
    )
    parser.add_argument(
        "--mtbench-path",
        type=str,
        default="",
        help="Path to locally downloaded MT-Bench question.jsonl.",
    )
    parser.add_argument(
        "--sharegpt-path",
        type=str,
        default="",
        help="Path to locally downloaded ShareGPT JSON file.",
    )
    parser.add_argument(
        "--hotpotqa-path",
        type=str,
        default="",
        help="Path to locally downloaded HotpotQA JSON file.",
    )
    parser.add_argument(
        "--squad-path",
        type=str,
        default="",
        help="Path to locally downloaded SQuAD dev-v2.0.json file.",
    )
    parser.add_argument(
        "--drop-path",
        type=str,
        default="",
        help="Path to locally downloaded DROP dataset directory.",
    )
    parser.add_argument(
        "--mbpp-path",
        type=str,
        default="",
        help="Path to locally downloaded MBPP jsonl file.",
    )

    # Dataset selection flags
    parser.add_argument(
        "--include-medium-length",
        action="store_true",
        help="Include all medium-length datasets (HotpotQA, SQuAD, DROP, MBPP).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of datasets to include (e.g., 'gsm8k,humaneval,hotpotqa'). "
        "If set, only these datasets will be included. Options: gsm8k, humaneval, mtbench, sharegpt, "
        "hotpotqa, squad, drop, mbpp",
    )

    args = parser.parse_args()

    cache_dir = args.cache_dir

    # Determine which datasets to include
    selected_datasets = None
    if args.datasets:
        selected_datasets = [s.strip().lower() for s in args.datasets.split(",")]

    def should_include(name):
        if selected_datasets is not None:
            return name in selected_datasets
        return True

    print("=" * 60)
    print("Preparing mixed benchmark dataset (mix-spec)")
    print(
        "Order: GSM8K -> HumanEval -> MT-Bench -> ShareGPT -> HotpotQA -> SQuAD -> DROP -> MBPP"
    )
    print("=" * 60)

    all_records = []
    dataset_counts = {}

    # Load base datasets
    if should_include("gsm8k"):
        gsm8k_records = load_gsm8k(args.gsm8k_path, args.num_gsm8k, cache_dir)
        all_records.extend(gsm8k_records)
        dataset_counts["GSM8K"] = len(gsm8k_records)

    if should_include("humaneval"):
        humaneval_records = load_humaneval(
            args.humaneval_path, args.num_humaneval, cache_dir
        )
        all_records.extend(humaneval_records)
        dataset_counts["HumanEval"] = len(humaneval_records)

    if should_include("mtbench"):
        mtbench_records = load_mtbench(args.mtbench_path, args.num_mtbench, cache_dir)
        all_records.extend(mtbench_records)
        dataset_counts["MT-Bench"] = len(mtbench_records)

    if should_include("sharegpt"):
        sharegpt_records = load_sharegpt(
            args.sharegpt_path, args.num_sharegpt, cache_dir
        )
        all_records.extend(sharegpt_records)
        dataset_counts["ShareGPT"] = len(sharegpt_records)

    # Load medium-length datasets
    if should_include("hotpotqa") and (args.include_medium_length or args.num_hotpotqa):
        hotpotqa_records = load_hotpotqa(
            args.hotpotqa_path, args.num_hotpotqa, cache_dir
        )
        all_records.extend(hotpotqa_records)
        dataset_counts["HotpotQA"] = len(hotpotqa_records)

    if should_include("squad") and (args.include_medium_length or args.num_squad):
        squad_records = load_squad(args.squad_path, args.num_squad, cache_dir)
        all_records.extend(squad_records)
        dataset_counts["SQuAD"] = len(squad_records)

    if should_include("drop") and (args.include_medium_length or args.num_drop):
        drop_records = load_drop(args.drop_path, args.num_drop, cache_dir)
        all_records.extend(drop_records)
        dataset_counts["DROP"] = len(drop_records)

    if should_include("mbpp") and (args.include_medium_length or args.num_mbpp):
        mbpp_records = load_mbpp(args.mbpp_path, args.num_mbpp, cache_dir)
        all_records.extend(mbpp_records)
        dataset_counts["MBPP"] = len(mbpp_records)

    # Write to output JSONL
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Mixed dataset written to: {args.output}")
    print(f"Total samples: {len(all_records)}")
    for name, count in dataset_counts.items():
        if count > 0:
            print(f"  - {name}: {count}")
        else:
            print(f"  - {name}: 0 (skipped, download failed)")

    if len(all_records) == 0:
        print("\nWARNING: No samples were loaded. Check download errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
