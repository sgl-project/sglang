import inspect
import json
import os

import numpy as np
import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

# LongBench V2 dataset configuration
# Reference: https://github.com/THUDM/LongBench
LONGBENCH_V2_DATASET = "THUDM/LongBench-v2"
LONGBENCH_V2_SPLIT = "train"
DEFAULT_NUM_SAMPLES = 48  # Number of samples to use
DEFAULT_PROMPT_TOKENS = 3000  # Maximum number of tokens to use
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".longbench_cache")

# In-memory cache for the current session
_cached_input_ids = {}


def format_longbench_v2_example(example):
    """Format a LongBench V2 example into a single text string (context + question only)."""
    context = example.get("context", "")
    question = example.get("question", "")
    return f"{context} {question}"


def get_input_ids(
    tokenizer_path, max_prompt_tokens=DEFAULT_PROMPT_TOKENS, num_samples=None
):
    """Get input_ids from LongBench V2 dataset with local caching."""
    # Create cache key based on parameters
    if num_samples is None:
        num_samples = DEFAULT_NUM_SAMPLES
    cache_key = f"{tokenizer_path}_{max_prompt_tokens}_{num_samples}"

    # Check in-memory cache first (fastest)
    if cache_key in _cached_input_ids:
        print(
            f"Using in-memory cached data ({len(_cached_input_ids[cache_key])} prompts)"
        )
        return _cached_input_ids[cache_key]

    # Check local file cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Use a safe filename
    safe_name = tokenizer_path.replace("/", "_").replace("\\", "_")
    cache_file = os.path.join(
        CACHE_DIR, f"input_ids_{safe_name}_{max_prompt_tokens}_{num_samples}.json"
    )

    if os.path.exists(cache_file):
        print(f"Loading from local cache: {cache_file}")
        with open(cache_file, "r") as f:
            input_ids = json.load(f)
        _cached_input_ids[cache_key] = input_ids
        print(f"Loaded {len(input_ids)} prompts from cache")
        return input_ids

    # Download from HuggingFace using streaming
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Please install the 'datasets' package: pip install datasets"
        ) from exc

    tokenizer = get_tokenizer(tokenizer_path)

    print(f"Downloading {num_samples} samples from LongBench V2 (streaming)...")
    dataset = load_dataset(
        LONGBENCH_V2_DATASET, split=LONGBENCH_V2_SPLIT, streaming=True
    )

    input_ids = []
    for i, example in enumerate(dataset):
        if len(input_ids) >= num_samples:
            break
        text = format_longbench_v2_example(example)
        tokens = tokenizer.encode(text)
        # Truncate to max_tokens
        input_ids.append(tokens[:max_prompt_tokens])

    # Save to local cache
    with open(cache_file, "w") as f:
        json.dump(input_ids, f)
    print(f"Saved {len(input_ids)} prompts to cache: {cache_file}")

    # Also cache in memory
    _cached_input_ids[cache_key] = input_ids

    return input_ids


def compare_kl_divergence(
    input_logprobs, output_logprobs, ACC_THRESHOLDS, model_name, test_name
):
    """Compare the KL divergence between input and output log probabilities."""
    kl_divs = []
    for input_logprob, output_logprob in zip(input_logprobs, output_logprobs):
        input_logprob = np.array(input_logprob)
        output_logprob = np.array(output_logprob)
        logr = input_logprob - output_logprob
        kl_approx = (np.exp(logr) - 1) - logr
        kl_divs.append(np.mean(kl_approx))

    print(f"kl_divs={kl_divs}")
    avg_kl_div = sum(kl_divs) / len(kl_divs)
    print(f"avg_kl_div={avg_kl_div}")
    print(f"ACC_THRESHOLDS={ACC_THRESHOLDS[model_name]}")
    assert avg_kl_div < ACC_THRESHOLDS[model_name]["kl_div"], (
        f"avg_kl_div={avg_kl_div} > threshold={ACC_THRESHOLDS[model_name]['kl_div']} "
        f"for {model_name} {test_name}"
    )


# Common request helpers
def _flush_cache(base_url):
    requests.post(base_url + "/flush_cache")


def _generate(
    base_url, input_ids, max_new_tokens, return_logprob=False, logprob_start_len=-1
):
    """Send generate request and return results."""
    json_data = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 1,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if return_logprob:
        json_data.update(
            {
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": logprob_start_len,
            }
        )
    response = requests.post(base_url + "/generate", json=json_data)
    return response.json()


def _get_input_logprobs(base_url, new_input_ids, output_logprobs):
    """Run prefill to get input logprobs matching output logprobs."""
    _flush_cache(base_url)
    results = _generate(
        base_url,
        new_input_ids,
        max_new_tokens=0,
        return_logprob=True,
        logprob_start_len=0,
    )
    assert len(results) == len(new_input_ids)

    input_logprobs = []
    for i, result in enumerate(results):
        logprob = result["meta_info"]["input_token_logprobs"]
        logprob = [x[0] for x in logprob][-len(output_logprobs[i]) :]
        input_logprobs.append(logprob)
    return input_logprobs


def _extract_output_logprobs(result):
    """Extract output logprobs from a result."""
    return [x[0] for x in result["meta_info"]["output_token_logprobs"]]


def test_input_output_logprobs_match_helper(
    base_url, ACC_THRESHOLDS, model_name, max_samples=None, max_new_tokens=16000
):
    num_samples = DEFAULT_NUM_SAMPLES
    if max_samples is not None and max_samples > num_samples:
        num_samples = max_samples
    input_ids = get_input_ids(tokenizer_path=model_name, num_samples=num_samples)
    if max_samples is not None:
        input_ids = input_ids[:max_samples]
    print(f"Running test_input_output_logprobs_match with {len(input_ids)} prompts")

    print("Flush Cache and Running generation to get output logprobs ...")
    _flush_cache(base_url)
    results = _generate(base_url, input_ids, max_new_tokens, return_logprob=True)
    assert len(results) == len(input_ids)

    new_input_ids = []
    output_logprobs = []
    for i, result in enumerate(results):
        new_input_ids.append(input_ids[i] + result["output_ids"])
        output_logprobs.append(_extract_output_logprobs(result))

    print("Running prefill to get input logprobs ...")
    input_logprobs = _get_input_logprobs(base_url, new_input_ids, output_logprobs)

    compare_kl_divergence(
        input_logprobs,
        output_logprobs,
        ACC_THRESHOLDS,
        model_name,
        inspect.currentframe().f_code.co_name,
    )


def test_input_output_logprobs_match_prefill_cache_hit_helper(
    base_url, ACC_THRESHOLDS, model_name, max_samples=None, max_new_tokens=8192
):
    server_info = requests.get(base_url + "/get_server_info").json()
    if server_info["disable_radix_cache"]:
        print("Radix cache is disabled, skipping test")
        return

    num_samples = DEFAULT_NUM_SAMPLES
    if max_samples is not None and max_samples > num_samples:
        num_samples = max_samples
    input_ids = get_input_ids(tokenizer_path=model_name, num_samples=num_samples)
    if max_samples is not None:
        input_ids = input_ids[:max_samples]
    print(
        f"Running test_input_output_logprobs_match_prefill_cache_hit with {len(input_ids)} prompts"
    )

    # Prefill to cache the input
    print("Flush Cache and Prefill to cache the input ...")
    _flush_cache(base_url)
    _generate(base_url, input_ids, max_new_tokens=0)

    # Generate with cache hit
    print("Running generation to get output logprobs ...")
    results = _generate(base_url, input_ids, max_new_tokens, return_logprob=True)
    assert len(results) == len(input_ids)

    new_input_ids = []
    output_logprobs = []
    for i, result in enumerate(results):
        if result["meta_info"]["cached_tokens"] == 0:
            print(f"Prefill cache miss for prompt {i}, skipping")
            continue
        new_input_ids.append(input_ids[i] + result["output_ids"])
        output_logprobs.append(_extract_output_logprobs(result))

    assert len(new_input_ids) > 0.5 * len(
        input_ids
    ), f"Too few prefill cache hits: {len(new_input_ids)}/{len(input_ids)}"

    print("Flush Cache and run prefill to get input logprobs ...")
    input_logprobs = _get_input_logprobs(base_url, new_input_ids, output_logprobs)

    compare_kl_divergence(
        input_logprobs,
        output_logprobs,
        ACC_THRESHOLDS,
        model_name,
        inspect.currentframe().f_code.co_name,
    )


def test_input_output_logprobs_match_decode_cache_hit_helper(
    base_url, ACC_THRESHOLDS, model_name, max_samples=None, max_new_tokens=8192
):
    server_info = requests.get(base_url + "/get_server_info").json()
    if server_info["disable_radix_cache"]:
        print("Radix cache is disabled, skipping test")
        return

    num_samples = DEFAULT_NUM_SAMPLES
    if max_samples is not None and max_samples > num_samples:
        num_samples = max_samples
    first_turn_input_ids = get_input_ids(
        tokenizer_path=model_name, num_samples=num_samples
    )
    if max_samples is not None:
        first_turn_input_ids = first_turn_input_ids[:max_samples]
    print(
        f"Running test_input_output_logprobs_match_decode_cache_hit with {len(first_turn_input_ids)} prompts"
    )

    # First turn: Prefill + Decode to cache
    print("Flush Cache and First turn: Prefill + Decode to cache decode ...")
    _flush_cache(base_url)
    results = _generate(
        base_url, first_turn_input_ids, max_new_tokens, return_logprob=True
    )
    assert len(results) == len(first_turn_input_ids)

    tokenizer = get_tokenizer(tokenizer_name=model_name)
    comma_token_id = tokenizer.encode(",")

    second_turn_input_ids = [
        first_turn_input_ids[i] + result["output_ids"] + comma_token_id
        for i, result in enumerate(results)
    ]

    # Second turn: should hit decode cache
    print("Running generation to get output logprobs ...")
    results = _generate(
        base_url, second_turn_input_ids, max_new_tokens, return_logprob=True
    )
    assert len(results) == len(second_turn_input_ids)

    new_input_ids = []
    output_logprobs = []
    for i, result in enumerate(results):
        if result["meta_info"]["cached_tokens"] <= len(first_turn_input_ids[i]) + 1:
            print(f"Decode cache miss for prompt {i}, skipping")
            continue
        new_input_ids.append(second_turn_input_ids[i] + result["output_ids"])
        output_logprobs.append(_extract_output_logprobs(result))

    assert len(new_input_ids) > 0.5 * len(
        second_turn_input_ids
    ), f"Too few decode cache hits: {len(new_input_ids)}/{len(second_turn_input_ids)}"

    print("Flush Cache and run prefill to get input logprobs ...")
    input_logprobs = _get_input_logprobs(base_url, new_input_ids, output_logprobs)

    compare_kl_divergence(
        input_logprobs,
        output_logprobs,
        ACC_THRESHOLDS,
        model_name,
        inspect.currentframe().f_code.co_name,
    )
