import inspect

import numpy as np
import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

# LongBench V2 dataset configuration
# Reference: https://github.com/THUDM/LongBench
LONGBENCH_V2_DATASET = "THUDM/LongBench-v2"
LONGBENCH_V2_SPLIT = "train"
NUM_SAMPLES = 48  # Number of samples to use


def format_longbench_v2_example(example):
    """Format a LongBench V2 example into a single text string (context + question only)."""
    context = example.get("context", "")
    question = example.get("question", "")
    return f"{context} {question}"


def get_input_ids(tokenizer_path, max_tokens=3000, num_samples=NUM_SAMPLES):
    """Get input_ids from LongBench V2 dataset using streaming (fast partial download)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Please install the 'datasets' package: pip install datasets"
        ) from exc

    tokenizer = get_tokenizer(tokenizer_path)

    print(f"Loading {num_samples} samples from LongBench V2 (streaming)...")
    # Use streaming to avoid downloading entire dataset
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
        input_ids.append(tokens[:max_tokens])

    print(f"Loaded {len(input_ids)} prompts (truncated to {max_tokens} tokens)")
    return input_ids


def compare_kl_divergence(
    input_logprobs, output_logprobs, ACC_THRESHOLDS, model_name, test_name
):
    """
    Compare the KL divergence between input and output log probabilities.
    """
    kl_divs = []
    for i, (input_logprob, output_logprob) in enumerate(
        zip(input_logprobs, output_logprobs)
    ):
        input_logprob = np.array(input_logprob)
        output_logprob = np.array(output_logprob)
        logr = input_logprob - output_logprob
        kl_approx = (np.exp(logr) - 1) - logr
        kl_divs.append(np.mean(kl_approx))

    print(f"kl_divs={kl_divs}")
    avg_kl_div = sum(kl_divs) / len(kl_divs)
    print(f"avg_kl_div={avg_kl_div}")
    print(f"ACC_THRESHOLDS={ACC_THRESHOLDS[model_name]}")
    assert (
        avg_kl_div < ACC_THRESHOLDS[model_name]["kl_div"]
    ), f"avg_kl_div={avg_kl_div} is greater than threshold={ACC_THRESHOLDS[model_name]['kl_div']} for {model_name} {test_name}"


def test_input_output_logprobs_match_helper(
    base_url, ACC_THRESHOLDS, model_name, max_samples=None, max_new_tokens=16000
):
    input_ids = get_input_ids(tokenizer_path=model_name)
    if max_samples is not None:
        input_ids = input_ids[:max_samples]
    print("Running test_input_output_logprobs_match with ", len(input_ids), "prompts")

    print("Flush Cache and Running generation to get output logprobs ...")
    print(base_url)
    requests.post(base_url + "/flush_cache")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
    )

    results = response.json()
    assert len(results) == len(input_ids)
    new_input_ids = []
    output_logprobs = []
    for i, result in enumerate(results):
        output_ids = result["output_ids"]
        new_input_ids.append(input_ids[i] + output_ids)
        output_logprob = result["meta_info"]["output_token_logprobs"]
        output_logprob = [x[0] for x in output_logprob]
        output_logprobs.append(output_logprob)

    print("Running prefill to get input logprobs ...")
    # Flush cache before running prefill
    requests.post(base_url + "/flush_cache")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": new_input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": 0,
        },
    )

    new_results = response.json()
    assert len(new_results) == len(new_input_ids)

    input_logprobs = []
    for i, result in enumerate(new_results):
        input_logprob = result["meta_info"]["input_token_logprobs"]
        input_logprob = [x[0] for x in input_logprob][-len(output_logprobs[i]) :]
        input_logprobs.append(input_logprob)

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
    # query server info to make sure disable_radix_cache is False
    server_info = requests.get(base_url + "/get_server_info").json()
    if server_info["disable_radix_cache"]:
        print(
            "Radix cache is disabled, skipping test_input_output_logprobs_match_prefill_cache_hit test"
        )
        return

    input_ids = get_input_ids(tokenizer_path=model_name)
    if max_samples is not None:
        input_ids = input_ids[:max_samples]
    print(
        "Running test_input_output_logprobs_match_prefill_cache_hit with ",
        len(input_ids),
        "prompts",
    )

    print("Flush Cache and Prefill to cache the input ...")
    requests.post(base_url + "/flush_cache")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": 0,
                "ignore_eos": True,
            },
        },
    )

    print("Running generation to get output logprobs ...")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
    )

    results = response.json()
    assert len(results) == len(input_ids)
    new_input_ids = []
    output_logprobs = []
    for i, result in enumerate(results):
        output_ids = result["output_ids"]
        cached_tokens = result["meta_info"]["cached_tokens"]
        if cached_tokens == 0:
            print(f"Prefill cache miss for prompt {i}, skipping this prompt")
            continue
        new_input_ids.append(input_ids[i] + output_ids)
        output_logprob = result["meta_info"]["output_token_logprobs"]
        output_logprob = [x[0] for x in output_logprob]
        output_logprobs.append(output_logprob)

    assert len(new_input_ids) > 0.5 * len(
        input_ids
    ), f"Too few prefill cache hits, {len(new_input_ids)=}, {len(input_ids)=}"

    print("Flush Cache and run prefill to get input logprobs ...")
    # Flush cache before running prefill
    requests.post(base_url + "/flush_cache")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": new_input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": 0,
        },
    )

    new_results = response.json()
    assert len(new_results) == len(new_input_ids)

    input_logprobs = []
    for i, result in enumerate(new_results):
        input_logprob = result["meta_info"]["input_token_logprobs"]
        input_logprob = [x[0] for x in input_logprob][-len(output_logprobs[i]) :]
        input_logprobs.append(input_logprob)

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
    # query server info to make sure disable_radix_cache is False
    server_info = requests.get(base_url + "/get_server_info").json()
    if server_info["disable_radix_cache"]:
        print(
            "Radix cache is disabled, skipping test_input_output_logprobs_match_decode_cache_hit test"
        )
        return

    first_turn_input_ids = get_input_ids(tokenizer_path=model_name)
    if max_samples is not None:
        first_turn_input_ids = first_turn_input_ids[:max_samples]
    print(
        "Running test_input_output_logprobs_match_decode_cache_hit with ",
        len(first_turn_input_ids),
        "prompts",
    )

    print("Flush Cache and First turn: Prefill + Decode to cache decode ...")
    requests.post(base_url + "/flush_cache")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": first_turn_input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
    )

    tokenizer = get_tokenizer(tokenizer_name=model_name)
    comma_token_id = tokenizer.encode(",")  # add comma to ensure cache hit

    results = response.json()
    assert len(results) == len(first_turn_input_ids)
    second_turn_input_ids = []
    for i, result in enumerate(results):
        output_ids = result["output_ids"]
        second_turn_input_ids.append(
            first_turn_input_ids[i] + output_ids + comma_token_id
        )

    print("Running generation to get output logprobs ...")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": second_turn_input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
    )

    results = response.json()
    assert len(results) == len(second_turn_input_ids)
    new_input_ids = []
    output_logprobs = []
    for i, result in enumerate(results):
        output_ids = result["output_ids"]
        cached_tokens = result["meta_info"]["cached_tokens"]
        if cached_tokens <= len(first_turn_input_ids[i]) + 1:
            print(f"Decode cache miss for prompt {i}, skipping this prompt")
            continue
        new_input_ids.append(second_turn_input_ids[i] + output_ids)
        output_logprob = result["meta_info"]["output_token_logprobs"]
        output_logprob = [x[0] for x in output_logprob]
        output_logprobs.append(output_logprob)

    assert len(new_input_ids) > 0.5 * len(
        second_turn_input_ids
    ), f"Too few decode cache hits, {len(new_input_ids)=}, {len(second_turn_input_ids)=}"

    print("Flush Cache and run prefill to get input logprobs ...")
    # Flush cache before running prefill
    requests.post(base_url + "/flush_cache")
    response = requests.post(
        base_url + "/generate",
        json={
            "input_ids": new_input_ids,
            "sampling_params": {
                "temperature": 1,
                "max_new_tokens": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": 0,
        },
    )

    new_results = response.json()
    assert len(new_results) == len(new_input_ids)

    input_logprobs = []
    for i, result in enumerate(new_results):
        input_logprob = result["meta_info"]["input_token_logprobs"]
        input_logprob = [x[0] for x in input_logprob][-len(output_logprobs[i]) :]
        input_logprobs.append(input_logprob)

    compare_kl_divergence(
        input_logprobs,
        output_logprobs,
        ACC_THRESHOLDS,
        model_name,
        inspect.currentframe().f_code.co_name,
    )
