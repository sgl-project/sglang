import inspect
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

MMLU_DATA_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
MMLU_DATA_DIR = os.path.join(os.path.dirname(__file__), "mmlu_data")


def download_and_extract_mmlu_data():
    """Download and extract MMLU dataset if not already present."""
    data_dir = os.path.join(MMLU_DATA_DIR, "data")
    if os.path.exists(data_dir):
        return data_dir

    os.makedirs(MMLU_DATA_DIR, exist_ok=True)
    tar_path = os.path.join(MMLU_DATA_DIR, "data.tar")

    if not os.path.exists(tar_path):
        print(f"Downloading MMLU dataset from {MMLU_DATA_URL}...")
        urllib.request.urlretrieve(MMLU_DATA_URL, tar_path)

    print(f"Extracting MMLU dataset to {MMLU_DATA_DIR}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(MMLU_DATA_DIR)

    return data_dir


def load_questions_from_mmlu(data_dir, tokenizer):
    """Load questions from MMLU dataset and tokenize them."""
    test_dir = os.path.join(data_dir, "test")
    input_ids = []
    for filename in sorted(os.listdir(test_dir)):
        if filename.endswith("_test.csv"):
            df = pd.read_csv(os.path.join(test_dir, filename), header=None)
            for i in range(df.shape[0]):
                question = df.iloc[i, 0]  # First column is the question
                answer = df.iloc[i, 1]  # Second column is the answer
                to_encode = str(question) + " " + str(answer)
                # print(f"{i}: {to_encode}")
                tokens = tokenizer.encode(to_encode)
                input_ids.append(tokens)
    return input_ids


def get_input_ids(tokenizer_path):
    """Get input_ids from MMLU dataset."""
    data_dir = download_and_extract_mmlu_data()
    tokenizer = get_tokenizer(tokenizer_path)
    input_ids = load_questions_from_mmlu(data_dir, tokenizer)
    input_ids = input_ids[::10]  # Sample every 10th prompt
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
