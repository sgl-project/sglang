import inspect
import os

import numpy as np
import pandas as pd
import requests

from sglang.srt.utils.hf_transformers_utils import get_tokenizer

NUMERICS_DATASET = "TODO"


def load_prompts_from_parquet(directory):
    tokens = []
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_parquet(file_path)
                if "tokens" in df.columns:
                    tokens.extend(df["tokens"].tolist())
            except Exception as e:
                print(f"[ERROR] Failed to read {filename}: {str(e)}")
    return tokens


def get_input_ids_and_disallowed_tokens():
    input_ids = load_prompts_from_parquet(NUMERICS_DATASET)
    input_ids = [x.tolist() for x in input_ids]
    input_ids = input_ids[::10]
    disallowed_tokens = [0, 2]
    return input_ids, disallowed_tokens


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
    input_ids, disallowed_tokens = get_input_ids_and_disallowed_tokens()
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
                "disallowed_token_ranges": ",".join(map(str, disallowed_tokens)),
                # "ignore_eos": True,
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
        # disallow tokens should not be in the output
        for disallowed_token in disallowed_tokens:
            assert disallowed_token not in output_ids
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
                # "ignore_eos": True,
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

    input_ids, disallowed_tokens = get_input_ids_and_disallowed_tokens()
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
                "disallowed_token_ranges": ",".join(map(str, disallowed_tokens)),
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
        # disallow tokens should not be in the output
        for disallowed_token in disallowed_tokens:
            assert disallowed_token not in output_ids
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
                # "ignore_eos": True,
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

    first_turn_input_ids, disallowed_tokens = get_input_ids_and_disallowed_tokens()
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
                "disallowed_token_ranges": ",".join(map(str, disallowed_tokens)),
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
        },
    )

    tokenizer = get_tokenizer("/data/datasets/tokenizers/model/v6/v6.xtok.json")
    comma_token_id = tokenizer.encode(",")  # add comma to ensure cache hit

    results = response.json()
    assert len(results) == len(first_turn_input_ids)
    second_turn_input_ids = []
    for i, result in enumerate(results):
        output_ids = result["output_ids"]
        # disallow tokens should not be in the output
        for disallowed_token in disallowed_tokens:
            assert disallowed_token not in output_ids
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
                "disallowed_token_ranges": ",".join(map(str, disallowed_tokens)),
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
        # disallow tokens should not be in the output
        for disallowed_token in disallowed_tokens:
            assert disallowed_token not in output_ids
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
