"""
Batch the same prompt in random batch sizes, and test if the results are consistent across different trials.

Usage:
# Single mode: test determinism with varying batch sizes
python3 -m sglang.test.test_deterministic --n-trials 50 --test-mode single

# Prefix mode: test with shared prefixes
python3 -m sglang.test.test_deterministic --n-start 1 --n-trials 50 --test-mode prefix

# Radix Cache Consistency mode: test radix cache determinism (cached vs uncached prefill)
python3 -m sglang.test.test_deterministic --test-mode radix_cache
"""

import argparse
import dataclasses
import json
import os
import random
from typing import Any, Dict, List, Optional

import requests

from sglang.profiler import run_profile

PROMPT_1 = "Tell me about Richard Feynman: "
PROMPT_2 = "Generate 1000 random numbers. Go directly into it, don't say Sure and don't say here are numbers. Just start with a number."
dirpath = os.path.dirname(__file__)
with open(os.path.join(dirpath, "long_prompt.txt"), "r") as f:
    LONG_PROMPT = f.read()


@dataclasses.dataclass
class BenchArgs:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 1
    temperature: float = 0.0
    sampling_seed: int = 42
    max_new_tokens: int = 100
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    return_logprob: bool = False
    stream: bool = False
    profile: bool = False
    profile_steps: int = 3
    profile_by_stage: bool = False
    test_mode: str = "single"
    n_trials: int = 50
    n_start: int = 1

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host", type=str, default=BenchArgs.host)
        parser.add_argument("--port", type=int, default=BenchArgs.port)
        parser.add_argument("--n-trials", type=int, default=BenchArgs.n_trials)
        parser.add_argument("--n-start", type=int, default=BenchArgs.n_start)
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument(
            "--sampling-seed", type=int, default=BenchArgs.sampling_seed
        )
        parser.add_argument(
            "--max-new-tokens", type=int, default=BenchArgs.max_new_tokens
        )
        parser.add_argument(
            "--frequency-penalty", type=float, default=BenchArgs.frequency_penalty
        )
        parser.add_argument(
            "--presence-penalty", type=float, default=BenchArgs.presence_penalty
        )
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument("--stream", action="store_true")
        parser.add_argument(
            "--test-mode",
            type=str,
            default=BenchArgs.test_mode,
            choices=[
                "single",
                "prefix",
                "radix_cache",
                "p_vs_d",
            ],
        )
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def send_single(
    args,
    profile: bool = False,
    profile_steps: int = 3,
    profile_by_stage: bool = False,
    return_full_response: bool = False,
    input_ids: List[int] = None,
    prompt: List[str] = None,
    max_new_tokens: int = None,
    extra_params: Optional[Dict[str, Any]] = None,
    pick_first_result: bool = True,
):
    base_url = f"http://{args.host}:{args.port}"

    # Use input_ids if provided, otherwise use text prompts
    if input_ids is not None:
        assert prompt is None
        json_data = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": (
                    max_new_tokens
                    if max_new_tokens is not None
                    else args.max_new_tokens
                ),
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
            },
            "return_logprob": args.return_logprob,
            "stream": args.stream,
            **(extra_params or {}),
        }
    else:
        assert input_ids is None
        json_data = {
            "text": prompt,
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": (
                    max_new_tokens
                    if max_new_tokens is not None
                    else args.max_new_tokens
                ),
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty,
            },
            "return_logprob": args.return_logprob,
            "stream": args.stream,
            **(extra_params or {}),
        }

    if args.sampling_seed is not None:
        # sglang server cannot parse None value for sampling_seed
        json_data["sampling_params"]["sampling_seed"] = args.sampling_seed

    if profile:
        run_profile(
            url=base_url,
            num_steps=profile_steps,
            activities=["CPU", "GPU"],
            profile_by_stage=profile_by_stage,
        )

    response = requests.post(
        f"{base_url}/generate",
        json=json_data,
        stream=args.stream,
    )

    if response.status_code != 200:
        ret = response.json()
        print(f"Error: {ret}")
        return None

    if args.stream:
        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if chunk and chunk.startswith("data:"):
                if chunk == "data: [DONE]":
                    break
                ret = json.loads(chunk[5:].strip("\n"))
    else:
        ret = response.json()

    if pick_first_result:
        ret = ret[0] if isinstance(ret, list) else ret

    if return_full_response:
        return ret
    else:
        return ret["text"]


def send_prefix(
    args, batch_size: int, prompts: List[str], return_full_response: bool = False
):
    requests.post(f"http://{args.host}:{args.port}/flush_cache")

    batch_data = []
    sampled_indices = []
    for _ in range(batch_size):
        sampled_index = random.randint(0, len(prompts) - 1)
        sampled_indices.append(sampled_index)
        batch_data.append(prompts[sampled_index])

    json_data = {
        "text": batch_data,
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
        },
        "return_logprob": args.return_logprob,
        "stream": args.stream,
    }

    if args.sampling_seed is not None:
        json_data["sampling_params"]["sampling_seed"] = args.sampling_seed

    response = requests.post(
        f"http://{args.host}:{args.port}/generate",
        json=json_data,
        stream=args.stream,
    )
    ret = response.json()
    if response.status_code != 200:
        print(ret)
        return -1, -1, -1

    if return_full_response:
        # Return full responses grouped by prompt index
        ret_dict = {i: [] for i in range(len(prompts))}
        for i in range(batch_size):
            ret_dict[sampled_indices[i]].append(ret[i])
        return ret_dict
    else:
        # Return only text grouped by prompt index
        ret_dict = {i: [] for i in range(len(prompts))}
        for i in range(batch_size):
            ret_dict[sampled_indices[i]].append(ret[i]["text"])
        return ret_dict


def compare_logprobs(logprobs1, logprobs2, tolerance=0):
    """Compare two logprobs sequences with a tolerance."""
    if len(logprobs1) != len(logprobs2):
        return False, f"Length mismatch: {len(logprobs1)} vs {len(logprobs2)}"

    for i, (lp1, lp2) in enumerate(zip(logprobs1, logprobs2)):
        # Each element is [logprob, token_id]
        if lp1[1] != lp2[1]:
            return False, f"Token ID mismatch at position {i}: {lp1[1]} vs {lp2[1]}"
        if abs(lp1[0] - lp2[0]) > tolerance:
            return (
                False,
                f"Logprob mismatch at position {i}: {lp1[0]} vs {lp2[0]} (diff: {abs(lp1[0] - lp2[0])})",
            )

    return True, "Logprobs match"


def _test_mode_p_vs_d(args, batch_size):
    print()
    print(f"Execute: test p_vs_d {batch_size=}")

    random.seed(42)
    args.return_logprob = True
    query_extra_params = {
        "logprob_start_len": 0,
        "return_text_in_logprobs": True,
    }

    def _create_prompts():
        ans = [PROMPT_1, PROMPT_2]
        for i in range(batch_size - len(ans)):
            end = random.randrange(1, 4096)
            if random.random() < 0.5:
                begin = 0
            else:
                begin = random.randrange(0, end)
            ans.append(LONG_PROMPT[begin:end])
        return ans[:batch_size]

    # warmup + flush
    send_single(args, input_ids=[1] * 64, max_new_tokens=65, return_full_response=True)
    requests.post(f"http://{args.host}:{args.port}/flush_cache")

    prompts = _create_prompts()

    resp_a = send_single(
        args,
        prompt=prompts,
        max_new_tokens=args.max_new_tokens,
        return_full_response=True,
        pick_first_result=False,
        extra_params=query_extra_params,
    )
    info_a = _extract_ids_and_logprobs(resp_a)

    requests.post(f"http://{args.host}:{args.port}/flush_cache")

    resp_b = send_single(
        args,
        input_ids=[x["io"].token_ids for x in info_a],
        max_new_tokens=1,
        return_full_response=True,
        pick_first_result=False,
        extra_params=query_extra_params,
    )
    info_b = _extract_ids_and_logprobs(resp_b)

    ans = []
    for i, (info_a_item, info_b_item) in enumerate(zip(info_a, info_b, strict=True)):
        print(f"Compare sequence {i} in batch...")
        correct = TokenIdsAndLogprobs.compare(info_a_item["io"], info_b_item["input"])
        ans.append(int(correct))

    return ans


@dataclasses.dataclass
class TokenIdsAndLogprobs:
    token_ids: List[int]
    logprobs: List[float]

    def __add__(self, other):
        return TokenIdsAndLogprobs(
            token_ids=self.token_ids + other.token_ids,
            logprobs=self.logprobs + other.logprobs,
        )

    @classmethod
    def compare(cls, a: "TokenIdsAndLogprobs", b: "TokenIdsAndLogprobs"):
        import numpy as np

        assert len(a.token_ids) == len(b.token_ids)
        token_match = a.token_ids == b.token_ids
        logprobs_match = a.logprobs == b.logprobs

        if token_match:
            print(f"✅ Token match")
        else:
            print(f"❌ Token mismatch: {a.token_ids=} {b.token_ids=}")

        if logprobs_match:
            print(f"✅ Logprobs match:", a.logprobs[:5])
        else:
            print(f"❌ Logprobs mismatch")
            # Only print first 5 elements for readability
            n_show = 5
            a_show = a.logprobs[:n_show]
            b_show = b.logprobs[:n_show]
            print(
                "    A:   ",
                [f"{x:.10f}" if x is not None else "None" for x in a_show],
                f"... ({len(a.logprobs)} total)" if len(a.logprobs) > n_show else "",
            )
            print(
                "    B:   ",
                [f"{x:.10f}" if x is not None else "None" for x in b_show],
                f"... ({len(b.logprobs)} total)" if len(b.logprobs) > n_show else "",
            )
            diff = [
                abs(x - y) if x is not None else float("nan")
                for x, y in zip(a.logprobs, b.logprobs)
            ]
            print(
                "    Diff:",
                [f"{x:.10e}" for x in diff[:n_show]],
                f"... ({len(diff)} total)" if len(diff) > n_show else "",
            )

            # Compute KL-divergence using K3 approximation
            # KL(P||Q) ≈ (exp(log(P) - log(Q)) - 1) - (log(P) - log(Q))
            # This is based on selected token logprobs only
            valid_pairs = [
                (lp_a, lp_b)
                for lp_a, lp_b in zip(a.logprobs, b.logprobs)
                if lp_a is not None and lp_b is not None
            ]
            if valid_pairs and token_match:
                logprobs_a = np.array([lp for lp, _ in valid_pairs])
                logprobs_b = np.array([lp for _, lp in valid_pairs])

                # K3 approximation: KL(A||B) ≈ (exp(logr) - 1) - logr, where logr = log_a - log_b
                logr = logprobs_a - logprobs_b
                kl_per_token = (np.exp(logr) - 1) - logr
                kl_mean = np.mean(kl_per_token)
                kl_max = np.max(kl_per_token)

                print(f"    KL(A||B) mean: {kl_mean:.10e}")
                print(f"    KL(A||B) max : {kl_max:.10e}")
                print(f"    Mean absolute logprob diff: {np.mean(np.abs(logr)):.10e}")

        return token_match and logprobs_match


def _extract_ids_and_logprobs(responses):
    def _extract_part(response, name):
        token_ids, logprobs = [], []
        for item in response["meta_info"][name]:
            logprob, token_id, text = item
            token_ids.append(token_id)
            logprobs.append(logprob)
        return TokenIdsAndLogprobs(token_ids=token_ids, logprobs=logprobs)

    def _extract_one_response(response):
        input = _extract_part(response, "input_token_logprobs")
        output = _extract_part(response, "output_token_logprobs")
        return dict(input=input, output=output, io=input + output)

    if not isinstance(responses, list):
        responses = [responses]
    return [_extract_one_response(x) for x in responses]


def test_deterministic(args):
    if args.test_mode == "single":
        # In single mode, we test the deterministic behavior by sending the same prompt in batch sizes ranging from 1 to n_trials.
        texts = []
        for i in range(1, args.n_trials + 1):
            batch_size = i
            text = send_single(args, args.profile, prompt=[PROMPT_1] * batch_size)
            text = text.replace("\n", " ")
            print(f"Trial {i} with batch size {batch_size}: {text}")
            texts.append(text)
        print(f"Total samples: {len(texts)}, Unique samples: {len(set(texts))}")
        return [len(set(texts))]

    elif args.test_mode == "prefix":
        # In prefix mode, we create prompts from the same long prompt, with different lengths of common prefix.
        len_prefix = [1, 511, 2048, 4097]
        num_prompts = len(len_prefix)
        outputs = {i: [] for i in range(4)}
        prompts = [LONG_PROMPT[: len_prefix[i]] for i in range(4)]

        # If return_logprob is enabled, store full responses for comparison
        if args.return_logprob:
            full_responses = {i: [] for i in range(4)}

        for i in range(args.n_start, args.n_start + args.n_trials):
            batch_size = i
            ret_dict = send_prefix(
                args, batch_size, prompts, return_full_response=args.return_logprob
            )
            msg = f"Testing Trial {i} with batch size {batch_size},"
            for i in range(num_prompts):
                msg += f" # prefix length {len_prefix[i]}: {len(ret_dict[i])},"
            print(msg)
            for i in range(num_prompts):
                if args.return_logprob:
                    # Store full response for logprob comparison
                    full_responses[i].extend(ret_dict[i])
                    # Extract text for determinism check
                    outputs[i].extend([resp["text"] for resp in ret_dict[i]])
                else:
                    outputs[i].extend(ret_dict[i])

        for i in range(num_prompts):
            print(
                f"Prompt {i} with prefix length {len_prefix[i]}: total samples: {len(outputs[i])}, Unique samples: {len(set(outputs[i]))}"
            )

        results = []
        for i in range(num_prompts):
            results.append(len(set(outputs[i])))

        # If logprobs are enabled, compare them across different batch sizes
        if args.return_logprob:
            print(f"\n{'='*60}")
            print("Logprobs Comparison Across Batch Sizes")
            print("=" * 60)

            logprob_results = []
            for prompt_idx in range(num_prompts):
                print(
                    f"\nPrompt {prompt_idx} (prefix length {len_prefix[prompt_idx]}):"
                )
                responses = full_responses[prompt_idx]

                if len(responses) < 2:
                    continue

                # Compare all responses against the first one
                reference = responses[0]
                all_match = True
                mismatches = []

                for j, resp in enumerate(responses[1:], start=1):
                    ref_logprobs = reference["meta_info"]["output_token_logprobs"]
                    resp_logprobs = resp["meta_info"]["output_token_logprobs"]

                    match, msg = compare_logprobs(ref_logprobs, resp_logprobs)

                    if not match:
                        print(f"  ✗ Sample {j+1}: {msg}")
                        mismatches.append((j + 1, msg))
                        all_match = False

                if all_match:
                    print(f"  ✓ All {len(responses)} samples have identical logprobs")
                    logprob_results.append(1)
                else:
                    print(
                        f"  ✗ Found {len(mismatches)} mismatches out of {len(responses)} samples"
                    )
                    logprob_results.append(0)

            print(f"\n{'='*60}")
            if all(r == 1 for r in logprob_results):
                print("✓✓✓ Logprobs are identical across all batch sizes! ✓✓✓")
            else:
                print("✗✗✗ Some logprobs differ across batch sizes! ✗✗✗")

        return results

    elif args.test_mode == "radix_cache":
        # Radix mode requires logprobs to compare results
        args.return_logprob = True

        print("\n=== Prefill Cache Consistency Test ===")
        print(
            "This test verifies prefill request produces consistent logprobs w/ and w/o cache.\n"
        )

        # We noticed that we cannot call flush cache before any request, otherwise it will hang.
        warmup_response = send_single(
            args, input_ids=[1] * 64, max_new_tokens=65, return_full_response=True
        )

        # Flush cache first to make sure there is no cache hit from previous tests
        flush_response = requests.post(f"http://{args.host}:{args.port}/flush_cache")

        print(f"Step 1: Generating random 64 token IDs...")
        # Use a reasonable token ID range (e.g., 1-50000 for most tokenizers)
        # Avoid special tokens like 0 (padding), 1 (BOS), 2 (EOS)
        # set seed for random.randint
        random.seed(42)
        initial_token_ids = [random.randint(100, 50000) for _ in range(64)]

        print(f"✓ Using {len(initial_token_ids)} initial tokens")
        print(f"  Initial token IDs: {initial_token_ids}")

        print(
            f"\nStep 2: Generating 2 tokens from {len(initial_token_ids)} token prefix..."
        )
        first_response = send_single(
            args,
            input_ids=initial_token_ids,
            max_new_tokens=100,
            return_full_response=True,
        )
        first_output_text = first_response["text"]
        first_output_token_ids = first_response["output_ids"]
        first_output_logprobs = first_response["meta_info"]["output_token_logprobs"]

        expected_token_id = first_output_token_ids[-1]
        expected_logprob = first_output_logprobs[-1][0]

        print(f"✓ Generated {len(first_output_token_ids)} tokens")
        print(f'  Output text: "{first_output_text}"')

        print(
            f"\nStep 3: Generating with radix cache (164 tokens prefill, should hit > 128 tokens cache, based on page size)..."
        )
        prefix_token_ids = initial_token_ids + first_output_token_ids[:-1]
        print(
            f"  Prefix: {len(initial_token_ids)} initial + 64 generated = {len(prefix_token_ids)} tokens"
        )
        print(f"Using Prompt: {prefix_token_ids}")
        cached_response = send_single(
            args,
            input_ids=prefix_token_ids,
            max_new_tokens=1,
            return_full_response=True,
        )
        cached_logprobs = cached_response["meta_info"]["output_token_logprobs"]
        cached_token_data = cached_logprobs[0]
        cached_logprob = cached_token_data[0]
        cached_token_id = cached_token_data[1]

        print(f"✓ Generated with cache:")
        print(f"  Token ID: {cached_token_id}")
        print(f"  Logprob:  {cached_logprob:.10f}")

        print(f"\nStep 4: Flushing cache...")
        flush_response = requests.post(f"http://{args.host}:{args.port}/flush_cache")

        print(
            f"\nStep 5: Generating without cache (same 164 tokens prefill, no cache)..."
        )
        print(f"Using Prompt: {prefix_token_ids}")

        uncached_response = send_single(
            args,
            input_ids=prefix_token_ids,
            max_new_tokens=1,
            return_full_response=True,
        )

        uncached_logprobs = uncached_response["meta_info"]["output_token_logprobs"]
        uncached_token_data = uncached_logprobs[0]
        uncached_logprob = uncached_token_data[0]
        uncached_token_id = uncached_token_data[1]

        print(f"✓ Generated without cache:")
        print(f"  Token ID: {uncached_token_id}")
        print(f"  Logprob:  {uncached_logprob:.10f}")

        # Step 6: Compare results
        print(f"\n{'='*60}")
        print("Comparison 1: Decode (Request 1) vs Prefill with Cache (Request 2)")
        print("=" * 60)

        # Compare first request (decode) vs second request (prefill with cache)
        # We expect them to be different (different kernels)
        decode_vs_prefill_token_match = expected_token_id == cached_token_id
        decode_vs_prefill_logprob_match = expected_logprob == cached_logprob

        print(
            f"  Decode token (Request 1):          ID={expected_token_id}, logprob={expected_logprob:.10f}"
        )
        print(
            f"  Prefill w/ cache token (Request 2): ID={cached_token_id}, logprob={cached_logprob:.10f}"
        )
        print(
            f"  Token ID match: {'✓ YES' if decode_vs_prefill_token_match else '✗ NO'}"
        )
        print(
            f"  Logprob match:  {'✓ YES' if decode_vs_prefill_logprob_match else '✗ NO'}"
        )
        if not decode_vs_prefill_logprob_match:
            diff = abs(expected_logprob - cached_logprob)
            print(f"  Logprob difference: {diff:.10e}")
        print(f"  Note: We expect these to be DIFFERENT (decode vs prefill kernels)")

        print(f"\n{'='*60}")
        print(
            "Comparison 2: Cached Prefill (Request 2) vs Uncached Prefill (Request 3)"
        )
        print("=" * 60)

        # Main test: compare cached vs uncached prefill (should be identical)
        token_match = cached_token_id == uncached_token_id
        logprob_match = cached_logprob == uncached_logprob

        print(
            f"  Cached prefill token (Request 2):   ID={cached_token_id}, logprob={cached_logprob:.10f}"
        )
        print(
            f"  Uncached prefill token (Request 3): ID={uncached_token_id}, logprob={uncached_logprob:.10f}"
        )
        print(f"  Token ID match: {'✓ YES' if token_match else '✗ NO'}")
        if not token_match:
            print(f"    Cached:   {cached_token_id}")
            print(f"    Uncached: {uncached_token_id}")

        print(f"  Logprob match:  {'✓ YES' if logprob_match else '✗ NO'}")
        if not logprob_match:
            print(f"    Cached:   {cached_logprob:.10f}")
            print(f"    Uncached: {uncached_logprob:.10f}")
            diff = abs(cached_logprob - uncached_logprob)
            print(f"    Difference: {diff:.10e}")
        print(f"  Note: We expect these to be IDENTICAL (both prefill kernels)")

        print(f"\n{'='*60}")
        if token_match and logprob_match:
            print("✓✓✓ TEST PASSED - Radix cache is consistent! ✓✓✓")
            return [1]
        else:
            print("✗✗✗ TEST FAILED - Radix cache produces different results! ✗✗✗")
            return [0]

    elif args.test_mode == "p_vs_d":
        # TODO also extract other modes to functions
        ans = []
        for i in range(1, args.n_trials + 1):
            ans += _test_mode_p_vs_d(args, batch_size=i)
        return ans

    else:
        raise ValueError(f"Invalid test mode: {args.test_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    if args.sampling_seed is None:
        args.sampling_seed = 42

    test_deterministic(args)
