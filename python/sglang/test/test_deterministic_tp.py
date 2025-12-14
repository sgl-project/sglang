"""
Compare the outputs and logprobs between two host serving systems with the same requests.

Usage:
# Prefix mode: test with shared prefixes on two hosts
python3 -m sglang.test.test_deterministic_tp --host-a localhost --port-a 30000 --host-b localhost --port-b 30001 --test-mode prefix --n-start 1 --n-trials 50 --return-logprob

# Radix Cache Consistency mode: compare radix cache behavior between two hosts
python3 -m sglang.test.test_deterministic_tp --host-a localhost --port-a 30000 --host-b localhost --port-b 30001 --test-mode radix_cache
"""

import argparse
import dataclasses
import json
import os
import random
from typing import Any, Dict, List, Optional

import requests

# Note: run_profile import kept but profiling might be specific to one host
from sglang.profiler import run_profile

PROMPT_1 = "Tell me about Richard Feynman: "
PROMPT_2 = "Generate 1000 random numbers. Go directly into it, don't say Sure and don't say here are numbers. Just start with a number."
dirpath = os.path.dirname(__file__)
# Assuming long_prompt.txt exists in the same directory
if os.path.exists(os.path.join(dirpath, "long_prompt.txt")):
    with open(os.path.join(dirpath, "long_prompt.txt"), "r") as f:
        LONG_PROMPT = f.read()
else:
    LONG_PROMPT = "A long prompt " * 1000


@dataclasses.dataclass
class BenchArgs:
    host_a: str = "localhost"
    port_a: int = 30000
    host_b: str = "localhost"
    port_b: int = 30001  # Default to a different port
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
    test_mode: str = "prefix"
    n_trials: int = 50
    n_start: int = 1

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--host-a", type=str, default=BenchArgs.host_a)
        parser.add_argument("--port-a", type=int, default=BenchArgs.port_a)
        parser.add_argument("--host-b", type=str, default=BenchArgs.host_b)
        parser.add_argument("--port-b", type=int, default=BenchArgs.port_b)
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
                "prefix",
                "radix_cache",
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
    base_url: str,
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

    try:
        response = requests.post(
            f"{base_url}/generate",
            json=json_data,
            stream=args.stream,
        )
    except Exception as e:
        print(f"Connection error to {base_url}: {e}")
        return None

    if response.status_code != 200:
        print(f"Error from {base_url}: {response.json()}")
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
    base_url: str,
    args,
    batch_size: int,
    prompts: List[str],
    sampled_indices: List[int],
    return_full_response: bool = False,
):
    try:
        requests.post(f"{base_url}/flush_cache")
    except Exception:
        pass

    batch_data = []
    for idx in sampled_indices:
        batch_data.append(prompts[idx])

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
        f"{base_url}/generate",
        json=json_data,
        stream=args.stream,
    )
    ret = response.json()
    if response.status_code != 200:
        print(ret)
        return -1

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
        # Each element is [logprob, token_id, token_text]
        if lp1[1] != lp2[1]:
            return False, f"Token ID mismatch at position {i}: {lp1[1]} vs {lp2[1]}"

        # Handle cases where logprob might be None (though unlikely in return)
        v1 = lp1[0] if lp1[0] is not None else 0.0
        v2 = lp2[0] if lp2[0] is not None else 0.0

        if abs(v1 - v2) > tolerance:
            return (
                False,
                f"Logprob mismatch at position {i}: {v1} vs {v2} (diff: {abs(v1 - v2)})",
            )

    return True, "Logprobs match"


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

        if len(a.token_ids) != len(b.token_ids):
            print(f"❌ Length mismatch: {len(a.token_ids)} vs {len(b.token_ids)}")
            return False

        token_match = np.array_equal(a.token_ids, b.token_ids)

        # Handle None in logprobs safely
        a_lp = np.array([x if x is not None else 0.0 for x in a.logprobs])
        b_lp = np.array([x if x is not None else 0.0 for x in b.logprobs])

        logprobs_match = np.allclose(a_lp, b_lp, atol=1e-5)

        if token_match:
            print(f"    ✅ Token match")
        else:
            print(
                f"    ❌ Token mismatch: {a.token_ids[:5]}... vs {b.token_ids[:5]}..."
            )

        if logprobs_match:
            print(f"    ✅ Logprobs match")
        else:
            print(f"    ❌ Logprobs mismatch")
            # Only print first 5 elements for readability
            n_show = 5
            print("      A:   ", [f"{x:.5f}" for x in a.logprobs[:n_show]], "...")
            print("      B:   ", [f"{x:.5f}" for x in b.logprobs[:n_show]], "...")

            diff = np.abs(a_lp - b_lp)
            print(f"      Max Diff: {np.max(diff):.5e}")
            print(f"      Mean Diff: {np.mean(diff):.5e}")

        return token_match and logprobs_match


def _extract_ids_and_logprobs(responses):
    def _extract_part(response, name):
        token_ids, logprobs = [], []
        if name in response["meta_info"]:
            for item in response["meta_info"][name]:
                logprob, token_id, text = item
                token_ids.append(token_id)
                logprobs.append(logprob)
        return TokenIdsAndLogprobs(token_ids=token_ids, logprobs=logprobs)

    def _extract_one_response(response):
        input_data = _extract_part(response, "input_token_logprobs")
        output_data = _extract_part(response, "output_token_logprobs")
        return dict(input=input_data, output=output_data, io=input_data + output_data)

    if not isinstance(responses, list):
        responses = [responses]
    return [_extract_one_response(x) for x in responses]


def test_deterministic(args):
    base_url_a = f"http://{args.host_a}:{args.port_a}"
    base_url_b = f"http://{args.host_b}:{args.port_b}"

    print(f"Comparing Host A ({base_url_a}) vs Host B ({base_url_b})")

    if args.test_mode == "prefix":
        # In prefix mode, we create prompts from the same long prompt, with different lengths of common prefix.
        len_prefix = [1, 511, 2048, 4097]
        num_prompts = len(len_prefix)
        prompts = [LONG_PROMPT[: len_prefix[i]] for i in range(4)]

        total_match = True

        for i in range(args.n_start, args.n_start + args.n_trials):
            batch_size = i

            # Generate deterministic indices for this batch so A and B get same inputs
            sampled_indices = []
            random.seed(
                i
            )  # Seed with trial number ensures A and B get same random indices
            for _ in range(batch_size):
                sampled_indices.append(random.randint(0, len(prompts) - 1))

            print(f"\n--- Trial {i} (Batch Size {batch_size}) ---")

            # Request A
            ret_dict_a = send_prefix(
                base_url_a,
                args,
                batch_size,
                prompts,
                sampled_indices,
                return_full_response=True,
            )

            # Request B
            ret_dict_b = send_prefix(
                base_url_b,
                args,
                batch_size,
                prompts,
                sampled_indices,
                return_full_response=True,
            )

            if ret_dict_a == -1 or ret_dict_b == -1:
                print("Skipping trial due to error.")
                continue

            # Compare A and B
            for p_idx in range(num_prompts):
                res_list_a = ret_dict_a[p_idx]
                res_list_b = ret_dict_b[p_idx]

                if not res_list_a:
                    continue

                print(f"Prompt Prefix {len_prefix[p_idx]}: {len(res_list_a)} samples")

                for j, (resp_a, resp_b) in enumerate(zip(res_list_a, res_list_b)):
                    # Compare Text
                    if resp_a["text"] != resp_b["text"]:
                        print(f"  ❌ Text Mismatch Sample {j}")
                        print(f"     A: {resp_a['text'][:50]}...")
                        print(f"     B: {resp_b['text'][:50]}...")
                        total_match = False

                    # Compare Logprobs if enabled
                    if args.return_logprob:
                        lp_a = resp_a["meta_info"]["output_token_logprobs"]
                        lp_b = resp_b["meta_info"]["output_token_logprobs"]
                        match, msg = compare_logprobs(lp_a, lp_b)
                        if not match:
                            print(f"  ❌ Logprob Mismatch Sample {j}: {msg}")
                            total_match = False
                        else:
                            # Optional: print checkmark only for first sample to avoid spam
                            if j == 0:
                                print("  ✅ Logprobs match")

        if total_match:
            print(
                "\n✓✓✓ All Prefix Tests Passed! Host A and Host B are consistent. ✓✓✓"
            )
        else:
            print("\n✗✗✗ Some Prefix Tests Failed! ✗✗✗")
        return [int(total_match)]

    elif args.test_mode == "radix_cache":
        # Radix mode requires logprobs to compare results
        args.return_logprob = True

        print("\n=== Radix Cache Consistency Test (Host A vs Host B) ===")
        print(
            "Running sequence: Warmup -> Gen (Prefill) -> Gen (Cache Hit) -> Flush -> Gen (Uncached)"
        )

        def run_radix_sequence(url, label):
            print(f"\n--- Running Sequence on {label} ({url}) ---")

            # Warmup
            send_single(
                url,
                args,
                input_ids=[1] * 64,
                max_new_tokens=65,
                return_full_response=True,
            )
            requests.post(f"{url}/flush_cache")

            # Step 1: Initial Tokens
            random.seed(42)
            initial_token_ids = [random.randint(100, 50000) for _ in range(64)]

            # Step 2: First Generation
            resp_1 = send_single(
                url,
                args,
                input_ids=initial_token_ids,
                max_new_tokens=100,
                return_full_response=True,
            )
            out_ids_1 = resp_1["output_ids"]

            # Step 3: Cached Generation (Hit)
            prefix_ids = initial_token_ids + out_ids_1[:-1]
            resp_cached = send_single(
                url,
                args,
                input_ids=prefix_ids,
                max_new_tokens=1,
                return_full_response=True,
            )

            # Step 4: Flush
            requests.post(f"{url}/flush_cache")

            # Step 5: Uncached Generation
            resp_uncached = send_single(
                url,
                args,
                input_ids=prefix_ids,
                max_new_tokens=1,
                return_full_response=True,
            )

            return {"gen_1": resp_1, "cached": resp_cached, "uncached": resp_uncached}

        # Run on A
        res_a = run_radix_sequence(base_url_a, "Host A")
        # Run on B
        res_b = run_radix_sequence(base_url_b, "Host B")

        print(f"\n{'='*60}")
        print("Comparing Host A vs Host B Results")
        print("=" * 60)

        all_passed = True

        # Compare Step 2 (First Gen)
        print("1. Initial Generation Comparison:")
        lp_a = res_a["gen_1"]["meta_info"]["output_token_logprobs"]
        lp_b = res_b["gen_1"]["meta_info"]["output_token_logprobs"]
        match, msg = compare_logprobs(lp_a, lp_b)
        if match:
            print("   ✅ Match")
        else:
            print(f"   ❌ Mismatch: {msg}")
            all_passed = False

        # Compare Step 3 (Cached)
        print("2. Cached Generation Comparison:")
        lp_a = res_a["cached"]["meta_info"]["output_token_logprobs"]
        lp_b = res_b["cached"]["meta_info"]["output_token_logprobs"]
        match, msg = compare_logprobs(lp_a, lp_b)
        if match:
            print("   ✅ Match")
        else:
            print(f"   ❌ Mismatch: {msg}")
            all_passed = False

        # Compare Step 5 (Uncached)
        print("3. Uncached Generation Comparison:")
        lp_a = res_a["uncached"]["meta_info"]["output_token_logprobs"]
        lp_b = res_b["uncached"]["meta_info"]["output_token_logprobs"]
        match, msg = compare_logprobs(lp_a, lp_b)
        if match:
            print("   ✅ Match")
        else:
            print(f"   ❌ Mismatch: {msg}")
            all_passed = False

        if all_passed:
            print("\n✓✓✓ Radix Cache Consistency Test Passed! ✓✓✓")
            return [1]
        else:
            print("\n✗✗✗ Radix Cache Consistency Test Failed! ✗✗✗")
            return [0]
    else:
        raise ValueError(f"Invalid test mode: {args.test_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    if args.sampling_seed is None:
        args.sampling_seed = 42

    test_deterministic(args)
