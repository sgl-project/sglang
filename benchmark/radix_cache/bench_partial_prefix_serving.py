"""Run SGLang's serving benchmark on an exact partial-page prefix workload.

This is a thin dataset/preparation adapter around
``sglang.benchmark.serving``.  The official serving benchmark still owns the
async HTTP client, request-rate/concurrency control, timing, and metrics.

The adapter builds a production RadixCache topology with a shared aligned
prefix, one fully matching child page, and a token-exact match of ``r`` tokens
inside the following page.  It then supplies unique divergent requests so a
measured request cannot accidentally become a full cache hit on an earlier
measured request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import sys
from typing import Any

import requests

from sglang.benchmark import serving
from sglang.benchmark.datasets.common import DatasetRow


def deterministic_tokens(seed: int, length: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 29000) for _ in range(length)]


def force_different(value: int, forbidden: int) -> int:
    if value != forbidden:
        return value
    return 1000 + ((value - 999) % 28000)


def make_divergent_suffix(
    *, seed: int, sample_index: int, length: int, forbidden: int
) -> list[int]:
    suffix = deterministic_tokens(seed + sample_index * 1009, length)
    candidate = 1000 + (sample_index % 13000)
    if candidate == forbidden:
        candidate = 15000 + (sample_index % 13000)
    suffix[0] = candidate
    return suffix


def post_generate(
    base_url: str,
    input_ids: list[int],
    output_len: int,
    cache_salt: str,
    timeout: float,
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": input_ids,
            "cache_salt": cache_salt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    host = args.host
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = args.port or 30000
    return f"http://{host}:{port}"


def parse_adapter_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pp-page-size", type=int, required=True)
    parser.add_argument("--pp-aligned-prefix-tokens", type=int, required=True)
    parser.add_argument("--pp-partial-len", type=int, required=True)
    parser.add_argument("--pp-suffix-len", type=int, required=True)
    parser.add_argument("--pp-output-len", type=int, default=1)
    parser.add_argument("--pp-cache-salt", required=True)
    parser.add_argument("--pp-manual-warmups", type=int, default=3)
    parser.add_argument("--pp-concurrent-warmups", type=int, default=0)
    parser.add_argument("--pp-sample-offset", type=int, default=0)
    parser.add_argument("--pp-expected-cached-tokens", type=int, required=True)
    parser.add_argument("--pp-request-timeout", type=float, default=900)
    adapter_args, serving_argv = parser.parse_known_args()
    if adapter_args.pp_page_size <= 1:
        parser.error("--pp-page-size must be greater than one")
    if not 0 < adapter_args.pp_partial_len < adapter_args.pp_page_size:
        parser.error("--pp-partial-len must be in [1, page_size)")
    if adapter_args.pp_aligned_prefix_tokens % adapter_args.pp_page_size:
        parser.error("--pp-aligned-prefix-tokens must be page aligned")
    if adapter_args.pp_suffix_len < 1:
        parser.error("--pp-suffix-len must be positive")
    if adapter_args.pp_concurrent_warmups < 0:
        parser.error("--pp-concurrent-warmups must be non-negative")
    return adapter_args, serving_argv


def main() -> None:
    adapter_args, serving_argv = parse_adapter_args()
    original_get_dataset = serving.get_dataset

    def get_exact_partial_dataset(args, tokenizer, model_id=None):
        del tokenizer, model_id
        if args.backend != "sglang":
            raise ValueError(
                "The exact partial-prefix adapter requires --backend sglang"
            )
        # The adapter replaces the CLI-selected built-in dataset. Keep the
        # persisted result metadata honest instead of reporting the parser's
        # default (usually ``sharegpt``).
        args.dataset_name = "exact-partial-prefix"
        if args.warmup_requests != 0:
            raise ValueError(
                "Pass --warmup-requests 0: this adapter performs exact-shape "
                "manual warmups that are excluded from the timed request list."
            )

        page_size = adapter_args.pp_page_size
        aligned_prefix = adapter_args.pp_aligned_prefix_tokens
        partial_len = adapter_args.pp_partial_len
        suffix_len = adapter_args.pp_suffix_len
        seed = 91_000_000 + aligned_prefix * 17 + page_size

        common = deterministic_tokens(seed, aligned_prefix)
        target_p0 = deterministic_tokens(seed + 10_000, page_size)
        target_p1 = deterministic_tokens(seed + 20_000, page_size)
        sibling_p0 = deterministic_tokens(seed + 30_000, page_size)
        sibling_p1 = deterministic_tokens(seed + 40_000, page_size)
        sibling_p0[0] = force_different(sibling_p0[0], target_p0[0])
        target_guard = deterministic_tokens(seed + 50_000, 1)
        sibling_guard = deterministic_tokens(seed + 60_000, 1)

        base_url = resolve_base_url(args)
        target_prompt = common + target_p0 + target_p1 + target_guard
        sibling_prompt = common + sibling_p0 + sibling_p1 + sibling_guard
        post_generate(
            base_url,
            target_prompt,
            1,
            adapter_args.pp_cache_salt,
            adapter_args.pp_request_timeout,
        )
        post_generate(
            base_url,
            sibling_prompt,
            1,
            adapter_args.pp_cache_salt,
            adapter_args.pp_request_timeout,
        )

        forbidden = target_p1[partial_len]
        for warmup_index in range(adapter_args.pp_manual_warmups):
            sample_index = adapter_args.pp_sample_offset + warmup_index
            suffix = make_divergent_suffix(
                seed=seed + 70_000,
                sample_index=sample_index,
                length=suffix_len,
                forbidden=forbidden,
            )
            prompt = common + target_p0 + target_p1[:partial_len] + suffix
            output = post_generate(
                base_url,
                prompt,
                adapter_args.pp_output_len,
                adapter_args.pp_cache_salt,
                adapter_args.pp_request_timeout,
            )
            cached = int((output.get("meta_info") or {}).get("cached_tokens", -1))
            if cached != adapter_args.pp_expected_cached_tokens:
                raise RuntimeError(
                    f"Warmup cached-token mismatch: expected "
                    f"{adapter_args.pp_expected_cached_tokens}, got {cached}"
                )

        concurrent_start = (
            adapter_args.pp_sample_offset + adapter_args.pp_manual_warmups
        )

        def run_concurrent_warmup(concurrent_index: int) -> None:
            sample_index = concurrent_start + concurrent_index
            suffix = make_divergent_suffix(
                seed=seed + 70_000,
                sample_index=sample_index,
                length=suffix_len,
                forbidden=forbidden,
            )
            prompt = common + target_p0 + target_p1[:partial_len] + suffix
            output = post_generate(
                base_url,
                prompt,
                adapter_args.pp_output_len,
                adapter_args.pp_cache_salt,
                adapter_args.pp_request_timeout,
            )
            cached = int((output.get("meta_info") or {}).get("cached_tokens", -1))
            if cached != adapter_args.pp_expected_cached_tokens:
                raise RuntimeError(
                    f"Concurrent warmup cached-token mismatch: expected "
                    f"{adapter_args.pp_expected_cached_tokens}, got {cached}"
                )

        if adapter_args.pp_concurrent_warmups:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=adapter_args.pp_concurrent_warmups
            ) as executor:
                list(
                    executor.map(
                        run_concurrent_warmup,
                        range(adapter_args.pp_concurrent_warmups),
                    )
                )

        rows: list[DatasetRow] = []
        first_measured = concurrent_start + adapter_args.pp_concurrent_warmups
        for request_index in range(args.num_prompts):
            sample_index = first_measured + request_index
            suffix = make_divergent_suffix(
                seed=seed + 70_000,
                sample_index=sample_index,
                length=suffix_len,
                forbidden=forbidden,
            )
            prompt = common + target_p0 + target_p1[:partial_len] + suffix
            rows.append(
                DatasetRow(
                    prompt=prompt,
                    prompt_len=len(prompt),
                    output_len=adapter_args.pp_output_len,
                    extra_request_body={"cache_salt": adapter_args.pp_cache_salt},
                )
            )

        aligned_match = aligned_prefix + page_size
        exact_match = aligned_match + partial_len
        print(
            "Exact partial-prefix workload prepared: "
            f"page_size={page_size}, aligned_base={aligned_prefix}, "
            f"aligned_match={aligned_match}, exact_match={exact_match}, "
            f"prompt_len={len(rows[0].prompt)}, requests={len(rows)}, "
            f"manual_warmups={adapter_args.pp_manual_warmups}, "
            f"concurrent_warmups={adapter_args.pp_concurrent_warmups}, "
            f"expected_cached={adapter_args.pp_expected_cached_tokens}"
        )
        return rows

    serving.get_dataset = get_exact_partial_dataset
    try:
        sys.argv = [sys.argv[0], *serving_argv]
        serving.cli_main()
    finally:
        serving.get_dataset = original_get_dataset


if __name__ == "__main__":
    main()
