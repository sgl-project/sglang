"""
Benchmark the latency of running a single batch with a server.

This script launches a server and uses the HTTP interface.
It accepts server arguments (the same as launch_server.py) and benchmark arguments (e.g., batch size, input lengths).

Usage:
python3 -m sglang.benchmark.one_batch_server --model meta-llama/Meta-Llama-3.1-8B --batch-size 1 16 64 --input-len 1024 --output-len 8

python3 -m sglang.benchmark.one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8
python3 -m sglang.benchmark.one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --show-report --profile --profile-by-stage
python3 -m sglang.benchmark.one_batch_server --model None --base-url http://localhost:30000 --batch-size 16 --input-len 1024 --output-len 8 --result-filename results.jsonl --profile
"""

import argparse
import dataclasses
import itertools
import json
import os
import random
import re
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
from pydantic import BaseModel
from tabulate import tabulate
from transformers import AutoProcessor, PreTrainedTokenizer

from sglang.benchmark.datasets import get_dataset
from sglang.benchmark.endpoint import acquire_endpoint
from sglang.benchmark.stream_metrics import (
    BatchStreamRecorder,
    SteadyStateWindow,
    validate_finish_reason,
)
from sglang.benchmark.utils import get_processor, get_tokenizer
from sglang.profiler import run_profile
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_blackwell
from sglang.test.nightly_bench_utils import save_results_as_pydantic_models
from sglang.test.test_utils import is_in_ci, write_github_step_summary

DEFAULT_TIMEOUT = 600

# Replay recorded text prompts (encoded client-side); recorded completion
# lens are ignored -- --output-len caps every request.
REPLAY_TEXT_DATASETS = ("sharegpt", "custom")


def get_cache_tokens_from_metrics(url: str) -> Optional[tuple]:
    """
    Get cached_tokens_total and prompt_tokens_total from Prometheus /metrics endpoint.
    Returns (cached_tokens_total, prompt_tokens_total) or None if metrics are not available.
    """
    try:
        response = requests.get(url + "/metrics", timeout=5)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            return None

        # Parse Prometheus text format
        # Looking for: sglang:cached_tokens_total{...} <value>
        #              sglang:prompt_tokens_total{...} <value>
        cached_tokens_total = 0.0
        prompt_tokens_total = 0.0

        for line in response.text.split("\n"):
            if line.startswith("sglang:cached_tokens_total{"):
                match = re.search(
                    r"sglang:cached_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    cached_tokens_total += float(match.group(1))
            elif line.startswith("sglang:prompt_tokens_total{"):
                match = re.search(
                    r"sglang:prompt_tokens_total\{[^}]*\}\s+([\d.eE+-]+)", line
                )
                if match:
                    prompt_tokens_total += float(match.group(1))

        return (cached_tokens_total, prompt_tokens_total)
    except Exception as e:
        print(f"Warning: Failed to get cache tokens from metrics: {e}")
        return None


def calculate_cache_hit_rate(
    before: Optional[tuple], after: Optional[tuple]
) -> Optional[float]:
    """
    Calculate cache hit rate from before/after metrics snapshots.
    Returns cached_tokens_delta / prompt_tokens_delta for the benchmark run.
    """
    if before is None or after is None:
        return None

    cached_delta = after[0] - before[0]
    prompt_delta = after[1] - before[1]

    if prompt_delta > 0:
        return cached_delta / prompt_delta
    return None


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    disable_ignore_eos: bool = False
    return_logprob: bool = False
    client_stream_interval: int = 1
    input_len_step_percentage: float = 0.0
    base_url: str = ""
    local_tokenizer_path: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_start_step: Optional[int] = None
    profile_steps: int = 5
    profile_by_stage: bool = False
    profile_prefix: Optional[str] = None
    profile_output_dir: Optional[str] = None
    dataset_path: str = ""
    dataset_name: str = "random"
    fixed_prompt_file: str = ""
    apply_chat_template: bool = False
    gsp_num_groups: int = 1
    gsp_system_prompt_len: int = 2048
    gsp_question_len: int = 128
    gsp_output_len: int = 256
    parallel_batch: bool = False
    result_filename: str = "result.jsonl"
    pydantic_result_filename: Optional[str] = None
    append_to_github_summary: bool = True
    seed: int = 42
    cache_hit_rate: float = 0.0
    backend: str = "sglang"
    fake_prefill: bool = False
    server_args_for_metrics: Optional[List[str]] = None
    lora_name: Optional[List[str]] = None
    lora_request_distribution: str = "uniform"
    lora_zipf_alpha: float = 1.1
    enable_multi_batch: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument(
            "--disable-ignore-eos",
            action="store_true",
            help=(
                "Let requests finish at EOS/stop tokens instead of sending "
                "ignore_eos=true. Use when benchmarking with real prompts "
                "(e.g. --dataset-name sharegpt/custom), where forcing decode "
                "past EOS shifts the output distribution toward gibberish. "
                "Requests then early-stop, so the whole-run output_throughput "
                "(actual tokens generated after the last request's first "
                "token, divided by the time since then) still includes the "
                "decaying-batch tail; read the steady-state metrics instead, "
                "which cover only the window where every request is still "
                "decoding (from the last request's first token to the first "
                "request's finish)."
            ),
        )
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--client-stream-interval",
            type=int,
            default=BenchArgs.client_stream_interval,
        )
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument(
            "--local-tokenizer-path",
            type=str,
            default=BenchArgs.local_tokenizer_path,
            help=(
                "Local tokenizer path to use when benchmarking an external "
                "SGLang server via --base-url. Defaults to the tokenizer path "
                "reported by /server_info."
            ),
        )
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=("CPU", "GPU"),
            choices=["CPU", "GPU", "XPU"],
            help="Profiler activities: CPU, GPU, XPU. use torch profiler.",
        )
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=BenchArgs.profile_start_step,
            help="Start profiling after this many forward steps. Useful for warmup.",
        )
        parser.add_argument(
            "--profile-steps", type=int, default=BenchArgs.profile_steps
        )
        parser.add_argument("--profile-by-stage", action="store_true")
        parser.add_argument(
            "--profile-prefix",
            type=str,
            default=BenchArgs.profile_prefix,
        )
        parser.add_argument(
            "--profile-output-dir",
            type=str,
            default=BenchArgs.profile_output_dir,
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default=BenchArgs.dataset_path,
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--dataset-name",
            type=str,
            default=BenchArgs.dataset_name,
            choices=[
                "mmmu",
                "random",
                "random-ids",
                "generated-shared-prefix",
                "sharegpt",
                "custom",
            ],
            help="Name of the dataset to benchmark on. sharegpt/custom replay "
            "recorded text prompts (custom reads --dataset-path JSONL); their "
            "recorded output lens are ignored in favor of --output-len. For "
            "OpenAI-format traces with per-request parameters, use "
            "sglang.benchmark.serving instead.",
        )
        parser.add_argument(
            "--fixed-prompt-file",
            type=str,
            default=BenchArgs.fixed_prompt_file,
            help="Use this file's prompt for every request in the batch, "
            "bypassing --dataset-name.",
        )
        parser.add_argument(
            "--apply-chat-template",
            action="store_true",
            help="Encode each prompt as a single user message through the "
            "model's chat template. Requires --fixed-prompt-file or a replay "
            "text dataset (sharegpt/custom).",
        )
        parser.add_argument(
            "--gsp-num-groups",
            type=int,
            default=BenchArgs.gsp_num_groups,
            help="Number of shared prefix groups. batch_size requests are distributed across groups.",
        )
        parser.add_argument(
            "--gsp-system-prompt-len",
            type=int,
            default=BenchArgs.gsp_system_prompt_len,
            help="Length of the shared system prompt in tokens per group.",
        )
        parser.add_argument(
            "--gsp-question-len",
            type=int,
            default=BenchArgs.gsp_question_len,
            help="Length of the unique question suffix in tokens per request.",
        )
        parser.add_argument(
            "--gsp-output-len",
            type=int,
            default=BenchArgs.gsp_output_len,
            help="Output length in tokens for generated-shared-prefix requests.",
        )
        parser.add_argument("--parallel-batch", action="store_true")
        parser.add_argument(
            "--result-filename",
            type=str,
            default=BenchArgs.result_filename,
            help="Store the results line by line in the JSON Line format to this file.",
        )
        parser.add_argument(
            "--pydantic-result-filename",
            type=str,
            default=BenchArgs.pydantic_result_filename,
            help="Store the results as pydantic models in the JSON format to this file.",
        )
        parser.add_argument(
            "--no-append-to-github-summary",
            action="store_false",
            dest="append_to_github_summary",
            help="Disable appending the output of this run to github ci summary",
        )
        parser.add_argument("--seed", type=int, default=BenchArgs.seed)
        parser.add_argument(
            "--cache-hit-rate",
            type=float,
            default=BenchArgs.cache_hit_rate,
            help="Cache hit rate for benchmarking (0.0-1.0). "
            "0.0 means no cache hits (flush all), 0.4 means 40%% of input tokens are cached.",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=BenchArgs.backend,
            choices=["sglang", "vllm"],
            help="Backend server type (sglang or vllm).",
        )
        parser.add_argument(
            "--fake-prefill",
            action="store_true",
            default=BenchArgs.fake_prefill,
            help="Enable fake prefill mode for decode-only benchmarking. "
            "Use with a decode server running --disaggregation-transfer-backend fake "
            "to benchmark pure decode performance without a real prefill node.",
        )
        parser.add_argument(
            "--server-args-for-metrics",
            type=str,
            nargs="*",
            default=None,
            help="Server launch arguments to record in metrics output (for tracking configurations).",
        )
        parser.add_argument(
            "--lora-name",
            type=str,
            nargs="*",
            default=BenchArgs.lora_name,
            help="Name(s) of pre-loaded LoRA adapter(s) to apply to the batch "
            "(sent as `lora_path` in the SGLang /generate payload). Requires "
            "the server to be launched with --enable-lora and --lora-paths "
            "<name>=<path> for every name listed here. Pass one name to apply "
            "a single adapter to every prompt, or multiple names to sample a "
            "per-prompt adapter per --lora-request-distribution.",
        )
        parser.add_argument(
            "--lora-request-distribution",
            type=str,
            default=BenchArgs.lora_request_distribution,
            choices=["uniform", "distinct", "skewed"],
            help="How to sample a LoRA adapter per prompt when more than one "
            "is listed in --lora-name. Mirrors serving.py. "
            "'uniform' picks uniformly at random, 'distinct' round-robins so "
            "consecutive prompts get different adapters, 'skewed' samples "
            "from a Zipf distribution over --lora-name (alpha controls the "
            "skew; see --lora-zipf-alpha).",
        )
        parser.add_argument(
            "--lora-zipf-alpha",
            type=float,
            default=BenchArgs.lora_zipf_alpha,
            help="Zipf exponent for 'skewed' LoRA sampling: the number of "
            "requests to adapter i is alpha times the number to adapter i+1. "
            "Must be > 1. Only used when --lora-request-distribution=skewed.",
        )
        parser.add_argument(
            "--enable-multi-batch",
            action="store_true",
            help=(
                "Allow --batch-size to exceed the server's "
                "effective_max_running_requests_per_dp * dp_size. The surplus "
                "requests are queued by the scheduler and promoted as slots "
                "free, so the batch is served as multiple sequential batches "
                "at the running-batch cap. Useful for stabilizing throughput "
                "measurements: driving more total prompts through a "
                "fixed running batch amortizes per-request prefill and "
                "first-step transients into steady-state decode. "
                "NOTE: only `overall_throughput` (= total_tokens / wall_time) "
                "is meaningful in this mode; input_throughput, "
                "output_throughput, last_ttft, and ITL assume one-shot "
                "batching and will be misleading."
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class BenchOneCaseResult(BaseModel):
    run_name: str
    batch_size: int
    input_len: int
    output_len: int
    latency: float
    input_throughput: float
    output_throughput: float
    overall_throughput: float
    last_ttft: float
    last_gen_throughput: float
    acc_length: float
    total_output_tokens: Optional[int] = None
    num_early_stopped: Optional[int] = None
    steady_state_output_throughput: Optional[float] = None
    steady_state_window_s: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    profile_link: Optional[str] = None

    def dump_to_jsonl(self, result_filename: str):
        with open(result_filename, "a") as fout:
            res = {
                "run_name": self.run_name,
                "batch_size": self.batch_size,
                "input_len": self.input_len,
                "output_len": self.output_len,
                "latency": round(self.latency, 4),
                "input_throughput": round(self.input_throughput, 2),
                "output_throughput": round(self.output_throughput, 2),
                "overall_throughput": round(self.overall_throughput, 2),
                "last_ttft": round(self.last_ttft, 4),
                "last_gen_throughput": round(self.last_gen_throughput, 2),
                "acc_length": round(self.acc_length, 2),
                "total_output_tokens": self.total_output_tokens,
                "num_early_stopped": self.num_early_stopped,
                "steady_state_output_throughput": (
                    round(self.steady_state_output_throughput, 2)
                    if self.steady_state_output_throughput is not None
                    else None
                ),
                "steady_state_window_s": (
                    round(self.steady_state_window_s, 4)
                    if self.steady_state_window_s is not None
                    else None
                ),
                "cache_hit_rate": (
                    round(self.cache_hit_rate, 4)
                    if self.cache_hit_rate is not None
                    else None
                ),
            }
            fout.write(json.dumps(res) + "\n")


def _warmup_cache(
    url: str,
    input_ids: List[List[int]],
    input_len: int,
    cache_hit_rate: float,
    dataset_name: str = "random",
    image_data: Optional[List] = None,
    backend: str = "sglang",
    model_name: Optional[str] = None,
):
    """Warm up the cache by sending prefix tokens to populate the radix/prefix cache.

    Args:
        url: Server URL
        input_ids: List of input token id lists
        input_len: Length of input tokens
        cache_hit_rate: Fraction of input tokens to cache (0.0-1.0)
        dataset_name: Name of the dataset (used to determine if image data should be included)
        image_data: Optional image data for VLM models
        backend: Backend server type ("sglang" or "vllm")
        model_name: Model name (required for vllm backend)
    """
    cached_token_len = int(input_len * cache_hit_rate)
    if cached_token_len <= 0:
        return

    print(
        f"Warming up cache with {cache_hit_rate * 100:.1f}% hit rate "
        f"({cached_token_len} tokens per request)"
    )
    # Create prefix input_ids for cache warming
    cache_warmup_input_ids = [ids[:cached_token_len] for ids in input_ids]

    if backend == "vllm":
        cache_warmup_payload = {
            "model": model_name,
            "prompt": cache_warmup_input_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        gen_url = url + "/v1/completions"
    else:
        cache_warmup_payload = {
            "input_ids": cache_warmup_input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,  # Minimal output, just to populate cache
                "ignore_eos": True,
            },
            "stream": False,
        }
        if dataset_name == "mmmu" and image_data is not None:
            # include image data in cache warmup
            cache_warmup_payload["image_data"] = image_data
        gen_url = url + "/generate"

    warmup_response = requests.post(
        gen_url,
        json=cache_warmup_payload,
        timeout=DEFAULT_TIMEOUT,
    )
    warmup_response.raise_for_status()
    print("Cache warmup completed")


def _flush_cache_with_retry(url: str, endpoint: str, max_retries: int = 3):
    """Post to a cache flush endpoint with retries on failure."""
    for attempt in range(max_retries):
        try:
            response = requests.post(url + endpoint, timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return
            if attempt >= max_retries - 1:
                response.raise_for_status()
        except requests.RequestException:
            if attempt >= max_retries - 1:
                raise
        time.sleep(2)


@lru_cache(maxsize=None)
def _load_hf_config(name_or_path: str):
    if not name_or_path:
        return None

    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    except Exception as e:
        print(
            f"Warning: could not load config for {name_or_path!r} ({e}); "
            "falling back to the HF chat template for --apply-chat-template."
        )
        return None


def _encode_fixed_prompt(
    tok_inner, prompt_text: str, apply_chat_template: bool
) -> List[int]:
    if not apply_chat_template:
        return tok_inner.encode(prompt_text)

    from sglang.srt.entrypoints.openai.chat_encoding import (
        encode_simple_chat,
        resolve_chat_encoding_spec,
    )

    hf_config = _load_hf_config(getattr(tok_inner, "name_or_path", "") or "")
    spec = (
        resolve_chat_encoding_spec(hf_config=hf_config, tokenizer=tok_inner)
        if hf_config is not None
        else None
    )
    return encode_simple_chat(
        tokenizer=tok_inner,
        spec=spec,
        messages=[{"role": "user", "content": prompt_text}],
    )


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    ignore_eos: bool,
    return_logprob: bool,
    stream_interval: int,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer: PreTrainedTokenizer | AutoProcessor,
    profile: bool = False,
    profile_activities: Tuple[str] = ("CPU", "GPU"),
    profile_start_step: Optional[int] = None,
    profile_steps: int = BenchArgs.profile_steps,
    profile_by_stage: bool = False,
    profile_prefix: Optional[str] = BenchArgs.profile_prefix,
    profile_output_dir: Optional[str] = BenchArgs.profile_output_dir,
    dataset_name: str = BenchArgs.dataset_name,
    dataset_path: str = BenchArgs.dataset_path,
    parallel_batch: bool = False,
    cache_hit_rate: float = BenchArgs.cache_hit_rate,
    backend: str = "sglang",
    model_name: Optional[str] = None,
    gsp_num_groups: int = BenchArgs.gsp_num_groups,
    gsp_system_prompt_len: int = BenchArgs.gsp_system_prompt_len,
    gsp_question_len: int = BenchArgs.gsp_question_len,
    gsp_output_len: int = BenchArgs.gsp_output_len,
    fake_prefill: bool = False,
    lora_name: Optional[List[str]] = None,
    lora_request_distribution: str = BenchArgs.lora_request_distribution,
    lora_zipf_alpha: float = BenchArgs.lora_zipf_alpha,
    fixed_prompt_file: str = "",
    apply_chat_template: bool = False,
):
    if backend == "vllm":
        # You need to have export VLLM_SERVER_DEV_MODE=1 in your environment to use this endpoint.
        _flush_cache_with_retry(url, "/reset_prefix_cache")
    else:
        _flush_cache_with_retry(url, "/flush_cache")

    if fixed_prompt_file:
        tok_inner = getattr(tokenizer, "tokenizer", tokenizer)
        with open(fixed_prompt_file) as f:
            prompt_ids = _encode_fixed_prompt(tok_inner, f.read(), apply_chat_template)
        input_ids = [list(prompt_ids) for _ in range(batch_size)]
        input_len = len(prompt_ids)
        image_data = None
    else:
        # Load input token ids via benchmark.datasets.get_dataset
        supported_datasets = (
            "random",
            "random-ids",
            "mmmu",
            "generated-shared-prefix",
        ) + REPLAY_TEXT_DATASETS
        if dataset_name not in supported_datasets:
            raise ValueError(
                f"Unsupported dataset for batch benchmark: {dataset_name}. "
                f"Supported: {supported_datasets}"
            )

        actual_gsp_groups = min(gsp_num_groups, batch_size)
        dataset_args = SimpleNamespace(
            dataset_name=dataset_name,
            num_prompts=batch_size,
            random_input_len=input_len,
            random_output_len=output_len,
            random_range_ratio=1.0,
            dataset_path=dataset_path,
            tokenize_prompt=dataset_name in ("random", "random-ids"),
            backend=backend,
            sharegpt_output_len=None,
            sharegpt_context_len=None,
            prompt_suffix="",
            apply_chat_template=apply_chat_template,
            seed=BenchArgs.seed,
            gsp_num_groups=actual_gsp_groups,
            gsp_prompts_per_group=(batch_size + actual_gsp_groups - 1)
            // actual_gsp_groups,
            gsp_system_prompt_len=gsp_system_prompt_len,
            gsp_question_len=gsp_question_len,
            gsp_output_len=gsp_output_len,
            # The generated-shared-prefix dataset's from_args requires these; the
            # batch-bench path only ever uses the uniform group distribution.
            gsp_group_distribution="uniform",
            gsp_zipf_alpha=None,
        )
        tok_inner = getattr(tokenizer, "tokenizer", tokenizer)
        dataset_model_id = model_name or getattr(tok_inner, "name_or_path", None)
        input_requests = get_dataset(dataset_args, tokenizer, model_id=dataset_model_id)

        if dataset_name == "generated-shared-prefix":
            input_requests = input_requests[:batch_size]
            input_ids = [tokenizer.encode(req.prompt) for req in input_requests]
            input_len = sum(len(ids) for ids in input_ids) // len(input_ids)
            output_len = gsp_output_len
            image_data = None
        elif dataset_name == "mmmu":
            input_ids = [tok_inner.encode(req.prompt) for req in input_requests]
            image_data = [req.image_data for req in input_requests]
        elif dataset_name in REPLAY_TEXT_DATASETS:
            if len(input_requests) < batch_size:
                raise ValueError(
                    f"{dataset_name} produced only {len(input_requests)} usable "
                    f"prompts for batch size {batch_size}; use a smaller batch "
                    f"size or a larger dataset."
                )
            input_ids = [tok_inner.encode(req.prompt) for req in input_requests]
            input_len = sum(len(ids) for ids in input_ids) // len(input_ids)
            image_data = None
        else:
            input_ids = [req.prompt for req in input_requests]
            image_data = None

    # Build payload based on backend
    if backend == "vllm":
        payload = {
            "model": model_name,
            "prompt": input_ids,
            "max_tokens": output_len,
            "temperature": temperature,
            "stream": True,
            "ignore_eos": ignore_eos,
        }
        if return_logprob:
            payload["logprobs"] = 1
        gen_url = url + "/v1/completions"
    else:
        # Load sampling parameters
        use_structured_outputs = False
        if use_structured_outputs:
            texts = []
            for _ in range(batch_size):
                texts.append(
                    "Human: What is the capital city of france? can you give as many trivial information as possible about that city? answer in json.\n"
                    * 50
                    + "Assistant:"
                )
            json_schema = "$$ANY$$"
        else:
            json_schema = None

        payload = {
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": output_len,
                "ignore_eos": ignore_eos,
                "json_schema": json_schema,
                "stream_interval": stream_interval,
            },
            "return_logprob": return_logprob,
            "stream": True,
            **({"parallel_batch": parallel_batch} if parallel_batch else {}),
        }
        payload["input_ids"] = input_ids
        if image_data is not None:
            payload["image_data"] = image_data
        if fake_prefill:
            payload["bootstrap_host"] = FAKE_BOOTSTRAP_HOST
            payload["bootstrap_room"] = 0
        if lora_name:
            # SGLang /generate accepts lora_path as either a string (applied
            # to every prompt) or a list matching the batch size (per-prompt
            # adapter). See io_struct.GenerateReqInput._normalize_lora_path.
            if len(lora_name) == 1:
                payload["lora_path"] = lora_name[0]
            elif lora_request_distribution == "uniform":
                payload["lora_path"] = [
                    random.choice(lora_name) for _ in range(batch_size)
                ]
            elif lora_request_distribution == "distinct":
                payload["lora_path"] = [
                    lora_name[i % len(lora_name)] for i in range(batch_size)
                ]
            elif lora_request_distribution == "skewed":
                weights = np.array([lora_zipf_alpha**-i for i in range(len(lora_name))])
                probs = weights / np.sum(weights)
                payload["lora_path"] = list(
                    np.random.choice(lora_name, size=batch_size, p=probs)
                )
            else:
                raise ValueError(
                    f"Unexpected lora_request_distribution: "
                    f"{lora_request_distribution!r}"
                )
        gen_url = url + "/generate"

    # Warm up cache if cache_hit_rate > 0.0
    if cache_hit_rate > 0.0:
        _warmup_cache(
            url=url,
            input_ids=input_ids,
            input_len=input_len,
            cache_hit_rate=cache_hit_rate,
            dataset_name=dataset_name,
            image_data=image_data,
            backend=backend,
            model_name=model_name,
        )

    # Turn on profiler
    profile_link = None
    if profile:
        profile_link: str = run_profile(
            url=url,
            num_steps=profile_steps,
            activities=profile_activities,
            output_dir=profile_output_dir,
            profile_by_stage=profile_by_stage,
            profile_prefix=profile_prefix,
            start_step=profile_start_step,
        )

    # Get metrics before the request (for cache hit rate calculation)
    metrics_before = get_cache_tokens_from_metrics(url)

    # Run the request
    recorder: Optional[BatchStreamRecorder] = None
    tic = time.perf_counter()
    with requests.post(
        gen_url,
        json=payload,
        stream=True,
        timeout=DEFAULT_TIMEOUT,
    ) as response:
        response.raise_for_status()

        # Get the TTFT of the last request in the batch
        last_ttft = 0.0
        if backend == "vllm":
            # Parse OpenAI-compatible streaming format from vLLM
            first_token_indices = set()
            for chunk in response.iter_lines(decode_unicode=False):
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    data_str = chunk[5:].strip()
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    if "error" in data:
                        raise RuntimeError(f"Request has failed. {data}.")
                    for choice in data.get("choices", []):
                        idx = choice["index"]
                        if idx not in first_token_indices:
                            first_token_indices.add(idx)
                            if len(first_token_indices) == batch_size:
                                last_ttft = time.perf_counter() - tic
        else:
            # mmmu can silently return fewer prompts than requested.
            recorder = BatchStreamRecorder(batch_size=len(input_ids))
            for chunk in response.iter_lines(decode_unicode=False):
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    if chunk == "data: [DONE]":
                        break
                    data = json.loads(chunk[5:].strip("\n"))
                    if "error" in data:
                        raise RuntimeError(f"Request has failed. {data}.")

                    finish_reason = data["meta_info"]["finish_reason"]
                    if finish_reason is not None:
                        validate_finish_reason(finish_reason, ignore_eos=ignore_eos)
                    recorder.record_chunk(
                        index=data["index"],
                        completion_tokens=data["meta_info"]["completion_tokens"],
                        finish_type=(
                            finish_reason["type"] if finish_reason is not None else None
                        ),
                        now=time.perf_counter(),
                    )
            missing = recorder.missing_indices()
            if missing:
                raise RuntimeError(
                    f"No stream chunk received for request indices {missing}."
                )
            last_ttft = recorder.all_started_time - tic

    # Compute metrics
    latency = time.perf_counter() - tic
    if recorder is None:
        # vllm chunks carry no token counts; assume the requested lengths.
        total_output_tokens = batch_size * output_len
        decode_window_tokens = float(total_output_tokens)
        num_early_stopped = None
        steady_state_window: Optional[SteadyStateWindow] = None
    else:
        total_output_tokens = recorder.total_output_tokens
        num_early_stopped = recorder.num_early_stopped
        steady_state_window = recorder.steady_state_window()
        if ignore_eos and dataset_name not in REPLAY_TEXT_DATASETS:
            # Historical numerator; CI throughput floors are calibrated to it.
            decode_window_tokens = float(total_output_tokens)
        else:
            # Count only tokens generated inside the denominator's window.
            decode_window_tokens = (
                total_output_tokens - recorder.tokens_before_all_started
            )
    steady_state_output_throughput = (
        steady_state_window.output_throughput
        if steady_state_window is not None
        else None
    )
    if (
        ignore_eos
        and recorder is not None
        and total_output_tokens != batch_size * output_len
    ):
        print(
            f"WARNING: generated {total_output_tokens} tokens but requested "
            f"{batch_size * output_len}; the server clamped max_new_tokens or "
            f"served fewer prompts, so throughput uses the actual count."
        )

    if dataset_name in REPLAY_TEXT_DATASETS:
        total_input_tokens = sum(len(ids) for ids in input_ids)
    else:
        total_input_tokens = batch_size * input_len
    input_throughput = total_input_tokens / last_ttft
    output_throughput = decode_window_tokens / (latency - last_ttft)
    overall_throughput = (total_input_tokens + total_output_tokens) / latency

    if backend == "vllm":
        # vLLM does not expose these metrics via API
        last_gen_throughput = -1
        acc_length = -1
    else:
        response = requests.get(url + "/server_info", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        server_info = response.json()
        internal_states = server_info.get("internal_states", [])
        acc_length = -1
        last_gen_throughput = -1
        for internal_state in internal_states:
            val_acc = internal_state.get("avg_spec_accept_length")
            if val_acc is not None:
                acc_length = val_acc
            val_thr = internal_state.get("last_gen_throughput")
            if val_thr is not None:
                last_gen_throughput = val_thr

    # Calculate cache hit rate from before/after metrics delta
    metrics_after = get_cache_tokens_from_metrics(url)
    metrics_cache_hit_rate = calculate_cache_hit_rate(metrics_before, metrics_after)

    # Print results
    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")
    print(f"last_ttft: {last_ttft:.2f} s")
    print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    if acc_length > 0:
        print(f"acc_length: {acc_length:.2f} ")
    if metrics_cache_hit_rate is not None:
        print(f"cache hit rate: {metrics_cache_hit_rate:.4f}")
    if not ignore_eos and recorder is not None:
        print(
            f"total output tokens: {total_output_tokens} "
            f"(requested {batch_size * output_len})"
        )
        print(f"early-stopped requests: {num_early_stopped}/{batch_size}")
        if steady_state_window is not None:
            print(
                f"steady-state output throughput: "
                f"{steady_state_output_throughput:.2f} tok/s "
                f"over a {steady_state_window.duration:.2f} s full-batch window"
            )
        else:
            print(
                "WARNING: no steady-state full-batch window (a request finished "
                "before the last one started decoding); steady-state metrics "
                "are n/a."
            )

    # Dump results
    result = BenchOneCaseResult(
        run_name=run_name,
        batch_size=batch_size,
        input_len=input_len,
        output_len=output_len,
        latency=latency,
        input_throughput=input_throughput,
        output_throughput=output_throughput,
        overall_throughput=overall_throughput,
        last_ttft=last_ttft,
        last_gen_throughput=last_gen_throughput,
        acc_length=acc_length,
        total_output_tokens=total_output_tokens,
        num_early_stopped=num_early_stopped,
        steady_state_output_throughput=steady_state_output_throughput,
        steady_state_window_s=(
            steady_state_window.duration if steady_state_window is not None else None
        ),
        cache_hit_rate=metrics_cache_hit_rate,
        profile_link=profile_link,
    )

    # Save and return the results
    if result_filename:
        result.dump_to_jsonl(result_filename)

    return result


def should_skip_due_to_token_capacity(
    batch_size, input_len, output_len, skip_token_capacity_threshold
):
    if batch_size * (input_len + output_len) > skip_token_capacity_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} * ({input_len=} + {output_len=}) = {batch_size * (input_len + output_len)} > {skip_token_capacity_threshold=} due to kv cache limit."
            + "=" * 8
        )
        return True
    return False


def should_skip_due_to_max_running_requests(
    batch_size, skip_max_running_requests_threshold
):
    if batch_size > skip_max_running_requests_threshold:
        print(
            "=" * 8
            + f"Skip benchmark {batch_size=} > {skip_max_running_requests_threshold=} due to max running requests limit."
            + "=" * 8
        )
        return True
    return False


def get_report_summary(
    results: List[BenchOneCaseResult], bench_args: BenchArgs, server_args: ServerArgs
):
    summary = (
        f"\nInput lens: {bench_args.input_len}. Output lens: {bench_args.output_len}."
    )
    if bench_args.cache_hit_rate > 0.0:
        summary += f" Cache hit rate: {bench_args.cache_hit_rate * 100:.1f}%."
    summary += "\n"

    if is_blackwell():
        hourly_cost_per_gpu = 4  # $4/hour for one B200
    else:
        hourly_cost_per_gpu = 2  # $2/hour for one H100
    input_util = 0.7

    # sort result by input_len
    results.sort(key=lambda x: x.input_len)
    rows = []
    headers = [
        "batch size",
        "input len",
        "latency (s)",
        "input throughput (tok/s)",
        "output throughput (tok/s)",
        "acc length",
        "ITL (ms)",
        "input cost ($/1M)",
        "output cost ($/1M)",
        "cache hit rate",
    ]
    if bench_args.disable_ignore_eos:
        headers += [
            "steady output throughput (tok/s)",
            "steady ITL (ms)",
            "early stops",
        ]
    if bench_args.profile:
        headers.append("profile")

    for res in results:
        hourly_cost = hourly_cost_per_gpu * server_args.tp_size
        accept_length = f"{res.acc_length:.2f}" if res.acc_length > 0 else "n/a"
        # 0 when every request's first chunk is also its finish chunk.
        if res.output_throughput > 0:
            itl_ms = f"{1000 * res.batch_size / res.output_throughput:.2f}"
            output_cost = f"{1e6 / res.output_throughput / 3600 * hourly_cost:.2f}"
        else:
            itl_ms = output_cost = "n/a"
        input_cost = 1e6 / (res.input_throughput * input_util) / 3600 * hourly_cost
        cache_hit_rate = (
            f"{res.cache_hit_rate:.4f}" if res.cache_hit_rate is not None else "n/a"
        )

        row = [
            res.batch_size,
            res.input_len,
            f"{res.latency:.2f}",
            f"{res.input_throughput:.2f}",
            f"{res.output_throughput:.2f}",
            accept_length,
            itl_ms,
            f"{input_cost:.2f}",
            output_cost,
            cache_hit_rate,
        ]
        if bench_args.disable_ignore_eos:
            if res.steady_state_output_throughput is not None:
                steady_tput = f"{res.steady_state_output_throughput:.2f}"
                steady_itl = (
                    f"{1000 * res.batch_size / res.steady_state_output_throughput:.2f}"
                )
            else:
                steady_tput = steady_itl = "n/a"
            row += [
                steady_tput,
                steady_itl,
                f"{res.num_early_stopped}/{res.batch_size}",
            ]
        if bench_args.profile:
            if res.profile_link:
                row.append(f"[Profile]({res.profile_link})")
            else:
                row.append("n/a")
        rows.append(row)

    summary += tabulate(rows, headers=headers, tablefmt="github")
    summary += "\n"

    return summary


def run_benchmark_internal(
    server_args: ServerArgs,
    bench_args: BenchArgs,
    launch_server_func: Callable = launch_server,
):
    # Validate flags that depend only on bench_args before launching a server.
    if bench_args.disable_ignore_eos:
        if bench_args.backend == "vllm":
            raise ValueError(
                "--disable-ignore-eos requires the sglang backend: vllm stream "
                "chunks carry no cumulative token counts, so early-stop-aware "
                "accounting is impossible."
            )
        if bench_args.fake_prefill:
            raise ValueError(
                "--disable-ignore-eos is incompatible with --fake-prefill: "
                "decode conditions on uninitialized KV, so EOS timing is "
                "meaningless."
            )
        if bench_args.client_stream_interval != 1:
            raise ValueError(
                "--disable-ignore-eos requires --client-stream-interval 1: the "
                "steady-state window needs per-token cumulative counts, and "
                "larger intervals quantize them by up to interval-1 tokens per "
                "request."
            )

    if bench_args.dataset_name in REPLAY_TEXT_DATASETS:
        if bench_args.backend == "vllm":
            raise ValueError(
                "Replay datasets require the sglang backend: vllm stream "
                "chunks carry no token counts, so the decode-window token "
                "accounting replay throughput relies on is impossible."
            )
        if bench_args.fixed_prompt_file:
            raise ValueError(
                "--fixed-prompt-file bypasses --dataset-name; drop one of them."
            )
        if bench_args.dataset_name == "custom" and not os.path.isfile(
            bench_args.dataset_path
        ):
            raise ValueError(
                "--dataset-name custom requires --dataset-path pointing at a "
                f"JSONL conversation file; got {bench_args.dataset_path!r}."
            )
        if len(bench_args.input_len) > 1:
            raise ValueError(
                "Replay datasets take prompt lengths from the recorded data; "
                "sweeping --input-len is meaningless. Pass at most one value "
                "(used only by the token-capacity skip guard)."
            )
        print(
            "NOTE: replay prompts have recorded lengths; the token-capacity "
            f"skip guard assumes --input-len ({bench_args.input_len[0]}) per "
            "prompt, so set it near the trace's mean length."
        )

    # set random seed
    random.seed(bench_args.seed)
    np.random.seed(bench_args.seed)

    # Resolve the benchmark target: launch a server, or connect to --base-url.
    endpoint = acquire_endpoint(server_args, bench_args.base_url, launch_server_func)
    base_url = endpoint.base_url

    # Get tokenizer and server info
    if bench_args.backend == "vllm":
        # For vLLM, get model name from /v1/models endpoint
        print(f"Connecting to vLLM server at {base_url}...")
        response = requests.get(base_url + "/v1/models", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        model_list = response.json().get("data", [])
        if not model_list:
            raise RuntimeError("No models found on vLLM server via /v1/models")
        model_name = model_list[0]["id"]
        print(f"Found model: {model_name}")
        print(f"Loading tokenizer for {model_name}...")
        if bench_args.dataset_name == "mmmu":
            tokenizer = get_processor(model_name)
        else:
            tokenizer = get_tokenizer(model_name)
        print("Tokenizer loaded.")

        server_info = {"model_name": model_name}
        # vLLM does not expose token capacity or max running requests via API
        skip_token_capacity_threshold = float("inf")
        skip_max_running_requests_threshold = float("inf")
    else:
        model_name = None
        response = requests.get(base_url + "/server_info", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        server_info = response.json()
        if bench_args.local_tokenizer_path:
            tokenizer_path = bench_args.local_tokenizer_path
        elif "tokenizer_path" in server_info:
            tokenizer_path = server_info["tokenizer_path"]
        elif "prefill" in server_info:
            tokenizer_path = server_info["prefill"][0]["tokenizer_path"]
        if bench_args.dataset_name == "mmmu":
            # mmmu implies this is a MLLM
            tokenizer = get_processor(tokenizer_path)
        else:
            tokenizer = get_tokenizer(tokenizer_path)

        internal_states = server_info.get("internal_states", [])
        internal_state = internal_states[0] if internal_states else {}
        dp_size = internal_state.get("dp_size", None) or 1

        # Get effective max running requests
        max_running_requests_per_dp = internal_state.get(
            "effective_max_running_requests_per_dp", -1
        )

        # Get token capacity
        skip_token_capacity_threshold = 0

        for state in internal_states:
            skip_token_capacity_threshold += state.get("memory_usage", {}).get(
                "token_capacity", 1000000000
            )

        # Router /get_server_info responses carry "router_manager"; worker
        # responses never do, so its presence confirms a router by design.
        if not internal_states and server_info.get("router_manager"):
            print(
                "WARNING: base_url points at a PD router; worker internal "
                "states are unavailable, so the max-running-requests and "
                "token-capacity skip guards are disabled."
            )
            skip_max_running_requests_threshold = float("inf")
            skip_token_capacity_threshold = float("inf")
        else:
            assert max_running_requests_per_dp > 0, (
                f"effective_max_running_requests_per_dp is not set, {max_running_requests_per_dp=}"
            )
            skip_max_running_requests_threshold = max_running_requests_per_dp * dp_size

        print(f"{max_running_requests_per_dp=}")
        print(f"{dp_size=}")
        print(f"{skip_max_running_requests_threshold=}")
        print(f"{skip_token_capacity_threshold=}")

    # Under --enable-multi-batch the client intentionally sends more prompts
    # than the server's running cap; surplus requests are queued (no KV
    # reservation) and promoted batch-by-batch. Peak live KV footprint is
    # bounded by the running cap, not by bs, so re-scope both guards:
    #   * max_running_requests: disabled (the whole point of the flag).
    #   * token_capacity: check against min(bs, running_cap) * (il + ol).
    effective_running_cap: Optional[int] = None
    if bench_args.enable_multi_batch:
        if skip_max_running_requests_threshold != float("inf"):
            effective_running_cap = skip_max_running_requests_threshold
        skip_max_running_requests_threshold = float("inf")

        # Multi-batch only kicks in when the client sends strictly more prompts
        # than the server's running cap; otherwise every prompt fits in a
        # single wave and the flag is a no-op for that case (but its metric
        # caveats — misleading input/output throughput and TTFT — still apply).
        # Warn loudly so the user can fix the batch-size sweep.
        if effective_running_cap is not None:
            noop_bs = sorted(
                {bs for bs in bench_args.batch_size if bs <= effective_running_cap}
            )
            if noop_bs:
                print(
                    f"WARNING: --enable-multi-batch is set but batch size(s) "
                    f"{noop_bs} are <= running cap ({effective_running_cap}); "
                    f"those cases will run as a single wave and the flag is a "
                    f"no-op for them. Use batch_size > {effective_running_cap} "
                    f"to actually exercise multi-batch."
                )

    # LoRA distribution args: mirror serving.py semantics so multi-LoRA
    # benchmarks behave consistently across harnesses.
    if bench_args.lora_request_distribution in ("distinct", "skewed"):
        assert bench_args.lora_name is not None and len(bench_args.lora_name) > 1, (
            "--lora-request-distribution=distinct/skewed requires more than "
            "one adapter via --lora-name."
        )
    assert bench_args.lora_zipf_alpha > 1, (
        f"--lora-zipf-alpha must be > 1, got {bench_args.lora_zipf_alpha}"
    )

    if bench_args.apply_chat_template and not (
        bench_args.fixed_prompt_file or bench_args.dataset_name in REPLAY_TEXT_DATASETS
    ):
        raise ValueError(
            "--apply-chat-template requires --fixed-prompt-file or a replay "
            "text dataset (sharegpt/custom): the other datasets generate "
            "token ids directly, so there is no prompt text to run through a "
            "chat template."
        )

    gsp_kwargs = dict(
        gsp_num_groups=bench_args.gsp_num_groups,
        gsp_system_prompt_len=bench_args.gsp_system_prompt_len,
        gsp_question_len=bench_args.gsp_question_len,
        gsp_output_len=bench_args.gsp_output_len,
    )

    # Warmup
    if not bench_args.skip_warmup:
        batch_size_unique = list(set(bench_args.batch_size))
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        print(f"Warmup with batch_size={batch_size_unique}")
        for bs in batch_size_unique:
            run_one_case(
                base_url,
                batch_size=bs,
                input_len=1024,
                output_len=16,
                temperature=bench_args.temperature,
                ignore_eos=True,
                return_logprob=bench_args.return_logprob,
                stream_interval=bench_args.client_stream_interval,
                input_len_step_percentage=bench_args.input_len_step_percentage,
                run_name="",
                result_filename="",
                tokenizer=tokenizer,
                dataset_name=bench_args.dataset_name,
                dataset_path=bench_args.dataset_path,
                parallel_batch=bench_args.parallel_batch,
                backend=bench_args.backend,
                model_name=model_name,
                fake_prefill=bench_args.fake_prefill,
                lora_name=bench_args.lora_name,
                lora_request_distribution=bench_args.lora_request_distribution,
                lora_zipf_alpha=bench_args.lora_zipf_alpha,
                fixed_prompt_file=bench_args.fixed_prompt_file,
                apply_chat_template=bench_args.apply_chat_template,
                **gsp_kwargs,
            )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    results = []
    profile_results = []
    try:
        # Benchmark all cases
        for bs, il, ol in itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        ):
            kv_footprint_bs = (
                bs if effective_running_cap is None else min(bs, effective_running_cap)
            )
            if should_skip_due_to_max_running_requests(
                bs, skip_max_running_requests_threshold
            ) or should_skip_due_to_token_capacity(
                kv_footprint_bs, il, ol, skip_token_capacity_threshold
            ):
                continue
            results.append(
                run_one_case(
                    base_url,
                    bs,
                    il,
                    ol,
                    temperature=bench_args.temperature,
                    ignore_eos=not bench_args.disable_ignore_eos,
                    return_logprob=bench_args.return_logprob,
                    stream_interval=bench_args.client_stream_interval,
                    input_len_step_percentage=bench_args.input_len_step_percentage,
                    run_name=bench_args.run_name,
                    result_filename=bench_args.result_filename,
                    tokenizer=tokenizer,
                    dataset_name=bench_args.dataset_name,
                    dataset_path=bench_args.dataset_path,
                    parallel_batch=bench_args.parallel_batch,
                    cache_hit_rate=bench_args.cache_hit_rate,
                    backend=bench_args.backend,
                    model_name=model_name,
                    fake_prefill=bench_args.fake_prefill,
                    lora_name=bench_args.lora_name,
                    lora_request_distribution=bench_args.lora_request_distribution,
                    lora_zipf_alpha=bench_args.lora_zipf_alpha,
                    fixed_prompt_file=bench_args.fixed_prompt_file,
                    apply_chat_template=bench_args.apply_chat_template,
                    **gsp_kwargs,
                )
            )

        # Profile all cases
        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    kv_footprint_bs = (
                        bs
                        if effective_running_cap is None
                        else min(bs, effective_running_cap)
                    )
                    if should_skip_due_to_max_running_requests(
                        bs, skip_max_running_requests_threshold
                    ) or should_skip_due_to_token_capacity(
                        kv_footprint_bs, il, ol, skip_token_capacity_threshold
                    ):
                        continue
                    profile_prefix = (
                        bench_args.profile_prefix or ""
                    ) + f"bs-{bs}-il-{il}"
                    profile_results.append(
                        run_one_case(
                            base_url,
                            bs,
                            il,
                            ol,
                            temperature=bench_args.temperature,
                            ignore_eos=not bench_args.disable_ignore_eos,
                            return_logprob=bench_args.return_logprob,
                            stream_interval=bench_args.client_stream_interval,
                            input_len_step_percentage=bench_args.input_len_step_percentage,
                            run_name=bench_args.run_name,
                            result_filename=bench_args.result_filename,
                            tokenizer=tokenizer,
                            dataset_name=bench_args.dataset_name,
                            dataset_path=bench_args.dataset_path,
                            parallel_batch=bench_args.parallel_batch,
                            cache_hit_rate=bench_args.cache_hit_rate,
                            profile=bench_args.profile,
                            profile_activities=bench_args.profile_activities,
                            profile_start_step=bench_args.profile_start_step,
                            profile_steps=bench_args.profile_steps,
                            profile_by_stage=bench_args.profile_by_stage,
                            profile_prefix=profile_prefix,
                            profile_output_dir=bench_args.profile_output_dir,
                            backend=bench_args.backend,
                            model_name=model_name,
                            fake_prefill=bench_args.fake_prefill,
                            lora_name=bench_args.lora_name,
                            lora_request_distribution=bench_args.lora_request_distribution,
                            lora_zipf_alpha=bench_args.lora_zipf_alpha,
                            fixed_prompt_file=bench_args.fixed_prompt_file,
                            apply_chat_template=bench_args.apply_chat_template,
                            **gsp_kwargs,
                        )
                    )
            except Exception as e:
                print(f"Error profiling, some profile traces may not be dumped: {e}")

            # Replace the profile link for any successful profile results
            for res, profile_res in zip(results, profile_results, strict=False):
                res.profile_link = profile_res.profile_link
    finally:
        endpoint.close()

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return results, server_info

    # Print summary
    summary = get_report_summary(results, bench_args, server_args)
    print(summary)

    if is_in_ci() and bench_args.append_to_github_summary:
        write_github_step_summary(summary)

    return results, server_info


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    cfg = resolving_view(server_args)
    results, server_info = run_benchmark_internal(server_args, bench_args)

    # Save results as pydantic models in the JSON format
    if bench_args.pydantic_result_filename:
        save_results_as_pydantic_models(
            results,
            pydantic_result_filename=bench_args.pydantic_result_filename,
            model_path=cfg.model_path,
            server_args=bench_args.server_args_for_metrics,
        )

    return results, server_info


def cli_main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    server_args.resolve_once()
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)


if __name__ == "__main__":
    cli_main()
