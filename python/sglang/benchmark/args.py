import json
from functools import reduce
from typing import Callable, List

import click

from sglang.benchmark.backends import BACKEND_MAPPING
from sglang.benchmark.datasets import DATASET_MAPPING


def composite_options(options: List[Callable]):
    def decorator(f):
        return reduce(lambda x, opt: opt(x), reversed(options), f)

    return decorator


def lora_path_callback(ctx, param, value):
    if not value:
        return None

    return list(value)


dataset_options = [
    click.option(
        "--dataset-name",
        type=click.Choice(DATASET_MAPPING.keys(), case_sensitive=False),
        default="sharegpt",
        show_default=True,
        help="Name of the dataset to benchmark on.",
    ),
    click.option(
        "--num-prompts",
        type=int,
        default=1000,
        show_default=True,
        help="Number of prompts to process.",
    ),
    click.option(
        "--dataset-path",
        type=str,
        default="",
        show_default=True,
        help="Path to the dataset.",
    ),
    click.option(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the ShareGPT dataset.",
    ),
    click.option(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the context length will be dropped.",
    ),
    click.option(
        "--random-input-len",
        type=int,
        default=1024,
        show_default=True,
        help="Number of input tokens per request, used only for random and image dataset.",
    ),
    click.option(
        "--random-output-len",
        type=int,
        default=1024,
        show_default=True,
        help="Number of output tokens per request, used only for random and image dataset.",
    ),
    click.option(
        "--random-range-ratio",
        type=float,
        default=0.0,
        show_default=True,
        help="Range of sampled ratio of input/output length, used only for random and image dataset.",
    ),
    click.option(
        "--image-count",
        type=int,
        default=1,
        show_default=True,
        help="Number of images per request (only available with the image dataset).",
    ),
    click.option(
        "--image-resolution",
        type=str,
        default="1080p",
        show_default=True,
        help="Resolution of images. Supports presets 4k/1080p/720p/360p or custom 'heightxwidth' (e.g., 1080x1920).",
    ),
    click.option(
        "--image-format",
        type=str,
        default="jpeg",
        show_default=True,
        help=("Format of images for image dataset. " "Supports jpeg and png."),
    ),
    click.option(
        "--image-content",
        type=str,
        default="random",
        show_default=True,
        help=("Content for images for image dataset. " "Supports random and blank."),
    ),
    click.option(
        "--use-trace-timestamps",
        is_flag=True,
        help="Use timestamps from the trace file for request scheduling ('mooncake' only).",
    ),
    click.option(
        "--mooncake-slowdown-factor",
        type=float,
        default=1.0,
        show_default=True,
        help="[Mooncake] Slowdown factor for replaying the trace (2.0 means twice as slow)."
        "NOTE: --request-rate is IGNORED in mooncake mode.",
    ),
    click.option(
        "--mooncake-num-rounds",
        type=int,
        default=1,
        show_default=True,
        help="[Mooncake] Number of conversation rounds per session (> 1 enables multi-turn).",
    ),
    click.option(
        "--mooncake-workload",
        type=click.Choice(
            ["mooncake", "conversation", "synthetic", "toolagent"], case_sensitive=False
        ),
        default="conversation",
        show_default=True,
        help="[Mooncake] Underlying workload for the dataset.",
    ),
]

serving_options = [
    click.option(
        "--backend",
        type=click.Choice(list(BACKEND_MAPPING.keys())),
        default="sglang",
        show_default=True,
        help="Must specify a backend, depending on the LLM Inference Engine.",
    ),
    click.option(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    ),
    click.option(
        "--host",
        type=str,
        default="0.0.0.0",
        show_default=True,
        help="Default host is 0.0.0.0.",
    ),
    click.option(
        "--port",
        type=int,
        default=None,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    ),
    click.option(
        "--model",
        type=str,
        default=None,
        help="Name or path of the model. If not set, the default model will request /v1/models for conf.",
    ),
    click.option(
        "--tokenizer",
        type=str,
        default=None,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    ),
    click.option(
        "--request-rate",
        type=float,
        default=float("inf"),
        show_default=True,
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    ),
    click.option(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    ),
]

common_benchmark_options = [
    click.option(
        "--output-file", type=str, default=None, help="Output JSONL file name."
    ),
    click.option(
        "--output-details", is_flag=True, help="Output details of benchmarking."
    ),
    click.option(
        "--seed", type=int, default=1, show_default=True, help="The random seed."
    ),
    click.option(
        "--extra-request-body",
        type=str,
        default=None,
        metavar="JSON_STRING",
        help="Append given JSON object (as a string) to the request payload. E.g., '{\"temperature\": 0.8}'.",
    ),
    click.option(
        "--lora-name",
        multiple=True,
        callback=lora_path_callback,
        help="The names of LoRA adapters. Can be specified multiple times.",
    ),
    click.option(
        "--prompt-suffix",
        type=str,
        default="",
        show_default=True,
        help="Suffix applied to the end of all user prompts.",
    ),
    click.option(
        "--warmup-requests",
        type=int,
        default=1,
        show_default=True,
        help="Number of warmup requests.",
    ),
    click.option("--disable-tqdm", is_flag=True, help="Disable tqdm progress bar."),
    click.option("--disable-stream", is_flag=True, help="Disable streaming mode."),
    click.option("--return-logprob", is_flag=True, help="Return logprob."),
    click.option("--disable-ignore-eos", is_flag=True, help="Disable ignoring EOS."),
    click.option("--apply-chat-template", is_flag=True, help="Apply chat template."),
    click.option(
        "--profile",
        is_flag=True,
        help="Use Torch Profiler (requires endpoint to be launched with profiler enabled).",
    ),
    click.option(
        "--pd-separated", is_flag=True, help="Benchmark PD disaggregation server."
    ),
    click.option(
        "--flush-cache",
        is_flag=True,
        help="Flush the cache before running the benchmark.",
    ),
    click.option(
        "--tokenize-prompt",
        is_flag=True,
        help="Use integer ids instead of string for inputs.",
    ),
    click.option(
        "--gsp-num-groups",
        type=int,
        default=64,
        show_default=True,
        help="[GSP] Number of system prompt groups.",
    ),
    click.option(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        show_default=True,
        help="[GSP] Number of prompts per system prompt group.",
    ),
    click.option(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        show_default=True,
        help="[GSP] Target length in tokens for system prompts.",
    ),
    click.option(
        "--gsp-question-len",
        type=int,
        default=128,
        show_default=True,
        help="[GSP] Target length in tokens for questions.",
    ),
    click.option(
        "--gsp-output-len",
        type=int,
        default=256,
        show_default=True,
        help="[GSP] Target length in tokens for outputs.",
    ),
]
