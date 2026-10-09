"""Run the benchmark through the Rust client (``sglang-bench``).

The Python client in :mod:`sglang.benchmark.serving` drives every request and
parses every response stream on one asyncio thread, so past a few thousand
concurrent streams it, not the server, caps the measured throughput. The Rust
client parses each stream on a Tokio worker thread instead. It reports the same
metrics, prints the same table, and appends the same result keys, so a run from
either side stays comparable.

Two ways in:

* ``python -m sglang.benchmark.rust_client --help`` forwards the command line
  to the Rust client, which owns the flag definitions. This path imports no
  model tooling, so it starts in well under a second.
* ``python -m sglang.benchmark.serving --rust-client ...`` routes an
  already-parsed argument namespace here, for callers on the existing entry
  point.

The Rust client covers fewer backends and datasets than the Python one; an
unsupported configuration raises here rather than quietly measuring something
else. See ``rust/sglang-bench/README.md``.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, Iterable, Optional

EXTENSION_MODULE = "sglang.srt.rust_extensions._bench"

#: Backends the Rust client implements. The Python client's remaining
#: backends (``trt``, ``gserver``, ``truss``, and the embedding ones) have no
#: Rust counterpart.
SUPPORTED_BACKENDS = frozenset(
    {
        "sglang",
        "sglang-native",
        "sglang-oai",
        "sglang-oai-chat",
        "vllm",
        "vllm-chat",
        "lmdeploy",
        "lmdeploy-chat",
    }
)

#: Datasets the Rust client implements.
SUPPORTED_DATASETS = frozenset({"random", "random-ids", "sharegpt"})

#: Namespace attributes that map onto a Rust flag of the same name. Anything
#: needing conversion is handled in :func:`build_config`.
_DIRECT_FIELDS = (
    "backend",
    "base_url",
    "host",
    "port",
    "ready_check_timeout_sec",
    "dataset_name",
    "dataset_path",
    "model",
    "served_model_name",
    "tokenizer",
    "num_prompts",
    "sharegpt_output_len",
    "sharegpt_context_len",
    "random_input_len",
    "random_output_len",
    "random_range_ratio",
    "max_concurrency",
    "warmup_requests",
    "output_file",
    "output_details",
    "disable_tqdm",
    "disable_stream",
    "disable_ignore_eos",
    "seed",
    "temperature",
    "top_p",
    "return_logprob",
    "top_logprobs_num",
    "logprob_start_len",
    "extra_request_body",
    "tokenize_prompt",
    "flush_cache",
    "flush_cache_timeout",
    "profile",
    "tag",
)

#: Options the Python client accepts that the Rust one does not implement.
#: Each maps to the value that means "not requested", so only a caller who
#: actually asked for one is turned away.
_UNSUPPORTED_OPTIONS = {
    "lora_name": None,
    "use_trace_timestamps": False,
    "apply_chat_template": False,
    "plot_throughput": False,
    "cache_report": False,
    "pd_separated": False,
    "return_routed_experts": False,
    "token_ids_logprob": None,
    "print_requests": False,
    "prompt_suffix": "",
    "fake_prefill": False,
    "profile_steps": None,
    "profile_num_steps": None,
    "profile_by_stage": False,
}


class UnsupportedByRustClient(ValueError):
    """The requested configuration has no Rust implementation."""


def check_supported(args: Any) -> None:
    """Raise unless the Rust client implements everything ``args`` asks for."""
    backend = getattr(args, "backend", None)
    if backend not in SUPPORTED_BACKENDS:
        raise UnsupportedByRustClient(
            f"--rust-client does not implement --backend {backend}; "
            f"supported: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )
    dataset = getattr(args, "dataset_name", None)
    if dataset not in SUPPORTED_DATASETS:
        raise UnsupportedByRustClient(
            f"--rust-client does not implement --dataset-name {dataset}; "
            f"supported: {', '.join(sorted(SUPPORTED_DATASETS))}"
        )
    requested = [
        name
        for name, unset in _UNSUPPORTED_OPTIONS.items()
        if getattr(args, name, unset) != unset
    ]
    if requested:
        raise UnsupportedByRustClient(
            "--rust-client does not implement: "
            f"{', '.join(sorted(requested))}. Drop --rust-client to run these "
            "on the Python client."
        )


def build_config(args: Any) -> Dict[str, Any]:
    """The Rust client's config object for an argument namespace.

    Only keys the caller actually set are sent; every omitted key takes the
    Rust flag's own default, which is the same value the Python flag has.
    """
    config: Dict[str, Any] = {}
    for field in _DIRECT_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            config[field] = value

    # The Rust flag takes a string so it can carry `inf`, which JSON cannot.
    rate = getattr(args, "request_rate", float("inf"))
    config["request_rate"] = "inf" if rate is None or math.isinf(rate) else str(rate)

    # The Python flag spells a header `Key=Value`; the Rust one `Key: Value`.
    config["headers"] = list(_as_headers(getattr(args, "header", None)))

    worker_threads = getattr(args, "rust_worker_threads", None)
    if worker_threads is not None:
        config["worker_threads"] = worker_threads
    return config


def _as_headers(header: Any) -> Iterable[str]:
    """`--header Key=Value ...` in the Rust client's `Key: Value` spelling."""
    if not header:
        return ()
    raw = (header,) if isinstance(header, str) else tuple(header)
    rewritten = []
    for entry in raw:
        name, separator, value = entry.partition("=")
        if not separator or not name or not value:
            raise UnsupportedByRustClient(
                f"--header must be `Key=Value`, got {entry!r}"
            )
        rewritten.append(f"{name}: {value}")
    return tuple(rewritten)


def _extension(extension: Any = None) -> Any:
    if extension is not None:
        return extension
    from sglang.srt.rust_extensions import load_rust_extension

    return load_rust_extension(EXTENSION_MODULE)


def run_rust_benchmark(args: Any, extension: Any = None) -> Dict[str, Any]:
    """Run the benchmark described by ``args`` and return its result document.

    The Rust client prints the report and appends the result line itself, so
    the returned mapping is for a caller that wants the numbers in process.
    """
    check_supported(args)
    config = build_config(args)
    result = _extension(extension).run_config(json.dumps(config))
    return json.loads(result)


def main(argv: Optional[list] = None, extension: Any = None) -> int:
    """Forward a command line to the Rust client, which owns the flags."""
    argv = list(sys.argv if argv is None else argv)
    if not argv:
        argv = ["sglang-bench"]
    # The Rust side expects argv[0] to be the program name, so clap's usage
    # and error messages name the command the user actually typed.
    argv[0] = "python -m sglang.benchmark.rust_client"
    try:
        _extension(extension).run_argv(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
