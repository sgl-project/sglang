#!/usr/bin/env python3
"""
Run decoupled speculative decoding for either an input prompt or a prompt dataset.

By default, this compares decoupled speculative decoding against normal decode.
Use `--skip-decode` to run decoupled speculation only and `--show-responses` to
print full response text. When `--output-dir` is set, JSON output records full
prompt and response text.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RAY_NAMESPACE = "dspec"
DEFAULT_PROMPT_COLUMN_CANDIDATES = [
    "prompt",
    "messages",
    "chat",
    "conversations",
    "text",
    "question",
    "instruction",
    "input",
    "query",
]
CODEFORCES_REQUIRED_COLUMNS = [
    "description",
    "input_format",
    "output_format",
    "time_limit",
    "memory_limit",
]
CODEFORCES_OPTIONAL_COLUMNS = [
    "title",
    "note",
    "examples",
    "input_mode",
    "interaction_format",
]
CODEFORCES_LANGUAGE_ALIASES = {
    "python": "Python 3",
    "py": "Python 3",
    "cpp": "C++17",
    "c++": "C++17",
}
_RUNTIME_IMPORTS_READY = False


def _ensure_runtime_imports() -> None:
    """Import Ray and decoupled-spec helpers after argparse handles --help."""
    global _RUNTIME_IMPORTS_READY
    global create_remote_decoupled_spec_topology
    global PlacementGroupSchedulingStrategy
    global PortActor
    global TargetActor
    global _build_chat_template_renderer
    global _build_dapo_math_17k_prompt
    global _get_real_verify_acceptance_stats
    global _messages_to_fallback_text
    global _normalize_prompt
    global get_decoupled_spec_actor_env_vars
    global infer_prompt_column
    global placement_group
    global ray
    global remove_placement_group
    global resolve_dapo_math_17k_prompt_column

    if _RUNTIME_IMPORTS_READY:
        return

    import ray as ray_module
    from ray.util.placement_group import placement_group as ray_placement_group
    from ray.util.placement_group import (
        remove_placement_group as ray_remove_placement_group,
    )
    from ray.util.scheduling_strategies import (
        PlacementGroupSchedulingStrategy as RayPlacementGroupSchedulingStrategy,
    )

    try:
        from . import decoupled_spec_common as common
    except ImportError:
        import decoupled_spec_common as common

    ray = ray_module
    placement_group = ray_placement_group
    remove_placement_group = ray_remove_placement_group
    PlacementGroupSchedulingStrategy = RayPlacementGroupSchedulingStrategy
    create_remote_decoupled_spec_topology = common.create_remote_decoupled_spec_topology
    PortActor = common.PortActor
    TargetActor = common.TargetActor
    _build_chat_template_renderer = common._build_chat_template_renderer
    _build_dapo_math_17k_prompt = common._build_dapo_math_17k_prompt
    _get_real_verify_acceptance_stats = common._get_real_verify_acceptance_stats
    _messages_to_fallback_text = common._messages_to_fallback_text
    _normalize_prompt = common._normalize_prompt
    get_decoupled_spec_actor_env_vars = common.get_decoupled_spec_actor_env_vars
    infer_prompt_column = common.infer_prompt_column
    resolve_dapo_math_17k_prompt_column = common.resolve_dapo_math_17k_prompt_column
    _RUNTIME_IMPORTS_READY = True


@dataclass
class PromptSample:
    row_index: int
    prompt: str
    prompt_input_ids: list[int]
    prompt_tokens: int


@dataclass
class ModeMetrics:
    mode: str
    generation_time_s: float
    total_generated_tokens: int
    output_throughput_tok_per_s: float
    per_request: list[dict[str, Any]]
    avg_spec_accept_length: float | None = None
    avg_spec_accept_rate: float | None = None
    avg_spec_valid_accept_rate: float | None = None
    total_spec_valid_draft_token_num: int = 0
    total_spec_valid_accept_token_num: int = 0


def parse_args() -> argparse.Namespace:
    def str_to_bool(value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        normalized = value.lower()
        if normalized in ("1", "true", "t", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "f", "no", "n", "off"):
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled speculation on one prompt or a parquet prompt batch, "
            "optionally comparing against normal decode."
        )
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Single prompt to generate from. When --batch-size is greater than 1, "
            "the prompt is repeated to fill the batch. Mutually exclusive with "
            "--dataset-path."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        "--parquet-path",
        dest="dataset_path",
        default=None,
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help=(
            "Prompt column in the parquet file. If omitted, common names are "
            f"searched in order: {DEFAULT_PROMPT_COLUMN_CANDIDATES}."
        ),
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "codeforces_raw", "dapo_math_17k"],
        default="auto",
        help=(
            "How to interpret the parquet rows. "
            "'auto' reads one prompt-like column. "
            "'codeforces_raw' builds a prompt from Codeforces problem fields and, "
            "when enabled, renders it through the model tokenizer's chat template. "
            "'dapo_math_17k' reads the DAPO-Math-17k structured prompt messages "
            "and renders them through the target model chat template."
        ),
    )
    parser.add_argument(
        "--code-language",
        choices=["python", "py", "cpp", "c++"],
        default="python",
        help=(
            "Target language used when --dataset-format=codeforces_raw. "
            "Ignored for normal prompt-column datasets."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        "--bs",
        dest="batch_size",
        type=int,
        default=1,
        help=(
            "Number of valid prompts to run in one generate call. When using "
            "multiple verifier replicas, this must be divisible by the number "
            "of verifier replicas."
        ),
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Disable tokenizer.apply_chat_template for chat-style prompt objects.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking-style generation when building chat prompts for "
            "models such as Qwen3/Qwen3.5. Disabled by default."
        ),
    )
    parser.add_argument(
        "--context-length",
        "--max-new-tokens",
        dest="context_length",
        type=int,
        required=True,
        help="Generation length. This is passed as max_new_tokens.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Optional prompt token upper bound. Prompts over this limit are skipped.",
    )
    parser.add_argument(
        "--target-model-path",
        required=True,
        help="Target/verifier model path.",
    )
    parser.add_argument(
        "--draft-model-path",
        required=True,
        help="Draft model path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path used for prompt length filtering. Defaults to target model.",
    )
    parser.add_argument("--target-tp-size", type=int, required=True)
    parser.add_argument(
        "--target-ep-size",
        type=int,
        default=None,
        help="Expert parallel size for the target/verifier engine.",
    )
    parser.add_argument(
        "--target-moe-a2a-backend",
        default=None,
        help="MoE A2A backend for the target/verifier engine, e.g. deepep.",
    )
    parser.add_argument(
        "--target-mamba-scheduler-strategy",
        "--mamba-scheduler-strategy",
        dest="target_mamba_scheduler_strategy",
        choices=["auto", "no_buffer", "extra_buffer"],
        default=None,
        help=(
            "Mamba scheduler strategy for the target/verifier engine. "
            "Decoupled verifier and drafter engines disable radix cache, so "
            "the default no_buffer strategy is normally sufficient."
        ),
    )
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--num-speculative-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable deterministic inference for both decoupled drafter and "
            "verifier engines."
        ),
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Set sampling_params.ignore_eos=True for both decoupled speculative "
            "decoding and normal decoding. Disabled by default."
        ),
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address. Use 'auto' for an existing cluster or local fallback on nnodes=1.",
    )
    parser.add_argument("--ray-namespace", default=DEFAULT_RAY_NAMESPACE)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument(
        "--n-gpu-per-node",
        type=int,
        default=None,
        help=(
            "GPU count available on each Ray node. Used with --nnodes to bound "
            "the total verifier and drafter GPU budgets."
        ),
    )
    parser.add_argument(
        "--verify-ngpus",
        dest="verify_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for verifier replicas. If omitted, all GPUs "
            "not reserved by --draft-ngpus are used for verifier replicas."
        ),
    )
    parser.add_argument(
        "--draft-ngpus",
        dest="draft_ngpus",
        type=int,
        default=None,
        help=(
            "Total GPUs reserved for all drafter replicas. The number of "
            "drafters is derived as draft_ngpus / draft_tp_size."
        ),
    )
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help=(
            "Optional SGLang distributed init address override. If omitted for "
            "multi-node runs, the script uses each verifier placement group's "
            "rank-0 host."
        ),
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=None,
        help=(
            "Base port for this run. With V verifier replicas, spec dist-init "
            "uses base..base+V-1, decode uses base+V..base+2V-1, verifier "
            "result endpoints use base+2V..base+3V-1, and drafter control "
            "endpoints start at base+3V."
        ),
    )
    parser.add_argument(
        "--num-draft-replicas",
        dest="num_draft_replicas",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional directory to write per-mode CSV/JSON outputs. "
            "A normal comparison run writes decoupled-spec.csv, "
            "decoupled-spec.json, decode.csv, and decode.json."
        ),
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Only run decoupled speculation and skip the normal decode baseline.",
    )
    parser.add_argument(
        "--show-responses",
        action="store_true",
        help=(
            "Print full response text in the terminal. When --output-dir is set, "
            "full prompt and response text is always included in per-mode JSON."
        ),
    )
    parser.add_argument(
        "--decoupled-spec-trace-dir",
        default=None,
        help="Directory for decoupled speculative decoding CSV trace files.",
    )
    parser.add_argument(
        "--decoupled-spec-allow-partial",
        type=str_to_bool,
        default=True,
        help=(
            "Whether the verifier may snapshot currently available partial draft "
            "tails. Set to false to block until every request in the verifier "
            "batch has enough draft tokens."
        ),
    )
    return parser.parse_args()


def _build_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to count prompt tokens for this benchmark."
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_path)


def _encode_prompt_tokens(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.encode(prompt))


def _stringify_optional_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _format_codeforces_examples(examples: Any) -> str:
    if not isinstance(examples, list):
        return ""

    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        if not isinstance(example, dict):
            continue
        input_text = _stringify_optional_text(example.get("input"))
        output_text = _stringify_optional_text(example.get("output"))
        if not input_text and not output_text:
            continue

        parts = [f"Example {index}"]
        if input_text:
            parts.append("Input:")
            parts.append(input_text)
        if output_text:
            parts.append("Output:")
            parts.append(output_text)
        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


def _build_codeforces_raw_prompt(
    row: dict[str, Any],
    *,
    row_index: int,
    chat_template_renderer,
    code_language: str,
) -> str:
    missing_values = [
        column
        for column in CODEFORCES_REQUIRED_COLUMNS
        if not _stringify_optional_text(row.get(column))
    ]
    if missing_values:
        raise ValueError(
            f"Row {row_index} is missing required Codeforces fields: {missing_values}"
        )

    normalized_language = CODEFORCES_LANGUAGE_ALIASES.get(
        code_language.lower(), code_language
    )
    title = _stringify_optional_text(row.get("title"))
    description = _stringify_optional_text(row.get("description"))
    input_format = _stringify_optional_text(row.get("input_format"))
    output_format = _stringify_optional_text(row.get("output_format"))
    note = _stringify_optional_text(row.get("note"))
    input_mode = _stringify_optional_text(row.get("input_mode"))
    interaction_format = _stringify_optional_text(row.get("interaction_format"))
    examples_text = _format_codeforces_examples(row.get("examples"))

    limit_parts = [
        f"Time limit: {_stringify_optional_text(row.get('time_limit'))} seconds",
        f"Memory limit: {_stringify_optional_text(row.get('memory_limit'))} MB",
    ]
    if input_mode:
        if input_mode == "stdio":
            limit_parts.append("I/O mode: standard input and standard output")
        else:
            limit_parts.append(f"I/O mode: {input_mode}")

    user_sections = [
        (
            f"Write a correct and efficient {normalized_language} solution for the "
            "following competitive programming problem."
        ),
        "\n".join(limit_parts),
    ]
    if title:
        user_sections.append(f"Title: {title}")
    user_sections.extend(
        [
            "Problem description:",
            description,
            "Input format:",
            input_format,
            "Output format:",
            output_format,
        ]
    )
    if interaction_format:
        user_sections.extend(["Interaction format:", interaction_format])
    if note:
        user_sections.extend(["Notes:", note])
    if examples_text:
        user_sections.extend(["Examples:", examples_text])
    user_sections.append(
        (
            f"Return only the final {normalized_language} source code. "
            "Do not include explanations or Markdown fences."
        )
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert competitive programmer. Produce a correct, "
                "efficient solution that respects the stated constraints."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(section for section in user_sections if section),
        },
    ]

    if chat_template_renderer is not None:
        return chat_template_renderer(messages)
    return _messages_to_fallback_text(messages)


def load_prompt_samples(
    args: argparse.Namespace,
) -> tuple[str, list[PromptSample], int]:
    if args.prompt is not None and args.dataset_path is not None:
        raise ValueError("--prompt and --dataset-path are mutually exclusive")
    if args.prompt is None and args.dataset_path is None:
        raise ValueError("Either --prompt or --dataset-path must be provided")
    if args.context_length <= 0:
        raise ValueError("context-length must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_prompt_length is not None and args.max_prompt_length <= 0:
        raise ValueError("max-prompt-length must be positive when set")

    tokenizer_path = args.tokenizer_path or args.target_model_path
    tokenizer = _build_tokenizer(tokenizer_path)

    if args.prompt is not None:
        prompt = args.prompt
        if not args.disable_chat_template:
            chat_template_renderer = _build_chat_template_renderer(
                tokenizer_path, enable_thinking=args.enable_thinking
            )
            if chat_template_renderer is not None:
                prompt = chat_template_renderer(
                    [{"role": "user", "content": args.prompt}]
                )
        prompt_input_ids = _encode_prompt_tokens(tokenizer, prompt)
        prompt_tokens = len(prompt_input_ids)
        if prompt_tokens == 0:
            raise ValueError("--prompt produced zero prompt tokens")
        if (
            args.max_prompt_length is not None
            and prompt_tokens > args.max_prompt_length
        ):
            raise ValueError(
                f"--prompt has {prompt_tokens} tokens, exceeding --max-prompt-length "
                f"{args.max_prompt_length}"
            )
        return (
            "prompt",
            [
                PromptSample(
                    row_index=index,
                    prompt=prompt,
                    prompt_input_ids=list(prompt_input_ids),
                    prompt_tokens=prompt_tokens,
                )
                for index in range(args.batch_size)
            ],
            args.batch_size,
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read parquet datasets.") from exc

    if args.offset < 0:
        raise ValueError("offset must be non-negative")

    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)
    total_rows = parquet_file.metadata.num_rows
    if args.offset >= total_rows:
        raise ValueError(
            f"offset {args.offset} is out of range for {dataset_path}; "
            f"total rows: {total_rows}"
        )

    column_names = parquet_file.schema_arrow.names
    if args.dataset_format == "codeforces_raw":
        missing_columns = [
            column
            for column in CODEFORCES_REQUIRED_COLUMNS
            if column not in column_names
        ]
        if missing_columns:
            raise ValueError(
                "dataset-format=codeforces_raw requires Codeforces problem fields. "
                f"Missing columns: {missing_columns}. Available columns: {column_names}"
            )
        prompt_column = f"codeforces_raw[{CODEFORCES_LANGUAGE_ALIASES.get(args.code_language.lower(), args.code_language)}]"
        read_columns = [
            column
            for column in [*CODEFORCES_REQUIRED_COLUMNS, *CODEFORCES_OPTIONAL_COLUMNS]
            if column in column_names
        ]
        selected_prompt_column = None
    elif args.dataset_format == "dapo_math_17k":
        selected_prompt_column = resolve_dapo_math_17k_prompt_column(
            column_names,
            prompt_column=args.prompt_column,
        )
        prompt_column = f"dapo_math_17k[{selected_prompt_column}]"
        read_columns = [selected_prompt_column]
    else:
        selected_prompt_column = args.prompt_column or infer_prompt_column(column_names)
        if selected_prompt_column not in column_names:
            raise ValueError(
                f"prompt column {selected_prompt_column!r} not found. Available columns: {column_names}"
            )
        prompt_column = selected_prompt_column
        read_columns = [selected_prompt_column]

    chat_template_renderer = None
    if not args.disable_chat_template:
        chat_template_renderer = _build_chat_template_renderer(
            tokenizer_path, enable_thinking=args.enable_thinking
        )

    samples: list[PromptSample] = []
    skipped_empty = 0
    skipped_too_long = 0
    skipped_invalid = 0
    current_row = 0
    remaining_skip = args.offset
    reader_batch_size = max(args.batch_size, 1024)

    for record_batch in parquet_file.iter_batches(
        batch_size=reader_batch_size,
        columns=read_columns,
    ):
        if args.dataset_format in {"codeforces_raw", "dapo_math_17k"}:
            batch_rows = record_batch.to_pylist()
        else:
            batch_rows = record_batch.column(0).to_pylist()

        if remaining_skip >= len(batch_rows):
            remaining_skip -= len(batch_rows)
            current_row += len(batch_rows)
            continue

        start_index = remaining_skip
        for local_index in range(start_index, len(batch_rows)):
            row_index = current_row + local_index
            try:
                if args.dataset_format == "codeforces_raw":
                    prompt = _build_codeforces_raw_prompt(
                        batch_rows[local_index],
                        row_index=row_index,
                        chat_template_renderer=chat_template_renderer,
                        code_language=args.code_language,
                    )
                elif args.dataset_format == "dapo_math_17k":
                    prompt = _build_dapo_math_17k_prompt(
                        batch_rows[local_index],
                        row_index=row_index,
                        prompt_column=selected_prompt_column or "prompt",
                        chat_template_renderer=chat_template_renderer,
                        enable_thinking=args.enable_thinking,
                    )
                else:
                    prompt = _normalize_prompt(
                        batch_rows[local_index],
                        row_index,
                        prompt_column,
                        chat_template_renderer,
                        enable_thinking=args.enable_thinking,
                    )
            except Exception:
                skipped_invalid += 1
                continue

            if not prompt:
                skipped_empty += 1
                continue

            prompt_input_ids = _encode_prompt_tokens(tokenizer, prompt)
            prompt_tokens = len(prompt_input_ids)
            if prompt_tokens == 0:
                skipped_empty += 1
                continue

            if (
                args.max_prompt_length is not None
                and prompt_tokens > args.max_prompt_length
            ):
                skipped_too_long += 1
                continue

            samples.append(
                PromptSample(
                    row_index=row_index,
                    prompt=prompt,
                    prompt_input_ids=prompt_input_ids,
                    prompt_tokens=prompt_tokens,
                )
            )
            if len(samples) >= args.batch_size:
                return prompt_column, samples, total_rows

        current_row += len(batch_rows)
        remaining_skip = 0

    raise ValueError(
        "Not enough valid prompts were found. "
        f"requested={args.batch_size}, collected={len(samples)}, "
        f"skipped_empty={skipped_empty}, skipped_too_long={skipped_too_long}, "
        f"skipped_invalid={skipped_invalid}, offset={args.offset}, "
        f"total_rows={total_rows}, prompt_column={prompt_column!r}"
    )


def _parse_host_port(addr: str) -> tuple[str, int | None]:
    if addr.count(":") == 1:
        host, raw_port = addr.rsplit(":", 1)
        if raw_port:
            return host, int(raw_port)
    return addr, None


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def derive_dist_init_addr(
    args: argparse.Namespace,
    *,
    port_offset: int = 0,
) -> str | None:
    if args.nnodes == 1 and args.dist_init_addr is None:
        if args.dist_init_port is not None:
            return f"127.0.0.1:{args.dist_init_port + port_offset}"
        return f"127.0.0.1:{_pick_free_local_port()}"

    if args.dist_init_addr is None:
        raise ValueError("dist-init-addr is required when nnodes > 1")

    host, parsed_port = _parse_host_port(args.dist_init_addr)
    base_port = args.dist_init_port if args.dist_init_port is not None else parsed_port
    if base_port is None:
        raise ValueError(
            "dist-init-addr must include a port or dist-init-port must be set"
        )

    return f"{host}:{base_port + port_offset}"


def derive_dist_init_addr_from_pg(
    args: argparse.Namespace,
    pg,
    *,
    port_offset: int = 0,
) -> str | None:
    if args.dist_init_addr is not None or args.nnodes == 1:
        return derive_dist_init_addr(args, port_offset=port_offset)

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    actor = PortActor.options(
        num_cpus=0,
        scheduling_strategy=scheduling_strategy,
    ).remote()
    try:
        preferred_port = (
            args.dist_init_port + port_offset
            if args.dist_init_port is not None
            else None
        )
        reservation = ray.get(actor.reserve_port.remote(preferred_port))
        host = reservation["host"]
        port = int(reservation["port"])
        ray.get(actor.release_port.remote())
    finally:
        ray.kill(actor, no_restart=True)

    return f"{host}:{port}"


def init_ray(address: str, namespace: str, nnodes: int) -> None:
    init_kwargs = dict(
        address=address,
        namespace=namespace,
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_level=logging.ERROR,
    )
    try:
        ray.init(**init_kwargs)
    except Exception:
        if address != "auto" or nnodes != 1:
            raise
        ray.init(
            namespace=namespace,
            ignore_reinit_error=True,
            log_to_driver=True,
            logging_level=logging.ERROR,
        )


def derive_target_layout(args: argparse.Namespace) -> tuple[int, int]:
    for candidate_nnodes in range(1, args.nnodes + 1):
        if args.target_tp_size % candidate_nnodes != 0:
            continue
        target_gpus_per_node = args.target_tp_size // candidate_nnodes
        if target_gpus_per_node <= args.n_gpu_per_node:
            return candidate_nnodes, target_gpus_per_node

    raise ValueError(
        f"target-tp-size ({args.target_tp_size}) cannot be packed evenly across up to "
        f"{args.nnodes} nodes with {args.n_gpu_per_node} GPUs per node"
    )


def validate_resources(args: argparse.Namespace) -> tuple[int, int]:
    if args.nnodes <= 0:
        raise ValueError("nnodes must be positive")
    if args.target_tp_size <= 0:
        raise ValueError("target-tp-size must be positive")
    if args.target_ep_size is not None and args.target_ep_size <= 0:
        raise ValueError("target-ep-size must be positive when set")
    if args.draft_tp_size <= 0:
        raise ValueError("draft-tp-size must be positive")
    if args.verify_ngpus is not None and args.verify_ngpus <= 0:
        raise ValueError("verify-ngpus must be positive when set")
    if args.draft_ngpus is not None and args.draft_ngpus <= 0:
        raise ValueError("draft-ngpus must be positive when set")
    if args.num_draft_replicas is not None and args.num_draft_replicas <= 0:
        raise ValueError("num-draft-replicas must be positive when set")
    if args.draft_ngpus is None:
        args.draft_ngpus = args.draft_tp_size * (args.num_draft_replicas or 1)
    if args.draft_ngpus % args.draft_tp_size != 0:
        raise ValueError(
            f"draft-ngpus ({args.draft_ngpus}) must be divisible by "
            f"draft-tp-size ({args.draft_tp_size})"
        )
    derived_num_draft_replicas = args.draft_ngpus // args.draft_tp_size
    if (
        args.num_draft_replicas is not None
        and args.num_draft_replicas != derived_num_draft_replicas
    ):
        raise ValueError(
            "num-draft-replicas must match draft-ngpus / draft-tp-size "
            f"({derived_num_draft_replicas}) when --draft-ngpus is set"
        )
    args.num_draft_replicas = derived_num_draft_replicas
    if args.verify_ngpus is not None and args.verify_ngpus % args.target_tp_size != 0:
        raise ValueError(
            f"verify-ngpus ({args.verify_ngpus}) must be divisible by "
            f"target-tp-size ({args.target_tp_size})"
        )

    if args.n_gpu_per_node is None:
        if args.nnodes != 1:
            raise ValueError("n-gpu-per-node is required when nnodes > 1")
        args.n_gpu_per_node = (
            args.verify_ngpus or args.target_tp_size
        ) + args.draft_ngpus
    if args.n_gpu_per_node <= 0:
        raise ValueError("n-gpu-per-node must be positive")

    total_cluster_gpus = args.n_gpu_per_node * args.nnodes
    if args.verify_ngpus is None:
        args.verify_ngpus = total_cluster_gpus - args.draft_ngpus
    if args.draft_ngpus + args.verify_ngpus > total_cluster_gpus:
        raise ValueError(
            f"verify-ngpus + draft-ngpus ({args.verify_ngpus} + "
            f"{args.draft_ngpus}) exceeds nnodes*n-gpu-per-node "
            f"({total_cluster_gpus})"
        )
    if args.verify_ngpus <= 0:
        raise ValueError(
            f"draft-ngpus ({args.draft_ngpus}) must leave GPUs for at least "
            "one verifier replica"
        )
    if args.verify_ngpus % args.target_tp_size != 0:
        raise ValueError(
            f"verify-ngpus ({args.verify_ngpus}) must be divisible by "
            f"target-tp-size ({args.target_tp_size})"
        )
    args.num_verifier_replicas = args.verify_ngpus // args.target_tp_size

    target_nnodes, target_gpus_per_node = derive_target_layout(args)

    if args.draft_tp_size > args.n_gpu_per_node:
        raise ValueError(
            f"each draft actor needs {args.draft_tp_size} GPUs on one node, "
            f"but n-gpu-per-node is only {args.n_gpu_per_node}"
        )

    ray_gpus = int(ray.cluster_resources().get("GPU", 0))
    if ray_gpus and total_cluster_gpus > ray_gpus:
        raise ValueError(
            f"Ray cluster reports {ray_gpus} GPUs, but this run requires "
            f"{total_cluster_gpus}"
        )

    alive_target_nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and float(node.get("Resources", {}).get("GPU", 0)) >= target_gpus_per_node
    ]
    if len(alive_target_nodes) < target_nnodes:
        raise ValueError(
            f"Ray cluster has {len(alive_target_nodes)} alive GPU nodes with at "
            f"least {target_gpus_per_node} GPUs, but target needs {target_nnodes} nodes"
        )

    return target_nnodes, target_gpus_per_node


def create_target_placement_group(target_nnodes: int, target_gpus_per_node: int):
    bundles = [{"CPU": 1, "GPU": target_gpus_per_node} for _ in range(target_nnodes)]
    strategy = "PACK" if target_nnodes == 1 else "STRICT_SPREAD"
    pg = placement_group(bundles, strategy=strategy)
    ray.get(pg.ready())
    return pg


def create_target_placement_groups(
    num_replicas: int,
    target_nnodes: int,
    target_gpus_per_node: int,
):
    return [
        create_target_placement_group(target_nnodes, target_gpus_per_node)
        for _ in range(num_replicas)
    ]


def launch_target_actors(
    *,
    args: argparse.Namespace,
    mode: str,
    dist_init_addr: str | None,
    target_nnodes: int,
    target_gpus_per_node: int,
    pg,
    bind_endpoint: str | None = None,
    connect_endpoints: list[str] | None = None,
    rank: int | None = None,
) -> list[Any]:
    actor_env_vars = get_decoupled_spec_actor_env_vars(args)
    actors = []
    for node_rank in range(target_nnodes):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=node_rank,
        )
        actor_options: dict[str, Any] = dict(
            num_gpus=target_gpus_per_node,
            num_cpus=1,
            scheduling_strategy=scheduling_strategy,
        )
        if actor_env_vars:
            actor_options["runtime_env"] = {"env_vars": actor_env_vars}
        actor = TargetActor.options(**actor_options).remote(
            mode=mode,
            model_path=args.target_model_path,
            tp_size=args.target_tp_size,
            ep_size=args.target_ep_size,
            moe_a2a_backend=args.target_moe_a2a_backend,
            mamba_scheduler_strategy=args.target_mamba_scheduler_strategy,
            nnodes=target_nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            speculative_num_steps=args.num_speculative_steps,
            bind_endpoint=bind_endpoint,
            connect_endpoints=connect_endpoints,
            rank=rank,
            deterministic=args.deterministic,
            decoupled_spec_trace_dir=args.decoupled_spec_trace_dir,
        )
        actors.append(actor)

    ray.get([actor.ready.remote() for actor in actors])
    return actors


def shutdown_actors(actors: list[Any]) -> None:
    if not actors:
        return
    try:
        ray.get([actor.shutdown.remote() for actor in actors], timeout=60)
    except Exception as exc:
        logger.warning("actor shutdown failed: %s", exc)
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_valid_draft_acceptance_stats(
    meta_info: dict[str, Any],
) -> tuple[int, int, float | None]:
    valid_draft_tokens = meta_info.get("spec_valid_draft_token_num")
    valid_accepted_tokens = meta_info.get("spec_valid_accept_token_num")
    if valid_draft_tokens is None or valid_accepted_tokens is None:
        return 0, 0, None

    valid_draft_tokens = int(valid_draft_tokens)
    valid_accepted_tokens = int(valid_accepted_tokens)
    if valid_draft_tokens <= 0:
        return valid_draft_tokens, valid_accepted_tokens, None

    valid_accept_rate = _float_or_none(meta_info.get("spec_valid_accept_rate"))
    if valid_accept_rate is None:
        valid_accept_rate = valid_accepted_tokens / valid_draft_tokens
    return valid_draft_tokens, valid_accepted_tokens, valid_accept_rate


def _get_decoupled_verify_acceptance_stats(
    meta_info: dict[str, Any],
) -> tuple[float | None, float | None, int, int, int]:
    """Return (accept_length, accept_rate, accepted, proposed_drafts, verify_ct).

    `accept_rate` uses the fixed verify-tree capacity as denominator
    (`spec_num_proposed_drafts` = verify_ct * (speculative_num_draft_tokens-1))
    so it is comparable to non-decoupled `spec_accept_rate`. The per-request
    `spec_valid_accept_rate` (computed separately via
    `_get_valid_draft_acceptance_stats`) uses the drafter-side denominator.
    Their ratio is the drafter fill rate; they coincide only when the drafter
    keeps up with every verify round.
    """
    verify_ct = meta_info.get("spec_verify_ct")
    valid_accepted_tokens = meta_info.get("spec_valid_accept_token_num")
    num_proposed_drafts = meta_info.get("spec_num_proposed_drafts")
    if (
        verify_ct is None
        or valid_accepted_tokens is None
        or num_proposed_drafts is None
    ):
        return None, None, 0, 0, 0

    verify_ct = int(verify_ct)
    valid_accepted_tokens = int(valid_accepted_tokens)
    num_proposed_drafts = int(num_proposed_drafts)
    if verify_ct <= 0:
        return None, None, 0, num_proposed_drafts, 0

    accept_length = valid_accepted_tokens / verify_ct
    accept_rate = (
        valid_accepted_tokens / num_proposed_drafts
        if num_proposed_drafts > 0
        else None
    )
    return (
        accept_length,
        accept_rate,
        valid_accepted_tokens,
        num_proposed_drafts,
        verify_ct,
    )


def collect_mode_metrics(
    *,
    mode: str,
    outputs: list[dict[str, Any]],
    prompt_samples: list[PromptSample],
    include_output_text: bool = True,
) -> ModeMetrics:
    if len(outputs) != len(prompt_samples):
        raise RuntimeError(
            f"{mode} returned {len(outputs)} outputs for {len(prompt_samples)} prompts"
        )

    total_generated_tokens = 0
    total_accepted_tokens = 0
    total_draft_tokens = 0
    total_valid_draft_tokens = 0
    total_valid_accepted_tokens = 0
    total_verify_ct = 0
    per_request = []
    for index, (sample, output) in enumerate(zip(prompt_samples, outputs, strict=True)):
        output_ids = output.get("output_ids", [])
        generated_tokens = len(output_ids) if isinstance(output_ids, list) else 0
        total_generated_tokens += generated_tokens

        meta_info = output.get("meta_info", {}) or {}
        (
            valid_draft_tokens,
            valid_accepted_tokens,
            valid_accept_rate,
        ) = _get_valid_draft_acceptance_stats(meta_info)
        if mode == "decoupled_spec":
            (
                accept_length,
                accept_rate,
                accepted_tokens,
                draft_tokens,
                verify_ct,
            ) = _get_decoupled_verify_acceptance_stats(meta_info)
        else:
            (
                accept_length,
                accept_rate,
                accepted_tokens,
                draft_tokens,
                verify_ct,
            ) = _get_real_verify_acceptance_stats(meta_info)
        total_accepted_tokens += accepted_tokens
        total_draft_tokens += draft_tokens
        total_valid_draft_tokens += valid_draft_tokens
        total_valid_accepted_tokens += valid_accepted_tokens
        total_verify_ct += verify_ct
        output_text = output.get("text", "")
        finish_reason = meta_info.get("finish_reason")
        request_latency_s = _float_or_none(meta_info.get("e2e_latency"))
        if request_latency_s is None:
            raise RuntimeError(
                f"{mode} output for batch_index={index}, row_index={sample.row_index} "
                "is missing meta_info['e2e_latency']; cannot compute request "
                "duration without script-side timing."
            )

        request_metrics = {
            "batch_index": index,
            "row_index": sample.row_index,
            "prompt_text": sample.prompt,
            "prompt_tokens": sample.prompt_tokens,
            "generated_tokens": generated_tokens,
            "request_latency_s": request_latency_s,
            "spec_accept_length": accept_length,
            "spec_accept_rate": accept_rate,
            "spec_valid_accept_rate": valid_accept_rate,
            "spec_valid_accept_token_num": valid_accepted_tokens or None,
            "spec_valid_draft_token_num": valid_draft_tokens or None,
            "spec_verify_ct": verify_ct or None,
            "finish_reason": finish_reason,
            "output_text_preview": (
                output_text[:512] if isinstance(output_text, str) else None
            ),
            "output_ids_head": (
                output_ids[:32] if isinstance(output_ids, list) else None
            ),
            "output_ids_tail": (
                output_ids[-32:] if isinstance(output_ids, list) else None
            ),
        }
        if include_output_text:
            request_metrics["output_text"] = (
                output_text if isinstance(output_text, str) else ""
            )
        per_request.append(request_metrics)

    generation_time_s = (
        max(item["request_latency_s"] for item in per_request) if per_request else 0.0
    )
    throughput = (
        total_generated_tokens / generation_time_s if generation_time_s > 0 else 0.0
    )
    avg_accept_length = (
        total_accepted_tokens / total_verify_ct if total_verify_ct > 0 else None
    )
    avg_accept_rate = (
        total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else None
    )
    avg_valid_accept_rate = (
        total_valid_accepted_tokens / total_valid_draft_tokens
        if total_valid_draft_tokens > 0
        else None
    )
    return ModeMetrics(
        mode=mode,
        generation_time_s=generation_time_s,
        total_generated_tokens=total_generated_tokens,
        output_throughput_tok_per_s=throughput,
        per_request=per_request,
        avg_spec_accept_length=avg_accept_length,
        avg_spec_accept_rate=avg_accept_rate,
        avg_spec_valid_accept_rate=avg_valid_accept_rate,
        total_spec_valid_draft_token_num=total_valid_draft_tokens,
        total_spec_valid_accept_token_num=total_valid_accepted_tokens,
    )


def _split_indices(num_items: int, num_shards: int) -> list[list[int]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if num_items % num_shards != 0:
        raise ValueError(
            f"batch size ({num_items}) must be divisible by verifier replicas "
            f"({num_shards})"
        )
    shard_size = num_items // num_shards
    return [
        list(range(shard_index * shard_size, (shard_index + 1) * shard_size))
        for shard_index in range(num_shards)
    ]


def run_mode(
    *,
    args: argparse.Namespace,
    mode: str,
    prompt_input_ids: list[list[int]],
    sampling_params: dict[str, Any],
    prompt_samples: list[PromptSample],
    dist_init_addrs: list[str | None],
    target_nnodes: int,
    target_gpus_per_node: int,
    pgs: list[Any] | None = None,
    endpoint_configs: list[Any] | None = None,
    include_output_text: bool = True,
) -> ModeMetrics:
    target_actor_groups: list[list[Any]] = []
    owns_pgs = pgs is None
    num_replicas = (
        len(endpoint_configs) if endpoint_configs is not None else len(dist_init_addrs)
    )
    if num_replicas <= 0:
        raise ValueError("run_mode requires at least one target replica")
    if len(dist_init_addrs) != num_replicas:
        raise ValueError(
            f"dist_init_addrs has {len(dist_init_addrs)} entries, expected {num_replicas}"
        )
    if pgs is not None and len(pgs) != num_replicas:
        raise ValueError(f"pgs has {len(pgs)} entries, expected {num_replicas}")

    replica_indices = _split_indices(len(prompt_samples), num_replicas)
    outputs_by_index: list[dict[str, Any] | None] = [None] * len(prompt_samples)
    try:
        if pgs is None:
            pgs = create_target_placement_groups(
                num_replicas,
                target_nnodes,
                target_gpus_per_node,
            )

        for replica_index in range(num_replicas):
            endpoint_config = (
                endpoint_configs[replica_index]
                if endpoint_configs is not None
                else None
            )
            actors = launch_target_actors(
                args=args,
                mode=mode,
                dist_init_addr=dist_init_addrs[replica_index],
                target_nnodes=target_nnodes,
                target_gpus_per_node=target_gpus_per_node,
                pg=pgs[replica_index],
                bind_endpoint=(
                    endpoint_config.bind_endpoint
                    if endpoint_config is not None
                    else None
                ),
                connect_endpoints=(
                    endpoint_config.connect_endpoints
                    if endpoint_config is not None
                    else None
                ),
                rank=endpoint_config.rank if endpoint_config is not None else None,
            )
            target_actor_groups.append(actors)

        result_refs = []
        for replica_index, indices in enumerate(replica_indices):
            shard_input_ids = [prompt_input_ids[index] for index in indices]
            result_refs.append(
                (
                    indices,
                    target_actor_groups[replica_index][0].generate_batch.remote(
                        shard_input_ids,
                        sampling_params,
                    ),
                )
            )

        for indices, result_ref in result_refs:
            result = ray.get(result_ref)
            shard_outputs = result["outputs"]
            if len(shard_outputs) != len(indices):
                raise RuntimeError(
                    f"{mode} returned {len(shard_outputs)} outputs for "
                    f"{len(indices)} prompts on one replica"
                )
            for index, output in zip(indices, shard_outputs, strict=True):
                outputs_by_index[index] = output
    finally:
        for actors in target_actor_groups:
            shutdown_actors(actors)
        if owns_pgs and pgs is not None:
            for pg in pgs:
                remove_placement_group(pg)

    if any(output is None for output in outputs_by_index):
        missing = [
            index for index, output in enumerate(outputs_by_index) if output is None
        ]
        raise RuntimeError(f"{mode} did not return outputs for indices {missing}")

    return collect_mode_metrics(
        mode=mode,
        outputs=[output for output in outputs_by_index if output is not None],
        prompt_samples=prompt_samples,
        include_output_text=include_output_text,
    )


def build_result(
    *,
    args: argparse.Namespace,
    target_nnodes: int,
    target_gpus_per_node: int,
    prompt_column: str,
    total_rows: int,
    prompt_samples: list[PromptSample],
    spec_metrics: ModeMetrics,
    decode_metrics: ModeMetrics | None = None,
) -> dict[str, Any]:
    speedup = (
        decode_metrics.generation_time_s / spec_metrics.generation_time_s
        if decode_metrics is not None and spec_metrics.generation_time_s > 0
        else None
    )
    result = {
        "config": {
            "dataset_path": args.dataset_path,
            "dataset_format": args.dataset_format,
            "prompt_column": prompt_column,
            "engine_input": "input_ids",
            "code_language": args.code_language,
            "offset": args.offset,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "max_new_tokens": args.context_length,
            "max_prompt_length": args.max_prompt_length,
            "target_model_path": args.target_model_path,
            "draft_model_path": args.draft_model_path,
            "tokenizer_path": args.tokenizer_path or args.target_model_path,
            "target_tp_size": args.target_tp_size,
            "target_ep_size": args.target_ep_size,
            "target_moe_a2a_backend": args.target_moe_a2a_backend,
            "target_mamba_scheduler_strategy": args.target_mamba_scheduler_strategy,
            "num_verifier_replicas": args.num_verifier_replicas,
            "verify_ngpus": args.verify_ngpus,
            "draft_tp_size": args.draft_tp_size,
            "draft_ngpus": args.draft_ngpus,
            "num_speculative_steps": args.num_speculative_steps,
            "temperature": args.temperature,
            "deterministic": args.deterministic,
            "ignore_eos": args.ignore_eos,
            "nnodes": args.nnodes,
            "n_gpu_per_node": args.n_gpu_per_node,
            "target_nnodes": target_nnodes,
            "target_gpus_per_node": target_gpus_per_node,
            "num_draft_replicas": args.num_draft_replicas,
            "skip_decode": args.skip_decode,
            "show_responses": args.show_responses,
            "decoupled_spec_trace_dir": args.decoupled_spec_trace_dir,
            "decoupled_spec_allow_partial": args.decoupled_spec_allow_partial,
        },
        "dataset": {
            "total_rows": total_rows,
            "loaded_rows": [sample.row_index for sample in prompt_samples],
            "total_prompt_tokens": sum(
                sample.prompt_tokens for sample in prompt_samples
            ),
            "prompt_samples": [
                {
                    "row_index": sample.row_index,
                    "prompt_tokens": sample.prompt_tokens,
                    "prompt": sample.prompt,
                    "prompt_head": sample.prompt[:1024],
                    "prompt_tail": sample.prompt[-1024:],
                }
                for sample in prompt_samples
            ],
        },
        "decoupled_spec": asdict(spec_metrics),
    }
    if decode_metrics is not None:
        result["decode"] = asdict(decode_metrics)
        result["e2e_speedup"] = speedup
    return result


def _iter_output_modes(result: dict[str, Any]):
    yield "decoupled_spec", "decoupled-spec"
    if "decode" in result:
        yield "decode", "decode"


def _request_output_record(item: dict[str, Any]) -> dict[str, Any]:
    steps = item["spec_verify_ct"] or item["generated_tokens"]
    return {
        "index": item["batch_index"],
        "offset": item["row_index"],
        "prompt-length": item["prompt_tokens"],
        "response-length": item["generated_tokens"],
        "steps": steps,
        "duration": item["request_latency_s"],
    }


def _csv_fieldnames_for_mode(mode_key: str) -> list[str]:
    fieldnames = [
        "index",
        "offset",
        "prompt-length",
        "response-length",
        "steps",
        "duration",
    ]
    if mode_key == "decoupled_spec":
        fieldnames.extend(
            [
                "spec_accept_length",
                "spec_accept_rate",
                "spec_valid_accept_rate",
                "spec_valid_accept_token_num",
                "spec_valid_draft_token_num",
            ]
        )
    return fieldnames


def _csv_output_record(mode_key: str, item: dict[str, Any]) -> dict[str, Any]:
    record = _request_output_record(item)
    if mode_key == "decoupled_spec":
        record.update(
            {
                "spec_accept_length": item["spec_accept_length"],
                "spec_accept_rate": item["spec_accept_rate"],
                "spec_valid_accept_rate": item["spec_valid_accept_rate"],
                "spec_valid_accept_token_num": item["spec_valid_accept_token_num"],
                "spec_valid_draft_token_num": item["spec_valid_draft_token_num"],
            }
        )
    return record


def write_output_files(result: dict[str, Any], output_dir: str) -> list[Path]:
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for mode_key, file_prefix in _iter_output_modes(result):
        mode_items = result[mode_key]["per_request"]

        csv_path = output_path / f"{file_prefix}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = _csv_fieldnames_for_mode(mode_key)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in mode_items:
                writer.writerow(_csv_output_record(mode_key, item))
        written_paths.append(csv_path)

        json_path = output_path / f"{file_prefix}.json"
        requests = []
        for item in mode_items:
            record = _request_output_record(item)
            if mode_key == "decoupled_spec":
                record.update(
                    {
                        "spec_accept_length": item["spec_accept_length"],
                        "spec_accept_rate": item["spec_accept_rate"],
                        "spec_valid_accept_rate": item["spec_valid_accept_rate"],
                        "spec_valid_accept_token_num": item[
                            "spec_valid_accept_token_num"
                        ],
                        "spec_valid_draft_token_num": item[
                            "spec_valid_draft_token_num"
                        ],
                    }
                )
            record.update(
                {
                    "prompt": item.get("prompt_text", ""),
                    "response": item.get("output_text", ""),
                }
            )
            requests.append(record)
        json_path.write_text(
            json.dumps(
                {"mode": mode_key, "requests": requests},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written_paths.append(json_path)

    return written_paths


def _print_response_block(label: str, text: str, *, indent: str = "    ") -> None:
    print(f"{indent}{label}:")
    if not text:
        print(f"{indent}  <empty>")
        return
    for line in text.splitlines():
        print(f"{indent}  {line}")


def print_summary(result: dict[str, Any]) -> None:
    spec = result["decoupled_spec"]
    decode = result.get("decode")
    speedup = result.get("e2e_speedup")
    title = (
        "decoupled_spec_vs_decode_batch"
        if decode is not None
        else "decoupled_spec_batch"
    )
    print(f"=== {title} ===")
    print(f"dataset_path: {result['config']['dataset_path']}")
    print(f"dataset_format: {result['config']['dataset_format']}")
    print(f"prompt_column: {result['config']['prompt_column']}")
    print(f"batch_size: {result['config']['batch_size']}")
    print(f"verify_ngpus: {result['config']['verify_ngpus']}")
    print(f"draft_ngpus: {result['config']['draft_ngpus']}")
    print(f"num_verifier_replicas: {result['config']['num_verifier_replicas']}")
    print(f"num_draft_replicas: {result['config']['num_draft_replicas']}")
    print(f"max_new_tokens: {result['config']['max_new_tokens']}")
    print(f"total_prompt_tokens: {result['dataset']['total_prompt_tokens']}")
    print(
        "decoupled_spec: "
        f"generation_time_s={spec['generation_time_s']:.3f}, "
        f"generated_tokens={spec['total_generated_tokens']}, "
        f"output_throughput={spec['output_throughput_tok_per_s']:.3f} tok/s, "
        f"avg_spec_accept_length={spec['avg_spec_accept_length']}, "
        f"avg_spec_accept_rate={spec['avg_spec_accept_rate']}, "
        f"avg_spec_valid_accept_rate={spec['avg_spec_valid_accept_rate']}, "
        f"valid_accept_tokens={spec['total_spec_valid_accept_token_num']}, "
        f"valid_draft_tokens={spec['total_spec_valid_draft_token_num']}"
    )
    if decode is not None:
        print(
            "decode: "
            f"generation_time_s={decode['generation_time_s']:.3f}, "
            f"generated_tokens={decode['total_generated_tokens']}, "
            f"output_throughput={decode['output_throughput_tok_per_s']:.3f} tok/s"
        )
        print(
            f"e2e_speedup: {speedup:.4f}"
            if speedup is not None
            else "e2e_speedup: None"
        )
    print("per_request:")
    for item in spec["per_request"]:
        print(
            "  "
            f"batch_index={item['batch_index']}, "
            f"row_index={item['row_index']}, "
            f"prompt_tokens={item['prompt_tokens']}, "
            f"generated_tokens={item['generated_tokens']}, "
            f"request_latency_s={item['request_latency_s']}, "
            f"spec_accept_length={item['spec_accept_length']}, "
            f"spec_accept_rate={item['spec_accept_rate']}, "
            f"spec_valid_accept_rate={item['spec_valid_accept_rate']}, "
            f"spec_valid_accept_token_num={item['spec_valid_accept_token_num']}, "
            f"spec_valid_draft_token_num={item['spec_valid_draft_token_num']}, "
            f"spec_verify_ct={item['spec_verify_ct']}"
        )
    if result["config"].get("show_responses"):
        print("responses:")
        decode_items = (
            decode["per_request"]
            if decode is not None
            else [None] * len(spec["per_request"])
        )
        for spec_item, decode_item in zip(
            spec["per_request"], decode_items, strict=True
        ):
            print(
                "  "
                f"batch_index={spec_item['batch_index']}, "
                f"row_index={spec_item['row_index']}"
            )
            _print_response_block(
                "decoupled_spec_response",
                spec_item.get("output_text", ""),
            )
            if decode_item is not None:
                if (
                    spec_item["batch_index"] != decode_item["batch_index"]
                    or spec_item["row_index"] != decode_item["row_index"]
                ):
                    raise RuntimeError(
                        "Mismatched per-request ordering between decoupled_spec and decode"
                    )
                _print_response_block(
                    "decode_response",
                    decode_item.get("output_text", ""),
                )


def main() -> None:
    args = parse_args()
    _ensure_runtime_imports()

    prompt_column, prompt_samples, total_rows = load_prompt_samples(args)
    prompt_input_ids = [list(sample.prompt_input_ids) for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    draft_actors: list[Any] = []
    spec_pgs = []
    try:
        init_ray(args.ray_address, args.ray_namespace, args.nnodes)
        target_nnodes, target_gpus_per_node = validate_resources(args)

        spec_pgs = create_target_placement_groups(
            args.num_verifier_replicas,
            target_nnodes,
            target_gpus_per_node,
        )
        spec_dist_init_addrs = [
            derive_dist_init_addr_from_pg(args, pg, port_offset=replica_index)
            for replica_index, pg in enumerate(spec_pgs)
        ]
        spec_dist_init_ports = {
            port
            for addr in spec_dist_init_addrs
            if addr is not None
            for _, port in [_parse_host_port(addr)]
            if port is not None
        }
        num_verifiers = args.num_verifier_replicas
        reserved_dist_init_ports = set(spec_dist_init_ports)
        if not args.skip_decode:
            if args.dist_init_port is not None:
                reserved_dist_init_ports.update(
                    args.dist_init_port + num_verifiers + replica_index
                    for replica_index in range(num_verifiers)
                )
            elif args.dist_init_addr is not None:
                _, base_port = _parse_host_port(args.dist_init_addr)
                if base_port is not None:
                    reserved_dist_init_ports.update(
                        base_port + num_verifiers + replica_index
                        for replica_index in range(num_verifiers)
                    )
        preferred_result_ports = (
            [args.dist_init_port + 2 * num_verifiers + i for i in range(num_verifiers)]
            if args.dist_init_port is not None
            else None
        )
        preferred_control_ports = (
            [
                args.dist_init_port + 3 * num_verifiers + i
                for i in range(args.num_draft_replicas)
            ]
            if args.dist_init_port is not None
            else None
        )
        topology = create_remote_decoupled_spec_topology(
            args,
            spec_pgs,
            avoid_ports=reserved_dist_init_ports,
            preferred_result_ports=preferred_result_ports,
            preferred_control_ports=preferred_control_ports,
        )
        draft_actors = topology.draft_actors or []
        spec_metrics = run_mode(
            args=args,
            mode="decoupled_spec",
            prompt_input_ids=prompt_input_ids,
            sampling_params=sampling_params,
            prompt_samples=prompt_samples,
            dist_init_addrs=spec_dist_init_addrs,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            pgs=spec_pgs,
            endpoint_configs=topology.verifier_configs,
            include_output_text=True,
        )
        shutdown_actors(draft_actors)
        draft_actors = []

        decode_metrics = None
        if not args.skip_decode:
            decode_dist_init_addrs = [
                derive_dist_init_addr_from_pg(
                    args,
                    pg,
                    port_offset=args.num_verifier_replicas + replica_index,
                )
                for replica_index, pg in enumerate(spec_pgs)
            ]
            decode_metrics = run_mode(
                args=args,
                mode="decode",
                prompt_input_ids=prompt_input_ids,
                sampling_params=sampling_params,
                prompt_samples=prompt_samples,
                dist_init_addrs=decode_dist_init_addrs,
                target_nnodes=target_nnodes,
                target_gpus_per_node=target_gpus_per_node,
                pgs=spec_pgs,
                include_output_text=True,
            )

        result = build_result(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
            decode_metrics=decode_metrics,
        )
        print_summary(result)
        if args.output_dir:
            print("output_files:")
            for output_path in write_output_files(result, args.output_dir):
                print(f"  {output_path}")
    finally:
        shutdown_actors(draft_actors)
        for pg in spec_pgs:
            remove_placement_group(pg)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
