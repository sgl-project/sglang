import argparse

from sglang.auto_benchmark_lib import (
    SUPPORTED_DATASETS,
    convert_dataset,
    run_auto_benchmark,
    validate_dataset,
)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kind",
        required=True,
        choices=sorted(SUPPORTED_DATASETS),
        help="Dataset kind: sharegpt, custom, random, or generated-shared-prefix.",
    )
    parser.add_argument(
        "--path",
        default="",
        help="Dataset file path. Leave empty for sharegpt auto-download.",
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--prompt-suffix", type=str, default="")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--random-input-len", type=int, default=1024)
    parser.add_argument("--random-output-len", type=int, default=256)
    parser.add_argument("--random-range-ratio", type=float, default=0.0)
    parser.add_argument("--gsp-num-groups", type=int, default=64)
    parser.add_argument("--gsp-prompts-per-group", type=int, default=16)
    parser.add_argument("--gsp-system-prompt-len", type=int, default=2048)
    parser.add_argument("--gsp-question-len", type=int, default=128)
    parser.add_argument("--gsp-output-len", type=int, default=256)
    parser.add_argument("--gsp-range-ratio", type=float, default=1.0)
    parser.add_argument("--gsp-fast-prepare", action="store_true")
    parser.add_argument("--gsp-send-routing-key", action="store_true")
    parser.add_argument("--gsp-num-turns", type=int, default=1)
    parser.add_argument("--gsp-ordered", action="store_true")
    parser.add_argument("--seed", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SGLang auto benchmark utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run auto benchmark from YAML config."
    )
    run_parser.add_argument("--config", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Prepare sharegpt/custom/random/generated-shared-prefix data into canonical autobench JSONL.",
    )
    add_dataset_args(convert_parser)
    convert_parser.add_argument("--output", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a canonical autobench JSONL dataset."
    )
    validate_parser.add_argument("--dataset-path", required=True)
    validate_parser.add_argument("--tokenizer", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_auto_benchmark(args.config)
    elif args.command == "convert":
        convert_dataset(args)
    elif args.command == "validate":
        validate_dataset(args)


if __name__ == "__main__":
    main()
