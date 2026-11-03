"""Launch a real SGLang server instrumented for simulator baseline collection."""

import argparse
import os
from pathlib import Path

from sglang_simulator.tools.collect_baseline.hooks import (
    COLLECTION_OUTPUT_DIR_ENV,
    TokenizerManagerCollectionHook,
    install_frontend_collection_hooks,
    run_collection_scheduler_process,
    validate_collection_hook,
)

# This must happen before importing the SGLang HTTP server.
install_frontend_collection_hooks()


def _prepare_output_dir(parser: argparse.ArgumentParser, value: str) -> Path:
    output_dir = Path(value).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(
            f"baseline collection output path is not a directory: {output_dir}"
        )
    if output_dir.is_dir() and any(output_dir.iterdir()):
        parser.error(
            f"baseline collection output directory must be empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ[COLLECTION_OUTPUT_DIR_ENV] = str(output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree

    validate_collection_hook(TokenizerManagerCollectionHook)

    parser = argparse.ArgumentParser()
    collection = parser.add_argument_group("baseline collection")
    collection.add_argument(
        "--collection-output-dir",
        default=os.getenv(COLLECTION_OUTPUT_DIR_ENV),
        required=os.getenv(COLLECTION_OUTPUT_DIR_ENV) is None,
        help=(
            "Empty directory for TP*.request.jsonl, raw_request.jsonl, "
            "and schedule_batch.jsonl"
        ),
    )
    ServerArgs.add_cli_args(parser.add_argument_group("sglang"))

    raw_args = parser.parse_args(argv)
    output_dir = _prepare_output_dir(parser, raw_args.collection_output_dir)
    server_args = ServerArgs.from_cli_args(raw_args)
    if not server_args.disable_overlap_schedule:
        parser.error(
            "--disable-overlap-schedule is required so synchronized batch "
            "latencies match ScheduleBatch execution boundaries"
        )

    print(f"SGLang Simulator baseline collection directory: {output_dir}")
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=run_collection_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
