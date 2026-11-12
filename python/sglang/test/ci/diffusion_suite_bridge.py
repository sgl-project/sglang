"""Bridge registered diffusion suites to their case-aware pytest adapter."""

import os
import sys
from pathlib import Path


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def run_diffusion_suite(suite: str) -> None:
    """Run one legacy-named diffusion suite without exposing a second CI CLI."""

    from sglang.multimodal_gen.test.runner.diffusion_suite_runner import main

    args = [sys.argv[0], "--suite", suite]
    optional_values = (
        ("DIFFUSION_PARTITION_ID", "--partition-id"),
        ("DIFFUSION_TOTAL_PARTITIONS", "--total-partitions"),
        ("DIFFUSION_PARTITION_PLAN_JSON", "--partition-plan-json"),
        ("DIFFUSION_PYTEST_FILTER", "--filter"),
    )
    for environment_name, option in optional_values:
        value = os.environ.get(environment_name)
        if value:
            args.extend([option, value])
    if _enabled("DIFFUSION_CONTINUE_ON_ERROR"):
        args.append("--continue-on-error")

    # Preserve the historical cwd: several diffusion fixtures emit artifacts
    # relative to ``python/`` rather than to their source file.
    os.chdir(Path(__file__).resolve().parents[4] / "python")
    sys.argv = args
    main()
