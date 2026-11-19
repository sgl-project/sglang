# SPDX-License-Identifier: Apache-2.0
"""Select reviewed GPU runners without executing code from the Mega checkout."""

import argparse
import json
import os
import re
from pathlib import Path

import yaml

# Runner labels are controlled here, never by a plugin PR's manifest.
RUNNERS = {"GB300": (4, "4-gpu-gb300"), "B300": (8, "8-gpu-b300")}
LEGACY = {
    "deepseek_v41": {
        "gpu": "GB300",
        "gpus": 4,
        "tests": ["test_deepseek_v41_server.py", "test_deepseek_v41_switch.py"],
    }
}


def build_matrix(e2e: Path, requested: list[str]) -> dict:
    manifest = e2e / "ci_models.yaml"
    models = yaml.safe_load(manifest.read_text()) if manifest.is_file() else LEGACY
    if not isinstance(models, dict) or not models:
        raise ValueError("Expected a nonempty model manifest")
    unknown = set(requested) - models.keys()
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    groups, listed = {}, set()
    for name, model in models.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"Invalid model name: {name!r}")
        if not isinstance(model, dict):
            raise ValueError(f"Invalid model entry: {name}")
        gpu, gpus, tests = model.get("gpu"), model.get("gpus"), model.get("tests")
        if not isinstance(gpu, str) or type(gpus) is not int or gpus < 1:
            raise ValueError(f"Invalid GPU requirement: {name}")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"No test files: {name}")
        for test in tests:
            if (
                not isinstance(test, str)
                or not re.fullmatch(r"test_\w+\.py", test)
                or not (e2e / test).is_file()
                or (e2e / test).resolve().parent != e2e.resolve()
            ):
                raise ValueError(f"Invalid test file for {name}: {test!r}")
        listed.update(tests)
        if requested and name not in requested:
            continue
        capacity, runner = RUNNERS.get(gpu, (0, None))
        if gpus > capacity:
            message = f"{name} requires {gpus} {gpu} GPUs; no configured runner"
            if requested:
                raise ValueError(message)
            print(f"::notice::{message}; not covered by this run")
            continue
        group = groups.setdefault(runner, {"runner": runner, "models": [], "tests": []})
        group["models"].append(name)
        group["tests"].extend(tests)
    for path in sorted(e2e.glob("test_*.py")):
        if path.name not in listed:
            print(f"::warning::{path.name} is not listed in ci_models.yaml")
    if not groups:
        raise ValueError("No selected model has a configured runner")
    return {
        "include": [
            {
                "runner": group["runner"],
                "models": " ".join(group["models"]),
                "tests": " ".join(
                    f"test/e2e/{t}" for t in dict.fromkeys(group["tests"])
                ),
            }
            for group in groups.values()
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("e2e", type=Path)
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()
    matrix = json.dumps(build_matrix(args.e2e, args.models))
    print(matrix)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            print(f"matrix={matrix}", file=output)
