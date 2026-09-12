"""Emit runner_config setup for GitHub Actions $GITHUB_OUTPUT.

runner_configs.py <runner_config>
    Per-field `key=value` lines (install / artifact_version /
    install_timeout / rdma_devices). `runs_on` is intentionally omitted —
    it carries the `$b200_runner` sentinel and is resolved via --map.
    Called per stage by _pr-test-stage.yml.

runner_configs.py --map <b200_runner_label>
    `runs_on_map={json}` — flat dict {runner_config: runs_on}, with
    `$b200_runner` substituted. Called once by _pr-test-check-changes.yml.
"""

import json
import os
import sys

import yaml

_YAML_PATH = os.path.join(os.path.dirname(__file__), "runner_configs.yml")
_B200_SENTINEL = "$b200_runner"


def load() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)["runner_configs"]


def _emit_single(rc: str) -> None:
    # runs_on goes through --map (resolves $b200_runner). Suppress it here so a
    # consumer can't accidentally read the raw sentinel value.
    cfg = load().get(rc)
    if cfg is None:
        sys.exit(f"unknown runner_config: {rc!r}")
    for key, value in cfg.items():
        if key == "runs_on":
            continue
        print(f"{key}={value}")


def _emit_map(b200_runner: str) -> None:
    runs_on = {
        name: (b200_runner if cfg.get("runs_on") == _B200_SENTINEL else cfg["runs_on"])
        for name, cfg in load().items()
    }
    print(f"runs_on_map={json.dumps(runs_on, separators=(',', ':'))}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        _emit_single(args[0])
    elif len(args) == 2 and args[0] == "--map":
        _emit_map(args[1])
    else:
        sys.exit(
            "usage:\n"
            "  runner_configs.py <runner_config>\n"
            "  runner_configs.py --map <b200_runner_label>"
        )
