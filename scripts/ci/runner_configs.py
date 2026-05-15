"""Emit runner_config setup for GitHub Actions $GITHUB_OUTPUT.

runner_configs.py <runner_config>
    Per-field `key=value` lines (install / artifact_version /
    install_timeout / runs_on / rdma_devices). Called per stage by
    _pr-test-stage.yml.

runner_configs.py --map --b200-runner <label>
    `runs_on_map={json}` — flat dict {runner_config: runs_on}, with
    `$b200_runner` substituted. Called once by _pr-test-check-changes.yml;
    used by stage / non-stage workflow `runs-on:` expressions.
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


def build_runs_on_map(b200_runner: str) -> dict:
    """Return flat {runner_config: runs_on} with `$b200_runner` substituted."""
    out = {}
    for name, cfg in load().items():
        runs_on = cfg.get("runs_on")
        if runs_on == _B200_SENTINEL:
            runs_on = b200_runner
        out[name] = runs_on
    return out


def _print_single(rc: str) -> None:
    config = load().get(rc)
    if config is None:
        sys.exit(f"unknown runner_config: {rc!r}")
    for key, value in config.items():
        print(f"{key}={value}")


def _print_map(b200_runner: str) -> None:
    payload = build_runs_on_map(b200_runner)
    print(f"runs_on_map={json.dumps(payload, separators=(',', ':'))}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--map":
        b200_runner = ""
        i = 1
        while i < len(args):
            if args[i] == "--b200-runner" and i + 1 < len(args):
                b200_runner = args[i + 1]
                i += 2
            else:
                sys.exit(f"unexpected arg: {args[i]!r}")
        if not b200_runner:
            sys.exit("usage: runner_configs.py --map --b200-runner <label>")
        _print_map(b200_runner)
    elif len(args) == 1:
        _print_single(args[0])
    else:
        sys.exit(
            "usage:\n"
            "  runner_configs.py <runner_config>\n"
            "  runner_configs.py --map --b200-runner <label>"
        )
