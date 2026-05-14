"""Emit runner_config setup details (install / artifact_version /
install_timeout / runs_on / rdma_devices) for use in GitHub Actions.

Two modes:

  runner_configs.py <runner_config>
      Single-config mode. Prints each field as `key=value` lines (one per
      field) suitable for `>> $GITHUB_OUTPUT`. Called once per stage by
      .github/workflows/_pr-test-stage.yml.

  runner_configs.py --map --b200-runner <label>
      Map mode. Prints a single line `runner_map={json}` for $GITHUB_OUTPUT,
      where {json} is a JSON object keyed by runner_config name. The literal
      `$b200_runner` in any `runs_on` field is substituted with <label>.
      Called once by .github/workflows/_pr-test-check-changes.yml so the
      resolved map is fanned out to every stage via check-changes outputs.
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


def build_map(b200_runner: str) -> dict:
    """Return the full runner_configs dict with `$b200_runner` substituted."""
    out = {}
    for name, cfg in load().items():
        resolved = dict(cfg)
        if resolved.get("runs_on") == _B200_SENTINEL:
            resolved["runs_on"] = b200_runner
        out[name] = resolved
    return out


def _print_single(rc: str) -> None:
    config = load().get(rc)
    if config is None:
        sys.exit(f"unknown runner_config: {rc!r}")
    for key, value in config.items():
        print(f"{key}={value}")


def _print_map(b200_runner: str) -> None:
    print(f"runner_map={json.dumps(build_map(b200_runner), separators=(',', ':'))}")


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
