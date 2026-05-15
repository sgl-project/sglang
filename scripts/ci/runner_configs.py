"""Emit a runner_config's setup details (install / artifact_version /
install_timeout) in $GITHUB_OUTPUT format. Reads scripts/ci/runner_configs.yml.
Called by .github/workflows/_pr-test-stage.yml.
"""

import os
import sys

import yaml

_YAML_PATH = os.path.join(os.path.dirname(__file__), "runner_configs.yml")


def load() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)["runner_configs"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: runner_configs.py <runner_config>")
    rc = sys.argv[1]
    config = load().get(rc)
    if config is None:
        sys.exit(f"unknown runner_config: {rc!r}")
    for key, value in config.items():
        print(f"{key}={value}")
