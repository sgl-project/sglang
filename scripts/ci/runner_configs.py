"""CLI wrapper around scripts/ci/runner_configs.yml. Used by
.github/actions/setup-cuda-test-stage to look up install script / artifact
version / install timeout for a given runner_config (see register_cuda_ci
declarations in test/registered/).
"""

import argparse
import os
import sys

import yaml

_YAML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runner_configs.yml"
)


def load() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)["runner_configs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("install", "artifact_version", "install_timeout", "github_output"),
        help="`github_output` emits all fields in $GITHUB_OUTPUT format; the rest print one value.",
    )
    parser.add_argument("--runner-config", required=True)
    args = parser.parse_args()

    configs = load()
    config = configs.get(args.runner_config)
    if config is None:
        print(
            f"error: unknown runner_config {args.runner_config!r}; "
            f"add it to {_YAML_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "github_output":
        for key, value in config.items():
            print(f"{key}={value}")
    else:
        print(config[args.mode])


if __name__ == "__main__":
    main()
