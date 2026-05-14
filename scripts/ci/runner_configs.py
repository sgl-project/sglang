"""Single source of truth for `runner_config` → CUDA-stage setup details.

Each `register_cuda_ci(stage="stage-X", runner_config="Y", ...)` registers a
test to one of the keys below. The matching pr-test.yml stage job calls
`./.github/actions/setup-cuda-test-stage` with `runner_config: Y`; the action
shells out to this script for install script, artifact download version,
and install timeout.
"""

import argparse
import sys

_CUDA_DEFAULT_INSTALL = "scripts/ci/cuda/ci_install_dependency.sh"
_CUDA_DEEPEP_INSTALL = "scripts/ci/cuda/ci_install_deepep.sh"
_CUDA_DSV4_INSTALL = "scripts/ci/cuda/ci_install_dsv4_dep.sh"

# Keep in sync with pr-test.yml stage jobs.
RUNNER_CONFIGS: dict[str, dict] = {
    "1-gpu-small": {
        "install": _CUDA_DEFAULT_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "1-gpu-large": {
        "install": _CUDA_DEFAULT_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "2-gpu-large": {
        "install": _CUDA_DEFAULT_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "4-gpu-b200": {
        "install": _CUDA_DEFAULT_INSTALL,
        "artifact_version": "v6",
        "install_timeout": "20",
    },
    "4-gpu-h100": {
        "install": _CUDA_DEFAULT_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "8-gpu-h200": {
        "install": _CUDA_DEFAULT_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "8-gpu-h20": {
        "install": _CUDA_DEEPEP_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "deepep-4-gpu-h100": {
        "install": _CUDA_DEEPEP_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "deepep-8-gpu-h200": {
        "install": _CUDA_DEEPEP_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "20",
    },
    "dsv4-4-gpu-b200": {
        "install": _CUDA_DSV4_INSTALL,
        "artifact_version": "v6",
        "install_timeout": "30",
    },
    "dsv4-8-gpu-h200": {
        "install": _CUDA_DSV4_INSTALL,
        "artifact_version": "v4",
        "install_timeout": "30",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("install", "artifact_version", "install_timeout", "github_output"),
        help="`github_output` emits all fields in $GITHUB_OUTPUT format; the rest print one value.",
    )
    parser.add_argument("--runner-config", required=True)
    args = parser.parse_args()

    config = RUNNER_CONFIGS.get(args.runner_config)
    if config is None:
        print(
            f"error: unknown runner_config {args.runner_config!r}; "
            f"add it to {__file__}",
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
