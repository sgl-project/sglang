"""Single source of truth for `runner_config` → CUDA-stage setup details.

Each `register_cuda_ci(stage="stage-X", runner_config="Y", ...)` registers a
test to one of the keys below. The matching pr-test.yml stage job calls
`./.github/actions/setup-cuda-test-stage` with `runner_config: Y`; the action
shells out to this script for the install script path.

Used by:
- `.github/actions/setup-cuda-test-stage/action.yml` (runtime install lookup)
- `scripts/ci/utils/compute_partitions.py` (validation: registered
  `runner_config` must exist here)
"""

import argparse
import sys

# Default CUDA install path used by most stages.
_CUDA_DEFAULT_INSTALL = "scripts/ci/cuda/ci_install_dependency.sh"
_CUDA_DEEPEP_INSTALL = "scripts/ci/cuda/ci_install_deepep.sh"
_CUDA_DSV4_INSTALL = "scripts/ci/cuda/ci_install_dsv4_dep.sh"

# CUDA per-commit runner configs. Keep in sync with pr-test.yml stage jobs.
RUNNER_CONFIGS: dict[str, dict] = {
    "1-gpu-small": {"install": _CUDA_DEFAULT_INSTALL},
    "1-gpu-large": {"install": _CUDA_DEFAULT_INSTALL},
    "2-gpu-large": {"install": _CUDA_DEFAULT_INSTALL},
    "4-gpu-b200": {"install": _CUDA_DEFAULT_INSTALL},
    "4-gpu-h100": {"install": _CUDA_DEFAULT_INSTALL},
    "8-gpu-h200": {"install": _CUDA_DEFAULT_INSTALL},
    "8-gpu-h20": {"install": _CUDA_DEEPEP_INSTALL},
    "deepep-4-gpu-h100": {"install": _CUDA_DEEPEP_INSTALL},
    "deepep-8-gpu-h200": {"install": _CUDA_DEEPEP_INSTALL},
    "dsv4-4-gpu-b200": {"install": _CUDA_DSV4_INSTALL},
    "dsv4-8-gpu-h200": {"install": _CUDA_DSV4_INSTALL},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", choices=("install",))
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
    print(config[args.field])


if __name__ == "__main__":
    main()
