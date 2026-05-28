"""Write the per-run config JSON snapshot consumed by the runner shell script.

Replaces the inline ``write_run_config`` heredoc. Reads its parameters from
env vars (preserving the original interface so the bash side just needs to
pass them through) and writes ``${OUTPUT_DIR}/run_config.json``.
"""

import json
import os
import sys
from pathlib import Path

REQUIRED_STR_KEYS = (
    "MODEL_LABEL",
    "MODEL_PATH",
    "PYTHON_BIN",
    "PYTHONPATH_BASE",
    "BASE_URL",
    "DTYPES",
    "EVALS",
    "MAMBA_SCHEDULER_STRATEGY",
    "ENABLE_DTYPE_PROBE",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK",
    "LOCAL_NO_PROXY",
)
INT_KEYS = (
    "TP_SIZE",
    "PORT",
    "NUM_THREADS",
    "MAX_TOKENS",
    "NUM_SHOTS",
    "CHUNKED_PREFILL_SIZE",
    "MAMBA_TRACK_INTERVAL",
)
FLOAT_KEYS = ("TEMPERATURE", "TOP_P")
OPTIONAL_STR_KEYS = ("EXTRA_SERVER_ARGS", "EXTRA_EVAL_ARGS")


def build_config_from_env(env: os._Environ | dict[str, str]) -> dict:
    config: dict[str, object] = {
        "model_label": env["MODEL_LABEL"],
        "model_path": env["MODEL_PATH"],
        "python_bin": env["PYTHON_BIN"],
        "pythonpath_base": env["PYTHONPATH_BASE"],
        "base_url": env["BASE_URL"],
        "dtypes": env["DTYPES"].split(),
        "evals": env["EVALS"].split(),
        "mamba_scheduler_strategy": env["MAMBA_SCHEDULER_STRATEGY"],
        "enable_dtype_probe": env["ENABLE_DTYPE_PROBE"],
        "sglang_skip_sgl_kernel_version_check": env[
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"
        ],
        "local_no_proxy": env["LOCAL_NO_PROXY"],
    }
    for key in INT_KEYS:
        config[key.lower()] = int(env[key])
    for key in FLOAT_KEYS:
        config[key.lower()] = float(env[key])
    for key in OPTIONAL_STR_KEYS:
        config[key.lower()] = env.get(key, "")
    return config


def main() -> int:
    output_dir = Path(os.environ["OUTPUT_DIR"])
    config = build_config_from_env(os.environ)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
