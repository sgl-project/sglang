# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RaiseNotImplementedAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        raise NotImplementedError(f"The {option_string} option is not yet implemented")


def launch_distributed(
    num_gpus: int, args: list[str], master_port: int | None = None
) -> int:
    """
    Launch a distributed job with the given arguments

    Args:
        num_gpus: Number of GPUs to use
        args: Arguments to pass to v1_sgl_diffusion_inference.py (defaults to sys.argv[1:])
        master_port: Port for the master process (default: random)
    """

    current_env = os.environ.copy()
    python_executable = sys.executable
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    main_script = os.path.join(
        project_root, "sgl_diffusion/sample/v1_sgl_diffusion_inference.py"
    )

    cmd = [
        python_executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
    ]

    if master_port is not None:
        cmd.append(f"--master_port={master_port}")

    cmd.append(main_script)
    cmd.extend(args)

    logger.info("Running inference with %d GPU(s)", num_gpus)
    logger.info("Launching command: %s", " ".join(cmd))

    current_env["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        cmd,
        env=current_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line.strip())

    return process.wait()
