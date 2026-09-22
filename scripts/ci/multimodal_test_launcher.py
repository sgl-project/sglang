"""Launch multimodal pytest specs with their required distributed workers."""

import shlex
import subprocess
import sys
from pathlib import PurePosixPath

# Most multi-GPU tests launch their own workers. Only these files expect the
# pytest process itself to run under torchrun.
_TORCHRUN_PROCESSES = {
    "python/sglang/multimodal_gen/test/single_test_file/component_accuracy/test_component_accuracy_2_gpu.py": 2,
    "python/sglang/multimodal_gen/test/unit/test_qwen_image21_distributed.py": 2,
}


def torchrun_processes(test_spec: str) -> int:
    test_file = shlex.split(test_spec)[0].split("::", 1)[0]
    return _TORCHRUN_PROCESSES.get(str(PurePosixPath(test_file)), 1)


def build_pytest_command(test_spec: str, python: str = "python3") -> list[str]:
    command = [python]
    num_processes = torchrun_processes(test_spec)
    if num_processes > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={num_processes}",
            ]
        )
    return [*command, "-m", "pytest", *shlex.split(test_spec), "-x"]


if __name__ == "__main__":
    command = build_pytest_command(sys.argv[1], python=sys.executable)
    print(f"Running command: {shlex.join(command)}", flush=True)
    sys.exit(subprocess.call(command))
