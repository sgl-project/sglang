import subprocess
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2400, suite="stage-b-test-1-gpu-large")


def main():
    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        "python3",
        "-m",
        "pytest",
        "-s",
        "-v",
        "python/sglang/multimodal_gen/test/server/test_accuracy_1_gpu_a.py",
    ]
    raise SystemExit(subprocess.call(cmd, cwd=repo_root))


if __name__ == "__main__":
    main()
