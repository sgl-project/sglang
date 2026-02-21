import subprocess

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2400, suite="stage-b-test-large-1-gpu")


def main():
    cmd = [
        "python3",
        "-m",
        "pytest",
        "-s",
        "python/sglang/multimodal_gen/test/server/test_accuracy_1_gpu_b.py",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
