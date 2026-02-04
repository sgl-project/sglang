import subprocess

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3600, suite="stage-b-test-large-2-gpu")


def main():
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "-m",
        "pytest",
        "-s",
        "python/sglang/multimodal_gen/test/server/test_accuracy_2_gpu_a.py",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
