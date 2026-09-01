from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

register_cuda_ci(
    est_time=3600,
    stage="base-b",
    runner_config="diffusion-bcg-1-gpu-h100",
)


if __name__ == "__main__":
    run_diffusion_suite("bcg-diffusion")
