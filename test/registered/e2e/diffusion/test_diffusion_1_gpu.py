from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

# ci-cost-override: compatibility bridge preserves the existing diffusion lane.
register_cuda_ci(
    est_time=14400,
    stage="base-b",
    runner_config="diffusion-1-gpu-h100",
)


if __name__ == "__main__":
    run_diffusion_suite("1-gpu")
