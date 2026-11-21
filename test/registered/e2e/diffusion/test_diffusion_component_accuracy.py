from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

# ci-cost-override: compatibility bridge preserves the existing diffusion lane.
register_cuda_ci(
    est_time=14400,
    stage="base-b-diffusion-component",
    runner_config="2-gpu-large",
)


if __name__ == "__main__":
    run_diffusion_suite("component-accuracy")
