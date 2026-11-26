from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.ci.diffusion_suite_bridge import run_diffusion_suite

# ci-cost-override: compatibility bridge for MAGI-2 4-GPU H100 cases.
register_cuda_ci(
    est_time=14400,
    stage="base-b-diffusion",
    runner_config="4-gpu-h100",
)


if __name__ == "__main__":
    run_diffusion_suite("4-gpu-h100")
