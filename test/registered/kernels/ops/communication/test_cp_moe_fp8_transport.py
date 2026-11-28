"""Production CP FP8 preparation and return collectives on four CUDA ranks."""

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.cp_moe_fp8_transport import run_transport_chain
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=240, stage="extra-b", runner_config="8-gpu-h200")


def test_cp_moe_fp8_transport_chain():
    # Includes changed-input graph replay, rank-order preservation, small-row
    # BF16 fallback, non-CP bypass and real output reduce-scatter.
    run_transport_chain()


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=[4], timeout=600)
