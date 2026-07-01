"""
Temporary diagnostic for Wan H100 torch_sdpa drift.
"""

from __future__ import annotations

from dataclasses import replace

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.gpu_cases import ONE_GPU_CASES
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


def _wan_sdpa_case(*, generator_device: str | None = None):
    base_case = next(case for case in ONE_GPU_CASES if case.id == "wan2_1_t2v_1.3b")
    sampling_params = base_case.sampling_params
    if generator_device is not None:
        sampling_params = replace(
            sampling_params,
            extras={
                **sampling_params.extras,
                "generator_device": generator_device,
            },
        )
    return replace(
        base_case,
        server_args=replace(
            base_case.server_args,
            extras=[
                *base_case.server_args.extras,
                "--attention-backend",
                "torch_sdpa",
            ],
        ),
        sampling_params=sampling_params,
        run_perf_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    )


class TestWanSdpaDiagnostic(DiffusionServerBase):
    case = diffusion_case_fixture([_wan_sdpa_case()])


class TestWanSdpaCpuGeneratorDiagnostic(DiffusionServerBase):
    case = diffusion_case_fixture([_wan_sdpa_case(generator_device="cpu")])
