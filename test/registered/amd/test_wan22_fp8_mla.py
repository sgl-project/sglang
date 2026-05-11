"""AMD test for Wan2.2-T2V-A14B with FP8 MLA attention (1-GPU and 8-GPU)."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("sgl_kernel", reason="sgl_kernel is required for FP8 MLA tests")

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionServerArgs,
    DiffusionTestCase,
    T2V_sampling_params,
)
from sglang.test.ci.ci_register import register_amd_ci

logger = logging.getLogger(__name__)

register_amd_ci(est_time=3600, suite="nightly-amd-fp8-mla-diffusion", nightly=True)

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
FP8_ENV = {"SGLANG_DIFFUSION_AITER_FP8_ATTN": "1"}

FP8_MLA_CASES = [
    DiffusionTestCase(
        "wan2_2_t2v_a14b_fp8_mla_1gpu",
        DiffusionServerArgs(
            model_path=MODEL,
            modality="video",
            num_gpus=1,
            extras=["--enable-torch-compile false"],
            env_vars=FP8_ENV,
        ),
        T2V_sampling_params,
        run_perf_check=False,
        run_consistency_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
    DiffusionTestCase(
        "wan2_2_t2v_a14b_fp8_mla_1gpu_compile",
        DiffusionServerArgs(
            model_path=MODEL,
            modality="video",
            num_gpus=1,
            extras=["--enable-torch-compile true"],
            env_vars=FP8_ENV,
        ),
        T2V_sampling_params,
        run_perf_check=False,
        run_consistency_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
    DiffusionTestCase(
        "wan2_2_t2v_a14b_fp8_mla_8gpu",
        DiffusionServerArgs(
            model_path=MODEL,
            modality="video",
            num_gpus=8,
            ulysses_degree=4,
            cfg_parallel=True,
            extras=["--enable-torch-compile false"],
            env_vars=FP8_ENV,
        ),
        T2V_sampling_params,
        run_perf_check=False,
        run_consistency_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
    DiffusionTestCase(
        "wan2_2_t2v_a14b_fp8_mla_8gpu_compile",
        DiffusionServerArgs(
            model_path=MODEL,
            modality="video",
            num_gpus=8,
            ulysses_degree=4,
            cfg_parallel=True,
            extras=["--enable-torch-compile true"],
            env_vars=FP8_ENV,
        ),
        T2V_sampling_params,
        run_perf_check=False,
        run_consistency_check=False,
        run_models_api_check=False,
        run_t2v_input_reference_check=False,
    ),
]


class TestWan22FP8MLA(DiffusionServerBase):
    """AMD test for FP8 MLA attention on Wan2.2-T2V-A14B."""

    @classmethod
    def teardown_class(cls):
        try:
            super().teardown_class()
        except AttributeError:
            pass

    @pytest.fixture(params=FP8_MLA_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        return request.param

    def test_diffusion_generation(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )

        perf_record, content = self.run_and_collect(
            diffusion_server, case.id, generate_fn
        )

        assert len(content) > 0, "FP8 MLA generation produced empty output"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
