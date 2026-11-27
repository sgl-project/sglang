# SPDX-License-Identifier: Apache-2.0
"""Run explicitly on an H100 with access to the gated Praxis checkpoints.

pytest python/sglang/multimodal_gen/test/server/manual_praxis_http.py
"""

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionServerArgs,
    DiffusionTestCase,
    PRAXIS_DROID_CI_sampling_params,
    PRAXIS_YAM_CI_sampling_params,
)


class TestPraxisHTTP(DiffusionServerBase):
    case = diffusion_case_fixture(
        [
            DiffusionTestCase(
                "praxis_droid_http",
                DiffusionServerArgs(model_path="runwayml/praxis-1-droid-lerobot"),
                PRAXIS_DROID_CI_sampling_params,
                perf_warmup_requests=1,
                run_consistency_check=False,
                run_component_accuracy_check=False,
                run_t2v_input_reference_check=False,
            ),
            DiffusionTestCase(
                "praxis_yam_http",
                DiffusionServerArgs(model_path="runwayml/praxis-1-yam-lerobot"),
                PRAXIS_YAM_CI_sampling_params,
                perf_warmup_requests=1,
                run_consistency_check=False,
                run_component_accuracy_check=False,
                run_t2v_input_reference_check=False,
            ),
        ]
    )
