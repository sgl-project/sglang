"""
End-to-end functional correctness tests for diffusion server.
Verifies functional logic, parameter propagation, and error robustness.
"""

from __future__ import annotations

import pytest
import requests

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import (
    CORRECTNESS_1_GPU_CASES,
    DiffusionTestCase,
)

logger = init_logger(__name__)


class CorrectnessTestMixin:
    """
    Mixin containing functional verification logic shared across GPU suites.
    """

    def test_functional_success(self, case: DiffusionTestCase, diffusion_server):
        """
        Verify that the model generates output successfully for its modality.
        Reuses project-standard polling and validation logic from get_generate_fn.
        """
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )
        
        # functional success check 
        self.run_and_collect(diffusion_server, case.id, generate_fn)
        logger.info(f"Functional success verified for {case.id}")

    def test_seed_determinism(self, case: DiffusionTestCase, diffusion_server):
        """
        Verify bit-identical results for identical seeds (Image only).
        """
        if case.server_args.modality != "image":
            pytest.skip("Seed determinism check restricted to image modality")

        client = self._client(diffusion_server)
        payload = {
            "model": case.server_args.model_path,
            "prompt": case.sampling_params.prompt or "A dog with sunglasses",
            "size": "512x512",
            "response_format": "b64_json",
            "seed": 42,
        }

        resp1 = client.images.generate(**payload)
        resp2 = client.images.generate(**payload)
        
        assert resp1.data[0].b64_json == resp2.data[0].b64_json
        logger.info(f"Seed determinism verified for {case.id}")

    def test_api_error_codes(self, case: DiffusionTestCase, diffusion_server):
        """
        Verify server returns correct HTTP error codes for malformed requests.
        """
        if case.server_args.modality != "image":
            pytest.skip("Error code boundary check restricted to image generations endpoint")

        base_url = f"http://localhost:{diffusion_server.port}/v1/images/generations"
        
        # Test 1: Invalid resolution (400 or 422)
        payload = {
            "model": case.server_args.model_path,
            "prompt": "test",
            "size": "invalid_size"
        }
        resp = requests.post(base_url, json=payload)
        assert resp.status_code in [400, 422]

        # Test 2: Missing mandatory prompt (422)
        payload = {"model": case.server_args.model_path}
        resp = requests.post(base_url, json=payload)
        assert resp.status_code == 422
        logger.info(f"Error handling verified for {case.id}")


class TestDiffusionCorrectness(CorrectnessTestMixin, DiffusionServerBase):
    """
    Functional correctness tests for 1-GPU diffusion cases.
    """

    @pytest.fixture(params=CORRECTNESS_1_GPU_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        return request.param

    def test_functional_correctness(self, case: DiffusionTestCase, diffusion_server):
        """Driver method for 1-GPU functional logic."""
        self.test_functional_success(case, diffusion_server)