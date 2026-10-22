import sys

import pytest

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.ppu_pr_utils import generate, prompt_ids, request, server

register_ppu_ci(est_time=60, suite="stage-a-test-1-gpu-ppu")


def test_http_startup_health_generation():
    with server(startup_timeout=90) as base:
        assert request(base, "GET", "/health").status_code == 200
        result = generate(base, prompt_ids(128))
        assert isinstance(result["text"], str)
        assert 0 < result["meta_info"]["completion_tokens"] <= 16


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
