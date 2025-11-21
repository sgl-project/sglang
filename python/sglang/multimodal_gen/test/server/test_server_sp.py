"""
SP Performance tests (A14B models) with --num-gpus 2 --ulysses-degree 2.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# Import the base class to reuse logic
from sglang.multimodal_gen.test.server.test_server_performance import (
    TestDiffusionPerformance,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerManager,
    WarmupRunner,
    download_image_from_url,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    SP_CASES,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port

logger = init_logger(__name__)


@pytest.fixture(params=SP_CASES, ids=lambda c: c.id)
def case(request) -> DiffusionTestCase:
    """Provide a DiffusionTestCase for each test."""
    return request.param


@pytest.fixture
def diffusion_server(case: DiffusionTestCase):
    """Start a diffusion server for a single case and tear it down afterwards."""
    default_port = get_dynamic_server_port()
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", default_port))

    # Append the required arguments
    extra_args = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")
    extra_args += f" --num-gpus {case.num_gpus} --ulysses-degree {case.num_gpus}"

    # start server
    manager = ServerManager(
        model=case.model_path,
        port=port,
        wait_deadline=float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200")),
        extra_args=extra_args,
    )
    ctx = manager.start()

    try:
        warmup = WarmupRunner(
            port=ctx.port,
            model=case.model_path,
            prompt=case.prompt or "A colorful raccoon icon",
            output_size=case.output_size,
        )
        warmup.run_text_warmups(case.warmup_text)

        if case.warmup_edit > 0 and case.edit_prompt and case.image_path:
            # Handle URL or local path
            image_path = case.image_path
            if case.is_image_url():
                image_path = download_image_from_url(str(case.image_path))
            else:
                image_path = Path(case.image_path)

            warmup.run_edit_warmups(
                count=case.warmup_edit,
                edit_prompt=case.edit_prompt,
                image_path=image_path,
            )
    except Exception as exc:
        logger.error("Warm-up failed for %s: %s", case.id, exc)
        ctx.cleanup()
        raise

    try:
        yield ctx
    finally:
        ctx.cleanup()


class TestDiffusionSP(TestDiffusionPerformance):
    """Performance tests for SP cases."""

    pass
