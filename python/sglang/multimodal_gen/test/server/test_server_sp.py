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
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port

logger = init_logger(__name__)

SP_CASES = [
    DiffusionTestCase(
        id="wan2_2_i2v_a14b",
        model_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        modality="video",
        prompt="generate",  # passing in something since failing if no prompt is passed
        warmup_text=0,  # warmups only for image gen models
        warmup_edit=0,
        output_size="832x1104",
        edit_prompt="generate",
        image_path="https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true",
        custom_validator="video",
        seconds=1,
    ),
    DiffusionTestCase(
        id="wan2_2_t2v_a14b",
        model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        modality="video",
        prompt="A curious raccoon",
        output_size="720x480",
        seconds=4,
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    DiffusionTestCase(
        id="wan2_1_t2v_14b",
        model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        modality="video",
        prompt="A curious raccoon",
        output_size="720x480",
        seconds=4,
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    DiffusionTestCase(
        id="wan2_1_i2v_14b_480P",
        model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        output_size="832x1104",
        modality="video",
        prompt="Animate this image",
        edit_prompt="Add dynamic motion to the scene",
        image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        warmup_text=0,  # warmups only for image gen models
        warmup_edit=0,
        custom_validator="video",
        seconds=1,
    ),
    DiffusionTestCase(
        id="wan2_1_i2v_14b_720P",
        model_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        modality="video",
        prompt="Animate this image",
        edit_prompt="Add dynamic motion to the scene",
        image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        output_size="832x1104",
        warmup_text=0,  # warmups only for image gen models
        warmup_edit=0,
        custom_validator="video",
        seconds=1,
    ),
]


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
    extra_args += " --num-gpus 2 --ulysses-degree 2"

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
