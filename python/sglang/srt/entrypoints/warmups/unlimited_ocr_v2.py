"""Unlimited-OCR v2 startup warmup."""

import base64
import io
import logging
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import GenerateReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


def _drawn_page(title: str) -> str:
    """A white page with a title and numbered lines, as a PNG data URL."""
    image = Image.new("RGB", (980, 1400), "white")
    draw = ImageDraw.Draw(image)
    # Default font: a sized one needs Pillow 10.1, which the pin does not require.
    draw.text((90, 90), title, fill="black")
    for i in range(8):
        draw.text((90, 220 + i * 60), f"{i + 1}. {title} line {i + 1}.", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


async def unlimited_ocr_v2_warmup(
    disaggregation_mode: str, tokenizer_manager: "TokenizerManager"
):
    """Run one drawn gundam page through the model; opt in with --warmups.

    Moves the first image request's one-time graph compiles to startup.
    """
    req = GenerateReqInput(
        text="<PAGE><image>Multi page parsing.",
        image_data=[_drawn_page("Warmup")],
        images_config={"image_mode": "gundam"},
        sampling_params={"max_new_tokens": 8, "temperature": 0},
    )
    if disaggregation_mode != "null":
        req.bootstrap_room = 0
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
    async for _ in tokenizer_manager.generate_request(req, None):
        pass
    logger.info("Unlimited-OCR v2 warmup done.")
