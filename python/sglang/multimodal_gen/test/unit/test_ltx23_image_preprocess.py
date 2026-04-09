from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
)


def test_ltx23_resize_center_crop_matches_official_fill_then_crop() -> None:
    img = Image.new("RGB", (600, 300))
    cropped = LTX2AVDenoisingStage._resize_center_crop(
        img, width=512, height=768
    )

    assert cropped.size == (512, 768)
