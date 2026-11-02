# SPDX-License-Identifier: Apache-2.0
"""Real-weight, single-GPU end-to-end check for BRIA Fibo 2.0.

Opt-in on purpose: it needs the released checkpoints, which are tens of gigabytes,
so a plain `pytest` run skips it. Point it at a local checkout to run it:

    SGLANG_FIBO2_TEST_MODEL=/models/Fibo-2.0-edit-turbo \
    SGLANG_FIBO2_TEST_DEV_MODEL=/models/Fibo-2.0-dev \
    pytest test/single_test_file/test_bria_fibo2_offline.py -q

What it covers beyond "it does not crash":

* the distilled checkpoint generates at four steps with guidance off, and the
  non-distilled one at thirty steps with guidance 5.0, which is a second forward
  pass on a null caption,
* editing runs the vision side of Qwen3-VL plus per-image reference latents,
* the decoded image has the canvas the pipeline promised, i.e. the shift schedule
  and the VAE round trip agree on the size.

Set `SGLANG_FIBO2_SAVE_DIR` to keep the produced PNGs for eyeballing.
"""

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

MODEL_ID = os.environ.get("SGLANG_FIBO2_TEST_MODEL")
DEV_MODEL_ID = os.environ.get("SGLANG_FIBO2_TEST_DEV_MODEL")
SAVE_DIR = os.environ.get("SGLANG_FIBO2_SAVE_DIR")

# A structured caption without `text_render`: the single reference implementation
# would also draw text hints, which v1 does not support, so both sides are kept on
# the same footing by not using them.
STRUCTURED_CAPTION = (
    '{"short_description":"A red bicycle leaning against a white brick wall '
    'on a sunny morning.","objects":[{"description":"a red city bicycle with a '
    'wicker basket","location":"center"}],"background_setting":"a white brick '
    'wall with a green door","lighting":{"conditions":"bright morning sun",'
    '"direction":"from the right"},"style_medium":"photograph"}'
)

EDIT_INSTRUCTION = "Make the background a snowy forest"

requires_turbo = pytest.mark.skipif(
    not MODEL_ID,
    reason="set SGLANG_FIBO2_TEST_MODEL to a Fibo 2.0 checkpoint to run this test",
)
requires_dev = pytest.mark.skipif(
    not DEV_MODEL_ID,
    reason="set SGLANG_FIBO2_TEST_DEV_MODEL to the Fibo-2.0-dev checkpoint",
)


def _save(image: Image.Image, name: str) -> None:
    if not SAVE_DIR:
        return
    directory = Path(SAVE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / name)


def _generate(model_path: str, request: dict, tmp_path: Path, file_name: str):
    from sglang.multimodal_gen import DiffGenerator

    with DiffGenerator.from_pretrained(
        model_path=model_path,
        num_gpus=1,
        warmup_mode="off",
        text_encoder_cpu_offload=True,
    ) as generator:
        result = generator.generate(
            {
                **request,
                "save_output": True,
                "output_path": str(tmp_path),
                "output_file_name": file_name,
            }
        )
    assert result is not None, "Fibo offline generation returned nothing"
    return Image.open(result.output_file_path).convert("RGB")


@requires_turbo
def test_turbo_text_to_image(tmp_path):
    """Four steps, guidance off: the distilled checkpoint's whole point."""
    image = _generate(
        MODEL_ID,
        {
            "prompt": STRUCTURED_CAPTION,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "height": 1024,
            "width": 1024,
            "seed": 0,
        },
        tmp_path,
        "fibo2_turbo_t2i.png",
    )
    assert image.size == (1024, 1024)
    _save(image, "fibo2_turbo_t2i.png")

    # A blank or single-colour frame is the usual symptom of a broken text path
    colours = image.getcolors(maxcolors=1024 * 1024)
    assert colours is None or len(colours) > 1000, (
        f"the image has only {len(colours)} distinct colours; the prompt likely "
        "did not reach the transformer"
    )


@requires_turbo
def test_turbo_edit(tmp_path):
    """Editing: Qwen3-VL sees the source image and its latents condition the run."""
    source = Image.new("RGB", (768, 512), (30, 90, 200))
    draw = ImageDraw.Draw(source)
    draw.rectangle([0, 256, 767, 511], fill=(40, 160, 60))
    source_path = tmp_path / "fibo2_source.png"
    source.save(source_path)

    image = _generate(
        MODEL_ID,
        {
            "prompt": EDIT_INSTRUCTION,
            "image_path": str(source_path),
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "seed": 0,
        },
        tmp_path,
        "fibo2_turbo_edit.png",
    )
    _save(image, "fibo2_turbo_edit.png")

    # Editing defaults the canvas to the source aspect ratio at about one
    # megapixel, 16-pixel aligned; 768x512 is 3:2.
    assert image.size[0] % 16 == 0 and image.size[1] % 16 == 0
    assert abs(image.size[0] / image.size[1] - 768 / 512) < 0.05
    area = image.size[0] * image.size[1]
    assert 0.9 * 1024 * 1024 <= area <= 1.1 * 1024 * 1024


@requires_dev
def test_dev_text_to_image_with_guidance(tmp_path):
    """Thirty steps with guidance 5.0: the negative pass must not break."""
    image = _generate(
        DEV_MODEL_ID,
        {
            "prompt": STRUCTURED_CAPTION,
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "height": 1024,
            "width": 1024,
            "seed": 0,
        },
        tmp_path,
        "fibo2_dev_t2i.png",
    )
    assert image.size == (1024, 1024)
    _save(image, "fibo2_dev_t2i.png")
