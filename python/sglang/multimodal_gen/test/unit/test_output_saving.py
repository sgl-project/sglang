import numpy as np
import pytest
from PIL import Image

import sglang.multimodal_gen.runtime.entrypoints.utils as output_utils
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import post_process_sample


def _rgb_frame() -> np.ndarray:
    return np.array(
        [
            [[0, 32, 255], [64, 128, 192], [255, 224, 16]],
            [[9, 17, 33], [127, 128, 129], [240, 12, 88]],
        ],
        dtype=np.uint8,
    )


@pytest.mark.parametrize("output_compression", [None, 0, 75])
def test_png_output_saving_preserves_pixels(tmp_path, output_compression):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    frames = post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert output_path.exists()
    np.testing.assert_array_equal(frames[0], frame)
    np.testing.assert_array_equal(np.array(Image.open(output_path)), frame)


@pytest.mark.parametrize(
    ("output_compression", "expected_compress_level"), [(None, 1), (0, 0), (75, 1)]
)
def test_png_output_saving_uses_fast_pillow_path(
    tmp_path, monkeypatch, output_compression, expected_compress_level
):
    frame = _rgb_frame()
    output_path = tmp_path / f"sample_{output_compression}.png"

    def fail_imageio_imwrite(*args, **kwargs):
        raise AssertionError("PNG output should use Pillow's PNG fast path")

    original_save = Image.Image.save
    save_calls = []

    def save_spy(self, fp, format=None, **params):
        save_calls.append((format, params.get("compress_level")))
        return original_save(self, fp, format=format, **params)

    monkeypatch.setattr(output_utils.imageio, "imwrite", fail_imageio_imwrite)
    monkeypatch.setattr(Image.Image, "save", save_spy)

    post_process_sample(
        frame,
        DataType.IMAGE,
        fps=1,
        save_file_path=str(output_path),
        output_compression=output_compression,
    )

    assert save_calls == [("PNG", expected_compress_level)]
