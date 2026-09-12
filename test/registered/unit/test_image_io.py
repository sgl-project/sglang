# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_save_base64_image_to_path():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "sglang"
        / "multimodal_gen"
        / "runtime"
        / "utils"
        / "image_io.py"
    )
    spec = importlib.util.spec_from_file_location("image_io", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.save_base64_image_to_path


def test_save_base64_image_to_path_rejects_invalid_payload(tmp_path):
    save_base64_image_to_path = _load_save_base64_image_to_path()

    with pytest.raises(ValueError, match="Failed to decode base64 image"):
        save_base64_image_to_path("data:image/png;base64,@@@@", str(tmp_path / "image"))

    assert not (tmp_path / "image.png").exists()


def test_save_base64_image_to_path_writes_valid_payload(tmp_path):
    save_base64_image_to_path = _load_save_base64_image_to_path()

    saved_path = save_base64_image_to_path(
        "data:image/png;base64,aW1hZ2U=",
        str(tmp_path / "image"),
    )

    assert saved_path == str(tmp_path / "image.png")
    assert Path(saved_path).read_bytes() == b"image"
