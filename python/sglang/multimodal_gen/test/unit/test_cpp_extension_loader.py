# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

from sglang.kernels.ops.diffusion import load_extension_with_recovery


def test_stale_torch_lock_is_removed_before_loading(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    build_directory.mkdir()
    torch_lock_path = build_directory / "lock"
    torch_lock_path.touch()

    expected = object()
    with (
        patch(
            "sglang.kernels.ops.diffusion.ext.loader._get_build_directory",
            return_value=build_directory,
        ),
        patch("torch.utils.cpp_extension.load", return_value=expected) as load,
    ):
        result = load_extension_with_recovery("test_extension", ["source.cpp"])

    assert result is expected
    assert not torch_lock_path.exists()
    load.assert_called_once_with(
        name="test_extension",
        sources=["source.cpp"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        build_directory=str(build_directory),
        verbose=False,
    )


def test_broken_extension_is_rebuilt_under_the_same_lock(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    build_directory.mkdir()
    expected = object()
    load_error = OSError(f"{build_directory}/test_extension.so: file too short")

    with (
        patch(
            "sglang.kernels.ops.diffusion.ext.loader._get_build_directory",
            return_value=build_directory,
        ),
        patch(
            "torch.utils.cpp_extension.load",
            side_effect=[load_error, expected],
        ) as load,
    ):
        result = load_extension_with_recovery("test_extension", ["source.cpp"])

    assert result is expected
    assert build_directory.is_dir()
    assert load.call_count == 2
