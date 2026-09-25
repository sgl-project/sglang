import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from sglang.srt.utils.cpp_extension_loader import load_extension_with_recovery
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_stale_torch_lock_is_removed_before_loading(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    build_directory.mkdir()
    torch_lock_path = build_directory / "lock"
    torch_lock_path.touch()

    expected = object()
    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
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
        extra_ldflags=None,
        build_directory=str(build_directory),
        with_cuda=None,
        verbose=False,
    )


def test_link_flags_and_cuda_toggle_reach_torch(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
            return_value=build_directory,
        ),
        patch("torch.utils.cpp_extension.load", return_value=object()) as load,
    ):
        load_extension_with_recovery(
            "test_extension",
            ["source.cpp"],
            extra_ldflags=["-lcrypto"],
            with_cuda=False,
        )

    kwargs = load.call_args.kwargs
    assert kwargs["extra_ldflags"] == ["-lcrypto"]
    assert kwargs["with_cuda"] is False


def test_broken_extension_is_rebuilt_under_the_same_lock(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    build_directory.mkdir()
    expected = object()
    load_error = OSError(f"{build_directory}/test_extension.so: file too short")

    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
