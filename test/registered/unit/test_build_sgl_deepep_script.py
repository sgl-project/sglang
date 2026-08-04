import os
import subprocess
import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "build_sgl_deepep.sh"


def call_bash_function(function: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            "-c",
            f'source "$1"; shift; {function} "$@"',
            "bash",
            str(SCRIPT),
            *args,
        ],
        capture_output=True,
        text=True,
    )


def write_executable(path: Path, body: str):
    path.write_text(f"#!/usr/bin/env bash\n{body}")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("arch", "cuda_major", "branch"),
    [
        ("x86_64", "12", "sgl-deepep-x86"),
        ("x86_64", "13", "sgl-deepep-x86"),
        ("aarch64", "12", "sgl-deepep-cu12-arm"),
        ("aarch64", "13", "sgl-deepep-arm"),
    ],
)
def test_select_deepep_branch(arch: str, cuda_major: str, branch: str):
    result = call_bash_function("select_deepep_branch", arch, cuda_major)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == branch


@pytest.mark.parametrize(
    ("arch", "cuda_major"),
    [("ppc64le", "13"), ("x86_64", "11"), ("aarch64", "14")],
)
def test_select_deepep_branch_rejects_unsupported_platform(arch: str, cuda_major: str):
    result = call_bash_function("select_deepep_branch", arch, cuda_major)

    assert result.returncode != 0
    assert "Unsupported architecture/CUDA combination" in result.stderr


@pytest.mark.parametrize(
    ("nvcc_output", "cuda_major"),
    [
        (
            "Cuda compilation tools, release 12.9, V12.9.86",
            "12",
        ),
        (
            "Cuda compilation tools, release 13.0, V13.0.88",
            "13",
        ),
    ],
)
def test_parse_cuda_major(nvcc_output: str, cuda_major: str):
    result = call_bash_function("parse_cuda_major", nvcc_output)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == cuda_major


def test_parse_cuda_major_rejects_unsupported_or_malformed_version():
    unsupported = call_bash_function(
        "parse_cuda_major", "Cuda compilation tools, release 11.8, V11.8.89"
    )
    malformed = call_bash_function("parse_cuda_major", "nvcc version unavailable")

    assert unsupported.returncode != 0
    assert "Unsupported CUDA toolkit" in unsupported.stderr
    assert malformed.returncode != 0
    assert "Could not parse CUDA toolkit version" in malformed.stderr


def test_resolve_cuda_home_from_nvcc_follows_symlink(tmp_path: Path):
    cuda_home = tmp_path / "cuda"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    write_executable(nvcc, "exit 0\n")
    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    nvcc_link = path_bin / "nvcc"
    nvcc_link.symlink_to(nvcc)

    result = call_bash_function("resolve_cuda_home_from_nvcc", str(nvcc_link))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(cuda_home)


def test_patch_cuda13_cccl_adds_include_once(tmp_path: Path):
    cuda_home = tmp_path / "cuda"
    (cuda_home / "include" / "cccl").mkdir(parents=True)
    setup_py = tmp_path / "setup.py"
    setup_py.write_text(
        "def extension():\n"
        "    include_dirs = ['csrc/']\n"
        "    return include_dirs\n"
    )

    first = call_bash_function("patch_cuda13_cccl", str(setup_py), str(cuda_home))
    second = call_bash_function("patch_cuda13_cccl", str(setup_py), str(cuda_home))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    cccl_line = f"    include_dirs.append('{cuda_home}/include/cccl')"
    assert setup_py.read_text().count(cccl_line) == 1


def test_patch_cuda13_cccl_rejects_missing_cccl_directory(tmp_path: Path):
    setup_py = tmp_path / "setup.py"
    setup_py.write_text("    include_dirs = ['csrc/']\n")

    result = call_bash_function(
        "patch_cuda13_cccl", str(setup_py), str(tmp_path / "cuda")
    )

    assert result.returncode != 0
    assert "CCCL include directory not found" in result.stderr


def test_patch_cuda13_cccl_rejects_changed_setup_layout(tmp_path: Path):
    cuda_home = tmp_path / "cuda"
    (cuda_home / "include" / "cccl").mkdir(parents=True)
    setup_py = tmp_path / "setup.py"
    setup_py.write_text('    include_dirs = ["csrc/"]\n')

    result = call_bash_function("patch_cuda13_cccl", str(setup_py), str(cuda_home))

    assert result.returncode != 0
    assert "Could not find DeepEP include_dirs insertion point" in result.stderr


def make_minimal_deepep_project(project_dir: Path):
    project_dir.mkdir()
    (project_dir / "deep_ep.py").write_text("VALUE = 'built-wheel'\n")
    (project_dir / "setup.py").write_text(
        "import os\n"
        "from setuptools import setup\n"
        "\n"
        "expected = '9.0;10.0;10.3'\n"
        "actual = os.environ.get('TORCH_CUDA_ARCH_LIST')\n"
        "if actual != expected:\n"
        "    raise RuntimeError(f'wrong arch list: {actual!r}')\n"
        "setup(name='deep_ep', version='0.0.0', py_modules=['deep_ep'])\n"
    )


def test_build_deepep_produces_installable_wheel_with_fixed_arch_list(
    tmp_path: Path,
):
    project_dir = tmp_path / "DeepEP"
    output_dir = tmp_path / "wheels"
    install_dir = tmp_path / "installed"
    make_minimal_deepep_project(project_dir)

    result = call_bash_function("build_deepep", str(project_dir), str(output_dir))

    assert result.returncode == 0, result.stderr
    wheels = list(output_dir.glob("deep_ep-*.whl"))
    assert len(wheels) == 1
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheels[0]),
        ],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr
    imported = subprocess.run(
        [sys.executable, "-c", "import deep_ep; print(deep_ep.VALUE)"],
        env={"PYTHONPATH": str(install_dir)},
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip() == "built-wheel"


def test_build_deepep_removes_stale_wheel_from_output(tmp_path: Path):
    project_dir = tmp_path / "DeepEP"
    output_dir = tmp_path / "wheels"
    output_dir.mkdir()
    stale_wheel = output_dir / "deep_ep-9.9.9-py3-none-any.whl"
    stale_wheel.touch()
    make_minimal_deepep_project(project_dir)

    result = call_bash_function("build_deepep", str(project_dir), str(output_dir))

    assert result.returncode == 0, result.stderr
    assert not stale_wheel.exists()
    assert len(list(output_dir.glob("deep_ep-*.whl"))) == 1


def test_build_deepep_propagates_build_failure(tmp_path: Path):
    project_dir = tmp_path / "DeepEP"
    output_dir = tmp_path / "wheels"
    project_dir.mkdir()
    (project_dir / "setup.py").write_text("raise RuntimeError('build failed')\n")

    result = call_bash_function("build_deepep", str(project_dir), str(output_dir))

    assert result.returncode != 0
    assert "build failed" in result.stderr


@pytest.mark.parametrize("wheel_count", [0, 2])
def test_find_single_wheel_rejects_invalid_wheel_count(
    tmp_path: Path, wheel_count: int
):
    for index in range(wheel_count):
        (tmp_path / f"deep_ep-{index}.whl").touch()

    result = call_bash_function("find_single_wheel", str(tmp_path))

    assert result.returncode != 0
    assert f"Expected exactly one DeepEP wheel, found {wheel_count}" in result.stderr


def test_install_apt_packages_accepts_failed_install_when_packages_exist(
    tmp_path: Path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_executable(bin_dir / "apt-get", "exit 1\n")
    write_executable(
        bin_dir / "dpkg",
        "printf 'ii  %s  1.0  all  installed\\n' \"${@: -1}\"\n",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; run_as_root() { "$@"; }; '
            "install_apt_packages package-one package-two",
            "bash",
            str(SCRIPT),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_install_apt_packages_rejects_failed_install_with_missing_package(
    tmp_path: Path,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_executable(bin_dir / "apt-get", "exit 1\n")
    write_executable(
        bin_dir / "dpkg",
        "if [[ \"${@: -1}\" == 'package-two' ]]; then exit 1; fi\n"
        "printf 'ii  %s  1.0  all  installed\\n' \"${@: -1}\"\n",
    )
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; run_as_root() { "$@"; }; '
            "install_apt_packages package-one package-two",
            "bash",
            str(SCRIPT),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Required package package-two is not installed" in result.stderr


def test_install_python_dependencies_uses_selected_python(tmp_path: Path):
    python_log = tmp_path / "python.log"
    fake_python = tmp_path / "python"
    write_executable(
        fake_python,
        'printf "%s\\n" "$*" > "$PYTHON_LOG"\n',
    )
    env = os.environ.copy()
    env["PYTHON_LOG"] = str(python_log)

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; PYTHON_BIN="$2"; install_python_dependencies',
            "bash",
            str(SCRIPT),
            str(fake_python),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert python_log.read_text().strip() == "-m pip install setuptools wheel ninja"


def test_remove_existing_deepep_propagates_uninstall_failure(tmp_path: Path):
    fake_python = tmp_path / "python"
    write_executable(fake_python, "exit 37\n")

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; PYTHON_BIN="$2"; remove_existing_deepep',
            "bash",
            str(SCRIPT),
            str(fake_python),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 37


def test_cli_rejects_more_than_one_output_directory():
    result = subprocess.run(
        ["bash", str(SCRIPT), "first-output", "second-output"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Usage: build_sgl_deepep.sh [OUTPUT_DIR]" in result.stderr


def test_cleanup_build_root_removes_registered_temporary_directory(tmp_path: Path):
    build_root = tmp_path / "sgl-deepep.build"
    build_root.mkdir()
    (build_root / "source.cu").touch()

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; DEEPEP_BUILD_ROOT="$2"; cleanup_build_root',
            "bash",
            str(SCRIPT),
            str(build_root),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not build_root.exists()


def test_build_and_report_ends_with_only_the_wheel_path(tmp_path: Path):
    project_dir = tmp_path / "DeepEP"
    output_dir = tmp_path / "wheels"
    make_minimal_deepep_project(project_dir)

    result = call_bash_function("build_and_report", str(project_dir), str(output_dir))

    assert result.returncode == 0, result.stderr
    wheel = next(output_dir.glob("deep_ep-*.whl"))
    assert result.stdout.splitlines()[-1] == f"DeepEP wheel: {wheel}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
