import hashlib
import json
import platform
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest
from sglang_wheel_artifact import (
    EXPECTED_EXTENSIONS,
    MANIFEST_NAME,
    _git_revision,
    _runtime_tuple,
    verify_artifact,
)


def _write_artifact(
    artifact_dir: Path,
    repo_root: Path,
    *,
    python_tag: str | None = None,
    abi_tag: str | None = None,
    platform_tag: str | None = None,
    extension_basename: str | None = None,
) -> Path:
    interpreter_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine().lower().replace("-", "_").replace(".", "_")
    python_tag = python_tag or interpreter_tag
    abi_tag = abi_tag or interpreter_tag
    platform_tag = platform_tag or f"linux_{machine}"
    wheel = artifact_dir / (f"sglang-0.0.0-{python_tag}-{abi_tag}-{platform_tag}.whl")
    extension_basename = (
        extension_basename or f"_core{sysconfig.get_config_var('EXT_SUFFIX')}"
    )
    members = {
        package: f"sglang/srt/{package}/{extension_basename}"
        for package in EXPECTED_EXTENSIONS
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for member in members.values():
            archive.writestr(member, b"test extension placeholder")

    manifest = {
        "schema": 1,
        "commit": _git_revision(repo_root),
        **_runtime_tuple(repo_root),
        "compiler": {
            "toolchain": _runtime_tuple(repo_root)["rust_channel"],
            "rustc": "rustc test identity",
            "cargo": "cargo test identity",
        },
        "max_glibc": "2.35",
        "required_glibc": "2.17",
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "extensions": members,
    }
    (artifact_dir / MANIFEST_NAME).write_text(json.dumps(manifest))
    return wheel


def test_verify_accepts_matching_artifact(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wheel = _write_artifact(tmp_path, repo_root)
    assert verify_artifact(tmp_path, repo_root) == wheel.resolve()


def test_verify_rejects_different_commit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _write_artifact(tmp_path, repo_root)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["commit"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="commit"):
        verify_artifact(tmp_path, repo_root)


def test_verify_rejects_modified_wheel(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    wheel = _write_artifact(tmp_path, repo_root)
    wheel.write_bytes(wheel.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_artifact(tmp_path, repo_root)


def test_verify_rejects_wheel_path_escape(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _write_artifact(tmp_path, repo_root)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["wheel"] = "../outside.whl"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="unsafe filename"):
        verify_artifact(tmp_path, repo_root)


@pytest.mark.parametrize(
    "tag_override",
    ({"python_tag": "cp399"}, {"abi_tag": "cp399"}),
)
def test_verify_rejects_wrong_python_or_abi_tag(
    tmp_path: Path, tag_override: dict[str, str]
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _write_artifact(tmp_path, repo_root, **tag_override)
    with pytest.raises(RuntimeError, match="wheel tags"):
        verify_artifact(tmp_path, repo_root)


def test_verify_rejects_wrong_architecture_tag(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _write_artifact(tmp_path, repo_root, platform_tag="linux_aarch64")
    with pytest.raises(RuntimeError, match="wheel tags"):
        verify_artifact(tmp_path, repo_root)


def test_verify_rejects_core_prefix_lookalike(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    _write_artifact(
        tmp_path,
        repo_root,
        extension_basename=f"_core_extra{extension_suffix}",
    )
    with pytest.raises(RuntimeError, match="expected only"):
        verify_artifact(tmp_path, repo_root)


def test_verify_rejects_newer_glibc_requirement(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _write_artifact(tmp_path, repo_root)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["required_glibc"] = "999.0"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="requires glibc"):
        verify_artifact(tmp_path, repo_root)


def test_verify_rejects_unverified_compiler_identity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _write_artifact(tmp_path, repo_root)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["compiler"] = {"toolchain": "different", "rustc": "", "cargo": ""}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="verified compiler identity"):
        verify_artifact(tmp_path, repo_root)
