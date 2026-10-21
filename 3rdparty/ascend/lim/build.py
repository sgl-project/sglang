#!/usr/bin/env python3
"""Prepare pinned LIM sources, build a local CANN package and its torch binding.

--prepare-only verifies/stages sources without importing torch or using an NPU.
All generated files and package installation remain under --work-dir.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_vendor() -> dict:
    lock = json.loads((HERE / "upstream.json").read_text())
    actual = {
        p.relative_to(HERE / "vendor").as_posix()
        for p in (HERE / "vendor").rglob("*")
        if p.is_file()
    }
    if actual != set(lock["files"]):
        raise RuntimeError("Vendored file inventory differs from upstream.json")
    for name, expected in lock["files"].items():
        path = HERE / "vendor" / name
        if (
            path.stat().st_size != expected["bytes"]
            or digest(path) != expected["sha256"]
        ):
            raise RuntimeError(f"Vendored source differs from upstream.json: {name}")
    return lock


def prepare(
    work: pathlib.Path, archive: pathlib.Path | None, lock: dict
) -> pathlib.Path:
    if archive is None:
        archive = work / "upstream.tar.gz"
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            request = urllib.request.Request(
                lock["archive_url"], headers={"User-Agent": "sglang-lim-build"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            partial.replace(archive)
    if digest(archive) != lock["archive_sha256"]:
        raise RuntimeError("Upstream archive SHA256 does not match upstream.json")

    source = work / f"upstream-{lock['commit']}"
    marker = source / "sglang-source.json"
    if source.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != lock:
            raise RuntimeError(
                "Work directory has a different source revision; use a fresh one"
            )
        for name, entry in lock["files"].items():
            if digest(source / name) != entry["sha256"]:
                raise RuntimeError(f"Prepared operator source was modified: {name}")
        return source

    source.mkdir()
    expected_root = f"vllm-ascend-{lock['commit']}"
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            parts = pathlib.PurePosixPath(member.name).parts
            if not parts or parts[0] != expected_root or ".." in parts:
                raise RuntimeError(f"Unexpected archive member: {member.name}")
            relative = pathlib.PurePosixPath(*parts[1:])
            if not parts[1:] or parts[1] != "csrc":
                continue
            if not member.isdir() and not member.isfile():
                raise RuntimeError(f"Unsupported archive link: {member.name}")
            target = source / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as content, target.open("wb") as output:
                    shutil.copyfileobj(content, output)
                target.chmod(member.mode & 0o777)
    for name, entry in lock["files"].items():
        # Verify both provenance and the local source actually used for the build.
        target = source / name
        if digest(target) != entry["sha256"]:
            raise RuntimeError(f"Archive and vendored source disagree: {name}")
        shutil.copyfile(HERE / "vendor" / name, target)
    marker.write_text(json.dumps(lock, indent=2) + "\n")
    return source


def build_package(source: pathlib.Path, work: pathlib.Path, args) -> pathlib.Path:
    if sys.platform != "linux":
        raise RuntimeError(
            "CANN compilation requires Linux; --prepare-only works elsewhere"
        )
    if not os.environ.get("ASCEND_HOME_PATH"):
        raise RuntimeError("Source the CANN environment to set ASCEND_HOME_PATH first")
    command = [
        "bash",
        "build.sh",
        "--pkg",
        "--soc=ascend910_93",
        "--ops=fused_lightning_indexer_manage",
        "--vendor_name=sglang_lim",
        f"-j{args.jobs}",
        "-O2",
    ]
    if args.cann_third_party:
        command.append(f"--cann_3rd_lib_path={args.cann_third_party.resolve()}")
    subprocess.run(command, cwd=source / "csrc", check=True)
    packages = list((source / "csrc/build").glob("*.run"))
    if len(packages) != 1:
        raise RuntimeError(f"Expected one CANN run package, found {packages}")
    install = work / "install/opp"
    install.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["bash", str(packages[0]), "--quiet", f"--install-path={install}"], check=True
    )
    libraries = list(install.rglob("op_api/lib/libcust_opapi.so"))
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one installed custom op API, found {libraries}")
    return libraries[0].parents[2]


def build_binding(
    work: pathlib.Path, vendor: pathlib.Path, jobs: int, lock: dict
) -> pathlib.Path:
    if sys.platform != "linux" or not os.environ.get("ASCEND_HOME_PATH"):
        raise RuntimeError(
            "Binding compilation requires Linux and the sourced CANN environment"
        )
    opapi = vendor / "op_api/lib/libcust_opapi.so"
    if not opapi.is_file():
        raise RuntimeError(f"Missing vendor op_api/lib/libcust_opapi.so under {vendor}")
    api = ctypes.CDLL(str(opapi))
    for name in (
        "aclnnFusedLightningIndexerManageGetWorkspaceSize",
        "aclnnFusedLightningIndexerManage",
    ):
        getattr(api, name)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + (
        ":" + os.environ["ASCEND_CUSTOM_OPP_PATH"]
        if os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        else ""
    )
    os.environ.setdefault("MAX_JOBS", str(jobs))
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    cann = pathlib.Path(os.environ["ASCEND_HOME_PATH"])
    npu = pathlib.Path(torch_npu.__file__).resolve().parent
    build = work / "binding"
    build.mkdir(exist_ok=True)
    source = HERE / "vendor/csrc"
    library = load(
        name="sglang_npu_lim",
        sources=[
            str(HERE / "binding.cpp"),
            str(source / "aclnn_torch_adapter/NPUBridge.cpp"),
            str(source / "aclnn_torch_adapter/NPUStorageImpl.cpp"),
        ],
        extra_include_paths=[str(source), str(npu / "include"), str(cann / "include")],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[
            f"-L{npu / 'lib'}",
            f"-L{cann / 'lib64'}",
            f"-Wl,-rpath,{npu / 'lib'}",
            f"-Wl,-rpath,{cann / 'lib64'}",
            "-ltorch_npu",
            "-lascendcl",
            "-ldl",
        ],
        build_directory=str(build),
        is_python_module=False,
        verbose=True,
    )
    schema = torch.ops._C_ascend.npu_fused_lightning_indexer_manage.default._schema
    if len(schema.arguments) != 19 or len(schema.returns) != 0:
        raise RuntimeError(f"Unexpected LIM schema: {schema}")
    result = work / "install/lib/sglang_npu_lim.so"
    result.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(library, result)
    (work / "binding-build.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "library": str(result),
                "library_sha256": digest(result),
                "vendor": str(vendor),
                "vendor_opapi_sha256": digest(opapi),
                "binding_source_upstream_commit": lock["commit"],
                "schema": str(schema),
                "note": "Schema registration check only; NPU execution is a separate check.",
            },
            indent=2,
        )
        + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--archive", type=pathlib.Path, help="Offline pinned upstream tar.gz"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--vendor-path",
        type=pathlib.Path,
        help="Use an existing LIM CANN vendor; build only the torch binding",
    )
    parser.add_argument("--cann-third-party", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1 or (args.prepare_only and args.vendor_path):
        parser.error(
            "--jobs must be positive; --prepare-only and --vendor-path are exclusive"
        )
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = verify_vendor()
    if args.vendor_path:
        vendor = args.vendor_path.resolve()
    else:
        source = prepare(work, args.archive.resolve() if args.archive else None, lock)
        print(
            f"Verified {len(lock['files'])} vendored files; staged {source}", flush=True
        )
        if args.prepare_only:
            return
        vendor = build_package(source, work, args)
    library = build_binding(work, vendor, args.jobs, lock)
    env = work / "env.sh"
    env.write_text(
        "# Generated by 3rdparty/ascend/lim/build.py; source before launching SGLang.\n"
        f"export ASCEND_CUSTOM_OPP_PATH={shlex.quote(str(vendor))}${{ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}}\n"
        f"export LD_LIBRARY_PATH={shlex.quote(str(vendor / 'op_api/lib'))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n"
        f"export SGLANG_NPU_LIM_LIBRARY={shlex.quote(str(library))}\n"
    )
    print(f"Built {library}\nSource {env} before launching SGLang.")


if __name__ == "__main__":
    main()
