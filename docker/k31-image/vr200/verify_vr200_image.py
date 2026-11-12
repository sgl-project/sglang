# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""CPU checks for the image built by docker/k31-vr200.Dockerfile (no GPU needed).

  python3 verify_vr200_image.py                 all checks below; exit 1 on any failure
  python3 verify_vr200_image.py includes PLUGIN_DIR HOST_INCLUDE_DIR
                                                only the include check, on any two directories

Checks:
  packages  every installed distribution and version equals served_packages.txt (the package set
            of the image the K3.1 VR200 points ran), nothing missing, nothing extra;
  parts     the FlashInfer DynBlock parts of install_k31_parts.sh are in place: the installed header
            has DynBlockKernelMaxNumExperts = 1024 and the sha256 its marker records, and the sm107a
            jit-cache fused_moe_trtllm_sm107 module carries that marker;
  sources   `sglang` and `sgl_nvidia_mega` import from the trees this image copied in;
  includes  every `#include <sgl_kernel/...>` in the plugin's CUDA sources exists in the host tree's
            JIT include directory, unless it is the fallback branch of an
            `#if __has_include(<X>)` whose X exists (the plugin compiles these at run time, so a
            missing header fails the first megakernel call, not the build).
"""

import hashlib
import importlib.metadata as md
import json
import re
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
SGLANG_TREE = Path("/sgl-workspace/sglang")
PLUGIN_TREE = Path("/sgl-workspace/sgl-nvidia-mega")
HOST_INCLUDE = SGLANG_TREE / "python/sglang/kernels/jit/include"

_INC = re.compile(r'^\s*#\s*include\s*<(sgl_kernel/[^>]+)>')
_IF_HAS = re.compile(r'^\s*#\s*if\s+__has_include\s*\(\s*<([^>]+)>\s*\)\s*$')
_IF = re.compile(r'^\s*#\s*if')
_ELSE = re.compile(r'^\s*#\s*else\b')
_ENDIF = re.compile(r'^\s*#\s*endif\b')


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_packages(errors: list) -> None:
    # (name, version) pairs, not a name -> version map: the served image itself carries two pip
    # dist-info directories, and both must stay.
    want = set()
    for line in (HERE / "served_packages.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==", 1)
            want.add((_norm(name), version))
    have = set()
    for dist in md.distributions():
        name = dist.metadata["Name"]
        if name:
            have.add((_norm(name), dist.version))
    for name, version in sorted(want - have):
        errors.append(f"packages: served {name}=={version} is not installed")
    for name, version in sorted(have - want):
        errors.append(f"packages: {name}=={version} is installed but not in the served set")


def check_parts(errors: list) -> None:
    site = Path(sysconfig.get_paths()["purelib"])
    header = (
        site / "flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
    )
    text = header.read_text()
    if "DynBlockKernelMaxNumExperts = 1024;" not in text:
        errors.append(f"parts: {header} is not patched to 1024")
    mods = list(
        site.glob(
            "flashinfer_jit_cache/providers/sm107a/jit_cache/fused_moe_trtllm_sm107/k31-dynblock.json"
        )
    )
    if not mods:
        errors.append("parts: no k31-dynblock.json next to the sm107a fused_moe_trtllm_sm107 module")
    for marker_path in mods:
        marker = json.loads(marker_path.read_text())
        if marker.get("header_sha256") != hashlib.sha256(header.read_bytes()).hexdigest():
            errors.append(f"parts: {marker_path} records another header than the installed one")
        if marker.get("module") != "fused_moe_trtllm_sm107" or marker.get("arch") != "10.7a":
            errors.append(f"parts: {marker_path} is not an sm107 DynBlock marker: {marker}")


def check_sources(errors: list) -> None:
    import importlib.util

    for module, root in (("sglang", SGLANG_TREE), ("sgl_nvidia_mega", PLUGIN_TREE)):
        spec = importlib.util.find_spec(module)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None or root.resolve() not in origin.parents:
            errors.append(f"sources: {module} resolves to {origin}, not under {root}")


def check_includes(plugin_dir: Path, host_include: Path, errors: list) -> int:
    """Returns the number of include lines checked."""
    checked = 0
    for src in sorted(plugin_dir.rglob("*")):
        if src.suffix not in (".cuh", ".cu", ".h", ".hpp", ".cpp") or not src.is_file():
            continue
        # Frames of the #if stack: (has_include target or None, in_else).
        stack = []
        for lineno, line in enumerate(src.read_text(errors="replace").splitlines(), 1):
            m = _IF_HAS.match(line)
            if m:
                stack.append([m.group(1), False])
                continue
            if _IF.match(line):
                stack.append([None, False])
                continue
            if _ELSE.match(line) and stack:
                stack[-1][1] = True
                continue
            if _ENDIF.match(line) and stack:
                stack.pop()
                continue
            m = _INC.match(line)
            if not m:
                continue
            checked += 1
            if (host_include / m.group(1)).is_file():
                continue
            covered = any(
                target is not None and in_else and (host_include / target).is_file()
                for target, in_else in stack
            )
            if not covered:
                errors.append(
                    f"includes: {src.relative_to(plugin_dir)}:{lineno} <{m.group(1)}> is not in {host_include}"
                )
    return checked


def main() -> None:
    errors: list = []
    if len(sys.argv) == 4 and sys.argv[1] == "includes":
        n = check_includes(Path(sys.argv[2]), Path(sys.argv[3]), errors)
        print(f"includes: {n} sgl_kernel include lines checked")
    elif len(sys.argv) == 1:
        check_packages(errors)
        check_parts(errors)
        check_sources(errors)
        n = check_includes(PLUGIN_TREE, HOST_INCLUDE, errors)
        print(f"includes: {n} sgl_kernel include lines checked")
        if n == 0:
            # The K3.1 plugin's vendored top-k includes sgl_kernel headers; none means another tree.
            errors.append(f"includes: no sgl_kernel include in {PLUGIN_TREE}; not the K3.1 plugin?")
    else:
        sys.exit(__doc__)
    for e in errors:
        print("FAIL", e)
    if errors:
        sys.exit(1)
    print("K31_VR200_IMAGE_OK")


if __name__ == "__main__":
    main()
