#!/usr/bin/env python3
"""Point torch's recorded Triton requirement at the Triton that is installed.

The ROCm torch 2.11 wheel hard-requires the triton-rocm distribution, while
AITER ships its Triton as plain `triton`; AITER's install_triton.sh uninstalls
triton-rocm before installing it. Neither of the two states that leaves behind
is usable:

  - triton-rocm removed, torch's metadata untouched: triton-rocm exists only on
    PyTorch's ROCm index, so any later resolution of torch fails with
    ResolutionImpossible. CI does exactly that when it installs the SGLang
    extras, which pin torch==2.11.0.
  - triton-rocm left installed alongside: both distributions own
    triton/__init__.py, so the one installed last wins on disk while pip keeps
    reporting the other. That silently drops `import triton` to 3.6.0 even
    though pip (and any metadata-based version check) still says 3.7.

So rewrite the one requirement line to name the Triton actually present.

Match any pinned triton / triton-rocm, not just the wheel's original
triton-rocm==3.6.0: the image build already rewrote that line, so a
version-specific pattern would no-op on the CI rebuild path this also runs on,
leaving torch naming a Triton that installer just replaced.

Both docker/rocm.Dockerfile and scripts/ci/amd/amd_ci_install_dependency.sh run
this. Self-skips on flavors whose torch never pinned triton/triton-rocm.
"""

import importlib.metadata as metadata
import pathlib
import re
import sys

PATTERN = (
    r"^Requires-Dist: (?P<name>triton|triton-rocm)==(?P<version>[^\s;]+)"
    r"(?P<marker>\s*;.*)?$"
)


def main(argv: list[str]) -> int:
    path = pathlib.Path(metadata.distribution("torch")._path) / "METADATA"
    source = path.read_text()

    match = re.search(PATTERN, source, flags=re.MULTILINE)
    if match is None:
        print(f"{path} does not pin triton/triton-rocm; nothing to rewrite")
        return 0

    triton_version = metadata.version("triton")
    if match.group("name") == "triton" and match.group("version") == triton_version:
        print(f"torch already requires triton=={triton_version}")
        return 0

    updated, count = re.subn(
        PATTERN,
        lambda match: f"Requires-Dist: triton=={triton_version}{match.group('marker') or ''}",
        source,
        flags=re.MULTILINE,
    )
    assert count == 1, f"FATAL: {path} pins triton/triton-rocm {count} times"
    path.write_text(updated)
    print(f"torch now requires triton=={triton_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
