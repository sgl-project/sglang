"""Demangling of kernel symbols for reports (CUPTI reports mangled names,
kineto demangled ones); falls back to the raw name when no demangler exists."""

from __future__ import annotations

import shutil
import subprocess
from typing import Iterable


def demangle(names: Iterable[str]) -> dict[str, str]:
    mangled = sorted({n for n in names if n.startswith("_Z")})
    out = {n: n for n in mangled}
    if not mangled:
        return out
    try:
        import cxxfilt

        for name in mangled:
            try:
                out[name] = cxxfilt.demangle(name)
            except cxxfilt.InvalidName:
                pass
        return out
    except ImportError:
        pass
    tool = shutil.which("c++filt")
    if tool is None:
        return out
    result = subprocess.run(
        [tool], input="\n".join(mangled), capture_output=True, text=True, check=False
    )
    lines = result.stdout.splitlines()
    if result.returncode == 0 and len(lines) == len(mangled):
        out.update(zip(mangled, lines))
    return out
