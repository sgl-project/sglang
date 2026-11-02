"""Convert CUDA sources with sailify before they are handed to hgcc.

hgcc cannot read CUDA: identifiers, API names and header paths first have to
be rewritten by sailify. Two properties of the converter shape everything in
this module:

* ``sailify_extra_files_recursive`` converts **in place** — its per-file
  output path is ``join(output_directory, relpath(input, output_directory))``,
  which normalizes back onto the input itself; only the generated
  ``.ppu_compat/`` headers actually land in ``output_directory``. The kernel
  sources in the tree must never be rewritten, so conversion runs on a
  mirror-copy of the sources, made under the JIT cache root.
* Conversion also rewrites every transitively ``#include``d header it can
  discover, so the mirror has to cover the whole include closure, and the
  digest that keys the mirror has to cover it too — otherwise a header edit
  would be invisible to both the mirror and, transitively, the build cache.

The mirror preserves each file's absolute path relative to ``/``, so relative
includes keep resolving inside it exactly as they did in the original tree.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import re
import shutil
import uuid
from typing import Dict, List, Optional, Set, Tuple

from sglang.kernels.jit.utils.common import cache_once
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Mirrors sailify's `_RE_INCLUDE` / `DEFAULT_EXTENSIONS`: the closure below
# must contain exactly the files sailify itself would discover and convert.
_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+[<"]([^>"]+)[>"]', re.MULTILINE)
_SOURCE_EXTENSIONS = (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".inl")

_KEY_HEX_LEN = 16


@cache_once
def sailify_version() -> str:
    """Installed sailify version, or 'unknown' (never raises)."""
    try:
        from importlib.metadata import version as dist_version

        return dist_version("sailify")
    except Exception:
        pass
    try:
        from sailify.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


@cache_once
def _ppu_sdk_version() -> str:
    """PPU SDK version as sailify detects it, or 'unknown' (never raises).

    The SDK version is baked into the generated ``compatible_wrapper.h``, so
    it is part of the mirror digest.
    """
    try:
        from sailify.ppu_sdk_version import (
            detect_ppu_sdk_version,
            format_version_string,
        )

        return format_version_string(detect_ppu_sdk_version())
    except Exception:
        return "unknown"


def _sailify_root() -> pathlib.Path:
    configured = envs.SGLANG_JIT_CACHE_DIR.get() or "~/.cache/sglang/jit"
    return pathlib.Path(configured).expanduser() / "sailified"


def _mirror_path(root: pathlib.Path, source: str) -> pathlib.Path:
    """The mirrored location of an absolute *source* path under *root*."""
    path = pathlib.Path(source)
    return root / path.relative_to(path.anchor)


def _discover_closure(
    cuda_files: List[str], include_dirs: List[str]
) -> Tuple[Set[str], Dict[str, Optional[str]]]:
    """Every file sailify would convert, with the base each resolved against.

    Replicates sailify's own discovery: an include is tried against the
    including file's directory first, then each include dir in order; the
    first candidate that exists on disk wins, whether or not it is a source
    extension. The base is ``None`` for includes resolved against the
    including file's directory (the mirror reproduces those relative
    relationships on its own) and the include dir otherwise (whose mirrored
    location has to reach the compile line for the converted header to win
    over the original).
    """
    visited: Set[str] = set()
    bases: Dict[str, Optional[str]] = {}
    queue: List[str] = []
    for filepath in cuda_files:
        filepath = os.path.abspath(filepath)
        if filepath not in visited:
            visited.add(filepath)
            bases[filepath] = None
            queue.append(filepath)

    while queue:
        filepath = queue.pop(0)
        try:
            source = pathlib.Path(filepath).read_text(errors="replace")
        except OSError:
            continue
        file_dir = os.path.dirname(filepath)
        for match in _INCLUDE_RE.finditer(source):
            inc = match.group(1)
            candidates = [(os.path.join(file_dir, inc), None)] + [
                (os.path.join(inc_dir, inc), os.path.abspath(inc_dir))
                for inc_dir in include_dirs
            ]
            for candidate, base in candidates:
                candidate = os.path.abspath(candidate)
                if candidate in visited:
                    break
                if os.path.isfile(candidate):
                    if os.path.splitext(candidate)[1] in _SOURCE_EXTENSIONS:
                        visited.add(candidate)
                        bases[candidate] = base
                        queue.append(candidate)
                    break
    return visited, bases


def _mirror_digest(closure: Set[str], include_dirs: List[str]) -> str:
    # Imported lazily: cache imports this module for the environment
    # fingerprint, so a top-level import would be circular.
    from sglang.kernels.jit.utils.compile import cache

    digest = hashlib.sha256()
    for part in (sailify_version(), _ppu_sdk_version(), tuple(include_dirs)):
        digest.update(repr(part).encode())
        digest.update(b"\0")
    for path in sorted(closure):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update((cache._file_digest(pathlib.Path(path)) or "").encode())
        digest.update(b"\0")
    return digest.hexdigest()[:_KEY_HEX_LEN]


def convert_cuda_sources(
    cuda_files: List[str], include_dirs: List[str]
) -> Tuple[List[str], List[str]]:
    """Convert .cu sources with sailify into a content-addressed dir.

    Returns (converted_cuda_files, extra_cuda_cflags) where extra flags add
    the mirrored include dirs, the generated .ppu_compat include dir, and a
    force-include of compatible_wrapper.h.
    """
    try:
        from sailify.sailify_python import sailify_extra_files_recursive
    except ImportError as error:
        raise RuntimeError(
            "JIT builds on SAIL require the sailify converter, which is not "
            "installed. Install it with `pip install -e cuda_free/sailify` from "
            "the sglang repository."
        ) from error

    closure, bases = _discover_closure(cuda_files, include_dirs)
    outdir = _sailify_root() / _mirror_digest(closure, include_dirs)
    converted = [str(_mirror_path(outdir, os.path.abspath(f))) for f in cuda_files]

    compat_header = outdir / ".ppu_compat" / "compatible_wrapper.h"
    if not (
        outdir.is_dir()
        and compat_header.is_file()
        and all(pathlib.Path(path).is_file() for path in converted)
    ):
        tmp = outdir.parent / f"{outdir.name}.tmp-{uuid.uuid4().hex}"
        try:
            mirror: Dict[str, str] = {}
            for source in sorted(closure):
                target = _mirror_path(tmp, source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                mirror[source] = str(target)
            sailify_extra_files_recursive(
                output_directory=str(tmp),
                extra_files=[mirror[source] for source in sorted(closure)],
                header_include_dirs=[],
            )
            outdir.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(tmp, outdir)
            except OSError:
                # A concurrent process published the identical content first.
                if not outdir.is_dir():
                    raise
        except Exception as error:
            raise RuntimeError(
                f"sailify failed to convert CUDA sources for hgcc: {error}"
            ) from error
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    missing = [path for path in converted if not pathlib.Path(path).is_file()]
    if missing:
        raise RuntimeError(
            "sailify did not produce the expected converted sources: "
            + ", ".join(missing)
        )
    for source, path in zip(cuda_files, converted):
        src_ext = os.path.splitext(source)[1]
        out_ext = os.path.splitext(path)[1]
        assert src_ext == out_ext, (
            f"sailify changed the extension of {source}: converted to {path}"
        )

    mirrored_include_dirs = [
        f"-I{_mirror_path(outdir, base)}"
        for base in dict.fromkeys(bases.values())
        if base is not None
    ]
    flags = mirrored_include_dirs + [
        f"-I{outdir / '.ppu_compat'}",
        "-include",
        "compatible_wrapper.h",
    ]
    return converted, flags
