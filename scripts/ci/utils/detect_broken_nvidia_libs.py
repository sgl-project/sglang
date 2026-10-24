#!/usr/bin/env python3
"""Print nvidia-* packages whose installed RECORD lists a .so that's gone on disk.

Deterministic filesystem-vs-manifest check used by ci_install_dependency.sh to
self-heal runners where a CUDA lib package's .so payload was deleted (interrupted
force-reinstall, or a node-level disk reclaim) while its dist-info stayed behind.
pip then treats the package as satisfied and won't restore it, so `import torch`
dies at job start with e.g. "libcudnn.so.9: cannot open shared object file".

Why a manifest check instead of catching the import error:
  - reinstalls only the actually-broken package(s) (KBs-MBs), not every nvidia-*
    wheel (several GB);
  - catches libs loaded lazily via dlopen (nvshmem, cufile, ...) that a plain
    `import torch` would not surface until much later;
  - no dependence on the dynamic-loader error string or an import->reinstall loop.

Emits one package name per line (sorted) on stdout; empty output == nothing broken.
"""

import csv
import os
import sys
from importlib import metadata


def _record_paths(dist):
    """RECORD-listed relative paths, read RAW.

    Deliberately NOT dist.files: importlib.metadata's Distribution.files runs the
    RECORD through skip_missing_files, so it silently omits any entry whose file
    is already gone on disk — precisely the entries we need to find. Verified on a
    runner: a package with 8 .so in RECORD but 2 deleted returned only the 6
    survivors via dist.files, so the breakage went undetected. Reading RECORD
    text directly gives the full manifest.
    """
    text = dist.read_text("RECORD")
    if not text:
        return []
    return [row[0] for row in csv.reader(text.splitlines()) if row]


def main() -> int:
    broken = set()
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"] or ""
        except Exception:
            continue
        if not name.lower().startswith("nvidia-"):
            continue
        for rel in _record_paths(dist):
            if ".so" not in rel:
                continue
            # locate_file resolves the RECORD-relative path to its real location.
            if not os.path.exists(dist.locate_file(rel)):
                broken.add(name)
                break
    print("\n".join(sorted(broken)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
