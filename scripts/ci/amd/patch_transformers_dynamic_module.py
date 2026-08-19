#!/usr/bin/env python3
"""Hot patch: transformers dynamic_module_utils symlink bug (v5.12.1).

_compute_local_source_files_hash calls Path(...).resolve() on custom-code
module files, following the HF-cache snapshots/<hash>/x.py -> blobs/<blob>
symlink. trust_remote_code models whose custom code uses relative imports
(e.g. Kimi-K2.6's kimi_k25_vision_processing.py: `from .media_utils import`)
then crash with FileNotFoundError: .../blobs/<name>.py at processor init.
Mirrors upstream transformers PR #46618 (merged, not yet released): drop the
.resolve() on the module file and its relative-import sources so the snapshot
.py names (not the blob targets) are used. Self-skips once transformers ships
the fix; fails the build loudly if the pattern is present but unpatched.
"""

import pathlib

import transformers.dynamic_module_utils as m

MARKS = ["Path(resolved_module_file).resolve()", "Path(source_file).resolve()"]
path = pathlib.Path(m.__file__)
src = path.read_text()
if not any(mark in src for mark in MARKS):
    print("transformers dynamic_module_utils already fixed; no patch needed")
else:
    patched = src.replace(
        "Path(resolved_module_file).resolve()", "Path(resolved_module_file)"
    ).replace("Path(source_file).resolve()", "Path(source_file)")
    assert patched != src, "FATAL: transformers symlink patch matched nothing"
    path.write_text(patched)
    print("patched transformers dynamic_module_utils.py (symlink hash fix)")
