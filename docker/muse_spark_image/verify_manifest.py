"""Verify the files an image (or an unpacked tree) carries against muse_image_manifest.json."""

import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/")
bad = []
for group in ("files", "aiter_files"):
    for path, digest in manifest[group].items():
        p = root / path.lstrip("/")
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(path)
if bad:
    sys.exit(f"{len(bad)} files differ from the manifest, e.g. {bad[:5]}")
print(
    f"manifest ok: tree {manifest['tree_commit']}, {len(manifest['files'])} package files, "
    f"{len(manifest['aiter_files'])} aiter files"
)
