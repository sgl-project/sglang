"""Check the exact FlashInfer package inputs or patched outputs."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("package", type=Path)
parser.add_argument("stage", choices=("before", "after"))
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
errors = []
for name, hashes in manifest["files"].items():
    path = args.package / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if actual != hashes[args.stage]:
        errors.append(name)
if errors:
    parser.exit(1, "FlashInfer package mismatch: " + ", ".join(errors) + "\n")
print(f"Verified {len(manifest['files'])} {args.stage} files")
