"""Generate peer-L1 KVCM bindings from the PR #44460 proto.

Run with grpcio-tools==1.62.0 and protobuf==4.25.3. --check verifies that the
committed bindings are reproducible without changing files. Source revision:
af2f07fbd3b15cce4ce3cfd6e5e318d52f4c78cf.
"""

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "sgl-router/sgl-kv-indexer/proto"
DEST = ROOT / "python/sglang/srt/mem_cache/l1_transfer/generated"
STEM = "tair_kvcache_meta_service"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, expected in (("grpcio-tools", "1.62.0"), ("protobuf", "4.25.3")):
        if importlib.metadata.version(name) != expected:
            raise SystemExit(f"Generation requires {name}=={expected}")
    with tempfile.TemporaryDirectory() as temp:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"-I{PROTO}",
                f"--python_out={temp}",
                f"--grpc_python_out={temp}",
                str(PROTO / f"{STEM}.proto"),
            ],
            check=True,
        )
        for filename in (f"{STEM}_pb2.py", f"{STEM}_pb2_grpc.py"):
            text = (Path(temp) / filename).read_text()
            if filename.endswith("_grpc.py"):
                old = f"import {STEM}_pb2 as "
                if text.count(old) != 1:
                    raise SystemExit("Unexpected generated protobuf import")
                text = text.replace(old, f"from . import {STEM}_pb2 as ")
            text = "# fmt: off\n# isort: skip_file\n# ruff: noqa\n" + text
            path = DEST / filename
            if args.check:
                if not path.exists() or path.read_text() != text:
                    raise SystemExit(f"Bindings differ: {path}")
            else:
                DEST.mkdir(parents=True, exist_ok=True)
                path.write_text(text)


if __name__ == "__main__":
    main()
