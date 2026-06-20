#!/usr/bin/env bash
# Regenerate the Python msgpack codec from the runtime proto (the IDL).
#
# The Rust codec is generated at build time by rust/sglang-grpc/build.rs; this
# script produces the committed Python counterpart. Run it after editing the
# proto, and commit the result.
#
#   proto/generate_msgspec.sh
#
# Requires: protoc, and `python -c "import google.protobuf"`.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
proto="sglang/runtime/v1/sglang.proto"
out="$repo_root/python/sglang/srt/grpc/messages.py"
python_bin="${PYTHON:-python3}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

protoc \
  --proto_path="$here" \
  --include_imports \
  --experimental_allow_proto3_optional \
  --descriptor_set_out="$tmp/sglang.desc" \
  "$proto"

"$python_bin" "$here/gen_msgspec.py" "$tmp/sglang.desc" > "$out"

echo "Wrote $out"
