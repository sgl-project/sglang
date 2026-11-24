#!/usr/bin/env bash
# Build one OCI layer (tar.gz) that moves the editable SGLang checkout of lmsysorg/sglang:qwen38-fn-rubin-a89019e
# (/sgl-workspace/sglang at a89019e3) to a later qwen38-fn-rubin commit, for `crane append` onto that image's
# arm64 manifest. Same result as rebuilding the image when the commits only add or change files that need no
# compilation (Python and JIT sources):
#   - every file the commits add or change, with its git mode (deletes, renames and compiled sources are refused);
#   - /sgl-workspace/sglang/.git replaced (opaque) by a depth-1 repository at the new commit, no auth header.
# Untracked build outputs in the image (the prebuilt Rust TreeCore .so, sgl-kernel) are untouched.
# usage: q38fn-gb300-overlay-layer.sh SGLANG_REPO_DIR NEW_COMMIT OUT.tar.gz
set -euo pipefail
SGL_REPO=$1 SGL_NEW=$2 OUT=$3
SGL_BASE=a89019e3c6118be44d820516aaeab3056433e7ea
SGL_URL=https://github.com/sgl-project/sglang
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT

if git -C "$SGL_REPO" diff --name-status --no-renames "$SGL_BASE" "$SGL_NEW" | grep -qv '^[AM]'; then
    echo "unexpected delete/rename between $SGL_BASE and $SGL_NEW" >&2; exit 1
fi
if git -C "$SGL_REPO" diff --name-only "$SGL_BASE" "$SGL_NEW" | grep -qE '^(sgl-kernel|rust|docker)/|(^|/)(setup\.py|pyproject\.toml|CMakeLists\.txt)$'; then
    echo "a compiled or packaging file changed between $SGL_BASE and $SGL_NEW: rebuild the image instead" >&2; exit 1
fi
dest=sgl-workspace/sglang
mkdir -p "$STAGE/$dest"
git -C "$SGL_REPO" diff --raw --no-renames -z "$SGL_BASE" "$SGL_NEW" | python3 -c '
import sys, subprocess, os
repo, new, root = sys.argv[1:4]
parts = sys.stdin.buffer.read().split(b"\0")
n = 0
for meta, path in zip(parts[0::2], parts[1::2]):
    if not meta: continue
    mode = meta.split()[1].decode(); path = path.decode()
    blob = subprocess.check_output(["git", "-C", repo, "show", f"{new}:{path}"])
    dst = os.path.join(root, path); os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(blob); os.chmod(dst, 0o755 if mode == "100755" else 0o644); n += 1
print(f"staged {n} files under {root}", file=sys.stderr)
' "$SGL_REPO" "$SGL_NEW" "$STAGE/$dest"
g="$STAGE/$dest/.git"
git init -q --bare "$g.tmp"
git -C "$g.tmp" fetch -q --depth 1 "file://$(cd "$SGL_REPO" && pwd)" "$SGL_NEW"
git -C "$g.tmp" config core.bare false
git -C "$g.tmp" config remote.origin.url "$SGL_URL"
git -C "$g.tmp" config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
echo "$SGL_NEW" > "$g.tmp/HEAD"
GIT_DIR="$g.tmp" git read-tree "$SGL_NEW"
rm -f "$g.tmp/FETCH_HEAD"
mv "$g.tmp" "$g"; touch "$g/.wh..wh..opq"
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-11-09 00:00:00Z' -C "$STAGE" -czf "$OUT" sgl-workspace
ls -la "$OUT"
