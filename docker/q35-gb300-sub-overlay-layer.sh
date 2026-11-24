#!/usr/bin/env bash
# Build one OCI layer (tar.gz) that moves the editable SGLang and sgl-nvidia-mega checkouts of the
# 2026-11-20 nightly (SGLang 03fde9cc, mega 7101de97) to the submission commits, for `crane append`
# onto the nightly's arm64 digest. Equivalent to Dockerfile.q35-gb300-sub's RUN step:
#   - every file the commits add or change, with its git mode (no file is deleted or renamed: checked);
#   - each tree's .git replaced (opaque) by a depth-1 repository at the new commit, no auth header;
#   - /sgl-workspace/sgl-nvidia-mega.commit.
# Untracked build outputs in the image (the prebuilt Rust TreeCore .so) are untouched.
# usage: make_overlay_layer.sh SGLANG_REPO_DIR MEGA_REPO_DIR OUT.tar.gz
set -euo pipefail
SGL_REPO=$1 MEGA_REPO=$2 OUT=$3
SGL_BASE=03fde9cc392a5f84a54a365635ffed0061cdf1b8 SGL_NEW=617bd6014ae3e2a2d234400d2c7371dcc9101796
MEGA_BASE=7101de977e2c25686d79d472661bdf0d9e8324d6 MEGA_NEW=fe8163cc788817ab6dd0f9ea09add97d7ac59004
SGL_URL=https://github.com/sgl-project/sglang MEGA_URL=https://github.com/sgl-project/sgl-nvidia-mega.git
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT

stage_tree() {  # repo base new dest url
    local repo=$1 base=$2 new=$3 dest=$4 url=$5
    if git -C "$repo" diff --name-status --no-renames "$base" "$new" | grep -qv '^[AM]'; then
        echo "unexpected delete/rename between $base and $new" >&2; exit 1
    fi
    mkdir -p "$STAGE/$dest"
    git -C "$repo" diff --raw --no-renames -z "$base" "$new" | python3 -c '
import sys, subprocess, os
repo, new, root = sys.argv[1:4]
parts = sys.stdin.buffer.read().split(b"\0")
for meta, path in zip(parts[0::2], parts[1::2]):
    if not meta: continue
    mode = meta.split()[1].decode(); path = path.decode()
    blob = subprocess.check_output(["git", "-C", repo, "show", f"{new}:{path}"])
    dst = os.path.join(root, path); os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(blob); os.chmod(dst, 0o755 if mode == "100755" else 0o644)
print(f"staged {sum(1 for m in parts[0::2] if m)} files under {root}", file=sys.stderr)
' "$repo" "$new" "$STAGE/$dest"
    # .git: opaque replacement with a depth-1 repository whose HEAD is the new commit.
    local g="$STAGE/$dest/.git"
    git init -q --bare "$g.tmp"
    git -C "$g.tmp" fetch -q --depth 1 "file://$(cd "$repo" && pwd)" "$new"
    git -C "$g.tmp" config core.bare false
    git -C "$g.tmp" config remote.origin.url "$url"
    git -C "$g.tmp" config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
    echo "$new" > "$g.tmp/HEAD"
    GIT_DIR="$g.tmp" git read-tree "$new"
    rm -f "$g.tmp/FETCH_HEAD"
    mv "$g.tmp" "$g"; touch "$g/.wh..wh..opq"
}

stage_tree "$SGL_REPO" "$SGL_BASE" "$SGL_NEW" sgl-workspace/sglang "$SGL_URL"
stage_tree "$MEGA_REPO" "$MEGA_BASE" "$MEGA_NEW" sgl-workspace/sgl-nvidia-mega "$MEGA_URL"
echo "$MEGA_NEW" > "$STAGE/sgl-workspace/sgl-nvidia-mega.commit"
tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-11-20 00:00:00Z' -C "$STAGE" -czf "$OUT" sgl-workspace
ls -la "$OUT"
