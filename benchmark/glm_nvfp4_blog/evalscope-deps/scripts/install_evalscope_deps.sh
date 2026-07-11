#!/usr/bin/bash
#
# Pre-install evalscope[all]'s dependencies WITHOUT downgrading any package the
# inference image already ships, so that `PIP_NO_DEPS=1 ./sweep.sh` works (the
# sweep then installs evalscope itself with no deps).
#
# How it works (see ../README.md for the full rationale):
#   1. Freeze the current env (minus editable/local installs) into a temporary
#      constraints file -> nothing already installed may change version.
#   2. `pip install --no-deps` only the curated missing-package list, so no
#      transitive resolution can pull an installed package down.
#
set -euo pipefail

DEPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODEPS="${DEPS_DIR}/evalscope-all-nodeps.txt"

if [[ ! -s "$NODEPS" ]]; then
    echo "Missing package list: $NODEPS" >&2
    exit 1
fi

CONSTRAINTS="$(mktemp)"
REQS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS" "$REQS"' EXIT

# Pin every currently-installed dist to its exact version (constraint, so the
# install can't change anything), AND filter the package list down to entries
# that are not already installed. The filter makes this robust to image
# updates: if a newer image now ships a package the list pinned to an older
# version, we skip it and keep the image's version instead of erroring on the
# (correctly) downgrade-blocking constraint.
python3 - "$CONSTRAINTS" "$NODEPS" "$REQS" <<'PY'
import importlib.metadata as md, json, sys

constraints_path, nodeps_path, reqs_path = sys.argv[1], sys.argv[2], sys.argv[3]

def is_local(dist):
    try:
        txt = dist.read_text("direct_url.json")
        if txt:
            d = json.loads(txt)
            return bool(d.get("dir_info", {}).get("editable")
                        or d.get("url", "").startswith("file://"))
    except Exception:
        pass
    return False

installed = {}      # normalized name -> version
seen, constraints = set(), []
for d in md.distributions():
    name = d.metadata["Name"]
    key = name.lower().replace("_", "-")
    installed[key] = d.version
    if key in seen or is_local(d):
        continue
    seen.add(key)
    constraints.append(f"{name}=={d.version}")
with open(constraints_path, "w") as f:
    f.write("\n".join(sorted(constraints, key=str.lower)) + "\n")

keep, skipped = [], []
for raw in open(nodeps_path):
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    pkg = line.split("==")[0].strip()
    key = pkg.lower().replace("_", "-")
    if key in installed:
        skipped.append(f"{pkg} (image has {installed[key]}, list pinned {line.split('==')[-1]})")
    else:
        keep.append(line)
with open(reqs_path, "w") as f:
    f.write("\n".join(keep) + "\n")

if skipped:
    print(f"Skipping {len(skipped)} package(s) already shipped by this image "
          "(keeping the image's version):")
    for s in skipped:
        print(f"  - {s}")
print(f"Installing {len(keep)} missing package(s) --no-deps.")
PY

pip install --no-deps -c "$CONSTRAINTS" -r "$REQS"

echo
echo "Done. evalscope[all] deps are present and no installed package changed."
echo "Expected, harmless 'pip check' mismatches (app/rag/aigc/opencompass extras"
echo "that 'evalscope perf' never imports): gradio/datasets/diffusers/ms-opencompass."
