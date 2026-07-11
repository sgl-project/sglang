#!/usr/bin/bash
#
# Recompute evalscope-all-nodeps.txt: resolve the full evalscope[all] dependency
# closure and keep only the packages NOT already installed in the current env,
# at evalscope's own resolved versions. Run this after bumping EVALSCOPE_COMMIT
# in sweep.sh (or when the base image's package set changes).
#
# Does NOT install anything (uses pip's --dry-run resolver).
#
set -euo pipefail

DEPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODEPS="${DEPS_DIR}/evalscope-all-nodeps.txt"

# Keep in sync with EVALSCOPE_COMMIT in ../../sweep.sh.
EVALSCOPE_COMMIT="${EVALSCOPE_COMMIT:-acd09b44384d53174768bb1063f675420f76fae9}"
SPEC="evalscope[all] @ git+https://github.com/modelscope/evalscope.git@${EVALSCOPE_COMMIT}"

REPORT="$(mktemp)"
trap 'rm -f "$REPORT"' EXIT

echo "Resolving ${SPEC} (dry-run)..."
pip install --dry-run --quiet --report "$REPORT" "$SPEC"

python3 - "$REPORT" "$NODEPS" "$EVALSCOPE_COMMIT" <<'PY'
import json, os, sys
import importlib.metadata as md

report_path, out_path, commit = sys.argv[1], sys.argv[2], sys.argv[3]
rep = json.load(open(report_path))
installed = {d.metadata["Name"].lower().replace("_", "-") for d in md.distributions()}

# Preserve PEP 508 environment markers ("; platform_machine == ...") attached
# to packages in the existing list. Some deps (notably decord, which only ships
# x86_64 wheels) need a platform marker so the install works on both x86_64 and
# aarch64 sglang images; without this carry-over, regen would wipe the marker.
existing_markers = {}
if os.path.exists(out_path):
    for raw in open(out_path):
        line = raw.strip()
        if not line or line.startswith("#") or ";" not in line:
            continue
        spec, marker = line.split(";", 1)
        pkg = spec.split("==")[0].strip().lower().replace("_", "-")
        existing_markers[pkg] = marker.strip()

missing = []
for it in rep["install"]:
    m = it["metadata"]
    key = m["name"].lower().replace("_", "-")
    if key == "evalscope":          # installed by the sweep itself (PIP_NO_DEPS)
        continue
    if key in installed:            # already shipped by the image -> keep as-is
        continue
    entry = f'{m["name"]}=={m["version"]}'
    if key in existing_markers:
        entry += f'; {existing_markers[key]}'
    missing.append(entry)

missing.sort(key=str.lower)
header = (
    "# evalscope[all] dependency closure, MINUS every package the inference image\n"
    "# already ships. Installed with `pip install --no-deps` so none of these can\n"
    "# drag an sglang pin (numpy/openai/markupsafe/datasets/transformers/...) down.\n"
    f"# Versions are evalscope's own sanctioned picks for commit {commit[:8]}.\n"
    "# Regenerate with scripts/regen_nodeps_list.sh after bumping EVALSCOPE_COMMIT.\n"
)
with open(out_path, "w") as f:
    f.write(header + "\n".join(missing) + "\n")
print(f"Wrote {len(missing)} packages to {out_path}")
PY
