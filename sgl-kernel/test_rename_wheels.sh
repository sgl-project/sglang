#!/usr/bin/env bash
# Smoke tests for sgl-kernel/rename_wheels.sh
#
# Angles covered:
#   C1  fresh linux_x86_64 wheel → correct rename + METADATA/WHEEL patch
#   C2  fresh linux_aarch64 wheel
#   C3  #20953 bug: filename already has +cu130+manylinux but METADATA is bare (mv-only prior run)
#   C4  different-suffix wheel (cu128 → re-run as cu130) — WHEEL tag must not double up
#   I   idempotency: run every scenario 10 more times, nothing should change
#   E1  no CUDA suffix detected — filename rename only, no version bump
#   E2  empty dist/ — no crash
#   E3  already-correct wheel — nothing changes from run 1
#
# Requirements: bash, python3, python3 -m wheel (pip install wheel)
# Run from repo root:  bash sgl-kernel/test_rename_wheels.sh
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/rename_wheels.sh"

# Resolve a real Python (with 'wheel' package).
# On Windows/Git-Bash the Microsoft Store python3 stub (exit 49) is useless.
_resolve_py() {
    for candidate in \
        /c/Users/nickc/miniconda3/python \
        /c/ProgramData/miniconda3/python \
        /opt/conda/bin/python3 \
        /opt/conda/bin/python \
        python python3; do
        local p
        p="$(command -v "$candidate" 2>/dev/null)" || continue
        [[ -x "$p" ]] || continue
        "$p" -m wheel version >/dev/null 2>&1 && { echo "$p"; return; }
    done
    echo "ERROR: no python with 'wheel' package found" >&2; exit 1
}
PYTHON="$(_resolve_py)"
PYTHON_DIR="$(dirname "$PYTHON")"

# Wrapper so inline `python3 - ...` calls in this script use the right binary.
# (Git-Bash python3 → Windows Store stub, which exits 49 and does nothing.)
python3() { "$PYTHON" "$@"; }
PASS=0; FAIL=0
TMPDIRS=()

cleanup() {
    for d in "${TMPDIRS[@]+"${TMPDIRS[@]}"}"; do rm -rf -- "$d"; done
}
trap cleanup EXIT

# ── colour helpers ────────────────────────────────────────────────────────────
_green(){ printf '\033[32m[PASS]\033[0m %s\n' "$*"; }
_red(){   printf '\033[31m[FAIL]\033[0m %s\n' "$*"; }
header(){ printf '\n\033[1;36m══ %s ══\033[0m\n' "$*"; }

# (( VAR++ )) returns exit 1 when VAR==0, which triggers set -e.
# Use $(( VAR + 1 )) assignment form which is always truthy.
ok()  { _green "$1"; PASS=$(( PASS + 1 )); }
fail(){ _red   "$1 — expected=«$2» got=«$3»"; FAIL=$(( FAIL + 1 )); }

# Use if/else, not `A && ok || fail`: if ok() fails, fail() would also run.
check_eq(){
    local lbl="$1" exp="$2" got="$3"
    if [[ "$exp" == "$got" ]]; then ok "$lbl"; else fail "$lbl" "$exp" "$got"; fi
}
check_contains(){
    local lbl="$1" needle="$2" hay="$3"
    if [[ "$hay" == *"$needle"* ]]; then ok "$lbl"; else fail "$lbl" "*${needle}*" "$hay"; fi
}
check_not_contains(){
    local lbl="$1" needle="$2" hay="$3"
    if [[ "$hay" != *"$needle"* ]]; then ok "$lbl"; else fail "$lbl (should NOT contain '$needle')" "" "$hay"; fi
}

# ── wheel helpers ─────────────────────────────────────────────────────────────
mktmp(){ local d; d=$(mktemp -d); TMPDIRS+=("$d"); echo "$d"; }

# make_wheel DIST_DIR NAME VERSION PTAG ATAG PLTAG [META_VER [WHEEL_PLTAG]]
#   PLTAG      = platform used in the zip filename
#   META_VER   = Version: in METADATA  (default = VERSION)
#   WHEEL_PLTAG= platform written into WHEEL Tag: line (default = PLTAG)
#
# Produces a wheel with correct RECORD hashes so `wheel unpack` accepts it.
make_wheel(){
    local dist="$1" name="$2" ver="$3" ptag="$4" atag="$5" pltag="$6"
    local mver="${7:-$ver}" wpltag="${8:-$pltag}"
    python3 - "$dist" "$name" "$ver" "$ptag" "$atag" "$pltag" "$mver" "$wpltag" <<'PY'
import sys, zipfile, hashlib, base64

dist, name, ver, ptag, atag, pltag, mver, wpltag = sys.argv[1:]
norm = name.replace('-', '_')
whl  = f"{dist}/{norm}-{ver}-{ptag}-{atag}-{pltag}.whl"
di   = f"{norm}-{mver}.dist-info"

def sha256_record(data: bytes) -> str:
    h = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(h).rstrip(b"=").decode()

init_data    = b""
wheel_data   = (
    f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
    f"Tag: {ptag}-{atag}-{wpltag}\n"
).encode()
metadata_data = (
    f"Metadata-Version: 2.1\nName: {norm}\nVersion: {mver}\n"
).encode()

record_lines = [
    f"{norm}/__init__.py,{sha256_record(init_data)},{len(init_data)}",
    f"{di}/WHEEL,{sha256_record(wheel_data)},{len(wheel_data)}",
    f"{di}/METADATA,{sha256_record(metadata_data)},{len(metadata_data)}",
    f"{di}/RECORD,,",
]
record_data = "\n".join(record_lines).encode()

with zipfile.ZipFile(whl, 'w') as z:
    z.writestr(f"{norm}/__init__.py",  init_data.decode())
    z.writestr(f"{di}/WHEEL",          wheel_data.decode())
    z.writestr(f"{di}/METADATA",       metadata_data.decode())
    z.writestr(f"{di}/RECORD",         record_data.decode())
PY
}

meta_version(){
    python3 - "$1" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in z.namelist():
        if n.endswith('.dist-info/METADATA'):
            for ln in z.read(n).decode().splitlines():
                if ln.startswith('Version:'):
                    print(ln.split(':',1)[1].strip()); raise SystemExit
PY
}

wheel_tag_line(){
    # Prints the Tag: line(s) from the WHEEL file inside the wheel archive
    python3 - "$1" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in z.namelist():
        if n.endswith('.dist-info/WHEEL'):
            for ln in z.read(n).decode().splitlines():
                if ln.startswith('Tag:'):
                    print(ln)
PY
}

whl_count(){ ls "$1"/*.whl 2>/dev/null | wc -l | tr -d ' '; }
first_whl(){ ls "$1"/*.whl 2>/dev/null | sort | head -1; }

# Run rename_wheels.sh in isolated tmp root (script hardcodes WHEEL_DIR=dist).
# Pass PYTHON_ROOT_PATH so resolve_python() inside the script skips the Store stub.
run(){
    local root="$1"; shift
    ( cd "$root" && env PYTHON_ROOT_PATH="$PYTHON_DIR" "$@" bash "$SCRIPT" ) >/dev/null 2>&1
}

idempotency_loop(){
    # idempotency_loop LABEL ROOT N SUFFIX_OVERRIDE
    local lbl="$1" root="$2" n="$3" sfx="$4"
    local name0 ver0 tag0
    name0=$(basename "$(first_whl "$root/dist")")
    ver0=$(meta_version "$(first_whl "$root/dist")")
    tag0=$(wheel_tag_line "$(first_whl "$root/dist")")
    for i in $(seq 1 "$n"); do
        run "$root" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE="$sfx"
        local nameN verN tagN
        nameN=$(basename "$(first_whl "$root/dist")")
        verN=$(meta_version "$(first_whl "$root/dist")")
        tagN=$(wheel_tag_line "$(first_whl "$root/dist")")
        check_eq "$lbl run $i: wheel count"        "1"      "$(whl_count "$root/dist")"
        check_eq "$lbl run $i: filename unchanged" "$name0" "$nameN"
        check_eq "$lbl run $i: version unchanged"  "$ver0"  "$verN"
        check_eq "$lbl run $i: WHEEL tag unchanged" "$tag0" "$tagN"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
header "C1  fresh linux_x86_64 wheel"
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel 0.4.0 cp310 abi3 linux_x86_64
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
W=$(first_whl "$R/dist")
check_eq           "C1: wheel count"               "1"                "$(whl_count "$R/dist")"
check_contains     "C1: filename has manylinux2014" manylinux2014      "$(basename "$W")"
check_contains     "C1: filename has +cu130"        +cu130             "$(basename "$W")"
check_not_contains "C1: no bare -linux_ in name"    -linux_            "$(basename "$W")"
check_eq           "C1: METADATA version"          "0.4.0+cu130"      "$(meta_version "$W")"
check_contains     "C1: WHEEL tag manylinux2014"   manylinux2014      "$(wheel_tag_line "$W")"
check_not_contains "C1: WHEEL tag no manymany"      manymanylinux     "$(wheel_tag_line "$W")"

header "C1-I  idempotency: 10 more runs on C1 result"
idempotency_loop "C1-I" "$R" 10 "+cu130"

# ─────────────────────────────────────────────────────────────────────────────
header "C2  fresh linux_aarch64 wheel"
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel 0.4.0 cp310 abi3 linux_aarch64
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
W=$(first_whl "$R/dist")
check_eq           "C2: wheel count"                    "1"                     "$(whl_count "$R/dist")"
check_contains     "C2: filename manylinux2014_aarch64"  manylinux2014_aarch64  "$(basename "$W")"
check_eq           "C2: METADATA version"               "0.4.0+cu130"           "$(meta_version "$W")"
check_contains     "C2: WHEEL tag manylinux2014_aarch64" manylinux2014_aarch64  "$(wheel_tag_line "$W")"

header "C2-I  idempotency: 10 more runs on C2 result"
idempotency_loop "C2-I" "$R" 10 "+cu130"

# ─────────────────────────────────────────────────────────────────────────────
header "C3  interrupted-run scenario (manylinux2014 in name, METADATA still bare)"
# Simulates a script run that completed the filename mv step (linux→manylinux2014)
# but was interrupted before unpack/repack. Filename has manylinux2014 but no
# version suffix yet, and WHEEL tag is still the original linux_x86_64.
# (Note: a wheel where filename version != dist-info version cannot be unpacked
# by `wheel unpack` at all — that state is not recoverable by this script.)
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel "0.4.0" cp310 abi3 manylinux2014_x86_64 \
           "0.4.0" "linux_x86_64"
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
W=$(first_whl "$R/dist")
check_eq           "C3: wheel count"                "1"             "$(whl_count "$R/dist")"
check_eq           "C3: METADATA version"           "0.4.0+cu130"   "$(meta_version "$W")"
check_contains     "C3: filename has +cu130"        "+cu130"        "$(basename "$W")"
check_not_contains "C3: no double +cu130"           "+cu130+cu130"  "$(basename "$W")"
check_not_contains "C3: no manymany in filename"    manymanylinux   "$(basename "$W")"
check_not_contains "C3: WHEEL tag no manymany"      manymanylinux   "$(wheel_tag_line "$W")"
check_contains     "C3: WHEEL tag manylinux2014"    manylinux2014   "$(wheel_tag_line "$W")"

header "C3-I  idempotency: 10 more runs on C3 result"
idempotency_loop "C3-I" "$R" 10 "+cu130"

# ─────────────────────────────────────────────────────────────────────────────
header "C4  suffix-change scenario (wheel was cu128, now re-running as cu130)"
# WHEEL tag and METADATA both reference +cu128; script now runs with +cu130.
# patch_wheel_platform_tags must not double-up the manylinux token.
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel "0.4.0+cu128" cp310 abi3 manylinux2014_x86_64 \
           "0.4.0+cu128" "manylinux2014_x86_64"
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
W=$(first_whl "$R/dist")
check_eq           "C4: METADATA version"           "0.4.0+cu128+cu130" "$(meta_version "$W")"
check_not_contains "C4: WHEEL tag no manymany"       manymanylinux       "$(wheel_tag_line "$W")"
check_contains     "C4: WHEEL tag manylinux2014"     manylinux2014       "$(wheel_tag_line "$W")"

header "C4-I  idempotency: 10 more runs on C4 result"
idempotency_loop "C4-I" "$R" 10 "+cu130"

# ─────────────────────────────────────────────────────────────────────────────
header "E1  no CUDA suffix (SGL_KERNEL_CUDA_SUFFIX_OVERRIDE='')"
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel 0.4.0 cp310 abi3 linux_x86_64
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=""
W=$(first_whl "$R/dist")
check_contains     "E1: manylinux2014 in name"  manylinux2014  "$(basename "$W")"
check_not_contains "E1: no +cu in name"         +cu            "$(basename "$W")"
check_eq           "E1: METADATA version bare"  "0.4.0"        "$(meta_version "$W")"

header "E1-I  idempotency: 10 more runs on E1 result"
idempotency_loop "E1-I" "$R" 10 ""

# ─────────────────────────────────────────────────────────────────────────────
header "E2  empty dist/ — no crash"
R=$(mktmp); mkdir -p "$R/dist"
set +e
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
RC=$?
set -e
check_eq "E2: no crash (exit 0)" "0" "$RC"
check_eq "E2: dist still empty"  "0" "$(whl_count "$R/dist")"

# ─────────────────────────────────────────────────────────────────────────────
header "E3  already-correct wheel — nothing changes"
# Build a fully-correct wheel (as if the script already ran once).
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel "0.4.0+cu130" cp310 abi3 manylinux2014_x86_64 \
           "0.4.0+cu130" "manylinux2014_x86_64"
NAME0=$(basename "$(first_whl "$R/dist")")
VER0=$(meta_version "$(first_whl "$R/dist")")
TAG0=$(wheel_tag_line "$(first_whl "$R/dist")")
idempotency_loop "E3" "$R" 10 "+cu130"

# ─────────────────────────────────────────────────────────────────────────────
printf '\n\033[1m══════════════════════════════════════════════\033[0m\n'
printf '  Total  \033[32m%d passed\033[0m  \033[31m%d failed\033[0m\n' "$PASS" "$FAIL"
printf '\033[1m══════════════════════════════════════════════\033[0m\n'
[[ "$FAIL" -eq 0 ]]
