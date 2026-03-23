#!/usr/bin/env bash
# Supplemental smoke tests for rename_wheels.sh
#
# Scenarios:
#   M1  multiple wheels in dist/ simultaneously (x86_64 + aarch64)
#   M2  verify real repacked wheel METADATA/WHEEL tag matches filename
#       (only runs if REAL_WHEEL env var points to the downloaded .whl)
#   V1  version with .post1 suffix
#   V2  version with .dev20251204 suffix (like rocm build wheels)
#   V3  version with multiple dots (0.4.1.post2)
#
# Requires: bash, python3 with 'wheel' package
# Run from sgl-kernel/:  bash test_rename_wheels_extra.sh
# With real wheel:       REAL_WHEEL=/tmp/sglang_kernel-0.4.0+cu130-...whl bash test_rename_wheels_extra.sh
set -euo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/rename_wheels.sh"
PASS=0; FAIL=0
TMPDIRS=()

cleanup() { for d in "${TMPDIRS[@]+"${TMPDIRS[@]}"}"; do rm -rf -- "$d"; done; }
trap cleanup EXIT

_green(){ printf '\033[32m[PASS]\033[0m %s\n' "$*"; }
_red(){   printf '\033[31m[FAIL]\033[0m %s\n' "$*"; }
header(){ printf '\n\033[1;36m══ %s ══\033[0m\n' "$*"; }

ok()  { _green "$1"; PASS=$(( PASS + 1 )); }
fail(){ _red   "$1 — expected=«$2» got=«$3»"; FAIL=$(( FAIL + 1 )); }

check_eq(){
    if [[ "$2" == "$3" ]]; then ok "$1"; else fail "$1" "$2" "$3"; fi
}
check_contains(){
    if [[ "$3" == *"$2"* ]]; then ok "$1"; else fail "$1" "*$2*" "$3"; fi
}
check_not_contains(){
    if [[ "$3" != *"$2"* ]]; then ok "$1"; else fail "$1 (must NOT contain $2)" "" "$3"; fi
}

_resolve_py() {
    for c in /c/Users/nickc/miniconda3/python /c/ProgramData/miniconda3/python \
              /opt/conda/bin/python3 /opt/conda/bin/python python python3; do
        local p; p="$(command -v "$c" 2>/dev/null)" || continue
        [[ -x "$p" ]] || continue
        "$p" -m wheel version >/dev/null 2>&1 && { echo "$p"; return; }
    done
    echo "ERROR: no python with 'wheel'" >&2; exit 1
}
PYTHON="$(_resolve_py)"
PYTHON_DIR="$(dirname "$PYTHON")"
python3() { "$PYTHON" "$@"; }

mktmp(){ local d; d=$(mktemp -d); TMPDIRS+=("$d"); echo "$d"; }

make_wheel(){
    local dist="$1" name="$2" ver="$3" ptag="$4" atag="$5" pltag="$6"
    local mver="${7:-$ver}" wpltag="${8:-$pltag}"
    python3 - "$dist" "$name" "$ver" "$ptag" "$atag" "$pltag" "$mver" "$wpltag" <<'PY'
import sys, zipfile, hashlib, base64
dist, name, ver, ptag, atag, pltag, mver, wpltag = sys.argv[1:]
norm = name.replace('-', '_')
whl  = f"{dist}/{norm}-{ver}-{ptag}-{atag}-{pltag}.whl"
di   = f"{norm}-{mver}.dist-info"
def sha256r(data):
    h = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(h).rstrip(b"=").decode()
init_d    = b""
wheel_d   = (f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
             f"Tag: {ptag}-{atag}-{wpltag}\n").encode()
meta_d    = (f"Metadata-Version: 2.1\nName: {norm}\nVersion: {mver}\n").encode()
record_d  = "\n".join([
    f"{norm}/__init__.py,{sha256r(init_d)},{len(init_d)}",
    f"{di}/WHEEL,{sha256r(wheel_d)},{len(wheel_d)}",
    f"{di}/METADATA,{sha256r(meta_d)},{len(meta_d)}",
    f"{di}/RECORD,,"]).encode()
with zipfile.ZipFile(whl, 'w') as z:
    z.writestr(f"{norm}/__init__.py",  init_d.decode())
    z.writestr(f"{di}/WHEEL",          wheel_d.decode())
    z.writestr(f"{di}/METADATA",       meta_d.decode())
    z.writestr(f"{di}/RECORD",         record_d.decode())
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
    python3 - "$1" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in z.namelist():
        if n.endswith('.dist-info/WHEEL'):
            for ln in z.read(n).decode().splitlines():
                if ln.startswith('Tag:'): print(ln)
PY
}

whl_count(){ ls "$1"/*.whl 2>/dev/null | wc -l | tr -d ' '; }
first_whl(){ ls "$1"/*.whl 2>/dev/null | sort | head -1; }
all_whls(){ ls "$1"/*.whl 2>/dev/null | sort; }

run(){
    local root="$1"; shift
    ( cd "$root" && env PYTHON_ROOT_PATH="$PYTHON_DIR" "$@" bash "$SCRIPT" ) >/dev/null 2>&1
}

idempotency_loop(){
    local lbl="$1" root="$2" n="$3" sfx="$4"
    local snap0; snap0=$(ls "$root/dist/"*.whl 2>/dev/null | sort | xargs -I{} bash -c 'echo "$(basename {}):$(python3 - {} <<PY
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in z.namelist():
        if n.endswith("/METADATA"):
            for l in z.read(n).decode().splitlines():
                if l.startswith("Version:"): print(l.split(":",1)[1].strip()); raise SystemExit
PY
)"')
    for i in $(seq 1 "$n"); do
        run "$root" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE="$sfx"
        local snapN; snapN=$(ls "$root/dist/"*.whl 2>/dev/null | sort | xargs -I{} bash -c 'echo "$(basename {}):$(python3 - {} <<PY
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in z.namelist():
        if n.endswith("/METADATA"):
            for l in z.read(n).decode().splitlines():
                if l.startswith("Version:"): print(l.split(":",1)[1].strip()); raise SystemExit
PY
)"')
        check_eq "$lbl run $i: snapshot unchanged" "$snap0" "$snapN"
        local cnt; cnt=$(whl_count "$root/dist")
        check_eq "$lbl run $i: wheel count stable" "${snap0_count:-$cnt}" "$cnt"
        snap0_count="$cnt"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
header "M1  multiple wheels in dist/ simultaneously (x86_64 + aarch64)"
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel 0.4.0 cp310 abi3 linux_x86_64
make_wheel "$R/dist" sglang_kernel 0.4.0 cp311 abi3 linux_aarch64
check_eq "M1 setup: two wheels in dist" "2" "$(whl_count "$R/dist")"

run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130

check_eq           "M1: still two wheels after run"   "2"   "$(whl_count "$R/dist")"
for W in $(all_whls "$R/dist"); do
    BN="$(basename "$W")"
    check_contains     "M1: $BN has +cu130"       "+cu130"       "$BN"
    check_contains     "M1: $BN has manylinux2014" "manylinux2014" "$BN"
    check_not_contains "M1: $BN no manymany"       "manymany"     "$BN"
    check_eq           "M1: $BN METADATA version"  \
        "$(echo "$BN" | python3 -c "import sys,re; m=re.search(r'-([^-]+)-cp',sys.stdin.read()); print(m.group(1))")" \
        "$(meta_version "$W")"
done

header "M1-I  idempotency x10 with two wheels"
N0=$(whl_count "$R/dist")
for i in $(seq 1 10); do
    run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
    check_eq "M1-I run $i: wheel count" "$N0" "$(whl_count "$R/dist")"
    for W in $(all_whls "$R/dist"); do
        check_not_contains "M1-I run $i: $(basename $W) no manymany" "manymany" "$(basename "$W")"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
header "M2  verify real repacked wheel METADATA + WHEEL tag"
REAL_WHEEL="${REAL_WHEEL:-}"
if [[ -z "$REAL_WHEEL" ]]; then
    # Auto-detect from sgl-kernel/dist/ if available
    CANDIDATES=(dist/sglang_kernel-0.4.0+cu130-*.whl)
    [[ -f "${CANDIDATES[0]}" ]] && REAL_WHEEL="${CANDIDATES[0]}"
fi
if [[ -n "$REAL_WHEEL" && -f "$REAL_WHEEL" ]]; then
    BN="$(basename "$REAL_WHEEL")"
    VER_IN_NAME="$(echo "$BN" | python3 -c "import sys,re; m=re.search(r'sglang_kernel-([^-]+)-', sys.stdin.read()); print(m.group(1))")"
    VER_IN_META="$(meta_version "$REAL_WHEEL")"
    TAG_LINE="$(wheel_tag_line "$REAL_WHEEL")"
    check_eq           "M2: METADATA version matches filename"  "$VER_IN_NAME"   "$VER_IN_META"
    check_contains     "M2: WHEEL tag has manylinux2014"         "manylinux2014"  "$TAG_LINE"
    check_not_contains "M2: WHEEL tag no manymany"               "manymany"       "$TAG_LINE"
    check_not_contains "M2: filename no manymany"                "manymany"       "$BN"
    check_not_contains "M2: filename no bare -linux_"            "-linux_"        "$BN"
    echo "    (wheel: $BN)"
else
    printf '\033[33m[SKIP]\033[0m M2: set REAL_WHEEL=/path/to/repacked.whl to enable\n'
fi

# ─────────────────────────────────────────────────────────────────────────────
header "V1  version with .post1 suffix"
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel 0.4.0.post1 cp310 abi3 linux_x86_64
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
W=$(first_whl "$R/dist")
check_eq           "V1: METADATA version"      "0.4.0.post1+cu130" "$(meta_version "$W")"
check_contains     "V1: filename has +cu130"   "+cu130"            "$(basename "$W")"
check_not_contains "V1: no manymany"           "manymany"          "$(basename "$W")"

header "V1-I  idempotency x10"
for i in $(seq 1 10); do
    run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
    check_eq "V1-I run $i: version stable" "0.4.0.post1+cu130" "$(meta_version "$(first_whl "$R/dist")")"
    check_not_contains "V1-I run $i: no manymany" "manymany" "$(basename "$(first_whl "$R/dist")")"
done

# ─────────────────────────────────────────────────────────────────────────────
header "V2  version with .dev date suffix (like ROCm build wheels)"
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel "0.4.0.dev20251204" cp310 abi3 linux_x86_64
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
W=$(first_whl "$R/dist")
check_eq           "V2: METADATA version"      "0.4.0.dev20251204+cu130" "$(meta_version "$W")"
check_contains     "V2: filename has +cu130"   "+cu130"                  "$(basename "$W")"
check_not_contains "V2: no manymany"           "manymany"                "$(basename "$W")"

header "V2-I  idempotency x10"
for i in $(seq 1 10); do
    run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu130
    check_eq "V2-I run $i: version stable" "0.4.0.dev20251204+cu130" "$(meta_version "$(first_whl "$R/dist")")"
    check_not_contains "V2-I run $i: no manymany" "manymany" "$(basename "$(first_whl "$R/dist")")"
done

# ─────────────────────────────────────────────────────────────────────────────
header "V3  version with multiple dots (0.4.1.post2)"
R=$(mktmp); mkdir -p "$R/dist"
make_wheel "$R/dist" sglang_kernel "0.4.1.post2" cp310 abi3 linux_x86_64
run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu128
W=$(first_whl "$R/dist")
check_eq           "V3: METADATA version"       "0.4.1.post2+cu128" "$(meta_version "$W")"
check_contains     "V3: filename has +cu128"    "+cu128"             "$(basename "$W")"
check_not_contains "V3: no manymany"            "manymany"           "$(basename "$W")"

header "V3-I  idempotency x10"
for i in $(seq 1 10); do
    run "$R" SGL_KERNEL_CUDA_SUFFIX_OVERRIDE=+cu128
    check_eq "V3-I run $i: version stable" "0.4.1.post2+cu128" "$(meta_version "$(first_whl "$R/dist")")"
done

# ─────────────────────────────────────────────────────────────────────────────
printf '\n\033[1m══════════════════════════════════════════════\033[0m\n'
printf '  Total  \033[32m%d passed\033[0m  \033[31m%d failed\033[0m\n' "$PASS" "$FAIL"
printf '\033[1m══════════════════════════════════════════════\033[0m\n'
[[ "$FAIL" -eq 0 ]]
