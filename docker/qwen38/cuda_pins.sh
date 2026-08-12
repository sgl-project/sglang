#!/usr/bin/env bash
# Keep the image's dependency reality in step with what this tree declares.
#
# The image installs SGLang with `--no-deps` on purpose: the base ships
# CUDA-tagged wheels (sglang-kernel X+cu129, sgl-deep-gemm Y+cu129) and a plain
# `pip install` would swap them for the untagged PyPI builds, which are compiled
# against a different libtorch and die at import with an undefined symbol.
#
# What `--no-deps` costs is the other direction: a pin this tree *raises* is
# silently dropped. The image keeps whatever the base had, builds green, gets
# pushed, gets pulled onto every node, and only fails when a server starts and
# srt/entrypoints/engine.py runs assert_pkg_version. These subcommands move each
# of those failures to build time, where one line of output is the whole answer.
#
#   check-torch <pyproject>
#       Fail if the base's torch is not the one this tree pins. Nothing below
#       can be satisfied when this is wrong: the CUDA-tagged kernel wheels are
#       built against a specific libtorch, and they declare no Requires-Dist,
#       so pip cannot see the conflict and will happily install a broken combo.
#       Depends only on the base image, so callers should run it as early as
#       possible -- a wrong base is knowable in two seconds, and every expensive
#       layer after it is wasted work.
#
#   reconcile <pyproject> <cuda-tag> <pkg>...
#       Install the +<cuda-tag> build of each pkg at the version <pyproject>
#       pins, from SGLang's own wheel index, then import it -- an ABI mismatch
#       shows up here rather than on the first forward pass. The import step is
#       skipped (loudly) on build hosts with no NVIDIA driver; see below.
#
#   verify <pyproject> <pkg>...
#       Run the same assert_pkg_version call the server makes at launch.
#
#   import-if-gpu <module>...
#       Import each module, but only where a CUDA device is reachable. For
#       modules that refuse to load without a driver this is the difference
#       between a check and a guaranteed build failure -- see have_cuda_device.
set -euo pipefail

# Point at the GitHub Pages origin directly. The docs.sglang.ai alias works but
# costs two cross-host redirects (docs.sglang.ai -> docs.sglang.io ->
# sgl-project.github.io) on every wheel fetch.
SGL_WHL_INDEX="${SGL_WHL_INDEX:-https://sgl-project.github.io/whl}"

# Echo the version <pyproject> pins for a dependency, or fail if it is unpinned.
pinned_version() {
    python3 - "$1" "$2" <<'PY'
import pathlib, sys, tomllib

data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
want = sys.argv[2]
for spec in data["project"]["dependencies"]:
    spec = spec.replace(" ", "")
    if spec.startswith(want + "=="):
        print(spec.split("==", 1)[1])
        break
else:
    raise SystemExit(f"{want} is not pinned in {sys.argv[1]}")
PY
}

# Distribution name -> import name. Deliberately an explicit table rather than a
# derivation: for these wheels the two names do not correspond, and a wrong
# guess surfaces as ModuleNotFoundError that reads exactly like the ABI failure
# this script exists to catch. Verified by listing the wheels' own contents:
#   sglang-kernel  ships  sgl_kernel/
#   sgl-deep-gemm  ships  deep_gemm/    <- NOT sgl_deep_gemm
module_for() {
    case "$1" in
        sglang-kernel) echo sgl_kernel ;;
        sgl-deep-gemm) echo deep_gemm ;;
        *)
            cat >&2 <<EOF
ERROR: no import-module mapping for '$1'.

  Add it to module_for() in $0. The mapping is not derived on purpose: these
  distributions do not name their modules after themselves (sgl-deep-gemm ships
  deep_gemm, sglang-kernel ships sgl_kernel), so guessing turns a missing entry
  into an error that looks like an ABI mismatch.
EOF
            return 1
            ;;
    esac
}

# True only where a CUDA extension can actually be loaded. Two distinct reasons
# an import fails on a build host, both of which this covers:
#   - libcuda.so.1 is the NVIDIA *driver* library. It is injected by the
#     container runtime on a GPU host and is absent from the image, so loading
#     any CUDA extension (sgl_kernel) fails however correct the wheel is.
#   - sgl-deep-ep goes further and refuses at import with "The NVIDIA driver
#     does not expose a usable CUDA device", so a present-but-deviceless driver
#     is not enough either.
# torch.cuda.device_count() answers both without raising: it returns 0 rather
# than throwing when there is no driver.
have_cuda_device() {
    python3 - <<'PY'
import sys

try:
    import torch

    sys.exit(0 if torch.cuda.device_count() > 0 else 1)
except Exception:
    sys.exit(1)
PY
}

cmd_check_torch() {
    local pyproject="$1"
    local want got
    want="$(pinned_version "$pyproject" "torch")"
    got="$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
    if [ "$want" != "$got" ]; then
        cat >&2 <<EOF
ERROR: this tree pins torch==${want}, but the base image ships torch ${got}.

  The CUDA-tagged sglang-kernel / sgl-deep-gemm wheels are compiled against a
  specific libtorch and declare no Requires-Dist, so installing the versions
  this tree pins on top of torch ${got} produces an image that imports with
  "undefined symbol: ...torch::headeronly..." on every rank at server start.

  The three move as a set -- upgrading the kernels alone or torch alone both
  leave a broken image. Fix one of:
    - bump the FROM base to one built on torch ${want}
    - move this tree back to dependency pins that match torch ${got}
EOF
        exit 1
    fi
    echo "[cuda_pins] torch OK: pinned ${want}, base ships ${got}"
}

# True when the installed version already satisfies the pin, using the same
# >=-on-release comparison assert_pkg_version applies. The local segment
# (+cu129) is deliberately ignored: it records which CUDA a wheel was built for,
# not a version ordering, and the base's own choice of build is the one to keep.
already_satisfied() {
    PKG="$1" WANT="$2" python3 - <<'PY'
import os, sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

try:
    got = version(os.environ["PKG"])
except PackageNotFoundError:
    sys.exit(1)
sys.exit(0 if Version(got) >= Version(os.environ["WANT"]) else 1)
PY
}

cmd_reconcile() {
    local pyproject="$1" cuda_tag="$2"
    shift 2
    local pkg ver mod
    for pkg in "$@"; do
        ver="$(pinned_version "$pyproject" "$pkg")"
        mod="$(module_for "$pkg")"
        # Reinstalling what the base already satisfies is not free: the base may
        # have installed an untagged build on purpose (upstream's CUDA 13 path
        # does exactly that), and replacing it with the +<tag> build swaps a
        # known-good artifact for a different one. Only act when the pin is
        # genuinely higher than what is installed.
        if already_satisfied "$pkg" "$ver"; then
            echo "[cuda_pins] ${pkg}: already satisfies the ${ver} pin" \
                 "(installed $(python3 -c "from importlib.metadata import version; print(version('${pkg}'))"))," \
                 "leaving the base's build in place"
            continue
        fi
        echo "[cuda_pins] reconciling ${pkg}==${ver}+${cuda_tag}"
        python3 -m pip install --no-deps "${pkg}==${ver}+${cuda_tag}" \
            --index-url "${SGL_WHL_INDEX}/${cuda_tag}"
        # Import now: a wheel built against another libtorch installs cleanly
        # and only fails when it is first loaded.
        if ! have_cuda_device; then
            echo "[cuda_pins] ${pkg}: SKIPPED the import check -- no reachable CUDA device on" \
                 "this build host, so no CUDA extension can load here regardless of correctness."
            echo "[cuda_pins] ${pkg}: 'import ${mod}' must be covered by a GPU smoke test." \
                 "The build has verified the installed version only."
            continue
        fi
        python3 -c "import ${mod}" || {
            echo "ERROR: ${pkg}==${ver}+${cuda_tag} installed but 'import ${mod}' fails" >&2
            echo "       against this base's torch -- the two were built apart." >&2
            exit 1
        }
    done
}

cmd_import_if_gpu() {
    local mod
    if ! have_cuda_device; then
        echo "[cuda_pins] SKIPPED importing $*: no reachable CUDA device on this build host."
        echo "[cuda_pins] These imports must be covered by a GPU smoke test."
        return 0
    fi
    for mod in "$@"; do
        python3 -c "import ${mod}; print('[cuda_pins] import ${mod} OK')"
    done
}

cmd_verify() {
    local pyproject="$1"
    shift
    python3 - "$pyproject" "$@" <<'PY'
import pathlib, sys, tomllib
from importlib.metadata import version

from sglang.srt.utils.common import assert_pkg_version

data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
pins = {}
for spec in data["project"]["dependencies"]:
    spec = spec.replace(" ", "")
    if "==" in spec:
        name, ver = spec.split("==", 1)
        pins[name] = ver

for pkg in sys.argv[2:]:
    if pkg not in pins:
        raise SystemExit(f"{pkg} is not pinned in {sys.argv[1]}")
    # The same call srt/entrypoints/engine.py makes when a server starts. Note
    # that engine.py hardcodes its minimum rather than reading pyproject, so the
    # two can drift apart; this asserts the tree's own pin, which is the stricter
    # reading of what the source expects.
    assert_pkg_version(pkg, pins[pkg], "image build did not reconcile this pin")
    print(f"[cuda_pins] {pkg} OK: pinned {pins[pkg]}, installed {version(pkg)}")
PY
}

case "${1:?usage: $0 check-torch|reconcile|verify|import-if-gpu ...}" in
    check-torch)   shift; cmd_check_torch "$@" ;;
    reconcile)     shift; cmd_reconcile "$@" ;;
    verify)        shift; cmd_verify "$@" ;;
    import-if-gpu) shift; cmd_import_if_gpu "$@" ;;
    *)             echo "unknown subcommand: $1" >&2; exit 2 ;;
esac
