#!/usr/bin/env bash
set -euo pipefail

phase=${1:?Usage: build_and_test.sh build|test}
case "$phase" in
    build|test) ;;
    *) printf 'Unknown phase: %s\n' "$phase" >&2; exit 2 ;;
esac
: "${PPU_CI_ARTIFACT_DIR:?}"
: "${PPU_CI_WHEEL_DIR:?}"
: "${PPU_CI_SOURCE_SHA:?}"
mkdir -p "$PPU_CI_ARTIFACT_DIR"
artifacts=$(realpath "$PPU_CI_ARTIFACT_DIR")
wheel_dir=$(realpath "$PPU_CI_WHEEL_DIR")
exec > >(tee "$artifacts/$phase.log") 2>&1

repo=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
checker="$repo/scripts/ci/ppu/check_dependencies.py"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 SGLANG_TEST_MAX_RETRY=0 SGLANG_IS_IN_CI=1

verify_wheels() {
    if ! (cd "$wheel_dir" && sha256sum source-sha.txt ./*.whl ./runtime-deps/*.whl | cmp -s SHA256SUMS -); then
        printf 'Wheel manifest does not match the complete wheel inventory\n' >&2
        return 1
    fi
}

if [[ "$phase" == build ]]; then
    if [[ "$(git -C "$repo" rev-parse HEAD)" != "$PPU_CI_SOURCE_SHA" ]]; then
        printf 'Source checkout does not match PPU_CI_SOURCE_SHA\n' >&2
        exit 2
    fi
    if [[ -n "$(git -C "$repo" status --porcelain --ignored)" ]]; then
        printf 'Source checkout must be clean, including untracked and ignored files\n' >&2
        exit 2
    fi
    : "${PPU_SDK:?}"
    if compgen -G "$wheel_dir/*.whl" >/dev/null; then
        printf 'Build wheel directory must be empty: %s\n' "$wheel_dir" >&2
        exit 2
    fi
    sdk_root=$(realpath "$PPU_SDK")
    set +u
    # shellcheck source=/dev/null
    source "$sdk_root/envsetup.sh" ppu
    set -u
    export PPU_SDK="$sdk_root" PPU_HOME="$sdk_root"
    export CUDA_HOME="$sdk_root" CUDA_PATH="$sdk_root"
    export PATH="$sdk_root/bin:$PATH"
    export LD_LIBRARY_PATH="$sdk_root/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$sdk_root/lib:${LIBRARY_PATH:-}"
    unset TORCH_CUDA_ARCH_LIST SGL_KERNEL_SKIP_SAILIFY SGL_KERNEL_SKIP_THIRD_PARTY
    export PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_10;ppu_15}"
    export SGL_KERNEL_USE_SAILIFY=1
    export MAX_JOBS="${MAX_JOBS:-8}" SGL_KERNEL_COMPILE_THREADS="${SGL_KERNEL_COMPILE_THREADS:-8}"

    python3 - <<'PY'
import torch
from torch.utils.cpp_extension import _get_cuda_arch_flags

assert getattr(torch.version, "sail", None), "A cuda-free SAIL Torch build is required for compilation"
flags = _get_cuda_arch_flags()
assert any(flag.startswith("--gpu-architecture=ppu_") for flag in flags), flags
assert not any("gencode" in flag for flag in flags), flags
print("Build Torch:", torch.__version__, "SAIL:", torch.version.sail, "Arch flags:", flags)
PY
    hgcc --version
    python3 -m pip install --index-url https://pypi.org/simple \
        'setuptools>=75' 'setuptools-scm>=8' 'setuptools-rust>=1.10' wheel build ninja
    mkdir -p "$wheel_dir/runtime-deps"
    python3 -m pip wheel --no-deps --no-build-isolation \
        --wheel-dir "$wheel_dir/runtime-deps" \
        git+https://github.com/t-head/sailify.git@f401ecae76d7b4991d44920bde9edaac7e655fd8
    python3 -m pip install --no-deps --force-reinstall "$wheel_dir/runtime-deps/"*.whl
    requirements=$(python3 "$checker" requirements)
    mapfile -t eval_requirements <<< "$requirements"
    python3 -m pip wheel --index-url https://pypi.org/simple --no-deps --no-build-isolation \
        --wheel-dir "$wheel_dir/runtime-deps" "${eval_requirements[@]}"
    python3 "$checker" snapshot "$artifacts/build-dependencies.json"

    work=$(mktemp -d)
    mkdir "$work/source"
    git -C "$repo" archive "$PPU_CI_SOURCE_SHA" | tar -C "$work/source" -xf -
    cp "$work/source/python/pyproject_other.toml" "$work/source/python/pyproject.toml"
    # This lane validates text inference, not the optional multimodal Rust extension.
    export SGLANG_BUILD_RUST_EXTS=none
    python3 -m build --wheel --no-isolation --outdir "$wheel_dir" "$work/source/python"
    cd "$work/source/python/sglang/kernels/aot"
    cp pyproject_sail.toml pyproject.toml
    python3 setup_sail.py bdist_wheel --dist-dir "$wheel_dir"
    cd "$wheel_dir"
    printf '%s\n' "$PPU_CI_SOURCE_SHA" > source-sha.txt
    sha256sum source-sha.txt ./*.whl ./runtime-deps/*.whl > SHA256SUMS
    exit 0
fi

verify_wheels
if [[ "$(<"$wheel_dir/source-sha.txt")" != "$PPU_CI_SOURCE_SHA" ]]; then
    printf 'Wheel source revision does not match PPU_CI_SOURCE_SHA\n' >&2
    exit 2
fi
: "${PPU_CI_TARGET_MODEL_PATH:?}"
: "${PPU_CI_EAGLE3_MODEL_PATH:?}"
: "${PPU_CI_DFLASH_MODEL_PATH:?}"
: "${PPU_CI_EVAL_CACHE:?}"
export HOME
HOME=$(mktemp -d)
export HF_HOME="$HOME/hf-cache" HF_HUB_CACHE="$HOME/hf-cache/hub" HF_HUB_OFFLINE=1
export UMD_ENABLE_COREDUMP_ON_EXCEPTION=1 UMD_ENABLE_LIGHTWEIGHT_COREDUMP=1
mkdir -p "$artifacts/eval-results"
ln -s "$artifacts/eval-results" "$HOME/.sgl_eval"
python3 "$checker" snapshot "$artifacts/dependencies-before.json"
finalize() {
    code=$?
    trap - EXIT
    set +e
    if ! python3 "$checker" snapshot "$artifacts/dependencies-after.json"; then
        code=1
    elif ! python3 "$checker" check "$artifacts/dependencies-before.json" "$artifacts/dependencies-after.json"; then
        code=1
    fi
    if ! verify_wheels; then
        code=1
    fi
    printf '%s\n' "$code" > "$artifacts/exit-code.txt"
    exit "$code"
}
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 "$repo/scripts/ci/ppu/check_ppu_environment.py"
python3 -c 'import torch; print("Runtime Torch:", torch.__version__, "SAIL build marker:", getattr(torch.version, "sail", None)); assert torch.cuda.device_count() == 1'
python3 -m pip uninstall -y sgl-kernel
python3 -m pip install --no-deps --no-index --force-reinstall "$wheel_dir/"*.whl
missing_wheels=$(python3 "$checker" missing-wheels "$wheel_dir/runtime-deps")
if [[ -n "$missing_wheels" ]]; then
    mapfile -t runtime_wheels <<< "$missing_wheels"
    python3 -m pip install --no-deps --no-index "${runtime_wheels[@]}"
fi
python3 "$checker" snapshot "$artifacts/dependencies-installed.json"
python3 "$checker" check "$artifacts/dependencies-before.json" "$artifacts/dependencies-installed.json"
python3 - <<'PY'
from sailify.sailify_python import sailify_extra_files_recursive
import sgl_kernel
import sglang
import torch
from sglang.srt.platforms import current_platform

assert current_platform.is_sail(), "The installed SGLang did not select SAIL"
print("Runtime Torch:", torch.__version__, "SAIL build marker:", getattr(torch.version, "sail", None))
print("SGLang:", sglang.__file__)
print("AOT extension:", sgl_kernel.common_ops.__file__)
PY
python3 "$repo/scripts/ci/ppu/prepare_runtime.py"

cd "$artifacts"
python3 "$repo/test/run_suite.py" --hw ppu --suite stage-a-test-1-gpu-ppu --timeout-per-file 1200 \
    2>&1 | tee "$artifacts/sanity.log"
# Require both sanity files; update this count when extending the suite.
grep -Fxq 'Test Summary: 2/2 passed' "$artifacts/sanity.log"
