#!/bin/bash
# Install the sgl-project/DeepGEMM `dev-0426` branch as `deep_gemm_v4`,
# so it can be imported alongside an existing `deep_gemm` install:
#
#     from deep_gemm_v4 import ...
#
# Usage: install_deepgemm_v4.sh [WORK_DIR]
#   WORK_DIR: where to clone the renamed source tree (default: /tmp/DeepGEMM_v4)
#
# This script clones DeepGEMM, rewrites every `deep_gemm` reference in the
# Python/C++/CMake/build files to `deep_gemm_v4`, then runs `bash install.sh`.
# Third-party submodules (cutlass, etc.) are intentionally left untouched.
set -ex

WORK_DIR="${1:-/tmp/DeepGEMM_v4}"
REPO_URL="https://github.com/sgl-project/DeepGEMM.git"
BRANCH="dev-0426"
OLD_NAME="deep_gemm"
NEW_NAME="deep_gemm_v4"
OLD_DIST="deep-gemm"
NEW_DIST="deep-gemm-v4"

echo "----------------------------------------"
echo "WORK_DIR: ${WORK_DIR}"
echo "REPO_URL: ${REPO_URL}"
echo "BRANCH:   ${BRANCH}"
echo "RENAME:   ${OLD_NAME} -> ${NEW_NAME}"
echo "----------------------------------------"

# 1. Fresh clone of the dev-0426 branch.
rm -rf "${WORK_DIR}"
git clone "${REPO_URL}" -b "${BRANCH}" "${WORK_DIR}"
cd "${WORK_DIR}"
git submodule update --init --recursive

# 2. Rename the top-level Python package directory.
if [ -d "${OLD_NAME}" ]; then
  mv "${OLD_NAME}" "${NEW_NAME}"
fi

# 3. Rewrite every reference to the old package name in source/build files.
#    Word-boundary regex avoids touching unrelated substrings like
#    `deep_gemm_cpp` (the pybind module name, which we want to keep).
#    Excludes: .git/, third-party submodules, build artefacts.
find . -type f \( \
        -name "*.py" -o \
        -name "*.pyi" -o \
        -name "*.cpp" -o \
        -name "*.cc" -o \
        -name "*.cu" -o \
        -name "*.cuh" -o \
        -name "*.h" -o \
        -name "*.hpp" -o \
        -name "*.cmake" -o \
        -name "CMakeLists.txt" -o \
        -name "setup.py" -o \
        -name "setup.cfg" -o \
        -name "pyproject.toml" -o \
        -name "MANIFEST.in" -o \
        -name "install.sh" -o \
        -name "*.sh" \
    \) \
    -not -path "./.git/*" \
    -not -path "./third-party/*" \
    -not -path "./third_party/*" \
    -not -path "./build/*" \
    -not -path "./dist/*" \
    -exec perl -pi -e "s/\\b${OLD_NAME}\\b/${NEW_NAME}/g; s/\\b${OLD_DIST}\\b/${NEW_DIST}/g" {} +

# 4. Sanity check: surface any leftover bare references inside our package.
echo "----------------------------------------"
echo "Leftover references to '${OLD_NAME}' (should be empty or third-party only):"
grep -rn --include="*.py" --include="*.cpp" --include="*.cu" --include="*.h" \
    --include="CMakeLists.txt" --include="*.cmake" --include="setup.py" \
    --include="pyproject.toml" "${OLD_NAME}\\b" . \
    | grep -v "/\\.git/" \
    | grep -v "/third-party/" \
    | grep -v "/third_party/" \
    || true
echo "----------------------------------------"

# 5. Make sure no stale install of the renamed package is in the way.
pip uninstall -y "${NEW_DIST}" "${NEW_NAME}" 2>/dev/null || true

# 6. Build & install.
bash install.sh

# 7. Verify.
python -c "import ${NEW_NAME}; print('${NEW_NAME} imported from:', ${NEW_NAME}.__file__)"

echo "Done. Use:  from ${NEW_NAME} import ..."
