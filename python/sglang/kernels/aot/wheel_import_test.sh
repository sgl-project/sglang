#!/usr/bin/env bash
set -euo pipefail

wheel_dir="$1"
python_bin="${PYTHON_BIN_PATH:-python3}"

shopt -s nullglob
wheels=("${wheel_dir}"/*.whl)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "expected exactly one wheel in ${wheel_dir}, found ${#wheels[@]}" >&2
  exit 1
fi

site_dir="${TEST_TMPDIR}/site-packages"
mkdir -p "${site_dir}"
"${python_bin}" - "${wheels[0]}" "${site_dir}" <<'PY'
import pathlib
import sys
import zipfile

wheel = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(destination)
PY

export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export PIP_NO_INDEX=1
export ROCR_VISIBLE_DEVICES=""
export UV_OFFLINE=1
export PYTHONPATH="${site_dir}${PYTHONPATH:+:${PYTHONPATH}}"
rocm_home="${ROCM_HOME:-/opt/rocm}"
export LD_LIBRARY_PATH="${rocm_home}/lib:${rocm_home}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${TEST_TMPDIR}"
"${python_bin}" - "${site_dir}" <<'PY'
import importlib.metadata
import pathlib
import sys

site_dir = pathlib.Path(sys.argv[1]).resolve()
import sgl_kernel

common_ops = pathlib.Path(sgl_kernel.common_ops.__file__).resolve()
if not common_ops.is_relative_to(site_dir):
    raise AssertionError(f"common_ops was not imported from the wheel: {common_ops}")
print(
    "imported sglang-kernel "
    f"{importlib.metadata.version('sglang-kernel')} from {common_ops}"
)
PY
