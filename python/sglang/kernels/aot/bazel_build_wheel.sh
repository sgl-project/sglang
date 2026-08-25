#!/usr/bin/env bash
set -euo pipefail

backend="$1"
output_dir="$2"
source_root="$3"

case "${output_dir}" in
  /*) ;;
  *) output_dir="${PWD}/${output_dir}" ;;
esac

stage="$(mktemp -d "${TMPDIR:-/tmp}/sgl-kernel-bazel.XXXXXX")"
trap 'rm -rf "${stage}"' EXIT
mkdir -p "${stage}/src"
cp -aL "${source_root}/." "${stage}/src/"

cd "${stage}/src"
case "${backend}" in
  cpu) cp pyproject_cpu.toml pyproject.toml ;;
  cuda) ;;
  *)
    echo "unsupported sgl-kernel backend: ${backend}" >&2
    exit 2
    ;;
esac

rm -rf build dist
if command -v uv >/dev/null 2>&1; then
  uv build --wheel -Cbuild-dir=build --no-build-isolation .
else
  python3 -m build --wheel --no-isolation .
fi

if [[ "${backend}" == "cuda" ]]; then
  PYTHON="$(command -v python3)" bash ./rename_wheels.sh
fi

shopt -s nullglob
wheels=(dist/*.whl)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "expected exactly one wheel, found ${#wheels[@]}" >&2
  exit 1
fi

mkdir -p "${output_dir}"
cp "${wheels[0]}" "${output_dir}/"
