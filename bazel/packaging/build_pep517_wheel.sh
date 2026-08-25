#!/usr/bin/env bash
set -euo pipefail

if [[ "${SGLANG_PEP517_TRANSITIONAL_ACTION:-}" != "1" ]]; then
  echo "This helper is only for //bazel/packaging:main_wheel_pep517." >&2
  exit 2
fi

if [[ -z "${SETUPTOOLS_SCM_PRETEND_VERSION:-}" ]]; then
  cat >&2 <<'EOF'
SETUPTOOLS_SCM_PRETEND_VERSION is required.
Compute the version from the checkout and pass it explicitly:
  VERSION="$(cd python && python3 -m setuptools_scm)"
  bazel build \
    --action_env=SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" \
    //bazel/packaging:main_wheel_pep517
EOF
  exit 2
fi

python_bin="${SGLANG_PEP517_PYTHON:-python3}"
if ! "$python_bin" -c "import build" >/dev/null 2>&1; then
  echo "$python_bin must provide the PEP 517 frontend module 'build'." >&2
  exit 2
fi

if [[ "$#" -ne 1 || "$1" != @* ]]; then
  echo "Expected one Bazel multiline parameter file." >&2
  exit 2
fi

mapfile -t params < "${1#@}"
if (( ${#params[@]} < 2 )); then
  echo "Parameter file must contain an output and at least one source." >&2
  exit 2
fi

wheel_dir="${params[0]}"
rm -rf "$wheel_dir"
mkdir -p "$wheel_dir"
stage="$wheel_dir/.source"
dist="$wheel_dir/.dist"
mkdir -p "$stage" "$dist"
cleanup() {
  rm -rf "$stage" "$dist"
}
trap cleanup EXIT

for source in "${params[@]:1}"; do
  case "$source" in
    /*|../*|*/../*)
      echo "Refusing source outside the Bazel execution root: $source" >&2
      exit 2
      ;;
  esac
  destination="$stage/$source"
  mkdir -p "$(dirname "$destination")"
  # Bazel source inputs are execution-root symlinks. Dereference those links so
  # the staged PEP 517 tree is independent of the checkout during the action.
  cp -Lp -- "$source" "$destination"
done

cp "$stage/README.md" "$stage/python/README.md"
cp "$stage/LICENSE" "$stage/python/LICENSE"

# setuptools-scm is both the version provider and setuptools' authoritative
# package-data file finder. The pretend version avoids reading checkout history,
# while this synthetic index makes the declared source set visible to the same
# file finder without exposing the real checkout's .git directory.
git -C "$stage" init --quiet
git -C "$stage" add --all
git -C "$stage" \
  -c user.name="Bazel PEP 517 bridge" \
  -c user.email="bazel-pep517@invalid" \
  commit --quiet --no-gpg-sign --message="Declared PEP 517 inputs"

cat >&2 <<EOF
WARNING: building a transitional, non-hermetic wheel.
  frontend: $python_bin -m build
  version:  $SETUPTOOLS_SCM_PRETEND_VERSION
  ambient:  host Python/build, Git, network, Cargo/Rust/cache, C linker, system libraries
EOF

export CARGO_TARGET_DIR="${SGLANG_PEP517_CARGO_TARGET_DIR:-$stage/cargo-target}"
"$python_bin" -m build \
  --wheel \
  --outdir "$dist" \
  "$stage/python"

shopt -s nullglob
wheels=("$dist"/*.whl)
if (( ${#wheels[@]} != 1 )); then
  echo "Expected exactly one wheel in $dist; found ${#wheels[@]}." >&2
  exit 1
fi
mv "${wheels[0]}" "$wheel_dir/"
