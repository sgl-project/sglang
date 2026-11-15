#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  printf '%s\n' 'PPU infrastructure selftests require Linux.' >&2
  exit 1
fi
for command in flock git jq python3; do
  if ! command -v "$command" >/dev/null; then
    printf 'Missing required command: %s\n' "$command" >&2
    exit 1
  fi
done

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_IS_IN_CI=0
export SGLANG_TEST_MAX_RETRY=0
export HF_HUB_OFFLINE=1

printf '%s\n' 'Running PPU infrastructure selftests with the current Python environment; no packages will be installed.'
for file in test_ppu_ci_dependencies.py test_ppu_ci_docker.py test_ppu_ci_suite.py; do
  printf '\nRunning %s\n' "$file"
  python3 "$ROOT/scripts/ci/ppu/$file" -f
done
printf '\n%s\n' 'PPU infrastructure selftests passed (3 files).'
