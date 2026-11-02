#!/bin/bash
# Install EvalScope into a virtual environment of its own, inside the worker pod.
#
# A separate environment rather than the image's, because EvalScope's dependency
# closure and the image's serving stack disagree where it is not negotiable:
# EvalScope requires modelscope[datasets]>=1.34, which resolves datasets>=4.0.0,
# while the v2.1.1 image ships datasets 3.1.0 pinned to dill<0.3.9 -- and
# ppu_install_dependency.sh installs dill deliberately inside that band because
# Python 3.12 unpickles _abc._abc_data only from 0.3.8 up.  Letting pip resolve
# EvalScope into the serving environment would move datasets and dill under the
# server this suite is here to measure.
#
# The isolation costs nothing here.  With --eval-type openai_api EvalScope is a
# pure HTTP client: it reads a staged dataset, calls /v1/chat/completions and
# scores the text that comes back.  Its core requirements carry no torch and it
# never imports the serving stack, so the two environments share only the socket.
#
# The version is pinned rather than floated, and above 1.11.0 rather than at any
# release: report schema v2 -- structured metric list, primary_metric_identity,
# execution_summary -- first appears there, and accuracy_eval_kit reads that
# schema and refuses v1 rather than misreading it (a v1 report has a report-level
# `score` whose value is a different thing).  The import check at the end is that
# contract, asserted before a checkpoint is loaded rather than after.
#
# Reads:
#   EVALSCOPE_VENV        where to build the environment (default /opt/evalscope-venv)
#   EVALSCOPE_SPEC        the requirement to install
#   EVALSCOPE_WHEELHOUSE  optional local wheel directory, tried before the index
#   EVALSCOPE_CACHE_DIR   shared archives (default /wl_nas/cache/evalscope)
#
# Usage: bash /workspace/source/scripts/ci/ppu/setup_evalscope.sh
set -euo pipefail

EVALSCOPE_VENV="${EVALSCOPE_VENV:-/opt/evalscope-venv}"
EVALSCOPE_SPEC="${EVALSCOPE_SPEC:-evalscope[ifeval]==1.11.1}"
EVALSCOPE_WHEELHOUSE="${EVALSCOPE_WHEELHOUSE:-/nas_aisw/datasets/packages/evalscope}"
EVALSCOPE_CACHE_DIR="${EVALSCOPE_CACHE_DIR:-/wl_nas/cache/evalscope}"

evalscope_bin="${EVALSCOPE_VENV}/bin/evalscope"
venv_python="${EVALSCOPE_VENV}/bin/python"
cache_key=$(python3 - "${EVALSCOPE_SPEC}" "${EVALSCOPE_VENV}" <<'PYEOF'
import hashlib
import json
from pathlib import Path
import platform
import sys
import sysconfig

interpreter = Path(sys.executable).resolve()
identity = {
    "format": 1,
    "spec": sys.argv[1],
    "venv": str(Path(sys.argv[2]).absolute()),
    "python": sys.version,
    "interpreter": str(interpreter),
    "interpreter_sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
    "base_prefix": sys.base_prefix,
    "abi": sysconfig.get_config_var("SOABI"),
    "arch": platform.machine(),
    "libc": platform.libc_ver(),
    "os": Path("/etc/os-release").read_text(),
}
print(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16])
PYEOF
)
key_dir="${EVALSCOPE_CACHE_DIR}/${cache_key}"
cache_archive="${key_dir}/venv.tar.gz"
echo "EvalScope cache key: ${cache_key}"

verify_environment() {
  [ -x "${evalscope_bin}" ] || return 1
  "${venv_python}" - "${EVALSCOPE_SPEC}" <<'PYEOF' || return 1
import evalscope
from evalscope.api.metric.semantics import MetricIdentity  # noqa: F401
from evalscope.report import Report
from importlib.metadata import version
from packaging.requirements import Requirement
import sys

requirement = Requirement(sys.argv[1])
installed = version(requirement.name)
if installed not in requirement.specifier:
    raise SystemExit(f"evalscope {installed} does not satisfy {requirement}")
for name in ("schema_version", "metrics", "primary_metric_identity"):
    if name not in Report.model_fields:
        raise SystemExit(f"evalscope {installed} has no Report.{name}")
print(f"evalscope {installed}, report schema v2 present")
PYEOF
  "${evalscope_bin}" --help >/dev/null || return 1
  "${venv_python}" -m pip check || return 1
}

ready=0
built=0
if [ -f "${EVALSCOPE_VENV}/.cache-key" ] \
    && [ "$(<"${EVALSCOPE_VENV}/.cache-key")" = "${cache_key}" ] \
    && verify_environment; then
  echo "EvalScope environment already present at ${EVALSCOPE_VENV}"
  ready=1
elif [ -r "${cache_archive}" ]; then
  echo "EvalScope cache HIT: ${cache_archive}"
  restore_stage=$(mktemp -d)
  if tar -xzf "${cache_archive}" -C "${restore_stage}" \
      && [ -f "${restore_stage}/.cache-key" ] \
      && [ "$(<"${restore_stage}/.cache-key")" = "${cache_key}" ]; then
    if [ -d "${EVALSCOPE_VENV}" ]; then
      python3 -m venv --clear "${EVALSCOPE_VENV}"
      cp -a "${restore_stage}/." "${EVALSCOPE_VENV}/"
    else
      mkdir -p "$(dirname "${EVALSCOPE_VENV}")"
      mv -T "${restore_stage}" "${EVALSCOPE_VENV}"
      restore_stage=""
    fi
    if verify_environment; then
      ready=1
    else
      echo "EvalScope cache validation failed, rebuilding locally"
    fi
  else
    echo "EvalScope cache restore failed, rebuilding locally"
  fi
  rm -rf "${restore_stage}"
fi

if [ "${ready}" != 1 ]; then
  echo "EvalScope cache MISS: building at ${EVALSCOPE_VENV}"
  python3 -m venv --clear "${EVALSCOPE_VENV}"
  "${venv_python}" -m pip install --no-cache-dir --upgrade pip wheel

  installed=0
  if [ -d "${EVALSCOPE_WHEELHOUSE}" ]; then
    echo "Trying the local wheel directory ${EVALSCOPE_WHEELHOUSE}"
    if "${venv_python}" -m pip install --no-cache-dir --no-index \
      --find-links "${EVALSCOPE_WHEELHOUSE}" "${EVALSCOPE_SPEC}"; then
      installed=1
    else
      echo "the local wheels do not close over ${EVALSCOPE_SPEC}, using the index"
    fi
  fi
  if [ "${installed}" != 1 ]; then
    "${venv_python}" -m pip install --no-cache-dir "${EVALSCOPE_SPEC}"
  fi
  verify_environment
  printf '%s\n' "${cache_key}" > "${EVALSCOPE_VENV}/.cache-key"
  built=1
fi

if [ "${built}" = 1 ]; then
  (
    local_stage=""
    nas_stage=""
    trap 'rm -rf "${local_stage}" "${nas_stage}"' EXIT
    mkdir -p "${key_dir}" || exit 1
    local_stage=$(mktemp -d) || exit 1
    nas_stage=$(mktemp -d "${EVALSCOPE_CACHE_DIR}/.${cache_key}.XXXXXX") || exit 1
    "${venv_python}" -m pip freeze > "${EVALSCOPE_VENV}/.cache-requirements.txt" || exit 1
    # Pack locally so NAS only receives one sequential file write.
    tar -I 'gzip -1' -cf "${local_stage}/venv.tar.gz" -C "${EVALSCOPE_VENV}" . || exit 1
    cp "${local_stage}/venv.tar.gz" "${nas_stage}/venv.tar.gz" || exit 1
    # Replacing a single archive is atomic even when workers publish concurrently.
    mv -fT "${nas_stage}/venv.tar.gz" "${cache_archive}" || exit 1
    echo "EvalScope cache published: ${cache_archive}"
  ) || echo "::warning::EvalScope cache publication failed; the local environment remains usable"
fi

echo "evalscope binary: ${evalscope_bin}"
