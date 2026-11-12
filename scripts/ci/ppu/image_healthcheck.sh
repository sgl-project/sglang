#!/usr/bin/env bash
# Health check for the image under test: it answers one question only -- is this
# artifact complete in itself?
#
# It does not stand in for functional regression. Learning that the image's
# setuptools had overridden torch's constraint took a whole nightly of 1300 cases;
# the checks here pull that class of "the image simply does not work" failure down
# to ten minutes and point at the field that broke.
#
# Only applies with PPU_TEST_MODE=image (testing the artifact). In source mode it
# exits 0 immediately and behaviour is as it was before this change.
#
# Reads:
#   PPU_TEST_MODE                image enters here, anything else skips
#   SGLANG_BUILD_COMMIT          baked into the image, the Dockerfile records the built commit
#   PPU_EXPECTED_COMMIT          head_sha of the build event; only compared when it is a full 40-char sha
#   PPU_EXPECTED_SGLANG_PREFIX   where sglang should load from, defaults to /sgl-workspace
#   PPU_MIN_KERNEL_OPS           lower bound on the sgl-kernel op surface, defaults to 20
#   PPU_HEALTHCHECK_STRICT_PIP   1 makes a failing pip check red, defaults to a warning
#   PPU_HEALTHCHECK_SERVE        0 skips the server step
#   PPU_HEALTHCHECK_MODEL        model for the server step, defaults to the small one the 1-GPU job already warms
#   PPU_HEALTHCHECK_PORT         port for the server step
#
# Writes:
#   <evidence>/image_healthcheck.json  per-check verdict, reused by the report and the trend
#
# Usage: bash scripts/ci/ppu/image_healthcheck.sh [evidence_dir]
set -uo pipefail

mode="${PPU_TEST_MODE:-source}"
if [ "${mode}" != "image" ]; then
  echo "[image-health] PPU_TEST_MODE=${mode}: not testing an image artifact, skipping"
  exit 0
fi

evidence_dir="${1:-${PPU_EVIDENCE_DIR:-}}"
expected_commit="${PPU_EXPECTED_COMMIT:-}"
expected_prefix="${PPU_EXPECTED_SGLANG_PREFIX:-/sgl-workspace}"
min_kernel_ops="${PPU_MIN_KERNEL_OPS:-20}"
strict_pip="${PPU_HEALTHCHECK_STRICT_PIP:-0}"
serve="${PPU_HEALTHCHECK_SERVE:-1}"
model="${PPU_HEALTHCHECK_MODEL:-/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B}"
port="${PPU_HEALTHCHECK_PORT:-39876}"

results=()
failures=0

pass() {
  echo "[image-health] ok   $1: $2"
  results+=("{\"check\":\"$1\",\"status\":\"pass\",\"detail\":\"${2//\"/\\\"}\"}")
}

# An image that was not built from this commit, or an sglang that never loaded
# from the image at all, means the wrong object is under test -- worse than any
# single failing case, so both are red here.
fail() {
  echo "::error::[image-health] $1: $2"
  results+=("{\"check\":\"$1\",\"status\":\"fail\",\"detail\":\"${2//\"/\\\"}\"}")
  failures=$((failures + 1))
}

warn() {
  echo "::warning::[image-health] $1: $2"
  results+=("{\"check\":\"$1\",\"status\":\"warn\",\"detail\":\"${2//\"/\\\"}\"}")
}

# ---------- 1. Does the image carry the commit we mean to test ---------- #
actual_commit="${SGLANG_BUILD_COMMIT:-}"
if [ -z "${actual_commit}" ] || [ "${actual_commit}" = "unknown" ]; then
  fail "image-commit" "no SGLANG_BUILD_COMMIT in the image, so there is no proof which build this is"
elif [[ "${expected_commit}" =~ ^[0-9a-fA-F]{40}$ ]] && [ "${actual_commit}" != "${expected_commit}" ]; then
  fail "image-commit" "image was built from ${actual_commit:0:7}, expected ${expected_commit:0:7}"
else
  pass "image-commit" "${actual_commit:0:12}"
fi

# ---------- 2. Does it import ---------- #
if out="$(python3 -c 'import sglang, torch; print("sglang", sglang.__version__, "torch", torch.__version__)' 2>&1)"; then
  pass "import" "${out}"
else
  fail "import" "$(printf '%s' "${out}" | tail -3 | tr '\n' ' ')"
fi

# ---------- 3. Does the code under test really come from the image ---------- #
# This is what the whole change rests on: pytest runs from /workspace/source, and
# the moment that checkout wins a place on sys.path the image is only a bystander
# while the report still looks green.
if out="$(python3 -c 'import os, sglang; print(os.path.realpath(sglang.__file__))' 2>&1)"; then
  case "${out}" in
    "${expected_prefix}"/*) pass "module-provenance" "${out}" ;;
    *) fail "module-provenance" "sglang loaded from ${out}, which is not under the image's ${expected_prefix}" ;;
  esac
else
  fail "module-provenance" "$(printf '%s' "${out}" | tail -1)"
fi

# ---------- 4. The version is not a placeholder ---------- #
version="$(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null || true)"
case "${version}" in
  ""|0.0.0|0.0.0*)
    # The Dockerfile installs editable over the local source; with .git missing
    # setuptools-scm falls back to 0.0.0, after which the version field in every
    # report is meaningless and the trend cannot be attributed.
    fail "version" "sglang version is '${version:-<empty>}', the image did not carry one in"
    ;;
  *) pass "version" "${version}" ;;
esac

# ---------- 5. The sgl-kernel op surface ---------- #
kernel_out="$(python3 - <<'PY' 2>&1
import sys
try:
    import torch, sgl_kernel
except Exception as exc:  # noqa: BLE001 - the point is to turn this into a readable failure
    print("import-error", exc)
    sys.exit(3)
registered = getattr(torch.ops, "sgl_kernel", None)
if registered is None:
    print("no-sgl-kernel-namespace")
    sys.exit(4)
names = [n for n in dir(registered) if not n.startswith("_")]
print("ops", len(names), "version", getattr(sgl_kernel, "__version__", "?"))
PY
)"
case "${kernel_out}" in
  ops*)
    op_count="$(printf '%s' "${kernel_out}" | awk '{print $2}')"
    # The count only means something against the history in the trend, so print it.
    if [ "${op_count:-0}" -lt "${min_kernel_ops}" ]; then
      fail "sgl-kernel" "only ${op_count} ops registered (floor ${min_kernel_ops}): ${kernel_out}"
    else
      pass "sgl-kernel" "${kernel_out}"
    fi
    ;;
  *) fail "sgl-kernel" "${kernel_out}" ;;
esac

# ---------- 6. Does the entry point still come up ---------- #
if python3 -m sglang.launch_server --help >/dev/null 2>&1; then
  pass "launch-server" "sglang.launch_server --help exited cleanly"
else
  fail "launch-server" "sglang.launch_server --help exited non-zero, the entry point in the image is broken"
fi

# ---------- 7. Bring up a server, hit /health and generate ---------- #
# Everything above only proves "it imports". What makes the image usable is that
# it serves and returns a token, which is the whole difference between a prod
# image and a pile of packages that install fine and never run.
if [ "${serve}" != "1" ]; then
  warn "serve" "PPU_HEALTHCHECK_SERVE=0, skipping the server step"
elif [ ! -d "${model}" ]; then
  warn "serve" "model directory ${model} is absent, skipping the server step"
else
  server_log="$(mktemp -t ppu-healthcheck-server.XXXXXX.log)"
  python3 -m sglang.launch_server \
    --model-path "${model}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --disable-cuda-graph >"${server_log}" 2>&1 &
  server_pid=$!

  ready=0
  for _ in $(seq 1 180); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      break
    fi
    if curl -fsS -m 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 5
  done

  if [ "${ready}" != "1" ]; then
    fail "serve" "/health was not ready within 15 minutes; tail of the server log: $(tail -5 "${server_log}" | tr '\n' ' ')"
  elif body="$(curl -fsS -m 120 -H 'Content-Type: application/json' \
      -d '{"text":"Hello","sampling_params":{"max_new_tokens":8,"temperature":0}}' \
      "http://127.0.0.1:${port}/generate" 2>&1)" &&
    printf '%s' "${body}" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("text") is not None else 1)' 2>/dev/null; then
    pass "serve" "/health came up and one 8-token generate returned text"
  else
    fail "serve" "/health came up but generate failed: $(printf '%s' "${body}" | head -c 300)"
  fi

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  rm -f "${server_log}"
fi

# ---------- 8. Dependencies are self-consistent ---------- #
# Warning only by default: this image carries a pile of PPU-private wheels and the
# historical pip check noise is not what this change is meant to fix. Look at it
# first, then decide whether it should gate.
if pip_out="$(python3 -m pip check 2>&1)"; then
  pass "pip-check" "dependencies are self-consistent"
elif [ "${strict_pip}" = "1" ]; then
  fail "pip-check" "$(printf '%s' "${pip_out}" | head -5 | tr '\n' ' ')"
else
  warn "pip-check" "$(printf '%s' "${pip_out}" | head -5 | tr '\n' ' ')"
fi

# ---------- Summary ---------- #
if [ -n "${evidence_dir}" ] && [ -d "${evidence_dir}" ]; then
  {
    printf '{"mode":"image","expected_commit":"%s","failures":%s,"checks":[' \
      "${expected_commit}" "${failures}"
    first=1
    for row in "${results[@]}"; do
      [ "${first}" = "1" ] || printf ','
      printf '%s' "${row}"
      first=0
    done
    printf ']}\n'
  } >"${evidence_dir}/image_healthcheck.json"
fi

if [ "${failures}" -gt 0 ]; then
  echo "::error::[image-health] ${failures} check(s) failed; the artifact itself does not hold up, so the cases that follow cannot be interpreted"
  exit 1
fi
echo "[image-health] all checks passed"
