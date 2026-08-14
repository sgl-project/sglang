#!/bin/bash
# Exercises the custom-image selection in amd_ci_start_container.sh by stubbing
# out docker/hostname/git, so the precedence rules can be checked without a
# runner or a real registry.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUB_DIR="$(mktemp -d)"
trap 'rm -rf "${STUB_DIR}"' EXIT

cat > "${STUB_DIR}/docker" <<'STUB'
#!/bin/bash
printf '%s\n' "$*" >> "${DOCKER_LOG}"
case "$1" in
  images) ;;              # pretend nothing is cached locally
  *) exit 0 ;;
esac
STUB

cat > "${STUB_DIR}/hostname" <<'STUB'
#!/bin/bash
echo "${FAKE_HOSTNAME}"
STUB

# Skip tag detection so the run does not depend on network access.
cat > "${STUB_DIR}/git" <<'STUB'
#!/bin/bash
exit 1
STUB

chmod +x "${STUB_DIR}"/docker "${STUB_DIR}"/hostname "${STUB_DIR}"/git

failures=0

# run_case <name> <hostname> <expected-image> <expect-custom-login: yes|no> [extra args...]
run_case() {
  local name=$1 host=$2 expected=$3 expect_login=$4; shift 4
  local log="${STUB_DIR}/docker.log"
  : > "${log}"

  DOCKER_LOG="${log}" FAKE_HOSTNAME="${host}" PATH="${STUB_DIR}:${PATH}" \
    bash "${SCRIPT_DIR}/amd_ci_start_container.sh" --rocm-version rocm724 "$@" \
    > "${STUB_DIR}/out.log" 2>&1
  local rc=$?

  local actual
  actual=$(grep '^run ' "${log}" | tail -n 1 | awk '{print $NF}')
  local logins
  logins=$(grep -c "^login -u ${AMD_CI_IMAGE_USERNAME:-__unset__} " "${log}")

  local problem=""
  [[ ${rc} -ne 0 ]] && problem="script exited ${rc}"
  [[ "${actual}" != "${expected}" ]] && problem="${problem} image='${actual}' want='${expected}'"
  if [[ "${expect_login}" == "yes" && "${logins}" -eq 0 ]]; then
    problem="${problem} missing login for the custom image"
  elif [[ "${expect_login}" == "no" && "${logins}" -ne 0 ]]; then
    problem="${problem} unexpected login for the custom image"
  fi

  if [[ -n "${problem}" ]]; then
    echo "FAIL ${name}:${problem}"
    sed 's/^/     /' "${STUB_DIR}/out.log" | tail -n 15
    failures=$((failures + 1))
  else
    echo "ok   ${name} -> ${actual}"
  fi
}

MI30X_HOST="linux-mi300-gpu-1-abc-runner-xyz"
MI35X_HOST="linux-mi35x-gpu-8-abc-runner-xyz"
A="staging/repo:mi30x-tag"
B="staging/repo:mi35x-tag"
C="staging/repo:cli-tag"

( export AMD_CI_IMAGE="${A}";                              run_case "mi30x takes AMD_CI_IMAGE"            "${MI30X_HOST}" "${A}" no )
( export AMD_CI_IMAGE="${A}" AMD_CI_IMAGE_MI35X="${B}";     run_case "mi35x takes AMD_CI_IMAGE_MI35X"      "${MI35X_HOST}" "${B}" no )
( export AMD_CI_IMAGE="${A}" AMD_CI_IMAGE_MI35X="${B}";     run_case "mi30x ignores AMD_CI_IMAGE_MI35X"    "${MI30X_HOST}" "${A}" no )
( export AMD_CI_IMAGE="${A}";                              run_case "mi35x falls back to AMD_CI_IMAGE"    "${MI35X_HOST}" "${A}" no )
( export AMD_CI_IMAGE="${A}" AMD_CI_IMAGE_MI35X="${B}";     run_case "--custom-image beats both"           "${MI35X_HOST}" "${C}" no --custom-image "${C}" )
( export AMD_CI_IMAGE="${A}" AMD_CI_IMAGE_USERNAME=someone AMD_CI_IMAGE_TOKEN=t; \
                                                            run_case "custom image logs in"                "${MI30X_HOST}" "${A}" yes )
( export AMD_CI_IMAGE_USERNAME=someone AMD_CI_IMAGE_TOKEN=t; \
  run_case "no override keeps auto-discovery" "${MI30X_HOST}" "rocm/sgl-dev:v0.5.5-rocm724-mi30x-$(date +%Y%m%d)" no )

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} case(s) failed"
  exit 1
fi
echo "all cases passed"
