#!/bin/bash
# Carry one Answer entry's report off the NAS onto the run page and into the
# artifact directory.
#
# Run by every entry of test-ppu-answer.yml in the orchestration shell, held
# here for the same reason the board-side body is: those jobs cannot call a local
# composite action, because their runner resolves `uses: ./` on its own
# filesystem rather than in the container the checkout populated (measured, run
# 34085102675).
#
# The action carries no files out of the worker pod, only its logs, so the report
# travels over the NAS both sides mount: the pod writes under /mnt/wl_nas and
# this shell reads the same bytes under /wl_nas.
#
# Reads:
#   ENTRY                     entry name, and what keeps eight parallel jobs'
#                             annotations apart on one run page -- the suite name
#                             would not, three entries share one
#   ANSWER_RESULTS_ON_RUNNER  the run's own report directory on the NAS
#
# Usage: bash scripts/ci/ppu/collect_answer_evidence.sh
set -euo pipefail

: "${ENTRY:?the entry name is required}"
: "${ANSWER_RESULTS_ON_RUNNER:?the NAS report directory is required}"

# The NAS copy is read, not moved. The pod writes the report as root and this
# shell is a different, non-root uid, so it can read those bytes but cannot
# unlink them: an rm here fails with EPERM on every file and turns this step red
# even when the suite passed, which is how the first run on this board reported
# itself. What is left behind is a few kilobytes under a directory this run owns
# exclusively, and it doubles as the on-cluster record once the artifact expires.
destination="${GITHUB_WORKSPACE}/ppu-answer-artifacts"
mkdir -p "${destination}"
if [ -d "${ANSWER_RESULTS_ON_RUNNER}" ]; then
  cp -a "${ANSWER_RESULTS_ON_RUNNER}/." "${destination}/"
else
  echo "::warning::the pod produced no Answer report at ${ANSWER_RESULTS_ON_RUNNER}"
fi

if [ -f "${destination}/summary.md" ]; then
  # Annotations are what a reader of this run page actually gets here. This
  # runner drives the job through a container hook that does not share the job
  # container's filesystem with the runner process, so bytes written to
  # GITHUB_STEP_SUMMARY inside the container are dropped without an error: the
  # check runs of the first two runs on this board both reported a summary of
  # length zero while their annotations arrived intact. Annotations travel over
  # this step's stdout, which does reach the runner, and they surface on the run
  # page and in the checks UI without downloading the artifact.
  #
  # The failing cases are read back from summary.md rather than from result.json
  # because this container is not guaranteed a JSON parser -- no python3, no jq.
  # render_summary emits one bullet per failing case prefixed with "- `", a
  # prefix no other line in that document has, and the evaluator's unit tests
  # lock that shape.
  while IFS= read -r failure; do
    echo "::error::${ENTRY}: ${failure#- }"
  done < <(grep '^- `' "${destination}/summary.md" || true)
  counts=$(grep '^- Cases: ' "${destination}/summary.md" | head -1 || true)
  if [ -n "${counts}" ]; then
    echo "::notice::${ENTRY}: ${counts#- }"
  fi
  cat "${destination}/summary.md" >> "${GITHUB_STEP_SUMMARY}"
fi
