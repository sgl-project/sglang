#!/bin/bash
# Carry one performance entry's report off the NAS onto the run page and into the
# artifact directory.
#
# Run by every entry of the three test-ppu-perf-*.yml workflows in the
# orchestration shell, held here for the same reason the board-side body is:
# those jobs cannot call a local composite action, because their runner resolves
# `uses: ./` on its own filesystem rather than in the container the checkout
# populated (measured, run 34085102675).
#
# The action carries no files out of the worker pod, only its logs, so the report
# travels over the NAS both sides mount: the pod writes under /mnt/wl_nas and this
# shell reads the same bytes under /wl_nas.
#
# What reaches the run page is the numbers themselves, as notices, one per
# measurement.  That is the whole delivery mechanism of this line: the suite
# judges nothing, so a green run whose numbers nobody can see would have measured
# nothing usable.  A measurement that produced no numbers becomes an error, and a
# measurement the evaluator flagged -- a prompt length that came back different
# from the one asked for, requests that failed -- becomes a warning.
#
# Reads:
#   ENTRY                   entry name, and what keeps parallel jobs'
#                           annotations apart on one run page
#   PERF_RESULTS_ON_RUNNER  the run's own report directory on the NAS
#
# Usage: bash scripts/ci/ppu/collect_perf_evidence.sh
set -euo pipefail

: "${ENTRY:?the entry name is required}"
: "${PERF_RESULTS_ON_RUNNER:?the NAS report directory is required}"

# The NAS copy is read, not moved. The pod writes the report as root and this
# shell is a different, non-root uid, so it can read those bytes but cannot
# unlink them: an rm here fails with EPERM on every file and turns this step red
# even when the suite passed, which is how the first Answer run on this board
# reported itself. What is left behind is a few kilobytes under a directory this
# run owns exclusively, and it doubles as the on-cluster record once the artifact
# expires.
destination="${GITHUB_WORKSPACE}/ppu-perf-artifacts"
mkdir -p "${destination}"
if [ -d "${PERF_RESULTS_ON_RUNNER}" ]; then
  cp -a "${PERF_RESULTS_ON_RUNNER}/." "${destination}/"
else
  echo "::warning::the pod produced no performance report at ${PERF_RESULTS_ON_RUNNER}"
fi

# A node's exit code is a workflow-level fact the report cannot carry: rank 0
# writes its verdict before it releases the workers, so a worker that then failed
# is visible only here and in its own log. The loop is a no-op for a single-board
# entry, which has no ranks directory.
for status_file in "${destination}"/ranks/rank-*.status; do
  [ -f "${status_file}" ] || continue
  node=$(basename "${status_file}" .status)
  code=$(cat "${status_file}" || echo "unreadable")
  if [ "${code}" = "0" ]; then
    echo "::notice::${ENTRY}: ${node} exited 0"
  else
    echo "::error::${ENTRY}: ${node} exited ${code}"
  fi
done

if [ -f "${destination}/summary.md" ]; then
  # Annotations are what a reader of this run page actually gets here. This
  # runner drives the job through a container hook that does not share the job
  # container's filesystem with the runner process, so bytes written to
  # GITHUB_STEP_SUMMARY inside the container are dropped without an error: the
  # check runs of the first two Answer runs on this board both reported a summary
  # of length zero while their annotations arrived intact. Annotations travel
  # over this step's stdout, which does reach the runner, and they surface on the
  # run page and in the checks UI without downloading the artifact.
  #
  # The three channels are read back from summary.md rather than from result.json
  # because this container is not guaranteed a JSON parser -- no python3, no jq.
  # render_summary prefixes a measured line with "- MEASURED ", an unmeasured one
  # with "- FAIL " and a flagged one with "- WARN ", no other line in that
  # document starts with any of them, and test_ppu_perf_eval_unit locks the
  # shape.
  while IFS= read -r measured; do
    echo "::notice::${ENTRY}: ${measured#- MEASURED }"
  done < <(grep '^- MEASURED ' "${destination}/summary.md" || true)
  while IFS= read -r failure; do
    echo "::error::${ENTRY}: ${failure#- FAIL }"
  done < <(grep '^- FAIL ' "${destination}/summary.md" || true)
  while IFS= read -r warning; do
    echo "::warning::${ENTRY}: ${warning#- WARN }"
  done < <(grep '^- WARN ' "${destination}/summary.md" || true)
  counts=$(grep '^- Measurements: ' "${destination}/summary.md" | head -1 || true)
  if [ -n "${counts}" ]; then
    echo "::notice::${ENTRY}: ${counts#- }"
  fi
  cat "${destination}/summary.md" >> "${GITHUB_STEP_SUMMARY}"
fi
